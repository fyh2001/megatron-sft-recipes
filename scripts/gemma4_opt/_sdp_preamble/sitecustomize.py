"""sitecustomize.py — auto-imported by every Python process started under
this directory's PYTHONPATH.  Pins SDPA backend + KV-expansion path for
gemma4 global-attention layers.

Why both knobs:
  1. gemma4 GLOBAL attn has head_dim=512 → modeling_gemma4.py patch falls back
     from FA2 to SDPA.  PyTorch's default SDPA dispatcher picks `math` backend
     for our shapes, which allocates O(N²) attn matrix = 20.5 GB at seq=16384
     → OOM on 80 GB cards.  Solution: only enable `mem_efficient` SDPA backend
     (CUTLASS, O(N) memory, supports head_dim=512 since PyTorch 2.5).
  2. transformers/integrations/sdpa_attention.py:`use_gqa_in_sdpa` returns True
     when `attention_mask is None and torch>=2.5`, which makes the wrapper pass
     `enable_gqa=True` to torch.nn.functional.scaled_dot_product_attention.
     PyTorch's `mem_efficient` backend does NOT support `enable_gqa=True` →
     dispatcher reports "Invalid backend" (verified by sdp_gqa_smoke.py).
     Solution: monkey-patch `use_gqa_in_sdpa` to always return False, forcing
     the wrapper to manually `repeat_kv`-expand KV before calling SDPA.  This
     is functionally equivalent (KV expansion is the historical default) and
     unlocks mem_efficient: 384 MB vs 20.5 GB peak attn alloc — 53× reduction.

Activated only when GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 is set, so the file is
harmless to leave on PYTHONPATH for non-gemma4 runs.

Usage:
    PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble:$PYTHONPATH \
    GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
    swift sft ...

Verified on PyTorch 2.10.0+cu129 / transformers 5.5.4. Numerical diff vs math:
max abs 1.56e-2, mean abs 3.06e-5 (bf16 atomics noise level).
"""
import os
import sys

if os.environ.get("GEMMA4_FORCE_MEM_EFFICIENT_SDP") == "1":
    _is_rank0 = os.environ.get("LOCAL_RANK", "0") == "0"

    # (1) SDPA backend pref — process-wide, persists for any later SDPA call.
    try:
        import torch
        # Pin TF32 state via new API consistently to avoid PyTorch 2.10 +
        # Inductor's "mixed legacy/new TF32 API" InductorError that breaks
        # torch.compile.  Setting precision="high" enables TF32 for matmul
        # (PyTorch 2.10 default for fp32 matmul); safe for bf16 training paths
        # which don't use cuBLAS fp32 matmul.  Verified via P6 retry.
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cuda.enable_flash_sdp(False)         # FA2 doesn't support head_dim=512
        torch.backends.cuda.enable_math_sdp(False)          # OOM trigger; O(N²) attn matrix
        torch.backends.cuda.enable_mem_efficient_sdp(True)  # CUTLASS, O(N) mem, head_dim=512 ok
        if _is_rank0:
            print(
                f"[gemma4 sdp_preamble] (1/3) backend prefs: "
                f"flash={torch.backends.cuda.flash_sdp_enabled()} "
                f"mem_eff={torch.backends.cuda.mem_efficient_sdp_enabled()} "
                f"math={torch.backends.cuda.math_sdp_enabled()}",
                file=sys.stderr, flush=True,
            )
    except Exception as _e:
        print(f"[gemma4 sdp_preamble] WARN: skip backend pref ({type(_e).__name__}: {_e})",
              file=sys.stderr, flush=True)

    # (2) Monkey-patch transformers GQA-in-SDPA detection to always pre-expand
    #     KV (avoids enable_gqa=True kwarg that mem_efficient backend rejects).
    try:
        from transformers.integrations import sdpa_attention as _sdpa_mod
        _orig_use_gqa = _sdpa_mod.use_gqa_in_sdpa
        def _force_no_gqa_kwarg(attention_mask, key):
            return False
        _sdpa_mod.use_gqa_in_sdpa = _force_no_gqa_kwarg
        if _is_rank0:
            print(
                "[gemma4 sdp_preamble] (2/3) patched transformers.integrations."
                "sdpa_attention.use_gqa_in_sdpa → always False (forces "
                "repeat_kv expansion, unblocks mem_efficient backend for GQA)",
                file=sys.stderr, flush=True,
            )
    except Exception as _e:
        print(f"[gemma4 sdp_preamble] WARN: skip GQA patch ({type(_e).__name__}: {_e})",
              file=sys.stderr, flush=True)

    # (5) Register gemma4 → _apply_liger_kernel_to_gemma4 in Liger's dispatch
    #     table.  Liger main (~v0.7.0) ships gemma/gemma2/gemma3 dispatches but
    #     NOT gemma4, so transformers' apply_liger_kernel(model, ...) silently
    #     no-ops for gemma4 ("There are currently no Liger kernels supported
    #     for model type: gemma4").  Our patch in
    #     scripts/benchmark/liger_gemma4_patch.py provides the dispatch:
    #     LigerRMSNorm (offset=0, casting=gemma) + LigerGEGLUMLP +
    #     LigerForCausalLMLoss (fused linear-CE).
    try:
        import sys as _sys
        # Make scripts/benchmark importable (where liger_gemma4_patch.py lives).
        _sb_path = "/home/ubuntu/fyh/megatron-sft-recipes/scripts/benchmark"
        if _sb_path not in _sys.path:
            _sys.path.insert(0, _sb_path)
        import liger_gemma4_patch
        if liger_gemma4_patch.register_gemma4_dispatch():
            if _is_rank0:
                print(
                    "[gemma4 sdp_preamble] (5/5) registered gemma4 → "
                    "_apply_liger_kernel_to_gemma4 in Liger's MODEL_TYPE_TO_APPLY_LIGER_FN "
                    "(unblocks --use_liger_kernel for gemma4: RMSNorm, GeGLU MLP, fused linear-CE)",
                    file=sys.stderr, flush=True,
                )
    except Exception as _e:
        # Don't fail Python startup if liger / swift not yet importable here.
        pass

    # (4) Force swift's Gemma4Template to declare support_padding_free=True.
    #     Default is None → falls back to `not is_multimodal` → False for VLM.
    #     For text-only training (freeze_vit=true, no images in dataset), the
    #     multimodal template's _encode() is functionally text-only (empty
    #     image/video/audio lists make processor a no-op).  Without this
    #     patch, --packing true / --padding_free true raise ValueError at
    #     swift/pipelines/train/sft.py:75.  Verified for sft-data/train.jsonl
    #     (text-only multi-turn).
    try:
        from swift.template.templates.gemma import Gemma4Template
        if Gemma4Template.support_padding_free in (None, False):
            Gemma4Template.support_padding_free = True
            if _is_rank0:
                print(
                    "[gemma4 sdp_preamble] (4/4) patched "
                    "swift.template.Gemma4Template.support_padding_free → True "
                    "(unblocks --packing / --padding_free for VLM template "
                    "in text-only training mode)",
                    file=sys.stderr, flush=True,
                )
    except Exception as _e:
        # Don't fail if swift not on path yet (sitecustomize runs early).
        pass

    # (3) Monkey-patch torch.distributed.init_process_group to use mixed
    #     backend (cpu:gloo + cuda:nccl) instead of the torchrun default
    #     ('nccl' only).  This is required when FSDP2 CPUOffloadPolicy puts
    #     gradients on CPU — DTensor.linalg.vector_norm in clip_grad_norm
    #     needs all_reduce on CPU tensors, which NCCL does not support.
    #     With gloo present, CPU collectives route through gloo and clip
    #     completes correctly.  PyTorch 2.6+ supports the device-typed
    #     backend syntax; verified on 2.10.0+cu129 (dist_backend_smoke.py).
    try:
        import torch.distributed as _dist
        _orig_init_pg = _dist.init_process_group
        def _patched_init_pg(*args, **kwargs):
            backend = kwargs.get("backend", args[0] if args else None)
            if backend in (None, "nccl"):
                # The default 'nccl' becomes 'cpu:gloo,cuda:nccl'.
                # This is harmless if grads stay on GPU (gloo idle),
                # and required when grads are offloaded to CPU.
                if args:
                    args = ("cpu:gloo,cuda:nccl",) + args[1:]
                else:
                    kwargs["backend"] = "cpu:gloo,cuda:nccl"
                if _is_rank0:
                    print(
                        "[gemma4 sdp_preamble] (3/3) intercepted "
                        "init_process_group: nccl → cpu:gloo,cuda:nccl "
                        "(enables clip_grad_norm with CPU-offloaded grads)",
                        file=sys.stderr, flush=True,
                    )
            return _orig_init_pg(*args, **kwargs)
        _dist.init_process_group = _patched_init_pg
    except Exception as _e:
        print(f"[gemma4 sdp_preamble] WARN: skip mixed-backend patch "
              f"({type(_e).__name__}: {_e})",
              file=sys.stderr, flush=True)
