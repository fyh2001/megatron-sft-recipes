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

    # (6) torchao adamw_torch_8bit / 4bit step bypass for FSDP2 + DTensor.
    #     torchao 0.17.0's `_AdamBase.step` unconditionally wraps the
    #     `single_param_adam` worker in `torch.compile(fullgraph=True,
    #     dynamic=False)`.  Inside that compile, dynamo's fake-tensor pipeline
    #     calls `aten.view(dtype=...)` on the `OptimState8bit` tensor subclass,
    #     which torchao hasn't lowered →
    #         torch._dynamo.exc.InternalTorchDynamoError: NotImplementedError:
    #         OptimState8bit dispatch: ...aten.view (overload='dtype')
    #     This blocks `--optim adamw_torch_8bit` on PyTorch 2.10 + FSDP2 (the
    #     same PR/issue area that breaks torchao + FSDP2 today).
    #
    #     We side-step it by replacing `_AdamBase.step` with a near-verbatim
    #     copy of the upstream method but **without** the `torch.compile`
    #     wrapper.  Per-step optim is a tiny fraction of wall (~ms vs seconds
    #     of fwd+bwd), so dropping compile-fusion has negligible throughput
    #     cost while unblocking 8-bit / 4-bit optim states on DTensor params.
    #
    #     Required to unlock the T1-A "drop FSDP_OFFLOAD" path for 25.8 B
    #     params: fp32 m,v costs 25.8 GB / rank, 8-bit costs ~6.5 GB → savings
    #     of 19 GB / rank.  See experiments/.../t1a_8bit_optim/run_*/FAILURE_ANALYSIS.md.
    try:
        from torchao.optim import adam as _ta_adam
        import torch as _t
        from torch import Tensor as _Tensor

        _orig_step_fn = _ta_adam._AdamBase.step

        @_t.no_grad()
        def _patched_step(self, closure=None):
            loss = None
            if closure is not None:
                with _t.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("Sparse gradient is not supported")
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = _t.tensor(0.0)
                        state["exp_avg"] = self._new_buffer(p, True)
                        state["exp_avg_sq"] = self._new_buffer(p, False)
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = self._new_buffer(p, False)
                    state["step"] += 1
                    if not isinstance(group["lr"], _Tensor):
                        raise RuntimeError(
                            "lr was changed to a non-Tensor object. If you want "
                            "to update lr, please use optim.param_groups[0]['lr'].fill_(new_lr)"
                        )
                    # --- the only difference from upstream: no torch.compile() ---
                    _ta_adam.single_param_adam(
                        p.detach(),
                        grad,
                        state["step"],
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state.get("max_exp_avg_sq", None),
                        group["lr"],
                        group["betas"][0],
                        group["betas"][1],
                        group["weight_decay"],
                        group["eps"],
                        self.is_adamw,
                        self.bf16_stochastic_round and p.dtype is _t.bfloat16,
                    )
            return loss

        _ta_adam._AdamBase.step = _patched_step
        if _is_rank0:
            print(
                "[gemma4 sdp_preamble] (6/6) patched torchao._AdamBase.step "
                "→ single_param_adam call without torch.compile() "
                "(unblocks adamw_torch_8bit / 4bit on FSDP2+DTensor for PyTorch 2.10)",
                file=sys.stderr, flush=True,
            )
    except ImportError:
        # torchao not installed; the patch is only relevant when --optim is
        # adamw_torch_8bit / 4bit, so a missing torchao means we're on a
        # different optim path and there's nothing to do.
        pass
    except Exception as _e:
        print(f"[gemma4 sdp_preamble] WARN: skip torchao step patch "
              f"({type(_e).__name__}: {_e})",
              file=sys.stderr, flush=True)

    # (7) accelerate ParallelismConfig injection from env vars.
    #     transformers 5.5.4 TrainingArguments has a `parallelism_config` field
    #     (default None) that, when set to a `accelerate.ParallelismConfig`,
    #     causes `Trainer._build_accelerator_args` to forward it to
    #     `Accelerator(parallelism_config=...)`, unlocking joint cp×dp_shard×tp
    #     sharding via PyTorch DeviceMesh.  ms-swift's CLI parser does not
    #     accept `--parallelism_config` (it's a complex object, not a JSON
    #     scalar), so we inject it via env vars during TrainingArguments
    #     `__post_init__`, which is called after CLI parse but before the
    #     Accelerator is constructed.
    #
    #     Env vars (all integer; absent / 0 means "don't set this dim"):
    #         ACCELERATE_CP_SIZE          : context-parallel size
    #         ACCELERATE_TP_SIZE          : tensor-parallel size
    #         ACCELERATE_DP_SHARD_SIZE    : FSDP shard size (model split)
    #         ACCELERATE_DP_REPLICATE_SIZE: pure DDP replica size
    #         ACCELERATE_CP_BACKEND       : "allgather" | "alltoall" (default allgather)
    #
    #     Critical preconditions checked elsewhere in this file / by transformers:
    #         - FSDP version 2 (transformers passes `--fsdp` config)
    #         - attn_impl == "sdpa" (Trainer raises if cp_enabled and not sdpa)
    #         - causal-only attention mask (Trainer validates per-step)
    #
    #     See `experiments/gemma4_opt/t1a_8bit_optim/run_*/FAILURE_ANALYSIS_v2*.md`
    #     for the bnb/torchao dead end that motivates this pivot.  The win:
    #     `ParallelismConfig(cp_size=2, dp_shard_size=4)` shards the model
    #     across the joint cp×dp_shard mesh (8 ranks), where swift Ulysses SP=2
    #     only sharded across DP=4 ranks.  This halves param/grad/optim memory
    #     per rank without any 8-bit optim-state plumbing — same fp32 AdamW.
    try:
        _cp = int(os.environ.get("ACCELERATE_CP_SIZE", "0") or "0")
        _tp = int(os.environ.get("ACCELERATE_TP_SIZE", "0") or "0")
        _dp_shard = int(os.environ.get("ACCELERATE_DP_SHARD_SIZE", "0") or "0")
        _dp_repl = int(os.environ.get("ACCELERATE_DP_REPLICATE_SIZE", "0") or "0")
        _cp_back = os.environ.get("ACCELERATE_CP_BACKEND", "")
        # Skip heavy `from transformers import TrainingArguments` import on
        # cold start unless the user explicitly opted into ParallelismConfig
        # via env vars. transformers' top-level import is several seconds of
        # work on this stack; defer it whenever possible.
        if (_cp > 1 or _tp > 1 or _dp_shard > 1 or _dp_repl > 1):
            # Defer `from accelerate import ...` to the patched function body —
            # importing accelerate at sitecustomize top-level can recurse via
            # subprocesses that re-execute sitecustomize before the import
            # finishes (observed as a `pip list` fork-bomb on this stack).
            from transformers import TrainingArguments

            _orig_post_init = TrainingArguments.__post_init__

            def _patched_post_init(self):
                _orig_post_init(self)
                if self.parallelism_config is None:
                    from accelerate import ParallelismConfig
                    from accelerate.utils import TorchContextParallelConfig
                    kwargs = {}
                    if _cp > 1:
                        kwargs["cp_size"] = _cp
                        if _cp_back:
                            kwargs["cp_handler"] = TorchContextParallelConfig(
                                cp_comm_strategy=_cp_back
                            )
                    if _tp > 1:
                        kwargs["tp_size"] = _tp
                    if _dp_shard > 1:
                        kwargs["dp_shard_size"] = _dp_shard
                    if _dp_repl > 1:
                        kwargs["dp_replicate_size"] = _dp_repl
                    self.parallelism_config = ParallelismConfig(**kwargs)
                    if _is_rank0:
                        print(
                            f"[gemma4 sdp_preamble] (7/8) injected "
                            f"ParallelismConfig({kwargs}) into TrainingArguments "
                            f"(unlocks accelerate cp×dp_shard×tp joint mesh)",
                            file=sys.stderr, flush=True,
                        )

            TrainingArguments.__post_init__ = _patched_post_init
    except Exception as _e:
        print(f"[gemma4 sdp_preamble] WARN: skip ParallelismConfig injection "
              f"({type(_e).__name__}: {_e})",
              file=sys.stderr, flush=True)

    # (8) ms-swift Seq2SeqTrainer.compute_loss does NOT call into
    #     `Trainer._prepare_context_parallel_inputs` (it overrides
    #     compute_loss entirely and bypasses the parent's
    #     `accelerator.maybe_context_parallel(...)` wrapper).  Patch
    #     `Seq2SeqTrainer.compute_loss` to wrap with the same context manager
    #     `Trainer.compute_loss` would have used.
    #
    #     IMPORTANT — fork-bomb avoidance:
    #     Importing `swift.trainers.seq2seq_trainer` inside sitecustomize
    #     itself triggers a `pip list` subprocess somewhere in swift's lazy
    #     dependency-detection chain. That subprocess re-runs sitecustomize,
    #     which re-imports swift, which spawns another `pip list`... → kernel
    #     gets DDoS'd in seconds.  We therefore install a hook on
    #     `transformers.Trainer.__init__`: only when a real Trainer is being
    #     constructed (in the main `swift sft` process, not in a `pip list`
    #     subprocess) do we import swift and apply the wrap. Idempotent via a
    #     class-level flag.
    if (_cp > 1 or _tp > 1 or _dp_shard > 1 or _dp_repl > 1):
        try:
            import transformers as _tfm
            import functools as _ft
            _orig_T_init = _tfm.Trainer.__init__

            # CRITICAL: preserve the original __init__ signature.  ms-swift's
            # SwiftMixin.__init__ does
            #     params = inspect.signature(HfTrainer.__init__).parameters
            #     key = 'processing_class' if 'processing_class' in params else 'tokenizer'
            # to pick between the new (transformers ≥4.46) and old (≤4.45)
            # tokenizer kwarg name.  If our patch erases the signature
            # (replacing with *args/**kwargs), swift falls back to the old
            # 'tokenizer=' kwarg, and the real Trainer.__init__ raises
            # TypeError: unexpected kwarg.  functools.wraps copies the
            # original __signature__/__wrapped__, keeping inspect.signature()
            # accurate.
            @_ft.wraps(_orig_T_init)
            def _patched_T_init(self, *args, **kwargs):
                _orig_T_init(self, *args, **kwargs)

                # (10) Force fixed-length padding so accelerate CP × SDPA
                #      mem_efficient ring attention chunks are aligned.
                #      Only when CP is opted in.
                if _cp > 1 and not getattr(self, "_gemma4_cp_pad_wrapped", False):
                    try:
                        if hasattr(self, "template") and getattr(self.template, "max_length", None):
                            _max_len = self.template.max_length
                            _inner = self.data_collator
                            def _padded_collate(batch, _i=_inner, _ml=_max_len):
                                try:
                                    return _i(batch, padding_to=_ml)
                                except TypeError:
                                    return _i(batch)
                            self.data_collator = _padded_collate
                            self._gemma4_cp_pad_wrapped = True
                            if _is_rank0:
                                print(
                                    f"[gemma4 sdp_preamble] (10/10) wrapped "
                                    f"swift data_collator with padding_to="
                                    f"max_length={_max_len} (CP × SDPA "
                                    f"mem_efficient alignment requirement)",
                                    file=sys.stderr, flush=True,
                                )
                    except Exception as _ce:
                        print(f"[gemma4 sdp_preamble] WARN: (10) collator pad "
                              f"wrap failed ({type(_ce).__name__}: {_ce})",
                              file=sys.stderr, flush=True)

                # (8) Try-once: as soon as a Trainer is instantiated, locate
                # swift.Seq2SeqTrainer (now loadable since swift's CLI is
                # already running) and wrap its compute_loss.
                # Also (11): under CP, force use_logits_to_keep=False because
                # swift would otherwise inject an [original_seq_len] mask that
                # mismatches CP-sharded hidden_states [B, seq/cp, H] in lm_head.
                try:
                    from swift.trainers.seq2seq_trainer import (
                        Seq2SeqTrainer as _SwSeq2Seq,
                    )
                    from swift.trainers.mixin import SwiftMixin as _SwMixin
                    if _cp > 1 and not getattr(_SwMixin, "_gemma4_cp_force_no_logits_keep", False):
                        def _patched_get_use_logits(self, default_value: bool = True):
                            self.args.use_logits_to_keep = False
                            return False
                        _SwMixin.get_use_logits_to_keep = _patched_get_use_logits
                        _SwMixin._gemma4_cp_force_no_logits_keep = True
                        if _is_rank0:
                            print(
                                "[gemma4 sdp_preamble] (11/11) forced "
                                "SwiftMixin.get_use_logits_to_keep → False "
                                "(prevents lm_head IndexError when CP shards "
                                "hidden_states but logits_to_keep mask is full)",
                                file=sys.stderr, flush=True,
                            )
                    if not getattr(_SwSeq2Seq, "_gemma4_cp_wrapped", False):
                        _orig_compute_loss = _SwSeq2Seq.compute_loss
                        import torch as _t_for_loss

                        def _patched_compute_loss(s, model, inputs, *a, **kw):
                            cp_ctx, inputs = s._prepare_context_parallel_inputs(
                                model, inputs
                            )
                            with cp_ctx():
                                loss = _orig_compute_loss(s, model, inputs, *a, **kw)
                            # (12) CP-shard rescue: when all valid labels on
                            # this rank fall outside the rank's seq-shard
                            # (common for SFT short samples padded to 16k and
                            # split by HeadTailLoadBalancer), modeling_gemma4
                            # returns `flat_logits.new_zeros(())` — a leaf
                            # scalar 0 with no grad_fn. Backward then raises
                            # "element 0 ... does not require grad". We
                            # synthesize a grad-connected zero from any
                            # trainable parameter so backward can walk the
                            # graph (this rank contributes 0 to gradients,
                            # which is the correct mathematical behavior for
                            # an all-padding shard).
                            if (_t_for_loss.is_tensor(loss)
                                    and not loss.requires_grad):
                                for _p in model.parameters():
                                    if _p.requires_grad:
                                        loss = _p.sum() * 0.0
                                        break
                            return loss

                        _SwSeq2Seq.compute_loss = _patched_compute_loss
                        _SwSeq2Seq._gemma4_cp_wrapped = True
                        if _is_rank0:
                            print(
                                "[gemma4 sdp_preamble] (8/8) wrapped "
                                "swift.Seq2SeqTrainer.compute_loss with "
                                "Trainer._prepare_context_parallel_inputs / "
                                "accelerator.maybe_context_parallel "
                                "(no-op when cp_size=1)",
                                file=sys.stderr, flush=True,
                            )
                except ImportError:
                    pass

            _tfm.Trainer.__init__ = _patched_T_init
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip swift compute_loss CP "
                  f"wrap setup ({type(_e).__name__}: {_e})",
                  file=sys.stderr, flush=True)

    # (9) Disable PyTorch CP head-tail load balancer when ParallelismConfig
    #     CP is opted in.  The default _HeadTailLoadBalancer enforces
    #         assert seq_length % (cp_world_size * 2) == 0
    #     in `_load_balancer.py:148`.  swift's batch collator uses dynamic
    #     padding to the per-batch max length, which is rarely a multiple of
    #     `cp_world_size * 2`. We could enable `--packing true` to force
    #     fixed-length seqs, but swift's packing requires flash attention,
    #     while accelerate CP requires SDPA — incompatible.
    #
    #     The load balancer is only useful for *non-causal* attention masks
    #     (it rearranges seq positions to balance computation over ranks).
    #     For SFT we always have causal masks, so disabling it is functionally
    #     a no-op for the math; we lose only a small comm/compute interleave
    #     optimization. Confirmed by reading PyTorch
    #     `_attention.py:397/444/496` and `_load_balancer.py:148`.
    # Note on (9): we KEEP _cp_options.enable_load_balance = True (default).
    # The HeadTailLoadBalancer is what redistributes valid tokens across CP
    # ranks; without it, when batches are heavily right-padded (most of our
    # SFT samples are ~1k–4k tokens but max_length=16384), the rank holding
    # the back half of the seq receives only padding tokens (labels=-100
    # everywhere), so its loss has no grad_fn and backward raises
    # "element 0 ... does not require grad".  The seq-divisibility assertion
    # the load balancer enforces is satisfied because (10) forces every batch
    # to pad to template.max_length=16384, and 16384 % (cp_size*2) == 0 for
    # cp_size in {1, 2, 4, 8}.
    if False:  # kept for documentation; re-enable only if a CP run hits the
               # `assert seq_length % (cp_size * 2) == 0` AssertionError again
        try:
            from torch.distributed.tensor.experimental._context_parallel._attention import (
                _cp_options as _cp_opt,
            )
            if _cp_opt.enable_load_balance:
                _cp_opt.enable_load_balance = False
                if _is_rank0:
                    print(
                        "[gemma4 sdp_preamble] (9/9) disabled PyTorch CP "
                        "_cp_options.enable_load_balance",
                        file=sys.stderr, flush=True,
                    )
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip load_balancer disable "
                  f"({type(_e).__name__}: {_e})",
                  file=sys.stderr, flush=True)

    # (12) Profiler: opt-in PyTorch profiler around 1+warmup+active steps.
    #      Triggered by GEMMA4_PROFILE=1 (see _profile_callback.py for env vars).
    #      Lazy hook on Trainer.__init__ so swift `inspect.signature(...)` keeps
    #      working (we re-attach via add_callback after the parent __init__).
    if os.environ.get("GEMMA4_PROFILE") == "1":
        try:
            import transformers as _tfm_p
            import functools as _ft_p
            _orig_T_init_p = _tfm_p.Trainer.__init__

            @_ft_p.wraps(_orig_T_init_p)
            def _patched_T_init_for_profile(self, *args, **kwargs):
                _orig_T_init_p(self, *args, **kwargs)
                if not getattr(self, "_gemma4_profile_attached", False):
                    try:
                        import sys as _sys_p
                        _gemma4_opt_path = (
                            "/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt"
                        )
                        if _gemma4_opt_path not in _sys_p.path:
                            _sys_p.path.insert(0, _gemma4_opt_path)
                        from _profile_callback import Gemma4ProfilerCallback
                        self.add_callback(Gemma4ProfilerCallback())
                        self._gemma4_profile_attached = True
                        if _is_rank0:
                            print(
                                "[gemma4 sdp_preamble] (12/12) attached "
                                "Gemma4ProfilerCallback to Trainer "
                                "(GEMMA4_PROFILE=1)",
                                file=sys.stderr, flush=True,
                            )
                    except Exception as _ce:
                        print(f"[gemma4 sdp_preamble] WARN: (12) profile "
                              f"callback attach failed "
                              f"({type(_ce).__name__}: {_ce})",
                              file=sys.stderr, flush=True)

            _tfm_p.Trainer.__init__ = _patched_T_init_for_profile
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip profile patch setup "
                  f"({type(_e).__name__}: {_e})",
                  file=sys.stderr, flush=True)

    # (14) Short-run benchmark stopper that preserves the full-run scheduler.
    #
    # `--max_steps 50` changes Trainer's total training steps to 50, so cosine
    # LR/warmup no longer matches the real 2-epoch (806-step) run.  For loss
    # comparisons we need the scheduler to be built from `num_train_epochs=2`
    # while stopping after N optimizer steps.  This callback does exactly that.
    _stop_after = int(os.environ.get("GEMMA4_STOP_AFTER_STEPS", "0") or "0")
    if _stop_after > 0:
        try:
            import transformers as _tfm_stop
            import functools as _ft_stop

            _orig_T_init_stop = _tfm_stop.Trainer.__init__

            class _Gemma4StopAfterStepsCallback(_tfm_stop.TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    if state.global_step >= _stop_after:
                        control.should_training_stop = True
                        control.should_save = False
                    return control

            @_ft_stop.wraps(_orig_T_init_stop)
            def _patched_T_init_for_stop(self, *args, **kwargs):
                _orig_T_init_stop(self, *args, **kwargs)
                if not getattr(self, "_gemma4_stop_after_attached", False):
                    self.add_callback(_Gemma4StopAfterStepsCallback())
                    self._gemma4_stop_after_attached = True
                    if _is_rank0:
                        print(
                            f"[gemma4 sdp_preamble] (14/14) attached "
                            f"StopAfterStepsCallback(stop_after={_stop_after}) "
                            f"(keeps full-run scheduler, stops benchmark early)",
                            file=sys.stderr, flush=True,
                        )

            _tfm_stop.Trainer.__init__ = _patched_T_init_for_stop
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip stop-after-steps setup "
                  f"({type(_e).__name__}: {_e})",
                  file=sys.stderr, flush=True)

    # (15) Force FSDP2 MixedPrecisionPolicy.reduce_dtype = float32.
    #
    # accelerate's FullyShardedDataParallelPlugin defaults to
    # reduce_dtype = param_dtype = bf16 (when --bf16 is on).  Gradient
    # reduce-scatter then happens in bf16, which causes a systematic
    # ~1.29x over-estimate of grad_norm vs DeepSpeed ZeRO-3 native (which
    # reduces gradients in fp32 master).  With max_grad_norm=1.0 the
    # downstream effect is that FSDP2 effectively scales the optimizer
    # update ~22% smaller than DS3, so loss converges slower.
    #
    # Activated only when GEMMA4_FSDP_REDUCE_FP32=1.
    # (16) Monkeypatch torch FSDP2 `FSDPParam.to_accumulated_grad_if_needed`
    # to fix a PyTorch 2.10 typo: it references `self._unsharded_param` which
    # is freed after `free_unsharded_param()`; the actual public accessor
    # right next door (`accumulate_unsharded_grad_if_needed`, same file) uses
    # `self.unsharded_param`.  Bug only triggers when reduce_dtype != param_dtype
    # (i.e. our reduce_fp32 path).  Activated unconditionally; safe because the
    # patched body matches the existing `accumulate_unsharded_grad_if_needed`
    # accessor pattern.
    try:
        from torch.distributed.fsdp._fully_shard import _fsdp_param as _tfp
        if not getattr(_tfp, "_fyh_to_accum_grad_typo_patched", False):
            def _patched_to_accum(self):
                # Access `_unsharded_param` directly (not the property) to
                # bypass the bug where the property raises AttributeError
                # when the attribute has been freed by free_unsharded_param().
                # If `_unsharded_param` doesn't exist, it means we are in a
                # state where there's no grad to accumulate (typically the
                # last reshard already happened).  Skip silently.
                if (
                    self.reduce_dtype is None
                    or not hasattr(self, "_unsharded_param")
                ):
                    return
                grad = self._unsharded_param.grad
                if grad is None or grad.dtype == self.reduce_dtype:
                    return
                self._unsharded_param.grad = None
                self.unsharded_accumulated_grad = grad.to(self.reduce_dtype)
            _tfp.FSDPParam.to_accumulated_grad_if_needed = _patched_to_accum
            _tfp._fyh_to_accum_grad_typo_patched = True
            if _is_rank0:
                print(
                    "[gemma4 sdp_preamble] (16/16) patched torch FSDP2 "
                    "FSDPParam.to_accumulated_grad_if_needed: guard "
                    "missing _unsharded_param attr (PyTorch 2.10 state-"
                    "machine bug; only triggers when reduce_dtype != "
                    "param_dtype)",
                    file=sys.stderr, flush=True,
                )
    except Exception as _e:
        print(f"[gemma4 sdp_preamble] WARN: skip FSDP2 to_accumulated_grad "
              f"typo patch ({type(_e).__name__}: {_e})",
              file=sys.stderr, flush=True)

    # (17) Replace accelerate's FSDP2 grad-clip path with a DS3-style fp64
    # norm computation.  Root cause of the systematic ~1.29× grad_norm
    # offset vs DS ZeRO-3 native (verified empirically across offload/no-
    # offload/A3-on/A3-off):
    #
    # accelerate's FSDP2 branch calls `torch.nn.utils.clip_grad_norm_`,
    # which calls `torch._foreach_norm(grads, p)`. When grads are bf16
    # (FSDP2 default reduce-scatter output), `torch._foreach_norm` returns
    # bf16 norms — squaring tiny grad values then summing in bf16 has
    # ~7-bit mantissa precision, and rounding-to-nearest-even biases the
    # sum upward.  Across 623 trainable tensors of gemma-4-E4B, the
    # accumulated bf16 sum-of-squares overestimates the true norm by ~29%.
    #
    # DS3 (deepspeed/runtime/zero/stage3.py:get_grad_norm_direct, line
    # 2129) explicitly upcasts each grad shard to fp64 before computing
    # `.norm(2)`, then SUMs squares across DP via all_reduce, then sqrt.
    # fp64 has 52-bit mantissa → essentially zero rounding error → the
    # canonical L2 norm.
    #
    # This patch reimplements exactly that path for FSDP2 by replacing
    # `accelerate.Accelerator.clip_grad_norm_`.  Activated only when
    # GEMMA4_GRAD_NORM_FP64=1 to keep the default codepath untouched.
    if os.environ.get("GEMMA4_GRAD_NORM_FP64") == "1":
        try:
            from accelerate import accelerator as _acc_mod
            from accelerate.utils import DistributedType as _DT
            import torch as _t_gn
            if not getattr(_acc_mod, "_fyh_fsdp2_fp64_norm_patched", False):
                _orig_clip = _acc_mod.Accelerator.clip_grad_norm_

                def _fp64_clip_grad_norm_(self, parameters, max_norm, norm_type=2):
                    # Only divert for FSDP2; everything else passes through.
                    if (
                        self.distributed_type != _DT.FSDP
                        or not getattr(self, "is_fsdp2", False)
                        or norm_type != 2
                    ):
                        return _orig_clip(self, parameters, max_norm, norm_type)
                    self.unscale_gradients()
                    parameters = list(parameters)
                    grads = [p.grad for p in parameters if p.grad is not None]
                    if not grads:
                        return _t_gn.tensor(0.0, dtype=_t_gn.float32)
                    # Upcast each shard's local tensor to fp64 and compute
                    # sum-of-squares.  DTensor.to_local() returns just this
                    # rank's shard; later we all-reduce SUM across DP.
                    try:
                        from torch.distributed.tensor import DTensor as _DT_t
                    except ImportError:
                        _DT_t = None
                    sum_sq = _t_gn.zeros((), dtype=_t_gn.float64, device=grads[0].device)
                    for g in grads:
                        local = g.to_local() if (_DT_t is not None and isinstance(g, _DT_t)) else g
                        local_fp64 = local.detach().to(_t_gn.float64)
                        sum_sq = sum_sq + (local_fp64 * local_fp64).sum()
                    if _t_gn.distributed.is_initialized():
                        _t_gn.distributed.all_reduce(sum_sq, op=_t_gn.distributed.ReduceOp.SUM)
                    total_norm = sum_sq.sqrt()
                    # Debug: print first few invocations from rank 0
                    if (os.environ.get("GEMMA4_GRAD_NORM_FP64_DEBUG") == "1"
                            and _is_rank0):
                        n_dtensor = sum(1 for g in grads
                                        if (_DT_t is not None and isinstance(g, _DT_t)))
                        sample_dtype = grads[0].dtype if grads else None
                        sample_local_dtype = (grads[0].to_local().dtype
                                              if (_DT_t is not None and isinstance(grads[0], _DT_t))
                                              else (grads[0].dtype if grads else None))
                        print(f"[gemma4 fp64-norm-dbg] n_grads={len(grads)} "
                              f"n_dtensor={n_dtensor} grad_dtype={sample_dtype} "
                              f"local_dtype={sample_local_dtype} "
                              f"sum_sq={sum_sq.item():.6f} "
                              f"total_norm_fp64={total_norm.item():.6f}",
                              file=sys.stderr, flush=True)
                    # Apply clip in-place
                    clip_coef = max_norm / (total_norm + 1e-6)
                    clip_coef = _t_gn.clamp(clip_coef, max=1.0)
                    for p in parameters:
                        if p.grad is None:
                            continue
                        # multiply DTensor or normal tensor by scalar — both work
                        p.grad.mul_(clip_coef.to(p.grad.dtype))
                    # Return fp32 for downstream HF Trainer logging
                    return total_norm.to(_t_gn.float32)

                _acc_mod.Accelerator.clip_grad_norm_ = _fp64_clip_grad_norm_
                _acc_mod._fyh_fsdp2_fp64_norm_patched = True
                if _is_rank0:
                    print(
                        "[gemma4 sdp_preamble] (17/17) patched "
                        "Accelerator.clip_grad_norm_ for FSDP2: compute "
                        "L2 norm in fp64 with explicit DP all-reduce SUM "
                        "(matches DS ZeRO-3 get_grad_norm_direct path; "
                        "fixes ~1.29x systematic over-estimate caused by "
                        "torch._foreach_norm running in bf16)",
                        file=sys.stderr, flush=True,
                    )
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip FSDP2 fp64 grad-norm "
                  f"patch ({type(_e).__name__}: {_e})",
                  file=sys.stderr, flush=True)

    # (22) Per-param grad-dump for FSDP2 vs DS3 layer-by-layer numerical
    # comparison.  Registers `register_hook` on every trainable param: hook
    # fires DURING autograd backward with the raw per-rank full unsharded
    # bf16 grad (BEFORE any FSDP2 reduce-scatter or DS3 reduce/partition
    # intervention).  Computes sum_sq in fp64 and appends to a TSV log:
    #   `step \t micro \t param_name \t sum_sq_fp64`
    #
    # Activated only when GEMMA4_GRAD_DUMP=1.  Output file: env
    # GEMMA4_GRAD_DUMP_DIR (default /tmp/grad_dump), one file per rank,
    # named `<engine>_rank<R>.tsv`.  Stops logging after
    # GEMMA4_GRAD_DUMP_MAX_STEPS (default 1) full steps.
    if os.environ.get("GEMMA4_GRAD_DUMP") == "1":
        try:
            import transformers as _tfm_gd
            import functools as _ft_gd
            import os as _os_gd
            import torch as _torch_gd

            _GD_DIR = _os_gd.environ.get("GEMMA4_GRAD_DUMP_DIR", "/tmp/grad_dump")
            try:
                _os_gd.makedirs(_GD_DIR, exist_ok=True)
            except Exception:
                pass

            _GD_MAX_STEPS = int(_os_gd.environ.get("GEMMA4_GRAD_DUMP_MAX_STEPS", "1"))
            _GD_MAX_MICROS = int(_os_gd.environ.get("GEMMA4_GRAD_DUMP_MAX_MICROS", "16"))

            # State (single Trainer per process; safe to use module-level)
            _GD_STATE = {
                "step": 0,
                "micro": 0,
                "fh": None,
                "engine": "?",
                "registered": False,
                "rank": int(_os_gd.environ.get("RANK", _os_gd.environ.get("LOCAL_RANK", "0"))),
            }

            # Raw-grad dump (for cos-sim cross-engine analysis); also used by DS3 hook below
            _RAW_PREFIXES_GLOBAL = _os_gd.environ.get(
                "GEMMA4_GRAD_DUMP_RAW_PREFIXES",
                "model.language_model.layers.22.self_attn.k_proj,"
                "model.language_model.layers.22.self_attn.v_proj,"
                "model.language_model.layers.22.mlp.down_proj,"
                "model.language_model.layers.0.self_attn.q_proj,"
                "model.language_model.layers.41.self_attn.q_proj"
            ).split(",")

            def _gd_make_hook(name):
                def _hook(grad):
                    if _GD_STATE["step"] >= _GD_MAX_STEPS:
                        return None
                    if _GD_STATE["fh"] is None:
                        return None
                    try:
                        sum_sq = float((grad.detach().double() ** 2).sum().item())
                    except Exception:
                        return None
                    _GD_STATE["fh"].write(
                        f"{_GD_STATE['step']}\t{_GD_STATE['micro']}\t{name}\t{sum_sq:.10e}\n"
                    )
                    # Raw grad dump for selected params (DS3 path)
                    try:
                        for prefix in _RAW_PREFIXES_GLOBAL:
                            if name.startswith(prefix.strip()):
                                fname = (f"raw_{_GD_STATE['rank']:02d}"
                                         f"_step{_GD_STATE['step']:02d}"
                                         f"_micro{_GD_STATE['micro']:02d}"
                                         f"_{prefix.strip().replace('.', '_').replace('/','_')}.pt")
                                fpath = _os_gd.path.join(_GD_DIR, fname)
                                _torch_gd.save(grad.detach().cpu(), fpath)
                                break
                    except Exception:
                        pass
                    return None
                return _hook

            _orig_T_init_gd = _tfm_gd.Trainer.__init__

            @_ft_gd.wraps(_orig_T_init_gd)
            def _patched_T_init_gd(self, *args, **kwargs):
                _orig_T_init_gd(self, *args, **kwargs)
                if _GD_STATE["registered"]:
                    return
                # Detect engine
                try:
                    from accelerate.utils import DistributedType as _DT_gd
                    if self.accelerator.distributed_type == _DT_gd.DEEPSPEED:
                        _GD_STATE["engine"] = "ds3"
                    elif self.accelerator.distributed_type == _DT_gd.FSDP:
                        _GD_STATE["engine"] = ("fsdp2" if getattr(self.accelerator, "is_fsdp2", False) else "fsdp1")
                    else:
                        _GD_STATE["engine"] = str(self.accelerator.distributed_type).lower()
                except Exception:
                    _GD_STATE["engine"] = "unknown"

                fname = f"{_GD_STATE['engine']}_rank{_GD_STATE['rank']}.tsv"
                fpath = _os_gd.path.join(_GD_DIR, fname)
                _GD_STATE["fh"] = open(fpath, "w", buffering=1)
                _GD_STATE["fh"].write("step\tmicro\tparam\tsum_sq_fp64\n")
                if _is_rank0:
                    print(f"[gemma4 grad-dump] writing to {fpath} (engine={_GD_STATE['engine']}, "
                          f"max_steps={_GD_MAX_STEPS}, max_micros_per_step={_GD_MAX_MICROS})",
                          file=sys.stderr, flush=True)

                # Cache model ref for dynamic id-to-name lookups.  We need
                # both id(param) -> fqn and id(module) -> fqn since FSDP2
                # ParamModuleInfo gives us (module, param_name) tuples.
                _GD_STATE["model_ref"] = self.model
                _GD_STATE["id2name"] = None  # param id -> full fqn
                _GD_STATE["mod_id2name"] = None  # module id -> module fqn

                def _gd_rebuild_maps():
                    m = _GD_STATE.get("model_ref")
                    if m is None:
                        return {}, {}
                    p_map = {id(p): name for name, p in m.named_parameters() if p.requires_grad}
                    mod_map = {id(mod): name for name, mod in m.named_modules()}
                    return p_map, mod_map

                def _gd_get_param_name(fp):
                    if _GD_STATE["id2name"] is None or _GD_STATE["mod_id2name"] is None:
                        p_map, mod_map = _gd_rebuild_maps()
                        _GD_STATE["id2name"] = p_map
                        _GD_STATE["mod_id2name"] = mod_map
                    # Try via FSDPParam._module_info (most reliable for FSDP2)
                    mi = getattr(fp, "_module_info", None)
                    if mi is not None:
                        mod_fqn = _GD_STATE["mod_id2name"].get(id(mi.module), "?")
                        return f"{mod_fqn}.{mi.param_name}" if mod_fqn else mi.param_name
                    # Fallback: try sharded_param id
                    sp = getattr(fp, "sharded_param", None)
                    if sp is not None:
                        return _GD_STATE["id2name"].get(id(sp), f"unknown_{id(fp):x}")
                    return f"unknown_{id(fp):x}"
                _GD_STATE["get_param_name_fn"] = _gd_get_param_name

                # For DS3: register_hook on each trainable param works
                # because autograd grads land on user-facing params.
                # For FSDP2: sharded params are not on the autograd graph
                # (the unsharded copies are), so register_hook never fires.
                # → patch FSDP2's `foreach_reduce` to log unsharded grads
                # right before reduce-scatter.
                n_hooks = 0
                if _GD_STATE["engine"] == "ds3":
                    for name, p in self.model.named_parameters():
                        if p.requires_grad:
                            p.register_hook(_gd_make_hook(name))
                            n_hooks += 1
                elif _GD_STATE["engine"].startswith("fsdp2"):
                    try:
                        from torch.distributed.fsdp._fully_shard import _fsdp_param_group as _fpg_gd
                        if not getattr(_fpg_gd, "_fyh_grad_dump_patched", False):
                            _orig_fr_gd = _fpg_gd.foreach_reduce
                            import inspect as _insp_gd
                            _sig = _insp_gd.signature(_orig_fr_gd)
                            _params_list = list(_sig.parameters.keys())

                            def _patched_fr_gd(*args, **kwargs):
                                # Collect (fsdp_params, unsharded_grads) by index/keyword
                                if "fsdp_params" in kwargs:
                                    _fp = kwargs["fsdp_params"]
                                else:
                                    _fp = args[_params_list.index("fsdp_params")]
                                if "unsharded_grads" in kwargs:
                                    _ug = kwargs["unsharded_grads"]
                                else:
                                    _ug = args[_params_list.index("unsharded_grads")]
                                # Log per param
                                if (_GD_STATE["fh"] is not None
                                    and _GD_STATE["step"] < _GD_MAX_STEPS):
                                    get_name = _GD_STATE["get_param_name_fn"]
                                    for fp, g in zip(_fp, _ug):
                                        try:
                                            name = get_name(fp)
                                            sum_sq = float((g.detach().double() ** 2).sum().item())
                                            _GD_STATE["fh"].write(
                                                f"{_GD_STATE['step']}\t"
                                                f"{_GD_STATE['micro']}\t"
                                                f"{name}\t{sum_sq:.10e}\n"
                                            )
                                            # Raw grad dump for selected params
                                            for prefix in _RAW_PREFIXES_GLOBAL:
                                                if name.replace("._checkpoint_wrapped_module.", ".").startswith(prefix.strip()):
                                                    fname = (f"raw_{_GD_STATE['rank']:02d}"
                                                             f"_step{_GD_STATE['step']:02d}"
                                                             f"_micro{_GD_STATE['micro']:02d}"
                                                             f"_{prefix.strip().replace('.', '_').replace('/','_')}.pt")
                                                    fpath = _os_gd.path.join(_GD_DIR, fname)
                                                    _torch_gd.save(g.detach().cpu(), fpath)
                                                    break
                                        except Exception:
                                            pass
                                return _orig_fr_gd(*args, **kwargs)

                            _fpg_gd.foreach_reduce = _patched_fr_gd
                            _fpg_gd._fyh_grad_dump_patched = True
                            n_hooks = 1  # symbolic
                            if _is_rank0:
                                print("[gemma4 grad-dump] patched FSDP2 "
                                      "foreach_reduce (logs unsharded grads "
                                      "pre-reduce-scatter)",
                                      file=sys.stderr, flush=True)
                    except Exception as _e:
                        print(f"[gemma4 grad-dump] WARN: skip FSDP2 fr-patch "
                              f"({type(_e).__name__}: {_e})",
                              file=sys.stderr, flush=True)
                if _is_rank0:
                    print(f"[gemma4 grad-dump] registered {n_hooks} param hooks "
                          f"(engine={_GD_STATE['engine']})",
                          file=sys.stderr, flush=True)

                # For FSDP2 grad-dump: optionally disable no_sync so foreach_reduce
                # fires per-micro instead of once per step (apples-to-apples
                # with DS3's per-micro autograd hook).
                if (_GD_STATE["engine"].startswith("fsdp2")
                    and _os_gd.environ.get("GEMMA4_GRAD_DUMP_FORCE_SYNC", "1") == "1"):
                    try:
                        from accelerate import accelerator as _acc_gd_ns
                        from accelerate.utils import DistributedType as _DT_gd_ns
                        if not getattr(_acc_gd_ns, "_fyh_grad_dump_no_sync_off", False):
                            _orig_no_sync_gd = _acc_gd_ns.Accelerator.no_sync
                            import contextlib as _ctx_gd
                            def _patched_no_sync_gd(s, model=None):
                                if (s.distributed_type == _DT_gd_ns.FSDP
                                    and getattr(s, "is_fsdp2", False)):
                                    @_ctx_gd.contextmanager
                                    def _nullctx():
                                        yield
                                    return _nullctx()
                                return _orig_no_sync_gd(s, model=model)
                            _acc_gd_ns.Accelerator.no_sync = _patched_no_sync_gd
                            _acc_gd_ns._fyh_grad_dump_no_sync_off = True
                            if _is_rank0:
                                print("[gemma4 grad-dump] FSDP2 no_sync DISABLED "
                                      "(force every micro to sync; foreach_reduce "
                                      "fires per-micro)",
                                      file=sys.stderr, flush=True)
                    except Exception as _e:
                        print(f"[gemma4 grad-dump] WARN: skip FSDP2 no_sync override "
                              f"({type(_e).__name__}: {_e})", file=sys.stderr, flush=True)

                # Hook accelerator.backward to track micro/step counter
                from accelerate import accelerator as _acc_gd
                _orig_backward_gd = _acc_gd.Accelerator.backward

                def _patched_backward_gd(_self, loss, **kw):
                    ret = _orig_backward_gd(_self, loss, **kw)
                    _GD_STATE["micro"] += 1
                    if _GD_STATE["micro"] >= _GD_MAX_MICROS:
                        _GD_STATE["micro"] = 0
                        _GD_STATE["step"] += 1
                        if _GD_STATE["step"] >= _GD_MAX_STEPS and _GD_STATE["fh"] is not None:
                            _GD_STATE["fh"].close()
                            _GD_STATE["fh"] = None
                            if _is_rank0:
                                print("[gemma4 grad-dump] reached max_steps, "
                                      "closed dump file", file=sys.stderr, flush=True)
                    return ret

                _acc_gd.Accelerator.backward = _patched_backward_gd
                _GD_STATE["registered"] = True

            _tfm_gd.Trainer.__init__ = _patched_T_init_gd
            if _is_rank0:
                print("[gemma4 sdp_preamble] (22) attached per-param grad-dump "
                      "(GEMMA4_GRAD_DUMP=1; output dir=" + _GD_DIR + ")",
                      file=sys.stderr, flush=True)
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip grad-dump patch "
                  f"({type(_e).__name__}: {_e})", file=sys.stderr, flush=True)

    # (21) Force FSDP2 reduce-scatter (NCCL) to run in fp32 instead of bf16.
    # Bypasses MixedPrecisionPolicy.reduce_dtype path which has unfixed
    # PyTorch 2.10 cublas bug (CUBLAS_STATUS_EXECUTION_FAILED).  Instead,
    # directly intercept `foreach_reduce`'s reduce_dtype argument before it
    # allocates the RS buffer.
    #
    # Why: FSDP2 default has `reduce_dtype=None` which falls back to grad
    # dtype (bf16).  NCCL reduce-scatter then averages 8 ranks of bf16 grads
    # with bf16 rounding (each pairwise sum loses ~7 bit mantissa precision).
    # DS3 with `contiguous_gradients=true` does NCCL reduce in fp32 — that's
    # likely the source of our 1.29x grad_norm bias.
    #
    # This patch:
    #   - imports `foreach_reduce` from FSDP2 internals
    #   - wraps it so `reduce_dtype=fp32` is passed unconditionally (when
    #     it would otherwise be None or bf16)
    #   - re-binds the wrapped version on `_fsdp_param_group` (where the
    #     actual call site lives, since it imports the function directly)
    #
    # Activated only when GEMMA4_FSDP_REDUCE_FP32_NCCL=1.
    if os.environ.get("GEMMA4_FSDP_REDUCE_FP32_NCCL") == "1":
        try:
            import torch as _t_rs
            from torch.distributed.fsdp._fully_shard import (
                _fsdp_collectives as _fc_rs,
                _fsdp_param_group as _fpg_rs,
            )
            if not getattr(_fpg_rs, "_fyh_fp32_rs_patched", False):
                _orig_foreach_reduce_rs = _fpg_rs.foreach_reduce
                # Find the index of `reduce_dtype` in the signature for
                # positional arg handling (PyTorch 2.10: it's index 6).
                import inspect as _insp_rs
                _sig_rs = _insp_rs.signature(_orig_foreach_reduce_rs)
                _params_rs = list(_sig_rs.parameters.keys())
                try:
                    _RD_IDX = _params_rs.index("reduce_dtype")
                except ValueError:
                    _RD_IDX = -1

                def _fp32_foreach_reduce(*args, **kwargs):
                    if "reduce_dtype" in kwargs:
                        if kwargs["reduce_dtype"] is None or kwargs["reduce_dtype"] == _t_rs.bfloat16:
                            kwargs["reduce_dtype"] = _t_rs.float32
                    elif _RD_IDX >= 0 and len(args) > _RD_IDX:
                        if args[_RD_IDX] is None or args[_RD_IDX] == _t_rs.bfloat16:
                            args = list(args)
                            args[_RD_IDX] = _t_rs.float32
                            args = tuple(args)
                    return _orig_foreach_reduce_rs(*args, **kwargs)

                _fpg_rs.foreach_reduce = _fp32_foreach_reduce
                _fc_rs.foreach_reduce = _fp32_foreach_reduce
                _fpg_rs._fyh_fp32_rs_patched = True
                if _is_rank0:
                    print(
                        "[gemma4 sdp_preamble] (21) patched FSDP2 foreach_reduce "
                        "to force reduce_dtype=fp32 (NCCL reduce-scatter in fp32; "
                        "bypasses MixedPrecisionPolicy + cublas bug; ~50ms/step "
                        "extra comm)",
                        file=sys.stderr, flush=True,
                    )

                # (21b) Also patch FSDPParam.init_dtype_attrs to NOT clamp
                # reduce_dtype to None when reduce_dtype == param_dtype.
                # The default clamp is a perf optimization that disables
                # cross-micro fp32 grad accumulation in to_accumulated_grad_if_needed
                # (which early-returns when self.reduce_dtype is None).
                # By keeping self.reduce_dtype=fp32 explicitly, we activate
                # the fp32 cross-micro accumulation path = DS3's contiguous_gradients
                # behavior.
                from torch.distributed.fsdp._fully_shard import _fsdp_param as _tfp_21b
                # Skip 21b when 21c or 21d is on — both supersede the
                # unsharded fp32 buffer path that 21b activates.
                _skip_21b = (
                    os.environ.get("GEMMA4_FSDP_GRAD_ACCUM_SHARDED") == "1"
                    or os.environ.get("GEMMA4_FSDP_GRAD_BUCKET_FUSION") == "1"
                )
                if (not getattr(_tfp_21b, "_fyh_init_dtype_no_clamp_patched", False)
                        and not _skip_21b):
                    _orig_init_dtype = _tfp_21b.FSDPParam.init_dtype_attrs

                    def _patched_init_dtype_attrs(self, mp_policy):
                        param_dtype, reduce_dtype = mp_policy.param_dtype, mp_policy.reduce_dtype
                        self.orig_dtype = self.sharded_param.dtype
                        # Force reduce_dtype = fp32 (matches patch 21's foreach_reduce override)
                        if reduce_dtype is None or reduce_dtype == _t_rs.bfloat16:
                            reduce_dtype = _t_rs.float32
                        # Original clamp for param_dtype (no-op if same)
                        if param_dtype == self.orig_dtype:
                            param_dtype = None
                        # Do NOT clamp reduce_dtype to None — keep fp32 explicitly
                        self.param_dtype = param_dtype
                        self.reduce_dtype = reduce_dtype

                    _tfp_21b.FSDPParam.init_dtype_attrs = _patched_init_dtype_attrs
                    _tfp_21b._fyh_init_dtype_no_clamp_patched = True
                    if _is_rank0:
                        print(
                            "[gemma4 sdp_preamble] (21b) patched FSDP2 "
                            "FSDPParam.init_dtype_attrs: keep reduce_dtype=fp32 "
                            "explicitly (don't clamp to None) → enables fp32 "
                            "cross-micro grad accumulation in to_accumulated_grad_if_needed "
                            "(matches DS3 contiguous_gradients=true)",
                            file=sys.stderr, flush=True,
                        )
                elif _skip_21b and _is_rank0:
                    print(
                        "[gemma4 sdp_preamble] (21b) SKIPPED — superseded by "
                        "patch 21c (sharded fp32 grad accum every micro, "
                        "no unsharded fp32 buffer needed)",
                        file=sys.stderr, flush=True,
                    )
        except ImportError as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip FSDP2 fp32 NCCL patch "
                  f"(ImportError: {_e})", file=sys.stderr, flush=True)
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip FSDP2 fp32 NCCL patch "
                  f"({type(_e).__name__}: {_e})", file=sys.stderr, flush=True)

    # (21c) Force per-micro sharded fp32 grad accumulation
    # (DS3-style contiguous_gradients).
    #
    # Problem solved: HF Trainer + accelerate wraps micros 1..N-1 in
    # `accelerator.no_sync()`, which calls
    # `model.set_requires_gradient_sync(False)` and flips `reduce_grads=False`
    # on every FSDP param-group.  When that flag is False, FSDP2's
    # `post_backward` skips reduce-scatter and stashes the *unsharded*
    # gradient (~32 GB/rank in fp32 with patch 21b, ~16 GB/rank in bf16
    # without 21b) until the final micro.  For Gemma-4-E4B (8B params)
    # this pushes peak GPU past 80 GiB even with activation_checkpointing
    # on (verified via cuda.memory_summary at OOM in micro 16: top
    # tensors are 42x (10240,2560) fp32 = ~13 GB plus 1x (262144,10752)
    # fp32 = 11 GB == exactly the unsharded grad accumulators).
    #
    # Fix: monkey-patch FSDPModule.set_requires_gradient_sync to ignore
    # the False arg and always keep `reduce_grads=True`.  Each micro then
    # does its own reduce-scatter; FSDP2's `foreach_reduce` already does
    # `sharded_param.grad._local_tensor += new_sharded_grad` when grad is
    # not None (i.e. on micros 2..N), which is exactly DS3's
    # contiguous_gradients semantics on the sharded level.
    #
    # Trade-off:
    #   + Saves ~32 GB / rank → can drop activation_cpu_offload entirely
    #   + Numerically equivalent to DS3 (sharded fp32 += sharded fp32)
    #   + Per-rank peak goes from ~80 GB OOM down to ~50 GB
    #   - 16 reduce-scatter calls per step vs 1 (extra ~50-100 ms latency
    #     but same total bytes transferred)
    #
    # Activated by GEMMA4_FSDP_GRAD_ACCUM_SHARDED=1.
    # Mutually exclusive with patch 21b (21b's no_sync-path code path
    # becomes unreachable when 21c forces reduce_grads=True every micro).
    if os.environ.get("GEMMA4_FSDP_GRAD_ACCUM_SHARDED") == "1":
        try:
            import torch.distributed.fsdp._fully_shard._fully_shard as _fsdp_21c
            if not getattr(_fsdp_21c.FSDPModule,
                           "_fyh_force_grad_sync_patched", False):
                _orig_set_sync_21c = _fsdp_21c.FSDPModule.set_requires_gradient_sync

                def _patched_set_sync_21c(self, requires_gradient_sync,
                                          *, recurse=True):
                    return _orig_set_sync_21c(self, True, recurse=recurse)

                _fsdp_21c.FSDPModule.set_requires_gradient_sync = (
                    _patched_set_sync_21c)
                _fsdp_21c.FSDPModule._fyh_force_grad_sync_patched = True
                if _is_rank0:
                    print(
                        "[gemma4 sdp_preamble] (21c) patched FSDP2 "
                        "FSDPModule.set_requires_gradient_sync: ignore "
                        "False, force per-micro reduce-scatter (sharded "
                        "fp32 grad accum, equivalent to DS3 "
                        "contiguous_gradients)",
                        file=sys.stderr, flush=True,
                    )
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip 21c patch "
                  f"({type(_e).__name__}: {_e})",
                  file=sys.stderr, flush=True)

    # (21d) Cross-group bucket fusion: defer no-sync path's reduce-scatter
    # into a single fused RS at the END of each micro batch.
    #
    # Goal: keep v5's numerical advantage (sharded fp32 grad accumulation
    # via patch 21's fp32 reduce-scatter) WITHOUT the 32 GB unsharded fp32
    # grad buffer that patch 21b creates.  Resulting peak memory ~50 GB
    # (vs 80 GB under 21b), so we can disable activation_cpu_offload and
    # save ~8-12 s/step of H2D/D2H overhead.
    #
    # Why this works:
    #   * 21c forces per-micro RS but each FSDP group RS-es independently
    #     → 16 micro × 45 groups = 720 RS calls = ~22 s latency bottleneck.
    #   * 21d defers per-group post_backward into a single global bucket;
    #     at the END of each micro (in _root_post_backward_final_callback)
    #     we fire ONE foreach_reduce across ALL groups → 16 fused RS/step
    #     (vs v5's 45 small RS, 21c's 720 small RS).
    #   * No unsharded fp32 buffer ever created (we cast bf16→fp32 inside
    #     the NCCL reduce-scatter via patch 21).
    #
    # FSDP2 internals leveraged:
    #   * foreach_reduce(list[FSDPParam], list[Tensor], group, stream, ...)
    #     accepts arbitrary lists of (params, grads) — coalesces internally.
    #   * _root_post_backward_final_callback runs after ALL groups'
    #     post_backward, before _finalize_backward, perfect flush point.
    #
    # Activated by GEMMA4_FSDP_GRAD_BUCKET_FUSION=1.
    # Mutually exclusive with patch 21b (skipped automatically).
    if os.environ.get("GEMMA4_FSDP_GRAD_BUCKET_FUSION") == "1":
        try:
            import torch as _t_21d
            from torch.distributed.fsdp._fully_shard import (
                _fsdp_param_group as _fpg_21d,
                _fsdp_state as _fst_21d,
            )
            from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
                foreach_reduce as _fr_21d,
            )
            from torch.distributed.fsdp._fully_shard._fsdp_common import (
                TrainingState as _TS_21d,
                FSDPMeshInfo as _FMI_21d,
                DDPMeshInfo as _DMI_21d,
            )
            from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
                ReduceScatterState as _RSS_21d,
                AllReduceState as _ARS_21d,
            )

            # Module-scope bucket (per backward iteration)
            _21D_STATE = {
                "params": [],
                "grads": [],
                "groups_seen": [],
                "groups_in_bucket": set(),
            }

            # Maximum bytes per fused RS chunk.  foreach_reduce internally
            # allocates a contiguous bf16 buffer (sum of all grad sizes) +
            # an fp32 buffer for the reduce_dtype cast.  For full-model
            # fusion (8B params), that is bf16 16 GB + fp32 32 GB = 48 GB
            # transient, which OOMs.  We chunk by bytes so that each call
            # uses at most ~CHUNK_BYTES of transient buffer.
            _21D_CHUNK_BYTES = int(os.environ.get(
                "GEMMA4_FSDP_FUSE_CHUNK_BYTES", str(2 * 1024**3)))  # 2 GB default

            def _21d_flush_bucket():
                """Chunked fused reduce-scatter across all bucketed groups."""
                if not _21D_STATE["params"]:
                    return
                proxy = _21D_STATE["groups_seen"][0]
                params_all = _21D_STATE["params"]
                grads_all = _21D_STATE["grads"]
                # Reset bucket up-front
                _21D_STATE["params"] = []
                _21D_STATE["grads"] = []
                _21D_STATE["groups_seen"] = []
                _21D_STATE["groups_in_bucket"] = set()

                rs_pg = (proxy._reduce_scatter_process_group
                         if isinstance(proxy.mesh_info, _FMI_21d) else None)
                ar_pg = (proxy._all_reduce_process_group
                         if isinstance(proxy.mesh_info, _DMI_21d) else None)
                all_reduce_stream = proxy.comm_ctx.all_reduce_stream

                # Chunk grads list so each foreach_reduce call uses bounded transient memory
                chunks = []
                cur_p, cur_g, cur_b = [], [], 0
                for p, g in zip(params_all, grads_all):
                    sz = g.numel() * g.element_size()
                    if cur_b + sz > _21D_CHUNK_BYTES and cur_p:
                        chunks.append((cur_p, cur_g))
                        cur_p, cur_g, cur_b = [], [], 0
                    cur_p.append(p)
                    cur_g.append(g)
                    cur_b += sz
                if cur_p:
                    chunks.append((cur_p, cur_g))

                last_rs_input = None
                last_rs_event = None
                last_post_reduce_event = None
                last_ar_input = None
                last_ar_event = None
                last_partial_reduce_output = None

                for ch_params, ch_grads in chunks:
                    (
                        rs_input,
                        rs_event,
                        post_reduce_event,
                        ar_input,
                        ar_event,
                        partial_reduce_output,
                    ) = _fr_21d(
                        ch_params,
                        ch_grads,
                        rs_pg,
                        proxy.comm_ctx.reduce_scatter_stream,
                        proxy._reduce_scatter_comm,
                        proxy._orig_dtype,
                        proxy._reduce_dtype,
                        proxy.device,
                        proxy.gradient_divide_factor,
                        ar_pg,
                        all_reduce_stream,
                        proxy.all_reduce_grads,
                        proxy._partial_reduce_output,
                        proxy._all_reduce_hook,
                        proxy.force_sum_reduction_for_comms,
                    )
                    last_rs_input = rs_input
                    last_rs_event = rs_event
                    last_post_reduce_event = post_reduce_event
                    last_ar_input = ar_input
                    last_ar_event = ar_event
                    last_partial_reduce_output = partial_reduce_output
                    # Drop refs to the chunk's grads so they can be freed before
                    # the next chunk's foreach_reduce allocates its transient buf
                    ch_grads.clear()
                grads_all.clear()
                params_all.clear()

                if last_rs_input is not None:
                    proxy.comm_ctx.reduce_scatter_state = _RSS_21d(
                        last_rs_input, last_rs_event)
                proxy._post_reduce_event = last_post_reduce_event
                proxy._partial_reduce_output = last_partial_reduce_output
                if last_ar_input is not None and last_ar_event is not None:
                    proxy._all_reduce_state = _ARS_21d(last_ar_input, last_ar_event)

                # Sync the RS stream into the current (compute) stream so that
                # any transient buffers (concat input, fp32 cast staging) are
                # safely freed BEFORE the next micro's forward starts.  Without
                # this, micro N+1's forward races with micro N's RS for memory
                # and OOMs.  Cost: forces serial RS, gives up some overlap.
                if last_post_reduce_event is not None:
                    _t_21d.cuda.current_stream().wait_event(
                        last_post_reduce_event)

            # ---- 1. Patch FSDPParamGroup.post_backward ----
            if not getattr(_fpg_21d.FSDPParamGroup,
                           "_fyh_21d_patched", False):
                _orig_post_bw_21d = _fpg_21d.FSDPParamGroup.post_backward

                def _patched_post_bw_21d(self, *unused):
                    self._training_state = _TS_21d.POST_BACKWARD
                    # Step 1: process previous-micro carry-over (rare with 21d)
                    for fsdp_param in self.fsdp_params:
                        fsdp_param.accumulate_unsharded_grad_if_needed()
                    # Step 2: collect (params, unsharded_grad) into global bucket
                    for fsdp_param in self.fsdp_params:
                        if not hasattr(fsdp_param, "_unsharded_param"):
                            continue
                        # Prefer accumulated_grad if present (carry-over from
                        # before 21d was active, or from edge cases); fall
                        # back to unsharded_param.grad
                        if fsdp_param.unsharded_accumulated_grad is not None:
                            _21D_STATE["params"].append(fsdp_param)
                            _21D_STATE["grads"].append(
                                fsdp_param.unsharded_accumulated_grad_data)
                            fsdp_param.unsharded_accumulated_grad = None
                        elif fsdp_param.unsharded_param.grad is not None:
                            _21D_STATE["params"].append(fsdp_param)
                            _21D_STATE["grads"].append(
                                fsdp_param.unsharded_grad_data)
                            fsdp_param.unsharded_param.grad = None
                    if id(self) not in _21D_STATE["groups_in_bucket"]:
                        _21D_STATE["groups_seen"].append(self)
                        _21D_STATE["groups_in_bucket"].add(id(self))
                    # Step 3: reshard now (we've extracted what we need)
                    if self.reshard_after_backward:
                        self.reshard()
                    # NB: do NOT call to_accumulated_grad_if_needed (21b path)
                    # NB: do NOT call foreach_reduce here — flush is done by
                    #     _root_post_backward_final_callback once per micro

                _fpg_21d.FSDPParamGroup.post_backward = _patched_post_bw_21d
                _fpg_21d.FSDPParamGroup._fyh_21d_patched = True

            # ---- 2. Patch _root_post_backward_final_callback ----
            if not getattr(_fst_21d.FSDPState,
                           "_fyh_21d_root_patched", False):
                _orig_root_cb_21d = _fst_21d.FSDPState._root_post_backward_final_callback

                def _patched_root_cb_21d(self):
                    # Flush our bucket once per micro batch BEFORE the original
                    # callback (which calls _finalize_backward and clears comm state)
                    if _21D_STATE["params"]:
                        _21d_flush_bucket()
                    return _orig_root_cb_21d(self)

                _fst_21d.FSDPState._root_post_backward_final_callback = _patched_root_cb_21d
                _fst_21d.FSDPState._fyh_21d_root_patched = True

            if _is_rank0:
                print(
                    "[gemma4 sdp_preamble] (21d) patched FSDP2 "
                    "FSDPParamGroup.post_backward + "
                    "FSDPState._root_post_backward_final_callback: "
                    "cross-group bucket fusion → 1 fused reduce-scatter "
                    "per micro (16 RS/step vs 45 in v5, 720 in 21c). "
                    "No unsharded fp32 grad buffer → can disable "
                    "activation_cpu_offload, peak ~50 GB.",
                    file=sys.stderr, flush=True,
                )
        except Exception as _e:
            import traceback as _tb_21d
            print(f"[gemma4 sdp_preamble] WARN: skip 21d patch "
                  f"({type(_e).__name__}: {_e})\n{_tb_21d.format_exc()}",
                  file=sys.stderr, flush=True)

    # (32) Make swift's activation_cpu_offload Dynamo-transparent.
    #
    # Why: when --torch_compile is enabled, accelerate calls
    # compile_regions() on each Gemma4TextDecoderLayer.  Inside the layer,
    # swift's `AsyncDoubleBufferGroupOffloadHandler` registers
    # saved-tensor hooks that call `torch.empty(..., pin_memory=True)`
    # to stage activations on CPU.  PyTorch inductor cannot lower
    # pin_memory and aborts with:
    #   torch._inductor.exc.InductorError: LoweringException:
    #   NotImplementedError: inductor does not support pin_memory
    #
    # Fix: wrap swift's `offload()` / `reload()` with
    # `torch.compiler.disable` so dynamo treats them as opaque
    # graph-break boundaries — they execute eagerly while the rest of
    # the layer compiles normally.  H2D/D2H copies don't benefit from
    # compile anyway, so this only loses the trivial empty() launch.
    #
    # Activated unconditionally (idempotent; no-op when torch.compile not used).
    try:
        import torch as _t_p32
        from swift.callbacks import activation_cpu_offload as _aco_p32
        _disable_p32 = getattr(getattr(_t_p32, "compiler", None), "disable", None)
        if _disable_p32 is not None:
            _patched_p32 = []
            # 1. Hook entry points on CpuOffloadHookWithOffloadHandler — these
            # are the saved_tensors_hooks that dynamo would otherwise inline
            # into the compiled graph (causing the pin_memory lowering error).
            _hook_cls = getattr(_aco_p32, "CpuOffloadHookWithOffloadHandler", None)
            if _hook_cls is not None:
                for _mname in ("on_save_for_backward", "on_get_saved_tensor"):
                    _fn = _hook_cls.__dict__.get(_mname)
                    if _fn is None or getattr(_fn, "_fyh_dyno_disabled", False):
                        continue
                    _wrapped = _disable_p32(_fn)
                    _wrapped._fyh_dyno_disabled = True
                    setattr(_hook_cls, _mname, _wrapped)
                    _patched_p32.append(f"CpuOffloadHookWithOffloadHandler.{_mname}")
            # 2. Static offload/reload methods (defense in depth — they may
            # also be reached from places dynamo can see)
            for _cname in ("SynchronizedGroupOffloadHandler",
                           "AsyncDoubleBufferGroupOffloadHandler"):
                _cls = getattr(_aco_p32, _cname, None)
                if _cls is None:
                    continue
                for _mname in ("offload", "reload", "tensor_push", "tensor_pop"):
                    if _mname not in _cls.__dict__:
                        continue
                    _attr = _cls.__dict__[_mname]
                    _is_static = isinstance(_attr, staticmethod)
                    _fn = _attr.__func__ if _is_static else _attr
                    if getattr(_fn, "_fyh_dyno_disabled", False):
                        continue
                    _wrapped = _disable_p32(_fn)
                    _wrapped._fyh_dyno_disabled = True
                    setattr(_cls, _mname,
                            staticmethod(_wrapped) if _is_static else _wrapped)
                    _patched_p32.append(f"{_cname}.{_mname}")
            if _patched_p32 and _is_rank0:
                print(
                    f"[gemma4 sdp_preamble] (32) wrapped {len(_patched_p32)} "
                    f"swift activation_cpu_offload methods with "
                    f"torch.compiler.disable: {_patched_p32}",
                    file=sys.stderr, flush=True,
                )
    except ImportError:
        pass
    except Exception as _e:
        print(f"[gemma4 sdp_preamble] WARN: skip 32 patch "
              f"({type(_e).__name__}: {_e})", file=sys.stderr, flush=True)

    # (29) Patch swift's enable_activation_offloading() to skip submodules
    # whose `_supports_gradient_checkpointing = False` (e.g. Gemma4AudioModel).
    #
    # Why: swift's callback first calls `gradient_checkpointing_disable()` on
    # every submodule via:
    #   for module in model.modules():
    #       if hasattr(module, 'gradient_checkpointing_disable'):
    #           module.gradient_checkpointing_disable()
    # transformers >= 5.0 raises ValueError when the model class has
    # `_supports_gradient_checkpointing = False`. For Gemma-4-E4B, the audio
    # tower (`Gemma4AudioModel`) is one such class. The result is the
    # whole `activation_cpu_offload` callback aborts before doing anything.
    #
    # Fix: wrap the inner call in try/except so failures are skipped silently.
    # Activated unconditionally (idempotent) — only applies when swift's
    # activation_cpu_offload callback actually runs.
    try:
        from swift.callbacks import activation_cpu_offload as _aco
        if not getattr(_aco, "_fyh_audio_skip_patched", False):
            _orig_enable_act_off = _aco.enable_activation_offloading

            def _patched_enable_act_off(model, strategy="fsdp2", enable_ckpt=True):
                # Replace `module.gradient_checkpointing_disable()` with a
                # try/except wrapper.
                _orig_disable_method = {}
                for module in model.modules():
                    if not hasattr(module, "gradient_checkpointing_disable"):
                        continue
                    cls = type(module)
                    if cls in _orig_disable_method:
                        continue
                    _orig_disable_method[cls] = cls.gradient_checkpointing_disable
                    if not getattr(cls, "_fyh_disable_safe_patched", False):
                        _orig = cls.gradient_checkpointing_disable

                        def _safe_disable(self, _orig=_orig):
                            try:
                                _orig(self)
                            except ValueError:
                                pass

                        cls.gradient_checkpointing_disable = _safe_disable
                        cls._fyh_disable_safe_patched = True
                # Now safe to call the original
                return _orig_enable_act_off(model, strategy=strategy, enable_ckpt=enable_ckpt)

            _aco.enable_activation_offloading = _patched_enable_act_off
            _aco._fyh_audio_skip_patched = True
            if _is_rank0:
                print(
                    "[gemma4 sdp_preamble] (29) patched swift activation_cpu_offload: "
                    "wrap gradient_checkpointing_disable in try/except (ValueError) "
                    "→ skip Gemma4AudioModel-style classes that don't support AC",
                    file=sys.stderr, flush=True,
                )
    except ImportError:
        pass
    except Exception as _e:
        print(f"[gemma4 sdp_preamble] WARN: skip activation_cpu_offload audio patch "
              f"({type(_e).__name__}: {_e})", file=sys.stderr, flush=True)

    # (30) Patch swift's AsyncDoubleBufferGroupOffloadHandler.tensor_pop()
    # to lazily reload tensors when the state is still a tuple (offloaded).
    # 
    # Why: swift's offload state machine assumes a fixed FSDP layer ordering
    # for `bulk_reload_group_backward`, but with custom wrap (A3 PLE separate +
    # gemma4 audio/vision branches), the backward group commit order doesn't
    # match the offload order. So `tensor_pop` may be called for a group
    # before its `bulk_reload_group` has fired → state is still tuple → assert
    # `not isinstance(tensor, tuple)` fails.
    # 
    # Fix: when state is a tuple, do the reload eagerly in tensor_pop itself.
    # Slightly serializes CPU→GPU h2d transfer (typically 100-300 MB at a
    # time, ~10ms over PCIe 5.0 × 16) but compatible with custom wrap.
    try:
        from swift.callbacks import activation_cpu_offload as _aco30
        if not getattr(_aco30, "_fyh_tensor_pop_patched", False):
            import torch as _t30
            _AsyncH = _aco30.AsyncDoubleBufferGroupOffloadHandler
            _Sync = _aco30.SynchronizedGroupOffloadHandler

            _orig_tensor_pop = _AsyncH.tensor_pop

            def _patched_tensor_pop(self, tensor_tag, **kwargs):
                if isinstance(tensor_tag, _t30.Tensor):
                    return tensor_tag
                assert tensor_tag in self.tensor_tag_to_state
                state = self.tensor_tag_to_state[tensor_tag]
                if isinstance(state, tuple):
                    # Still offloaded — eagerly reload this group now.
                    group_id, _idx = tensor_tag
                    if group_id in self.group_offload_mapping:
                        try:
                            from swift.callbacks.activation_cpu_offload import get_torch_device as _gtd
                            device = _gtd()
                        except Exception:
                            device = _t30.cuda
                        with device.stream(self.h2d_stream):
                            offload_mapping = self.group_offload_mapping.pop(group_id)
                            for key, s in offload_mapping.items():
                                offload_mapping[key] = _Sync.reload(s)
                            # Restore all tuple-states for this group
                            for tag2, st2 in list(self.tensor_tag_to_state.items()):
                                gid2, _ = tag2
                                if gid2 == group_id and isinstance(st2, tuple):
                                    key, shape = st2
                                    self.tensor_tag_to_state[tag2] = offload_mapping[key].view(shape)
                            # Wait for h2d to finish before consuming
                        device.current_stream().wait_stream(self.h2d_stream)
                        state = self.tensor_tag_to_state[tensor_tag]
                # Still a tuple? then we lost the offload mapping; surface that.
                if isinstance(state, tuple):
                    raise RuntimeError(
                        f"[gemma4 patch 30] tensor_pop: group {tensor_tag[0]} "
                        f"not in group_offload_mapping; cannot lazily reload. "
                        f"Likely a logic bug in swift's commit ordering."
                    )
                self.tensor_tag_to_state.pop(tensor_tag)
                self.tensor_tag_to_buf.pop(tensor_tag, None)
                return state

            _AsyncH.tensor_pop = _patched_tensor_pop
            _aco30._fyh_tensor_pop_patched = True
            if _is_rank0:
                print(
                    "[gemma4 sdp_preamble] (30) patched swift "
                    "AsyncDoubleBufferGroupOffloadHandler.tensor_pop: lazily "
                    "reload tuple-state on demand (fixes assert when FSDP wrap "
                    "order != commit order)",
                    file=sys.stderr, flush=True,
                )
    except ImportError:
        pass
    except Exception as _e:
        print(f"[gemma4 sdp_preamble] WARN: skip activation_cpu_offload tensor_pop patch "
              f"({type(_e).__name__}: {_e})", file=sys.stderr, flush=True)

    # (31) GPU memory profiling: dump cuda.memory_summary() at the end of
    # backward of step 1 (after AC + grad accum done, before optimizer step).
    # Activated by GEMMA4_MEM_PROFILE=1.  Output: /tmp/mem_profile_rank<R>.txt
    if os.environ.get("GEMMA4_MEM_PROFILE") == "1":
        try:
            import transformers as _tfm_mp
            if not getattr(_tfm_mp.Trainer, "_fyh_mem_profile_attached", False):
                _orig_training_step_mp = _tfm_mp.Trainer.training_step
                _state_mp = {"step": 0}

                def _dump_mem(rank, tag):
                    import torch as _t_mp
                    fname = f"/tmp/mem_profile_rank{rank}_{tag}.txt"
                    with open(fname, "w") as f:
                        f.write(f"=== rank {rank} {tag} ===\n")
                        f.write(f"alloc: {_t_mp.cuda.memory_allocated()/1e9:.2f} GB\n")
                        f.write(f"reserved: {_t_mp.cuda.memory_reserved()/1e9:.2f} GB\n")
                        f.write(f"max_alloc: {_t_mp.cuda.max_memory_allocated()/1e9:.2f} GB\n")
                        f.write("\n=== memory_summary ===\n")
                        f.write(_t_mp.cuda.memory_summary(abbreviated=False))
                        f.write("\n=== top 80 storages by ACTUAL allocated bytes ===\n")
                        import gc
                        # Track unique storage by data_ptr; sum over all tensors viewing it
                        storages = {}
                        for obj in gc.get_objects():
                            try:
                                if _t_mp.is_tensor(obj) and obj.is_cuda:
                                    sto = obj.untyped_storage()
                                    ptr = sto.data_ptr()
                                    nbytes = sto.nbytes()
                                    if ptr not in storages:
                                        storages[ptr] = (nbytes, [])
                                    desc = f"shape={tuple(obj.shape)} dtype={str(obj.dtype).replace('torch.','')}"
                                    is_dtensor = "DTensor" in type(obj).__name__
                                    if is_dtensor:
                                        desc += " [DTensor]"
                                    if hasattr(obj, "_local_tensor"):
                                        desc += f" local={tuple(obj._local_tensor.shape)}"
                                    storages[ptr][1].append(desc)
                            except Exception:
                                pass
                        sorted_storages = sorted(storages.items(),
                                                 key=lambda x: -x[1][0])
                        total_seen = 0
                        for ptr, (nbytes, descs) in sorted_storages[:80]:
                            total_seen += nbytes
                            f.write(f"  {nbytes/1e9:.3f} GB  ptr=0x{ptr:x}\n")
                            for d in descs[:3]:
                                f.write(f"      view: {d}\n")
                            if len(descs) > 3:
                                f.write(f"      ... +{len(descs)-3} more views\n")
                        f.write(f"\nTotal seen in top 80: {total_seen/1e9:.2f} GB\n")
                        f.write(f"Total all storages: {sum(v[0] for v in storages.values())/1e9:.2f} GB\n")

                def _patched_training_step_mp(s, model, inputs, *a, **kw):
                    import torch as _t_mp
                    rank = int(os.environ.get("RANK",
                                              os.environ.get("LOCAL_RANK", "0")))
                    _state_mp["step"] += 1
                    if _state_mp["step"] == 1:
                        _dump_mem(rank, "before_micro1")
                        if _is_rank0:
                            print(f"[gemma4 mem-profile] before_micro1: "
                                  f"alloc={_t_mp.cuda.memory_allocated()/1e9:.2f} GB",
                                  file=sys.stderr, flush=True)
                    try:
                        ret = _orig_training_step_mp(s, model, inputs, *a, **kw)
                    except _t_mp.cuda.OutOfMemoryError:
                        _dump_mem(rank, f"OOM_micro{_state_mp['step']}")
                        raise
                    return ret

                _tfm_mp.Trainer.training_step = _patched_training_step_mp
                _tfm_mp.Trainer._fyh_mem_profile_attached = True
                if _is_rank0:
                    print("[gemma4 sdp_preamble] (31) attached memory profiler "
                          "to transformers.Trainer.training_step "
                          "(GEMMA4_MEM_PROFILE=1)",
                          file=sys.stderr, flush=True)
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip mem profile patch "
                  f"({type(_e).__name__}: {_e})", file=sys.stderr, flush=True)

    # (20) Manual fp32 grad accumulator (bypasses FSDP2 default bf16 +=).
    # Goal: replicate DS3 ZeRO-3 numerical behavior where grads are accumulated
    # across micro-batches in fp32, regardless of param/reduce dtype.
    #
    # Why: FSDP2 default (param=bf16 reduce=bf16) accumulates in bf16 and gives
    # systematic ~1.29x grad_norm overestimate (cancellation loss in bf16 +=).
    # MixedPrecisionPolicy(reduce_dtype=fp32) is the canonical fix but PyTorch
    # 2.10 has unfixed bugs in that path (CUBLAS_STATUS_EXECUTION_FAILED).
    #
    # Approach: monkeypatch
    #   1. Accelerator.backward: after each micro's loss.backward(), copy each
    #      param's local bf16 shard to a per-param fp32 buffer, then zero the
    #      bf16 local shard so FSDP2's next reduce-scatter writes a "fresh"
    #      grad (effectively replacing rather than +='ing into stale bf16).
    #   2. Accelerator.clip_grad_norm_: before clipping, copy fp32 buffer back
    #      to p.grad (downcast to bf16 for storage).  Then zero buffer for
    #      next GAS round.
    #
    # Activated only when GEMMA4_FP32_GRAD_ACCUM=1.
    if os.environ.get("GEMMA4_FP32_GRAD_ACCUM") == "1":
        try:
            from accelerate import accelerator as _acc_mod_fp32
            from accelerate.utils import DistributedType as _DT_fp32
            import torch as _t_fp32
            if not getattr(_acc_mod_fp32, "_fyh_fp32_grad_accum_patched", False):
                # Per-Accelerator-instance state: id(p) -> fp32 buffer
                _STATE_KEY = "_fyh_fp32_grad_state"
                _MICRO_KEY = "_fyh_fp32_grad_micro_count"

                try:
                    from torch.distributed.tensor import DTensor as _DT_t_fp32
                except ImportError:
                    _DT_t_fp32 = None

                def _get_local(g):
                    if _DT_t_fp32 is not None and isinstance(g, _DT_t_fp32):
                        return g.to_local()
                    return g

                def _accumulate_fp32(self):
                    """Walk all FSDP-managed params, copy bf16 local shard to
                    fp32 buffer, zero the bf16 shard."""
                    if not hasattr(self, _STATE_KEY):
                        setattr(self, _STATE_KEY, {})
                        setattr(self, _MICRO_KEY, 0)
                    state = getattr(self, _STATE_KEY)
                    n_accum = 0
                    n_total = 0
                    n_no_req = 0
                    n_no_grad = 0
                    n_zero_local = 0
                    sum_sq_in_bf16 = 0.0
                    for model in self._models:
                        for p in model.parameters():
                            n_total += 1
                            if not p.requires_grad:
                                n_no_req += 1
                                continue
                            if p.grad is None:
                                n_no_grad += 1
                                continue
                            local = _get_local(p.grad)
                            if local.numel() == 0:
                                n_zero_local += 1
                                continue
                            pid = id(p)
                            if pid not in state:
                                state[pid] = _t_fp32.zeros_like(local, dtype=_t_fp32.float32)
                            local_fp32 = local.detach().to(_t_fp32.float32)
                            sum_sq_in_bf16 += float((local_fp32 * local_fp32).sum().item())
                            state[pid].add_(local_fp32)
                            local.zero_()
                            n_accum += 1
                    setattr(self, _MICRO_KEY, getattr(self, _MICRO_KEY, 0) + 1)
                    if (_is_rank0 and getattr(self, _MICRO_KEY) <= 18):
                        print(f"[gemma4 fp32-accum] micro #{getattr(self, _MICRO_KEY)}: "
                              f"n_accum={n_accum}/{n_total} (no_req={n_no_req}, "
                              f"no_grad={n_no_grad}, zero_local={n_zero_local}) "
                              f"this_micro_local_sum_sq_rank0={sum_sq_in_bf16:.4f}",
                              file=sys.stderr, flush=True)

                def _writeback_fp32(self):
                    """Before clip/optim.step, copy fp32 buffer back to p.grad
                    (downcast to bf16 local shard); zero buffer for next step."""
                    state = getattr(self, _STATE_KEY, None)
                    if not state:
                        if _is_rank0:
                            print("[gemma4 fp32-accum] writeback: state EMPTY, skipping",
                                  file=sys.stderr, flush=True)
                        return 0
                    n_writeback = 0
                    n_skip_pid = 0
                    n_skip_grad_none = 0
                    n_total = 0
                    sum_sq_buf = 0.0
                    for model in self._models:
                        for p in model.parameters():
                            n_total += 1
                            if not p.requires_grad:
                                continue
                            pid = id(p)
                            if pid not in state:
                                n_skip_pid += 1
                                continue
                            if p.grad is None:
                                n_skip_grad_none += 1
                                continue
                            local = _get_local(p.grad)
                            buf = state[pid]
                            sum_sq_buf += float((buf * buf).sum().item())
                            local.copy_(buf.to(local.dtype))
                            buf.zero_()
                            n_writeback += 1
                    setattr(self, _MICRO_KEY, 0)
                    if _is_rank0:
                        cnt = getattr(self, "_fyh_fp32_grad_sync_count", 0) + 1
                        setattr(self, "_fyh_fp32_grad_sync_count", cnt)
                        if cnt <= 4:
                            print(f"[gemma4 fp32-accum] writeback step {cnt}: "
                                  f"n_writeback={n_writeback}/{n_total} "
                                  f"(skip_pid={n_skip_pid}, skip_grad_none={n_skip_grad_none}) "
                                  f"buffer_sum_sq_rank0={sum_sq_buf:.4f} "
                                  f"state_size={len(state)}",
                                  file=sys.stderr, flush=True)
                    return n_writeback

                _orig_backward_fp32 = _acc_mod_fp32.Accelerator.backward
                _orig_clip_fp32 = _acc_mod_fp32.Accelerator.clip_grad_norm_
                _orig_no_sync_fp32 = _acc_mod_fp32.Accelerator.no_sync

                # Force every micro to sync (reduce-scatter) so our hook can
                # capture each micro's grad in `sharded_param.grad`.  HF Trainer
                # uses `accelerator.no_sync(model=model)` on non-final micros to
                # defer FSDP2 reduce-scatter, which causes 15 micros' grads to
                # bf16 += accumulate inside FSDP2's `unsharded_param.grad`
                # (where the cancellation error originates).  Patching no_sync
                # to a null context disables this deferral.  Cost: 16x more
                # reduce-scatter calls per step (negligible: ~50ms / 36s step).
                import contextlib as _ctx_fp32

                def _patched_no_sync_fp32(self, model=None):
                    if (self.distributed_type == _DT_fp32.FSDP
                        and getattr(self, "is_fsdp2", False)):
                        @_ctx_fp32.contextmanager
                        def _nullctx():
                            yield
                        return _nullctx()
                    return _orig_no_sync_fp32(self, model=model)

                def _patched_backward_fp32(self, loss, **kwargs):
                    # FSDP2 only.  Skip for DS or other engines.
                    if (self.distributed_type != _DT_fp32.FSDP
                        or not getattr(self, "is_fsdp2", False)):
                        return _orig_backward_fp32(self, loss, **kwargs)
                    out = _orig_backward_fp32(self, loss, **kwargs)
                    _accumulate_fp32(self)
                    return out

                def _patched_clip_fp32(self, parameters, max_norm, norm_type=2):
                    if (self.distributed_type == _DT_fp32.FSDP
                        and getattr(self, "is_fsdp2", False)):
                        _writeback_fp32(self)
                    return _orig_clip_fp32(self, parameters, max_norm, norm_type)

                _acc_mod_fp32.Accelerator.backward = _patched_backward_fp32
                _acc_mod_fp32.Accelerator.clip_grad_norm_ = _patched_clip_fp32
                _acc_mod_fp32.Accelerator.no_sync = _patched_no_sync_fp32
                _acc_mod_fp32._fyh_fp32_grad_accum_patched = True
                if _is_rank0:
                    print(
                        "[gemma4 sdp_preamble] (20) attached manual fp32 grad "
                        "accumulator (GEMMA4_FP32_GRAD_ACCUM=1; replicates DS3 "
                        "fp32 inter-step accumulation behavior; FSDP2 only)",
                        file=sys.stderr, flush=True,
                    )
        except ImportError:
            pass
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip fp32 grad accum patch "
                  f"({type(_e).__name__}: {_e})", file=sys.stderr, flush=True)

    # (19) Instrument accelerator.backward to show what scaling is applied
    # along the FSDP2 path (loss / GAS happens here for non-DS engines).
    if os.environ.get("GEMMA4_BACKWARD_DBG") == "1":
        try:
            from accelerate import accelerator as _acc_mod_dbg
            from accelerate.utils import DistributedType as _DT_dbg
            if not getattr(_acc_mod_dbg, "_fyh_backward_dbg_patched", False):
                _orig_backward = _acc_mod_dbg.Accelerator.backward
                _state_b = {"step": 0, "max": 32}

                def _patched_backward(self, loss, **kwargs):
                    _state_b["step"] += 1
                    if _is_rank0 and _state_b["step"] <= _state_b["max"]:
                        try:
                            loss_in = float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)
                        except Exception:
                            loss_in = None
                        gas = getattr(self, "gradient_accumulation_steps", None)
                        is_ds = (self.distributed_type == _DT_dbg.DEEPSPEED)
                        will_div = (not is_ds)
                        loss_after = (loss_in / gas) if (loss_in is not None and gas and will_div) else loss_in
                        print(f"[gemma4 backward-dbg] micro={_state_b['step']} "
                              f"distributed_type={self.distributed_type.value if hasattr(self.distributed_type, 'value') else self.distributed_type} "
                              f"gas={gas} loss_in={loss_in} "
                              f"will_divide_by_gas={will_div} "
                              f"loss_for_backward={loss_after} "
                              f"kwargs_keys={list(kwargs.keys())}",
                              file=sys.stderr, flush=True)
                    return _orig_backward(self, loss, **kwargs)

                _acc_mod_dbg.Accelerator.backward = _patched_backward
                _acc_mod_dbg._fyh_backward_dbg_patched = True
                if _is_rank0:
                    print("[gemma4 sdp_preamble] (19) attached accelerator.backward dbg "
                          "(GEMMA4_BACKWARD_DBG=1)", file=sys.stderr, flush=True)
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip backward-dbg patch "
                  f"({type(_e).__name__}: {_e})", file=sys.stderr, flush=True)

    # (18) Instrument loss / grad book-keeping for diagnosing the systematic
    # 1.29x grad_norm offset vs DS ZeRO-3 native.  Prints (rank 0 only):
    #   - raw model output loss (mean per micro-batch, before swift rescale)
    #   - swift's rescaled compute_loss return value
    #   - non-padding token count of this micro-batch
    #   - num_items_in_batch (HF Trainer's accumulated denominator for GAS)
    #   - total micro-batch tokens (incl. padding) from inputs shape
    # Activated only when GEMMA4_LOSS_DBG=1.
    if os.environ.get("GEMMA4_LOSS_DBG") == "1":
        try:
            import transformers as _tfm_dbg
            import functools as _ft_dbg
            _orig_T_init_dbg = _tfm_dbg.Trainer.__init__

            @_ft_dbg.wraps(_orig_T_init_dbg)
            def _patched_T_init_for_loss_dbg(self, *args, **kwargs):
                _orig_T_init_dbg(self, *args, **kwargs)
                if getattr(self, "_fyh_loss_dbg_attached", False):
                    return
                try:
                    from swift.trainers.seq2seq_trainer import Seq2SeqTrainer as _S
                    if not getattr(_S, "_fyh_loss_dbg_attached", False):
                        _orig_compute_loss = _S.compute_loss

                        _state = {"step": 0, "max_log_steps": 16}

                        def _patched_compute_loss(s, model, inputs, *a, **kw):
                            num_items_in_batch = kw.get("num_items_in_batch", None)
                            labels = inputs.get("labels", None)
                            input_ids = inputs.get("input_ids", None)
                            non_pad = None
                            tot = None
                            if labels is not None:
                                # match swift's logic: labels[:, 1:] != -100
                                lab = labels
                                try:
                                    non_pad = (lab[:, 1:] != -100).sum().item()
                                    tot = lab.numel()
                                except Exception:
                                    pass
                            elif input_ids is not None:
                                tot = input_ids.numel()
                            ret = _orig_compute_loss(s, model, inputs, *a, **kw)
                            try:
                                if isinstance(ret, tuple):
                                    loss_v = ret[0]
                                else:
                                    loss_v = ret
                                loss_f = float(loss_v.detach().item()) if hasattr(loss_v, "detach") else float(loss_v)
                            except Exception:
                                loss_f = None
                            _state["step"] += 1
                            if _is_rank0 and _state["step"] <= _state["max_log_steps"]:
                                print(
                                    f"[gemma4 loss-dbg] micro={_state['step']} "
                                    f"loss_returned={loss_f} "
                                    f"non_pad_tokens={non_pad} "
                                    f"total_tokens={tot} "
                                    f"num_items_in_batch={num_items_in_batch} "
                                    f"ratio_non_pad/items="
                                    f"{(non_pad/num_items_in_batch) if (non_pad and num_items_in_batch) else None}",
                                    file=sys.stderr, flush=True,
                                )
                            return ret

                        _S.compute_loss = _patched_compute_loss
                        _S._fyh_loss_dbg_attached = True
                        if _is_rank0:
                            print(
                                "[gemma4 sdp_preamble] (18) attached "
                                "swift Seq2SeqTrainer.compute_loss instrumentation "
                                "(GEMMA4_LOSS_DBG=1; logs first 16 micro-batches)",
                                file=sys.stderr, flush=True,
                            )
                except ImportError:
                    pass
                self._fyh_loss_dbg_attached = True

            _tfm_dbg.Trainer.__init__ = _patched_T_init_for_loss_dbg
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip loss-dbg patch "
                  f"({type(_e).__name__}: {_e})",
                  file=sys.stderr, flush=True)

    # (24) Selectively unwrap CheckpointWrapper from KV-source layers (14-22)
    # to fix the patch (8) × AC interaction that inflates source-layer
    # gradients by 3-4x in FSDP2 path.
    #
    # Detection: Gemma-4-E4B has 42 layers, num_kv_shared_layers=18, so
    # first_kv_shared_layer_idx=24.  Layers 24-41 read kv from layers
    # (idx-18), i.e. layers 14-23.  Per the layer-by-layer dump comparison
    # with DS3 (see docs §18), source layers 14-22 show grad inflation,
    # layer 23 is mild, layer 24+ is clean.
    #
    # Mechanism (hypothesis): patch (8)'s `_FYH_CURRENT_SHARED_KV` cell
    # interacts with AC `use_reentrant=False` checkpoint backward recompute
    # in a way that creates extra gradient flow through source layer
    # k_proj/k_norm.  Disabling AC on source layers avoids the interaction
    # at the cost of higher peak memory for those 10 layers.
    #
    # Activated only when GEMMA4_FSDP_NO_AC_KV_SOURCE=1.  Optional env
    # GEMMA4_FSDP_NO_AC_LAYER_RANGE=14:23 to override layer range.
    if os.environ.get("GEMMA4_FSDP_NO_AC_KV_SOURCE") == "1":
        try:
            from accelerate import accelerator as _acc_mod_ac
            import functools as _ft_ac

            _AC_SKIP_RANGE = os.environ.get("GEMMA4_FSDP_NO_AC_LAYER_RANGE", "14:23")
            _ac_lo, _ac_hi = (int(x) for x in _AC_SKIP_RANGE.split(":"))

            _orig_prepare_ac = _acc_mod_ac.Accelerator.prepare

            @_ft_ac.wraps(_orig_prepare_ac)
            def _patched_prepare_ac(self, *args, **kwargs):
                results = _orig_prepare_ac(self, *args, **kwargs)
                if getattr(self, "_fyh_no_ac_done", False):
                    return results
                # Walk all _models that were just prepared (FSDP-wrapped)
                # to find CheckpointWrapper-like modules in source layers
                # and replace with their inner module.
                unwrapped = 0
                models_to_walk = list(getattr(self, "_models", []))
                for arg in args:
                    import torch.nn as _nn_ac
                    if isinstance(arg, _nn_ac.Module) and arg not in models_to_walk:
                        models_to_walk.append(arg)
                for model in models_to_walk:
                    to_replace = []
                    for name, module in model.named_modules():
                        if not hasattr(module, "_checkpoint_wrapped_module"):
                            continue
                        if "language_model.layers." not in name:
                            continue
                        after_layers = name.split("language_model.layers.", 1)[1]
                        layer_idx_str = after_layers.split(".", 1)[0]
                        try:
                            layer_idx = int(layer_idx_str)
                        except ValueError:
                            continue
                        if not (_ac_lo <= layer_idx <= _ac_hi):
                            continue
                        parent_path, _, child_key = name.rpartition(".")
                        inner = module._checkpoint_wrapped_module
                        try:
                            parent = model.get_submodule(parent_path)
                        except AttributeError:
                            continue
                        to_replace.append((parent, child_key, inner, layer_idx))
                    for parent, key, inner, idx in to_replace:
                        setattr(parent, key, inner)
                        unwrapped += 1
                if _is_rank0:
                    print(f"[gemma4 sdp_preamble] (24) unwrapped "
                          f"{unwrapped} CheckpointWrapper(s) on layers "
                          f"{_ac_lo}-{_ac_hi} (KV-source) post-accelerator.prepare; "
                          f"skipped layers will run without AC backward recompute",
                          file=sys.stderr, flush=True)
                self._fyh_no_ac_done = True
                return results

            _acc_mod_ac.Accelerator.prepare = _patched_prepare_ac
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip (24) "
                  f"({type(_e).__name__}: {_e})", file=sys.stderr, flush=True)

    # (25) Selectively disable `reshard_after_forward` on KV-source layers.
    #
    # Root cause for the layers 0-13 deficit (see docs §20):
    # FSDP2 reshards source-layer (e.g. 22, 23) k_proj/v_proj weights after
    # forward.  When the 16 reader layers' backward fires, each reader's
    # backward chain re-gathers the source weight from sharded → creates a
    # NEW unsharded tensor with NEW autograd nodes per reader → 16 separate
    # backward paths to the weight.  Inflates source-layer grads 3-4x and
    # poisons the share→source backward edge our autograd-anchor patch
    # (KV_SHARE_ANCHOR=1) was supposed to merge.
    #
    # Fix: keep source-layer weights unsharded between forward and ALL
    # readers' backward.  Then 16 reader backwards all hit the same
    # unsharded tensor → autograd accumulates into one grad → 1x backward
    # call to k_proj/v_proj weight.
    #
    # Memory cost: only layers 22 and 23 need this (the only true KV
    # sources for E4B; verified via store_full_length_kv check in patch 8).
    # Each layer ~60M params bf16 = ~120 MB unsharded.  Sharded across 8
    # ranks would be ~15 MB.  Increment per rank = ~210 MB for 2 layers.
    #
    # GEMMA4_FSDP_NO_RESHARD_KV_SOURCE=1 to enable.
    # GEMMA4_FSDP_NO_RESHARD_LAYERS=22,23 to override layer set.
    if os.environ.get("GEMMA4_FSDP_NO_RESHARD_KV_SOURCE") == "1":
        try:
            from accelerate import accelerator as _acc_mod_nrs
            import functools as _ft_nrs

            _layers_str_nrs = os.environ.get(
                "GEMMA4_FSDP_NO_RESHARD_LAYERS", "22,23")
            _NO_RESHARD_LAYERS = set(int(x) for x in _layers_str_nrs.split(","))

            if not getattr(_acc_mod_nrs, "_fyh_no_reshard_patched", False):
                _orig_prepare_nrs = _acc_mod_nrs.Accelerator.prepare

                @_ft_nrs.wraps(_orig_prepare_nrs)
                def _patched_prepare_nrs(self, *args, **kwargs):
                    results = _orig_prepare_nrs(self, *args, **kwargs)
                    if getattr(self, "_fyh_no_reshard_done", False):
                        return results
                    # FSDP2 attaches state via `module._get_fsdp_state()` (private
                    # API) or the state lives at `module._modules` accessible
                    # via internal attrs.  Most robust path: each fully_shard'd
                    # module has a property `_fsdp_param_group` which holds
                    # `_reshard_after_forward` (bool).
                    import torch.nn as _nn_nrs
                    models_to_walk = list(getattr(self, "_models", []))
                    for arg in args:
                        if isinstance(arg, _nn_nrs.Module) and arg not in models_to_walk:
                            models_to_walk.append(arg)
                    n_changed = 0
                    for model in models_to_walk:
                        for name, module in model.named_modules():
                            if "language_model.layers." not in name:
                                continue
                            after_layers = name.split("language_model.layers.", 1)[1]
                            tail = after_layers.split(".", 1)
                            try:
                                li = int(tail[0])
                            except ValueError:
                                continue
                            if li not in _NO_RESHARD_LAYERS:
                                continue
                            # Only act on the decoder layer module itself, not
                            # its sub-modules (FSDP wrap is at decoder-layer level
                            # for TRANSFORMER_BASED_WRAP).
                            if len(tail) > 1 and tail[1] != "":
                                continue
                            # Try several internal-API spellings to flip the
                            # reshard policy.
                            flipped = False
                            for state_attr in ("_get_fsdp_state",):
                                _fn = getattr(module, state_attr, None)
                                if _fn is None:
                                    continue
                                try:
                                    state = _fn()
                                except Exception:
                                    continue
                                pg = getattr(state, "_fsdp_param_group", None)
                                if pg is None:
                                    continue
                                # PyTorch 2.10 FSDP2: setting
                                # `post_forward_mesh_info=None` disables reshard
                                # after forward (keeps tensors on the full mesh).
                                # `_reshard_after_forward` is a derived property
                                # without a setter and reads from this attr.
                                if hasattr(pg, "post_forward_mesh_info"):
                                    pg.post_forward_mesh_info = None
                                    flipped = True
                            if flipped:
                                n_changed += 1
                                if _is_rank0:
                                    print(f"[gemma4 sdp_preamble] (25) "
                                          f"L{li}: reshard_after_forward=False "
                                          f"({name})",
                                          file=sys.stderr, flush=True)
                    if _is_rank0:
                        print(f"[gemma4 sdp_preamble] (25/25) flipped "
                              f"reshard_after_forward=False on "
                              f"{n_changed} FSDP unit(s) "
                              f"(target layers={_NO_RESHARD_LAYERS}); "
                              f"keeps source-layer weights unsharded between "
                              f"forward and reader backward → autograd "
                              f"merges 16 reader paths into 1",
                              file=sys.stderr, flush=True)
                    self._fyh_no_reshard_done = True
                    return results

                _acc_mod_nrs.Accelerator.prepare = _patched_prepare_nrs
                _acc_mod_nrs._fyh_no_reshard_patched = True
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip (25) "
                  f"({type(_e).__name__}: {_e})", file=sys.stderr, flush=True)

    # (23) Force FSDP2 MixedPrecisionPolicy.cast_forward_inputs=False.
    # Default accelerate sets this to True so FSDP2's wrapped forward
    # `_apply_to_tensors` recursively rebuilds kwargs dicts to cast tensor
    # leaves to param_dtype.  This rebuilds the `shared_kv_states` dict
    # for each Gemma4TextDecoderLayer call, breaking KV-sharing across
    # layers 24-41 (KeyError 22).
    #
    # Patch (8) worked around this with a module-level cell, but the cell
    # interacts badly with activation_checkpointing's backward recompute
    # path: source layers 14-22 see ~3-4x inflated gradients (verified
    # via patch 22 layer-by-layer dump vs DS3 baseline).
    #
    # Cleaner fix: just disable cast_forward_inputs entirely.  Our model
    # is already loaded in bf16, so the cast is a no-op anyway; the only
    # purpose this flag serves for us is the dict-rebuild side effect,
    # which is exactly what we don't want.
    #
    # With this patch active, patch (8) becomes redundant and should be
    # disabled (set GEMMA4_SKIP_KV_SHARE_PATCH=1).
    if os.environ.get("GEMMA4_FSDP_NO_CAST_FORWARD_INPUTS") == "1":
        try:
            from accelerate.utils import dataclasses as _acc_dc_cfi
            import torch as _t_for_cfi
            if not getattr(_acc_dc_cfi, "_fyh_no_cast_forward_inputs_patched", False):
                _orig_set_mp_cfi = _acc_dc_cfi.FullyShardedDataParallelPlugin.set_mixed_precision

                def _patched_set_mp_cfi(self, mixed_precision, buffer_autocast=False, override=False):
                    _orig_set_mp_cfi(self, mixed_precision, buffer_autocast=buffer_autocast, override=override)
                    mp = self.mixed_precision_policy
                    if mp is None:
                        return
                    try:
                        if hasattr(mp, "cast_forward_inputs") and mp.cast_forward_inputs:
                            try:
                                mp.cast_forward_inputs = False
                                if _is_rank0:
                                    print(
                                        "[gemma4 sdp_preamble] (23) forced FSDP2 "
                                        "MixedPrecisionPolicy.cast_forward_inputs=False "
                                        "(model already bf16; avoids kwargs dict rebuild "
                                        "that broke KV-sharing in patch 8)",
                                        file=sys.stderr, flush=True,
                                    )
                            except Exception:
                                # MixedPrecisionPolicy may be a frozen dataclass
                                if self.fsdp_version == 2:
                                    from torch.distributed.fsdp import MixedPrecisionPolicy as _MP_cfi
                                else:
                                    from torch.distributed.fsdp import MixedPrecision as _MP_cfi
                                kwargs = {}
                                for k in ("param_dtype", "reduce_dtype", "buffer_dtype", "output_dtype"):
                                    v = getattr(mp, k, None)
                                    if v is not None:
                                        kwargs[k] = v
                                kwargs["cast_forward_inputs"] = False
                                self.mixed_precision_policy = _MP_cfi(**kwargs)
                                if _is_rank0:
                                    print(
                                        "[gemma4 sdp_preamble] (23) rebuilt "
                                        "FSDP2 MixedPrecisionPolicy with "
                                        "cast_forward_inputs=False",
                                        file=sys.stderr, flush=True,
                                    )
                    except Exception as _e:
                        print(f"[gemma4 sdp_preamble] WARN: (23) "
                              f"cast_forward_inputs override failed "
                              f"({type(_e).__name__}: {_e})",
                              file=sys.stderr, flush=True)

                _acc_dc_cfi.FullyShardedDataParallelPlugin.set_mixed_precision = _patched_set_mp_cfi
                _acc_dc_cfi._fyh_no_cast_forward_inputs_patched = True
        except ImportError:
            pass
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip (23) "
                  f"({type(_e).__name__}: {_e})",
                  file=sys.stderr, flush=True)

    if os.environ.get("GEMMA4_FSDP_REDUCE_FP32") == "1":
        try:
            from accelerate.utils import dataclasses as _acc_dc
            import torch as _t_for_mp
            if not getattr(_acc_dc, "_fyh_fsdp_reduce_fp32_patched", False):
                _orig_set_mp = _acc_dc.FullyShardedDataParallelPlugin.set_mixed_precision

                def _patched_set_mp(self, mixed_precision, buffer_autocast=False, override=False):
                    _orig_set_mp(self, mixed_precision, buffer_autocast=buffer_autocast, override=override)
                    mp = self.mixed_precision_policy
                    if mp is None:
                        return
                    try:
                        if getattr(mp, "reduce_dtype", None) is _t_for_mp.float32:
                            return
                        try:
                            mp.reduce_dtype = _t_for_mp.float32
                            if _is_rank0:
                                print(
                                    "[gemma4 sdp_preamble] (15/15) forced "
                                    "FSDP2 MixedPrecisionPolicy.reduce_dtype "
                                    "= float32 (param_dtype="
                                    f"{mp.param_dtype}; matches DS3 fp32 "
                                    "grad-reduce path)",
                                    file=sys.stderr, flush=True,
                                )
                        except Exception:
                            if self.fsdp_version == 2:
                                from torch.distributed.fsdp import MixedPrecisionPolicy as _MP
                            else:
                                from torch.distributed.fsdp import MixedPrecision as _MP
                            kwargs = {"param_dtype": getattr(mp, "param_dtype", None),
                                      "reduce_dtype": _t_for_mp.float32}
                            for k in ("buffer_dtype", "output_dtype", "cast_forward_inputs"):
                                v = getattr(mp, k, None)
                                if v is not None:
                                    kwargs[k] = v
                            self.mixed_precision_policy = _MP(**{kk: vv for kk, vv in kwargs.items() if vv is not None})
                            if _is_rank0:
                                print(
                                    "[gemma4 sdp_preamble] (15/15) rebuilt "
                                    "FSDP2 MixedPrecisionPolicy with "
                                    "reduce_dtype=float32 (matches DS3 fp32 "
                                    "grad-reduce path)",
                                    file=sys.stderr, flush=True,
                                )
                    except Exception as _ee:
                        print(f"[gemma4 sdp_preamble] WARN: (15) reduce_dtype "
                              f"override failed ({type(_ee).__name__}: {_ee})",
                              file=sys.stderr, flush=True)

                _acc_dc.FullyShardedDataParallelPlugin.set_mixed_precision = _patched_set_mp
                _acc_dc._fyh_fsdp_reduce_fp32_patched = True
        except ImportError:
            pass
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip FSDP2 reduce_fp32 patch "
                  f"({type(_e).__name__}: {_e})",
                  file=sys.stderr, flush=True)

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

    # (13) Pre-wrap gemma4 embed_tokens_per_layer (PLE) as its own FSDP unit.
    #
    # Default fsdp_config wraps Gemma4*Layer classes; the PLE
    # (`model.language_model.embed_tokens_per_layer`, ~2.8B params, 5.6 GiB
    # bf16) is plain `nn.Embedding`, not in any wrap list, so it ends up
    # inside the *root* FSDP unit (= Gemma4ForConditionalGeneration).  Root
    # all-gather during backward then has to pull 5.6+ GiB at once —
    # observed as the OOM at §10 / §15 (`torch.OutOfMemoryError: Tried to
    # allocate 5.25 GiB / 904 MiB ... root unit`).
    #
    # DeepSpeed ZeRO-3 chunks gathers via `stage3_max_live_parameters: 1e8`;
    # FSDP2 has no equivalent knob.  But we CAN pull the PLE out of the
    # root group by giving it its own `fully_shard()` call.  Then root
    # gathers ~700 MB (lm_head/embed_tokens etc.) and PLE gathers
    # separately, each tiny.
    #
    # Activated only when GEMMA4_FSDP_WRAP_PLE=1.
    if os.environ.get("GEMMA4_FSDP_WRAP_PLE") == "1":
        try:
            from accelerate.utils import fsdp_utils as _fsdp_utils

            if not getattr(_fsdp_utils, "_fyh_ple_wrap_patched", False):
                _orig_prep_policy = _fsdp_utils.fsdp2_prepare_auto_wrap_policy

                def _patched_prep_policy(plugin, model):
                    base_func = _orig_prep_policy(plugin, model)
                    # Locate the PLE.  Layout differs between text-only and
                    # multimodal:
                    #   multimodal: model.model.language_model.embed_tokens_per_layer
                    #   text-only : model.language_model.embed_tokens_per_layer
                    ple_module = None
                    for path in (
                        ("model", "language_model", "embed_tokens_per_layer"),
                        ("language_model", "embed_tokens_per_layer"),
                    ):
                        m = model
                        try:
                            for attr in path:
                                m = getattr(m, attr)
                            ple_module = m
                            break
                        except AttributeError:
                            continue
                    if ple_module is None:
                        if _is_rank0:
                            print(
                                "[gemma4 sdp_preamble] (13) WARN: could not "
                                "locate embed_tokens_per_layer; PLE wrap "
                                "skipped",
                                file=sys.stderr, flush=True,
                            )
                        return base_func

                    def _augmented(module):
                        if module is ple_module:
                            return True
                        if base_func is None:
                            return False
                        return base_func(module)

                    if _is_rank0:
                        n_params = sum(p.numel() for p in ple_module.parameters())
                        print(
                            f"[gemma4 sdp_preamble] (13/13) augmented FSDP2 "
                            f"auto_wrap_policy: pull "
                            f"embed_tokens_per_layer ({n_params/1e9:.2f}B params) "
                            f"out of root unit into its own FSDP unit "
                            f"(prevents 5.6+ GiB root-gather peak)",
                            file=sys.stderr, flush=True,
                        )
                    return _augmented

                _fsdp_utils.fsdp2_prepare_auto_wrap_policy = _patched_prep_policy
                _fsdp_utils._fyh_ple_wrap_patched = True
        except ImportError:
            pass
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip PLE wrap patch "
                  f"({type(_e).__name__}: {_e})",
                  file=sys.stderr, flush=True)

    # (8) Gemma-4-E4B (any model with num_kv_shared_layers>0) × FSDP2
    #     mixed-precision shared_kv_states fix.
    #
    # Bug: gemma4 modeling routes a `shared_kv_states: dict[int, (k,v)]`
    # through Gemma4TextDecoderLayer kwargs.  Last-non-shared layer
    # (e.g. layer 22 for E4B) writes via dict side-effect; later
    # kv-shared layers (24..41) read.
    #
    # FSDP2 default MixedPrecisionPolicy(cast_forward_inputs=True,
    # param_dtype=bf16) triggers `torch.distributed.utils._apply_to_tensors(
    # cast_fn, kwargs)` in `_FSDPState._pre_forward`.  That helper
    # *unconditionally* rebuilds dicts via `{k: apply(v) for k,v in x.items()}`
    # — even dicts that contain no tensors.  Result: each decoder layer call
    # receives a NEW empty `shared_kv_states` dict, layer-22 writes into a
    # throwaway copy, layer-24 reads an empty fresh copy → KeyError 22.
    #
    # 26B-A4B-it has num_kv_shared_layers=0, so no-op for that variant; only
    # E2B/E4B trip this.
    #
    # Fix: bypass the kwarg channel entirely — attach a single shared dict to
    # each Gemma4TextAttention instance via `self._fyh_shared_kv` (instance
    # attribute lookup is unaffected by FSDP kwargs cloning), and have
    # attention forward read/write from that attribute when present.
    # Opt-out: GEMMA4_SKIP_KV_SHARE_PATCH=1 (e.g., for DS3, which doesn't have
    # FSDP2's `cast_forward_inputs` dict-rebuild issue and works fine with the
    # original gemma4 forward).  Applying this patch on DS3 breaks layer 24-41
    # kv-share reads and causes loss to explode (~27 vs ~2.2 expected).
    try:
        import transformers.models.gemma4.modeling_gemma4 as _mg4

        if (os.environ.get("GEMMA4_SKIP_KV_SHARE_PATCH") == "1"):
            if _is_rank0:
                print(
                    "[gemma4 sdp_preamble] (8/8) SKIPPED kv-share patch "
                    "(GEMMA4_SKIP_KV_SHARE_PATCH=1; for DS3 / non-FSDP2 path)",
                    file=sys.stderr, flush=True,
                )
        elif not getattr(_mg4, "_fyh_kv_share_patch_applied", False):
            _orig_textmodel_forward = _mg4.Gemma4TextModel.forward
            _orig_attn_forward = _mg4.Gemma4TextAttention.forward

            import torch as _torch_kva
            import torch.nn.functional as _F_kva

            # ─── Identity anchor (used by GEMMA4_KV_SHARE_ANCHOR=1, kept for
            #     debugging; does NOT fix grad mismatch). ───
            class _KVShareAnchor(_torch_kva.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, grad_output):
                    return grad_output

            # ─── Weight-detached linear (used by GEMMA4_KV_SHARE_BETA0_DELTA1=1).
            #
            # forward: F.linear(x, W) — same value as nn.Linear
            # backward: dx via W (KEPT); dW := None (CUT)
            #
            # Goal: when source-layer (layer 22/23) stores k/v in cell, route
            # k_proj/v_proj computation through this op for the cell-bound
            # branch.  16 reader-side dk_share gradients then accumulate at
            # cell tensor → flow back through k_norm (gamma kept) → back to
            # this op → dx propagates to layer 22 input (residual to layers
            # 0-13 ✓) BUT k_proj.weight does not see share contribution
            # (matches DS3 stage_3's "share→source W = 0" behaviour ✓).
            #
            # Layer 22's OWN attention path remains untouched (uses normal
            # F.linear), so k_proj.weight gets its full own-attention grad
            # (matches DS3 own-attention contribution).
            class _DetachWeightLinear(_torch_kva.autograd.Function):
                @staticmethod
                def forward(ctx, x, W):
                    ctx.save_for_backward(W)
                    return _F_kva.linear(x, W)

                @staticmethod
                def backward(ctx, dy):
                    W, = ctx.saved_tensors
                    # F.linear: y = x @ W.T  =>  dx = dy @ W
                    dx = dy @ W
                    return dx, None  # CUT W.grad accumulation

            # Module-level "current" shared-KV dict.  textmodel.forward sets
            # this to a fresh dict at the start of each call; attn forward
            # overrides the kwargs `shared_kv_states` with this current dict.
            # Avoids both (a) FSDP2 cast_forward_inputs cloning kwarg dicts
            # AND (b) the observation that the inner self_attn module identity
            # at seed-time (textmodel.forward) does not match self_attn at
            # forward-time (after FSDP2 + CheckpointWrapper wrap).  Single-
            # threaded forward path makes module-level state safe (no
            # interleaved batches).
            _mg4._FYH_CURRENT_SHARED_KV = None

            def _patched_textmodel_forward(self, *args, **kwargs):
                # GEMMA4_KV_SHARE_CLEAR_ON_EXIT=1: clear cell after forward
                # exit (forces AC backward recompute to regenerate kv via
                # its own forward call, which reaches into the same cell
                # via this wrapper, ending up writing fresh tensors with
                # an autograd graph local to the recompute).  Hypothesis:
                # avoids the source-layer grad inflation seen when cell
                # persists across forward/AC-backward boundaries.
                _fyh_skv = {}
                _prev = _mg4._FYH_CURRENT_SHARED_KV
                _mg4._FYH_CURRENT_SHARED_KV = _fyh_skv
                if os.environ.get("GEMMA4_KV_SHARE_DEBUG") == "1" and _is_rank0:
                    print(f"[gemma4 kv-share] textmodel.forward set "
                          f"_FYH_CURRENT_SHARED_KV id={id(_fyh_skv)}",
                          file=sys.stderr, flush=True)
                _clear_on_exit = os.environ.get("GEMMA4_KV_SHARE_CLEAR_ON_EXIT") == "1"
                try:
                    return _orig_textmodel_forward(self, *args, **kwargs)
                finally:
                    if _clear_on_exit:
                        _mg4._FYH_CURRENT_SHARED_KV = _prev

            # Per-layer attn-forward call counter (to detect AC recompute).
            # Maps (layer_idx, micro_idx) -> count
            from collections import defaultdict as _dd_attnfc
            _ATTN_FWD_CNT = _dd_attnfc(int)
            _mg4._FYH_ATTN_FWD_CNT = _ATTN_FWD_CNT

            def _patched_attn_forward(self, *args, **kwargs):
                # Override shared_kv_states kwarg with the textmodel-set
                # current dict.  This works even when the kwarg was cloned
                # to a new empty dict by FSDP2's cast_forward_inputs (the
                # rebuild path that broke the original implementation).
                _ref = _mg4._FYH_CURRENT_SHARED_KV
                if _ref is not None:
                    kwargs["shared_kv_states"] = _ref
                # Count this attn forward call (for AC recompute detection)
                if os.environ.get("GEMMA4_ATTN_FWD_COUNT") == "1":
                    _li_cnt = getattr(self, "layer_idx", -1)
                    _ATTN_FWD_CNT[_li_cnt] += 1
                    # rank 0 only, layer 22 only, print count
                    if _li_cnt == 22 and _is_rank0:
                        print(f"[gemma4 attn-fwd] L22 fire #{_ATTN_FWD_CNT[_li_cnt]}",
                              file=sys.stderr, flush=True)
                _out = _orig_attn_forward(self, *args, **kwargs)
                # GEMMA4_KV_SHARE_DETACH=1: after a source layer stores its
                # (key, value) into the cell, replace with detached copies.
                # This breaks the share→source backward path that DS3
                # ZeRO-3 stage_3 also doesn't propagate (verified via
                # layer-by-layer dump: layer 22 k_proj has 10x inflation
                # in FSDP2 vs DS3, matching the ~15 sliding kv-shared
                # readers count).  With detach, the inflated extra path
                # is removed and FSDP2 grads match DS3.
                if (os.environ.get("GEMMA4_KV_SHARE_DETACH") == "1"
                    and _ref is not None
                    and getattr(self, "store_full_length_kv", False)):
                    _li = getattr(self, "layer_idx", None)
                    if _li is not None and _li in _ref:
                        try:
                            _k, _v = _ref[_li]
                            _ref[_li] = (_k.detach(), _v.detach())
                            if (os.environ.get("GEMMA4_KV_DETACH_DEBUG") == "1"
                                and _is_rank0):
                                print(f"[gemma4 kv-detach] L{_li}: detached "
                                      f"k/v in cell after orig_forward",
                                      file=sys.stderr, flush=True)
                        except Exception as _e:
                            if _is_rank0:
                                print(f"[gemma4 kv-detach] L{_li}: WARN "
                                      f"detach failed ({type(_e).__name__}: {_e})",
                                      file=sys.stderr, flush=True)
                # GEMMA4_KV_SHARE_ANCHOR=1: alternative to detach.  Wrap the
                # source-layer (k, v) through a custom autograd Function
                # whose forward+backward are pure identity.  Goal: force
                # FSDP2's autograd graph to merge the 16 reader paths into
                # a SINGLE gradient that flows back ONCE through k_proj /
                # v_proj weights — same effect DS3 stage_3 achieves natively.
                # Performance impact: ~zero (2 extra autograd nodes per
                # source layer per micro, no compute, no memory).
                #
                # Mutually exclusive with KV_SHARE_DETACH; if both set,
                # detach takes priority and ANCHOR is no-op.
                elif (os.environ.get("GEMMA4_KV_SHARE_ANCHOR") == "1"
                      and _ref is not None
                      and getattr(self, "store_full_length_kv", False)):
                    _li = getattr(self, "layer_idx", None)
                    if _li is not None and _li in _ref:
                        try:
                            _k, _v = _ref[_li]
                            _ref[_li] = (
                                _KVShareAnchor.apply(_k),
                                _KVShareAnchor.apply(_v),
                            )
                            if (os.environ.get("GEMMA4_KV_ANCHOR_DEBUG") == "1"
                                and _is_rank0):
                                print(f"[gemma4 kv-anchor] L{_li}: anchored "
                                      f"k/v in cell after orig_forward",
                                      file=sys.stderr, flush=True)
                        except Exception as _e:
                            if _is_rank0:
                                print(f"[gemma4 kv-anchor] L{_li}: WARN "
                                      f"anchor failed ({type(_e).__name__}: {_e})",
                                      file=sys.stderr, flush=True)
                # GEMMA4_KV_SHARE_BETA0_DELTA1=1: replicate DS3 stage_3
                # "β=0, δ=1" behaviour exactly:
                #   - β=0: layer 22/23 k_proj.weight grad has NO share-back
                #          contribution (matches DS3 stage_3's natural cut)
                #   - δ=1: cell tensor grad still propagates to source layer's
                #          input (residual to layers 0-13 preserved, fixing
                #          the "0.30x deficit" of plain detach mode)
                #
                # Strategy: re-compute (k, v) FOR CELL via _DetachWeightLinear
                # (forward identical, backward dx kept but dW cut).  Replace
                # the cell entry written by orig_forward with this new pair.
                # Layer's OWN attention computation already used the
                # normally-computed (k, v) and is unchanged.
                #
                # Memory cost: ~50-100 MB extra activations on source layers
                # (k_view, k_norm_out for backward).
                # Speed cost: extra k_proj/k_norm/rotary forward (~1%).
                #
                # Mutually exclusive with KV_SHARE_DETACH and KV_SHARE_ANCHOR.
                elif (os.environ.get("GEMMA4_KV_SHARE_BETA0_DELTA1") == "1"
                      and _ref is not None
                      and getattr(self, "store_full_length_kv", False)):
                    _li = getattr(self, "layer_idx", None)
                    if _li is not None and _li in _ref:
                        try:
                            # We need: hidden_states, cos, sin from the call args
                            # forward signature: (self, hidden_states, position_embeddings,
                            #                    attention_mask, shared_kv_states, ...)
                            _hidden = args[0] if args else kwargs.get("hidden_states")
                            _pos_emb = args[1] if len(args) > 1 else kwargs.get("position_embeddings")
                            _cos, _sin = _pos_emb
                            _input_shape = _hidden.shape[:-1]
                            _hidden_shape = (*_input_shape, -1, self.head_dim)
                            from transformers.models.gemma4.modeling_gemma4 import (
                                apply_rotary_pos_emb as _aroteb,
                            )
                            # Recompute k for cell with weight-detached linear
                            _k_raw = _DetachWeightLinear.apply(_hidden, self.k_proj.weight)
                            _k_view = _k_raw.view(_hidden_shape)
                            _k_normed = self.k_norm(_k_view)
                            _k_rot = _aroteb(_k_normed, _cos, _sin, unsqueeze_dim=2)
                            _k_for_cell = _k_rot.transpose(1, 2)
                            # Recompute v for cell.  v_proj may be None
                            # (use_alternative_attention) — but that path
                            # also disables store_full_length_kv via Gemma4
                            # init logic, so we shouldn't hit it here.
                            if self.v_proj is not None:
                                _v_raw = _DetachWeightLinear.apply(_hidden, self.v_proj.weight)
                                _v_view = _v_raw.view(_hidden_shape)
                                _v_normed = self.v_norm(_v_view)
                                _v_for_cell = _v_normed.transpose(1, 2)
                            else:
                                # Aliased to k path; should not reach here for
                                # E4B Gemma4, but be safe.
                                _v_for_cell = _k_for_cell
                            # Replace cell entry — own attention already used
                            # the original (k, v) computed in orig_forward
                            _ref[_li] = (_k_for_cell, _v_for_cell)
                            if (os.environ.get("GEMMA4_KV_BD_DEBUG") == "1"
                                and _is_rank0):
                                print(f"[gemma4 kv-β0δ1] L{_li}: replaced "
                                      f"cell entry with weight-detached k/v "
                                      f"(shape k={_k_for_cell.shape})",
                                      file=sys.stderr, flush=True)
                        except Exception as _e:
                            # AC's _StopRecomputationError is control-flow
                            # signalling — must propagate, NOT catch.
                            try:
                                from torch.utils.checkpoint import (
                                    _StopRecomputationError as _SRE_bd,
                                )
                                if isinstance(_e, _SRE_bd):
                                    raise
                            except ImportError:
                                pass
                            if _is_rank0:
                                import traceback as _tb_bd
                                print(f"[gemma4 kv-β0δ1] L{_li}: WARN "
                                      f"replacement failed "
                                      f"({type(_e).__name__}: {_e})\n"
                                      f"{_tb_bd.format_exc()}",
                                      file=sys.stderr, flush=True)
                if os.environ.get("GEMMA4_KV_SHARE_DEBUG") == "1":
                    _idx = getattr(self, "layer_idx", "?")
                    _ref_id = id(_ref) if _ref is not None else None
                    _ref_keys = sorted(_ref.keys()) if isinstance(_ref, dict) else None
                    _is_shared = getattr(self, "is_kv_shared_layer", None)
                    _is_writer = getattr(self, "store_full_length_kv", None)
                    _kv_idx = getattr(self, "kv_shared_layer_index", None)
                    _is_rank_local = os.environ.get("LOCAL_RANK", "0") == "0"
                    if _is_rank_local and _idx in (22, 23, 24, 41):
                        print(f"[gemma4 kv-share] L{_idx} attn forward: "
                              f"is_shared={_is_shared} writer={_is_writer} "
                              f"kv_shared_idx={_kv_idx} "
                              f"skv id={_ref_id} keys={_ref_keys}",
                              file=sys.stderr, flush=True)
                return _out

            _mg4.Gemma4TextModel.forward = _patched_textmodel_forward
            _mg4.Gemma4TextAttention.forward = _patched_attn_forward
            _mg4._fyh_kv_share_patch_applied = True

            if _is_rank0:
                print(
                    "[gemma4 sdp_preamble] (8/8) patched Gemma4TextModel."
                    "forward + Gemma4TextAttention.forward to route "
                    "shared_kv_states via module-level cell "
                    "(_mg4._FYH_CURRENT_SHARED_KV).  Bypasses both FSDP2 "
                    "cast_forward_inputs dict-rebuild AND post-FSDP "
                    "self_attn identity remap that broke instance-attr "
                    "approach for E4B kv-shared layers 24..41.",
                    file=sys.stderr, flush=True,
                )
    except ImportError:
        # transformers / gemma4 not loaded yet — patch is no-op for non-gemma4
        # processes (e.g. dataloader workers that don't import the model).
        pass
    except Exception as _e:
        print(f"[gemma4 sdp_preamble] WARN: skip kv-share patch "
              f"({type(_e).__name__}: {_e})",
              file=sys.stderr, flush=True)

    # (9) accelerate fsdp2_load_full_state_dict × tied embeddings fix.
    #
    # Bug: with cpu_ram_efficient_loading=True, accelerate calls
    # fsdp2_prepare_model() which:
    #   1. moves model to meta device,
    #   2. calls model.tie_weights() (so lm_head.weight is embed_tokens.weight),
    #   3. wraps each transformer layer with fully_shard(),
    #   4. wraps the root model with fully_shard(),
    #   5. calls fsdp2_load_full_state_dict() to broadcast weights from rank 0.
    #
    # In step 4, fully_shard() shards root-level params via setattr. For tied
    # weights, only one of the two underlying nn.Parameter slots gets replaced
    # with a DTensor — the OTHER one (e.g. lm_head.weight) is now a stale
    # regular Tensor pointing at the freed pre-shard storage. model.state_dict()
    # at step 5 still returns BOTH keys, so the load loop iterates over a
    # mixture of DTensors and plain Tensors, then trips when it does
    # `sharded_param.device_mesh` on the plain Tensor:
    #     File ".../accelerate/utils/fsdp_utils.py", line 537
    #     AttributeError: 'Tensor' object has no attribute 'device_mesh'
    #
    # The post-load code already calls model.tie_weights() again, which would
    # re-tie lm_head.weight = embed_tokens.weight as the DTensor — the load
    # loop just doesn't survive long enough to reach it.
    #
    # Fix: wrap fsdp2_load_full_state_dict so that:
    #   * non-DTensor entries are passed through as-is (no broadcast / no
    #     distribute_tensor) — they are skipped on every rank uniformly so
    #     dist.broadcast pairs stay matched;
    #   * model.load_state_dict(..., assign=True) is called with strict=False
    #     so missing tied keys (if full_sd has only one of the pair) and
    #     unexpected keys (if full_sd has the saved-once tied weight under
    #     a different name) don't raise.
    # tie_weights() afterwards reattaches the DTensor to the tied attribute.
    #
    # Applies to gemma-4 (tie_word_embeddings=True). Same root-cause class as
    # patches (8) and earlier — accelerate / FSDP2 plumbing assumes invariants
    # that break on real multimodal-with-tied-LM-head models.
    try:
        from accelerate.utils import fsdp_utils as _fsdp_utils

        if not getattr(_fsdp_utils, "_fyh_load_full_sd_patched", False):
            _orig_load_full_sd = _fsdp_utils.fsdp2_load_full_state_dict

            def _patched_load_full_sd(accelerator, model, full_sd, cpu_offload=False):
                import torch.distributed as dist
                from torch.distributed.tensor import DTensor, distribute_tensor

                meta_sharded_sd = model.state_dict()
                sharded_sd = {}

                def _infer_dtype(model_, name_, empty_):
                    try:
                        old_param = model_.get_parameter_or_buffer(name_)
                    except AttributeError:
                        base, local = name_.rsplit(".", 1)
                        sub = model_.get_submodule(base)
                        old_param = getattr(sub, local)
                    cast_dt = None
                    is_e4m3 = (hasattr(__import__('torch'), 'float8_e4m3fn')
                               and empty_.dtype == __import__('torch').float8_e4m3fn)
                    if empty_.dtype.is_floating_point and not is_e4m3:
                        cast_dt = old_param.dtype
                    return old_param is not None and old_param.is_contiguous(), cast_dt

                def _cast_contig(t, contig, dt):
                    if dt is not None:
                        t = t.to(dtype=dt)
                    if contig:
                        t = t.contiguous()
                    return t

                _n_skipped = 0
                if accelerator.is_main_process:
                    for (pname, full_p), sp in zip(full_sd.items(), meta_sharded_sd.values()):
                        if not isinstance(sp, DTensor):
                            sharded_sd[pname] = sp
                            _n_skipped += 1
                            continue
                        dm = sp.device_mesh
                        full_p = full_p.detach().to(dm.device_type)
                        if isinstance(full_p, DTensor):
                            full_p = full_p.to_local()
                        dist.broadcast(full_p, src=0, group=dist.group.WORLD)
                        st = distribute_tensor(full_p, dm, sp.placements)
                        c, dt = _infer_dtype(model, pname, full_p)
                        st = _cast_contig(st, c, dt)
                        if cpu_offload:
                            st = st.to("cpu")
                        sharded_sd[pname] = st
                else:
                    import torch as _t_local
                    for pname, sp in meta_sharded_sd.items():
                        if not isinstance(sp, DTensor):
                            sharded_sd[pname] = sp
                            _n_skipped += 1
                            continue
                        dm = sp.device_mesh
                        ft = _t_local.empty(sp.size(), device=dm.device_type, dtype=sp.dtype)
                        dist.broadcast(ft, src=0, group=dist.group.WORLD)
                        st = distribute_tensor(ft, dm, sp.placements)
                        c, dt = _infer_dtype(model, pname, ft)
                        st = _cast_contig(st, c, dt)
                        if cpu_offload:
                            st = st.to("cpu")
                        sharded_sd[pname] = st

                model.load_state_dict(sharded_sd, assign=True, strict=False)
                if _is_rank0:
                    print(
                        f"[gemma4 sdp_preamble] (9) fsdp2_load_full_state_dict "
                        f"completed: {_n_skipped} non-DTensor entries skipped "
                        f"(tied weights / unsharded buffers — handled by "
                        f"model.tie_weights() post-load)",
                        file=sys.stderr, flush=True,
                    )
                return model

            _fsdp_utils.fsdp2_load_full_state_dict = _patched_load_full_sd
            _fsdp_utils._fyh_load_full_sd_patched = True
            if _is_rank0:
                print(
                    "[gemma4 sdp_preamble] (9/9) patched accelerate."
                    "utils.fsdp_utils.fsdp2_load_full_state_dict to skip "
                    "non-DTensor entries (tied lm_head.weight after "
                    "fully_shard) and use strict=False on load_state_dict",
                    file=sys.stderr, flush=True,
                )
    except ImportError:
        pass
    except Exception as _e:
        print(f"[gemma4 sdp_preamble] WARN: skip fsdp2 load tied-weight patch "
              f"({type(_e).__name__}: {_e})",
              file=sys.stderr, flush=True)

    # (28) Layer-level backward fire-count instrumentation.
    #
    # Goal: determine if FSDP2's backward post-hook fires more than once per
    # micro per layer in no-detach mode (vs DS3 which fires once).  If yes,
    # this is the source of the layer 14-21 inflation we cannot fix at the
    # cell level.
    #
    # Mechanism: register a `register_post_accumulate_grad_hook` on every
    # layer's params; the hook increments a per-(layer_idx, param_name,
    # micro_idx) counter.  At end of micro, dump.
    #
    # Activate via GEMMA4_BWD_FIRE_COUNT=1.
    if os.environ.get("GEMMA4_BWD_FIRE_COUNT") == "1":
        try:
            import os as _os_fc
            import sys as _sys_fc
            import functools as _ft_fc
            import torch as _torch_fc
            import transformers as _tfm_fc
            from collections import defaultdict as _dd_fc

            _FC_DIR = _os_fc.environ.get("GEMMA4_BWD_FIRE_DIR", "/tmp/bwd_fire_count")
            _os_fc.makedirs(_FC_DIR, exist_ok=True)
            _FC_RANK = int(_os_fc.environ.get("RANK",
                          _os_fc.environ.get("LOCAL_RANK", "0")))
            _FC_STATE = {
                "step": 0, "micro": 0, "fh": None,
                "registered": False,
                "counts": _dd_fc(int),  # (layer_idx, param_name, micro) -> count
            }

            _orig_T_init_fc = _tfm_fc.Trainer.__init__

            @_ft_fc.wraps(_orig_T_init_fc)
            def _patched_T_init_fc(self, *args, **kwargs):
                _orig_T_init_fc(self, *args, **kwargs)
                if _FC_STATE["registered"]:
                    return
                _FC_STATE["registered"] = True
                fpath = _os_fc.path.join(_FC_DIR, f"bwd_fire_rank{_FC_RANK}.tsv")
                _FC_STATE["fh"] = open(fpath, "w", buffering=1)
                _FC_STATE["fh"].write("step\tmicro\tlayer\tevent\tcount\n")
                # Register MODULE-level hooks on each transformer layer.
                # `register_full_backward_hook` fires when the module's
                # backward fires.  In normal autograd: 1x per micro per layer.
                # If FSDP2 backward duplicates: N>1.  Also register
                # forward_pre to count how many times forward runs (AC
                # recompute multiplies this).
                n_hooks = 0
                # Walk decoder layers
                def _walk_layers(model):
                    out = []
                    for name, mod in model.named_modules():
                        if "language_model.layers." not in name:
                            continue
                        after = name.split("language_model.layers.", 1)[1]
                        # Match `layers.N` (no further dots, or only checkpoint wrapper)
                        if "." in after:
                            tail = after.split(".", 1)[1]
                            if tail and tail != "_checkpoint_wrapped_module":
                                continue
                        try:
                            li = int(after.split(".", 1)[0])
                        except ValueError:
                            continue
                        out.append((li, name, mod))
                    return out

                seen_li = set()
                for li, name, mod in _walk_layers(self.model):
                    if li in seen_li: continue  # only one entry per layer
                    seen_li.add(li)
                    def _make_fwd_hook(li_):
                        def _h(module, args):
                            mi = _FC_STATE["micro"]
                            si = _FC_STATE["step"]
                            _FC_STATE["counts"][(si, mi, li_, "forward")] += 1
                        return _h
                    def _make_bwd_hook(li_):
                        def _h(module, grad_input, grad_output):
                            mi = _FC_STATE["micro"]
                            si = _FC_STATE["step"]
                            _FC_STATE["counts"][(si, mi, li_, "backward")] += 1
                        return _h
                    mod.register_forward_pre_hook(_make_fwd_hook(li))
                    mod.register_full_backward_hook(_make_bwd_hook(li))
                    n_hooks += 2

                # Also hook FSDP2 _fsdp_param_group.{foreach_reduce, post_backward,
                # pre_backward} to count fires per layer per micro.
                try:
                    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup as _PG_fc
                    if not getattr(_PG_fc, "_fyh_fc_patched", False):
                        # Build (param_group_module fqn root) -> layer_idx map
                        # by post-FSDP traversal. Each layer has one FSDPParamGroup
                        # whose modules contain the layer's modules.
                        # We'll resolve via fsdp_params[0]._module_info.module name → layer_idx
                        _id2li = {}
                        for li, name, mod in _walk_layers(self.model):
                            if li in _id2li.values(): continue
                            _id2li[id(mod)] = li
                            # also walk children & add their ids → li
                            for sub in mod.modules():
                                if id(sub) not in _id2li:
                                    _id2li[id(sub)] = li
                        _FC_STATE["id2li"] = _id2li

                        def _resolve_li(self_pg):
                            for fp in getattr(self_pg, "fsdp_params", []):
                                mi = getattr(fp, "_module_info", None)
                                if mi is None: continue
                                m = getattr(mi, "module", None)
                                if m is None: continue
                                if id(m) in _FC_STATE["id2li"]:
                                    return _FC_STATE["id2li"][id(m)]
                            return -1

                        _orig_pre_bwd = _PG_fc.pre_backward
                        _orig_post_bwd = _PG_fc.post_backward

                        def _patched_pre_bwd(self_pg, *args, **kwargs):
                            li = _resolve_li(self_pg)
                            _FC_STATE["counts"][(_FC_STATE["step"],
                                _FC_STATE["micro"], li, "fsdp_pre_bwd")] += 1
                            return _orig_pre_bwd(self_pg, *args, **kwargs)

                        def _patched_post_bwd(self_pg, *args, **kwargs):
                            li = _resolve_li(self_pg)
                            _FC_STATE["counts"][(_FC_STATE["step"],
                                _FC_STATE["micro"], li, "fsdp_post_bwd")] += 1
                            return _orig_post_bwd(self_pg, *args, **kwargs)

                        _PG_fc.pre_backward = _patched_pre_bwd
                        _PG_fc.post_backward = _patched_post_bwd
                        _PG_fc._fyh_fc_patched = True
                        if _is_rank0:
                            print(f"[gemma4 bwd-fire] hooked FSDPParamGroup."
                                  f"{{pre,post}}_backward "
                                  f"({len(_id2li)} module-id mappings)",
                                  file=_sys_fc.stderr, flush=True)
                except ImportError:
                    pass

                if _is_rank0:
                    print(f"[gemma4 bwd-fire] registered {n_hooks} module hooks "
                          f"({n_hooks//2} layers × fwd+bwd); output={fpath}",
                          file=_sys_fc.stderr, flush=True)

            _tfm_fc.Trainer.__init__ = _patched_T_init_fc

            _orig_train_step_fc = _tfm_fc.Trainer.training_step

            @_ft_fc.wraps(_orig_train_step_fc)
            def _patched_train_step_fc(self, *args, **kwargs):
                ret = _orig_train_step_fc(self, *args, **kwargs)
                # Flush counts for THIS micro to file
                fh = _FC_STATE.get("fh")
                if fh is not None:
                    si = _FC_STATE["step"]
                    mi = _FC_STATE["micro"]
                    rows = [(k, v) for k, v in _FC_STATE["counts"].items()
                            if k[0] == si and k[1] == mi]
                    for (s, m, li, ev), cnt in sorted(rows):
                        fh.write(f"{s}\t{m}\t{li}\t{ev}\t{cnt}\n")
                # Advance micro counter (same logic as patch 26)
                _FC_STATE["micro"] += 1
                gas = getattr(self.args, "gradient_accumulation_steps", 1) or 1
                if _FC_STATE["micro"] >= gas:
                    _FC_STATE["micro"] = 0
                    _FC_STATE["step"] += 1
                    if _FC_STATE["step"] >= int(_os_fc.environ.get(
                            "GEMMA4_BWD_FIRE_MAX_STEPS", "1")):
                        if fh is not None:
                            fh.close()
                            _FC_STATE["fh"] = None
                            if _is_rank0:
                                print("[gemma4 bwd-fire] reached max_steps, "
                                      "closed dump", file=_sys_fc.stderr, flush=True)
                return ret

            _tfm_fc.Trainer.training_step = _patched_train_step_fc
            if _is_rank0:
                print(f"[gemma4 sdp_preamble] (28) installed bwd-fire counter "
                      f"(GEMMA4_BWD_FIRE_COUNT=1; output dir={_FC_DIR})",
                      file=_sys_fc.stderr, flush=True)
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip (28) bwd-fire "
                  f"({type(_e).__name__}: {_e})", file=sys.stderr, flush=True)

    # (27) Hook ms-swift BatchSamplerShard.__iter__ to dump yielded indices
    # per-rank.  Helps verify both DS3 and FSDP2 see the same dataloader
    # output even though they go through different accelerate.prepare paths.
    if os.environ.get("GEMMA4_SAMPLER_DUMP") == "1":
        try:
            import os as _os_smp
            import sys as _sys_smp
            _SMP_DIR = _os_smp.environ.get("GEMMA4_SAMPLER_DUMP_DIR", "/tmp/sampler_dump")
            _os_smp.makedirs(_SMP_DIR, exist_ok=True)
            _SMP_MAX_BATCHES = int(_os_smp.environ.get("GEMMA4_SAMPLER_DUMP_MAX_BATCHES", "16"))
            _SMP_RANK = int(_os_smp.environ.get("RANK",
                          _os_smp.environ.get("LOCAL_RANK", "0")))

            from swift.dataloader.shard import BatchSamplerShard as _MSwfBSS
            _orig_iter_smp = _MSwfBSS.__iter__
            _orig_set_epoch_smp = _MSwfBSS.set_epoch

            def _patched_iter_smp(self):
                fpath = _os_smp.path.join(_SMP_DIR, f"sampler_rank{_SMP_RANK}.tsv")
                with open(fpath, "a", buffering=1) as fh:
                    fh.write(f"# __iter__ called: curr_seed={self.curr_seed} "
                             f"world_size={self.world_size} rank={self.rank} "
                             f"batch_size={self.batch_size}\n")
                    fh.write("batch_idx\tindices\n")
                    n = 0
                    for batch in _orig_iter_smp(self):
                        if n < _SMP_MAX_BATCHES:
                            fh.write(f"{n}\t{','.join(str(i) for i in batch)}\n")
                        n += 1
                        yield batch
                    fh.write(f"# __iter__ done: yielded {n} batches\n")
                if _is_rank0:
                    print(f"[gemma4 sampler-dump] rank{_SMP_RANK} epoch "
                          f"(seed={self.curr_seed}) dumped {n} batches",
                          file=_sys_smp.stderr, flush=True)

            def _patched_set_epoch_smp(self, epoch):
                fpath = _os_smp.path.join(_SMP_DIR, f"sampler_rank{_SMP_RANK}.tsv")
                with open(fpath, "a", buffering=1) as fh:
                    fh.write(f"# set_epoch({epoch}) called -> curr_seed will be "
                             f"{self.base_seed + epoch}\n")
                if _is_rank0:
                    print(f"[gemma4 sampler-dump] rank{_SMP_RANK} "
                          f"set_epoch({epoch}) -> curr_seed={self.base_seed+epoch}",
                          file=_sys_smp.stderr, flush=True)
                return _orig_set_epoch_smp(self, epoch)

            _MSwfBSS.__iter__ = _patched_iter_smp
            _MSwfBSS.set_epoch = _patched_set_epoch_smp
            if _is_rank0:
                print(f"[gemma4 sdp_preamble] (27) hooked "
                      f"swift.dataloader.shard.BatchSamplerShard.__iter__ "
                      f"(GEMMA4_SAMPLER_DUMP=1; output dir={_SMP_DIR})",
                      file=_sys_smp.stderr, flush=True)
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip (27) sampler-dump "
                  f"({type(_e).__name__}: {_e})",
                  file=sys.stderr, flush=True)

    # (26) Per-(rank, micro) input_ids fingerprint dump.
    #
    # Verifies whether DS3 and FSDP2 dataloaders are yielding the same data
    # at the same (step, micro, rank).  Hooks Trainer.training_step;
    # for each call, hashes the input_ids tensor and writes one line:
    #   step \t micro \t rank \t n_tokens \t fingerprint
    # to GEMMA4_DATA_DUMP_DIR/{engine}_rank{N}.tsv.
    #
    # Activated only when GEMMA4_DATA_DUMP=1.
    # GEMMA4_DATA_DUMP_MAX_STEPS (default 1) limits how many full steps to dump.
    if os.environ.get("GEMMA4_DATA_DUMP") == "1":
        try:
            import os as _os_dd
            import sys as _sys_dd
            import functools as _ft_dd
            import hashlib as _hash_dd
            import transformers as _tfm_dd
            try:
                from accelerate.utils import DistributedType as _DT_dd
                _has_DT = True
            except Exception:
                _has_DT = False

            _DD_DIR = _os_dd.environ.get("GEMMA4_DATA_DUMP_DIR", "/tmp/data_dump")
            _os_dd.makedirs(_DD_DIR, exist_ok=True)
            _DD_MAX_STEPS = int(_os_dd.environ.get("GEMMA4_DATA_DUMP_MAX_STEPS", "1"))
            _DD_STATE = {
                "step": 0, "micro": 0, "fh": None, "engine": "?",
                "registered": False,
                "rank": int(_os_dd.environ.get("RANK",
                            _os_dd.environ.get("LOCAL_RANK", "0"))),
            }

            _orig_T_init_dd = _tfm_dd.Trainer.__init__

            @_ft_dd.wraps(_orig_T_init_dd)
            def _patched_T_init_dd(self, *args, **kwargs):
                _orig_T_init_dd(self, *args, **kwargs)
                if _DD_STATE["registered"]:
                    return
                _DD_STATE["registered"] = True
                # Detect engine
                if _has_DT:
                    try:
                        if self.accelerator.distributed_type == _DT_dd.DEEPSPEED:
                            _DD_STATE["engine"] = "ds3"
                        elif self.accelerator.distributed_type == _DT_dd.FSDP:
                            _DD_STATE["engine"] = ("fsdp2"
                                if getattr(self.accelerator, "is_fsdp2", False)
                                else "fsdp1")
                        else:
                            _DD_STATE["engine"] = str(
                                self.accelerator.distributed_type).lower()
                    except Exception:
                        _DD_STATE["engine"] = "unknown"
                fname = f"{_DD_STATE['engine']}_rank{_DD_STATE['rank']}.tsv"
                fpath = _os_dd.path.join(_DD_DIR, fname)
                _DD_STATE["fh"] = open(fpath, "w", buffering=1)
                _DD_STATE["fh"].write(
                    "step\tmicro\trank\tn_tokens\tinput_ids_sha\tlabels_sha\t"
                    "first16_ids\tlast16_ids\tn_loss_tokens\tpreview\n")
                # Cache tokenizer so we can decode previews
                _DD_STATE["tokenizer"] = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
                if _is_rank0:
                    print(f"[gemma4 data-dump] writing to {fpath} "
                          f"(engine={_DD_STATE['engine']}, "
                          f"max_steps={_DD_MAX_STEPS})",
                          file=_sys_dd.stderr, flush=True)

            _tfm_dd.Trainer.__init__ = _patched_T_init_dd

            _orig_train_step_dd = _tfm_dd.Trainer.training_step

            @_ft_dd.wraps(_orig_train_step_dd)
            def _patched_train_step_dd(self, model, inputs, *args, **kwargs):
                if (_DD_STATE["fh"] is not None
                        and _DD_STATE["step"] < _DD_MAX_STEPS
                        and "input_ids" in inputs):
                    try:
                        ids = inputs["input_ids"]
                        if hasattr(ids, "detach"):
                            ids_cpu = ids.detach().cpu().contiguous().view(-1).tolist()
                        else:
                            ids_cpu = list(ids)
                        n_tokens = len(ids_cpu)
                        ids_str = ",".join(str(int(x)) for x in ids_cpu)
                        h_ids = _hash_dd.sha1(ids_str.encode()).hexdigest()[:16]
                        h_lab = "-"
                        n_loss_tokens = -1
                        if "labels" in inputs:
                            lb = inputs["labels"]
                            lb_cpu = lb.detach().cpu().contiguous().view(-1).tolist()
                            lb_str = ",".join(str(int(x)) for x in lb_cpu)
                            h_lab = _hash_dd.sha1(lb_str.encode()).hexdigest()[:16]
                            n_loss_tokens = sum(1 for x in lb_cpu if int(x) != -100)
                        first16 = "|".join(str(int(x)) for x in ids_cpu[:16])
                        last16 = "|".join(str(int(x)) for x in ids_cpu[-16:])
                        # Decode a 80-char preview of the FIRST loss-target tokens
                        preview = "?"
                        tk = _DD_STATE.get("tokenizer")
                        if tk is not None and "labels" in inputs:
                            try:
                                lb_cpu_arr = lb_cpu
                                first_loss_idx = next((i for i, x in enumerate(lb_cpu_arr)
                                                        if int(x) != -100), 0)
                                snippet = ids_cpu[first_loss_idx:first_loss_idx+40]
                                preview = tk.decode(snippet, skip_special_tokens=False)
                                preview = preview.replace("\n", "\\n").replace("\t", " ")[:80]
                            except Exception:
                                preview = "<decode_failed>"
                        _DD_STATE["fh"].write(
                            f"{_DD_STATE['step']}\t{_DD_STATE['micro']}\t"
                            f"{_DD_STATE['rank']}\t{n_tokens}\t{h_ids}\t{h_lab}\t"
                            f"{first16}\t{last16}\t{n_loss_tokens}\t{preview}\n")
                    except Exception as _e:
                        if _is_rank0:
                            print(f"[gemma4 data-dump] WARN: row write failed "
                                  f"({type(_e).__name__}: {_e})",
                                  file=_sys_dd.stderr, flush=True)
                # Advance micro counter; reset on optimizer step
                _DD_STATE["micro"] += 1
                gas = getattr(self.args, "gradient_accumulation_steps", 1) or 1
                if _DD_STATE["micro"] >= gas:
                    _DD_STATE["micro"] = 0
                    _DD_STATE["step"] += 1
                    if _DD_STATE["step"] >= _DD_MAX_STEPS:
                        if _DD_STATE["fh"] is not None:
                            try:
                                _DD_STATE["fh"].close()
                            except Exception:
                                pass
                            _DD_STATE["fh"] = None
                            if _is_rank0:
                                print("[gemma4 data-dump] reached max_steps, "
                                      "closed dump file",
                                      file=_sys_dd.stderr, flush=True)
                return _orig_train_step_dd(self, model, inputs, *args, **kwargs)

            _tfm_dd.Trainer.training_step = _patched_train_step_dd
            if _is_rank0:
                print("[gemma4 sdp_preamble] (26) installed input_ids data-dump "
                      f"(GEMMA4_DATA_DUMP=1; output dir={_DD_DIR})",
                      file=_sys_dd.stderr, flush=True)
        except Exception as _e:
            print(f"[gemma4 sdp_preamble] WARN: skip data-dump patch "
                  f"({type(_e).__name__}: {_e})",
                  file=sys.stderr, flush=True)
