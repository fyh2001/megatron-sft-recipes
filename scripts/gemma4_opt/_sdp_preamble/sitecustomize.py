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
    try:
        import transformers.models.gemma4.modeling_gemma4 as _mg4

        if not getattr(_mg4, "_fyh_kv_share_patch_applied", False):
            _orig_textmodel_forward = _mg4.Gemma4TextModel.forward
            _orig_attn_forward = _mg4.Gemma4TextAttention.forward

            def _patched_textmodel_forward(self, *args, **kwargs):
                # Allocate a fresh shared dict per forward (per-batch) and
                # attach it to every layer's attention before the loop runs.
                _fyh_skv = {}
                for _layer in self.layers[: self.config.num_hidden_layers]:
                    _layer.self_attn._fyh_shared_kv = _fyh_skv
                return _orig_textmodel_forward(self, *args, **kwargs)

            def _patched_attn_forward(self, *args, **kwargs):
                # If we've been seeded by the patched TextModel.forward,
                # override the (cloned-by-FSDP) shared_kv_states kwarg with
                # our intact reference.  Otherwise leave kwargs untouched
                # (preserves non-distributed / non-FSDP code paths).
                _ref = getattr(self, "_fyh_shared_kv", None)
                if _ref is not None:
                    kwargs["shared_kv_states"] = _ref
                return _orig_attn_forward(self, *args, **kwargs)

            _mg4.Gemma4TextModel.forward = _patched_textmodel_forward
            _mg4.Gemma4TextAttention.forward = _patched_attn_forward
            _mg4._fyh_kv_share_patch_applied = True

            if _is_rank0:
                print(
                    "[gemma4 sdp_preamble] (8/8) patched Gemma4TextModel."
                    "forward + Gemma4TextAttention.forward to route "
                    "shared_kv_states via instance attr (bypasses FSDP2 "
                    "MixedPrecisionPolicy cast_forward_inputs dict-rebuild "
                    "that breaks E4B kv-shared layers 24..41 with KeyError)",
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
