"""Runtime compat shim for swift 4.1.2 Ulysses SP + transformers 5.5.x.

In transformers 5.5 the causal-mask interface was rewritten:
  - `flash_attention_mask` / `sdpa_mask` now take `q_length` as the 2nd positional
    arg instead of `cache_position`. All other arguments are passed as kwargs.
  - `create_causal_mask` keeps `cache_position` but only for back-compat; the
    real arguments now come through `past_key_values` / `position_ids` (kwargs).

swift 4.1.2's `SequenceParallel._prepare_flash_attn` (in
`swift/sequence_parallel/ulysses.py`) was written for the old signature, so on
tf 5.5 it raises

    TypeError: flash_attention_mask() missing 1 required positional argument:
    'cache_position'

during the first forward. This module replaces `_prepare_flash_attn` with a
version whose mask hooks use the new signature. Everything outside of the
signature is preserved (same world_size=1 fast path, same attention_mask
"all True -> None" pass-through for SP>1, same zigzag-style input_embeds
re-inflation for `create_causal_mask`).

Import this module **before** `swift.cli.sft`; the monkey-patch targets the
class method so any instance built afterwards picks up the new behaviour.
"""
from __future__ import annotations

import os


def apply() -> None:
    # Keep the patch idempotent: if swift is unavailable (e.g. running under
    # a stripped-down interpreter for tooling), bail silently rather than
    # crashing user code.
    try:
        from swift.sequence_parallel import ulysses as _ulysses
    except ImportError:
        return

    # Already patched once this process; don't re-apply.
    if getattr(_ulysses.SequenceParallel, "_tf55_patched", False):
        return

    def _prepare_flash_attn(self, base_model):
        # --- mask hooks ---------------------------------------------------
        from transformers import masking_utils

        _origin_flash_attention_mask = masking_utils.flash_attention_mask

        def flash_attention_mask(
            batch_size,
            q_length,
            kv_length,
            q_offset=0,
            kv_offset=0,
            mask_function=masking_utils.causal_mask_function,
            attention_mask=None,
            **kwargs,
        ):
            # SP=1 degenerate case -> delegate to the real tf 5.5 impl
            # (kwargs-only so we don't depend on arg ordering drifting again).
            if self.world_size == 1:
                return _origin_flash_attention_mask(
                    batch_size=batch_size,
                    q_length=q_length,
                    kv_length=kv_length,
                    q_offset=q_offset,
                    kv_offset=kv_offset,
                    mask_function=mask_function,
                    attention_mask=attention_mask,
                    **kwargs,
                )
            # SP>1: flash_attn is "unpadded"; swift only cares whether every
            # token is valid. Returning None triggers the fully-causal fast
            # path; returning the 2D mask triggers the per-seqlen fallback.
            if attention_mask is not None and attention_mask.all():
                attention_mask = None
            return attention_mask

        masking_utils.flash_attention_mask = flash_attention_mask
        masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping[
            "flash_attention_2"
        ] = flash_attention_mask
        # tf 5.5 also aliases flash_attention_3 / flash_attention_4 to the same
        # mask function; patch them too so any later attn_impl switch keeps SP.
        for _key in ("flash_attention_3", "flash_attention_4"):
            if _key in masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
                masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping[_key] = (
                    flash_attention_mask
                )

        # --- sdpa_mask -----------------------------------------------------
        # swift's original sdpa path needs `cache_position` which tf 5.5 no
        # longer passes. We're not using sdpa in this benchmark (attn_impl is
        # always flash_attention_2), so keep the SP>1 path as a
        # "just-use-attention_mask" stub and delegate the SP=1 path.
        def sdpa_mask(
            batch_size,
            q_length,
            kv_length,
            q_offset=0,
            kv_offset=0,
            mask_function=masking_utils.causal_mask_function,
            attention_mask=None,
            **kwargs,
        ):
            orig = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping[
                "sdpa_origin"
            ]
            if self.world_size == 1:
                return orig(
                    batch_size=batch_size,
                    q_length=q_length,
                    kv_length=kv_length,
                    q_offset=q_offset,
                    kv_offset=kv_offset,
                    mask_function=mask_function,
                    attention_mask=attention_mask,
                    **kwargs,
                )
            # For SP>1 with sdpa we fall back to eager mask-only; any downstream
            # caller that actually needs sdpa on SP will need a proper
            # port of swift's zigzag path. Surfaces loudly via NotImplemented
            # rather than hiding as a silent wrong result.
            raise NotImplementedError(
                "swift SP + sdpa is not supported under transformers 5.5; "
                "use --attn_impl flash_attention_2."
            )

        if "sdpa" in masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
            masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.setdefault(
                "sdpa_origin",
                masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping["sdpa"],
            )
            masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping["sdpa"] = sdpa_mask

        # --- create_causal_mask -------------------------------------------
        # tf 5.5 signature:
        #   create_causal_mask(config, inputs_embeds, attention_mask,
        #                      cache_position=None, *,
        #                      past_key_values=None,
        #                      position_ids=None,
        #                      or_mask_function=None,
        #                      and_mask_function=None)
        # We need to intercept it to fake a "full-sequence" inputs_embeds so
        # that the downstream mask_interface reconstructs kv_length = full_seq
        # rather than the per-rank shard length.
        import torch as _torch

        _origin_create_causal_mask = masking_utils.create_causal_mask

        def create_causal_mask(
            config,
            inputs_embeds,
            attention_mask,
            cache_position=None,
            *,
            past_key_values=None,
            position_ids=None,
            or_mask_function=None,
            and_mask_function=None,
            **kwargs,
        ):
            if self.world_size == 1:
                return _origin_create_causal_mask(
                    config,
                    inputs_embeds,
                    attention_mask,
                    cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    or_mask_function=or_mask_function,
                    and_mask_function=and_mask_function,
                    **kwargs,
                )
            # Inflate inputs_embeds on the sequence axis so the mask path
            # sees the full (pre-split) sequence length. We only need the
            # shape/dtype/device; the tensor values aren't read by the mask.
            full_embeds = _torch.ones(
                (
                    inputs_embeds.shape[0],
                    inputs_embeds.shape[1] * self.sp_world_size,
                    inputs_embeds.shape[2],
                ),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            full_cache_position = _torch.arange(
                0, full_embeds.shape[1], device=inputs_embeds.device
            )
            return _origin_create_causal_mask(
                config,
                full_embeds,
                attention_mask,
                full_cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
                or_mask_function=or_mask_function,
                and_mask_function=and_mask_function,
                **kwargs,
            )

        masking_utils.origin_create_causal_mask = _origin_create_causal_mask
        masking_utils.create_causal_mask = create_causal_mask

        # --- rest of swift's original prepare (attention_forward hook) ----
        # Re-implement the part of the original _prepare_flash_attn that wraps
        # the per-layer flash-attention forward; this half has no signature
        # mismatch with tf 5.5, so we just call it verbatim.
        if hasattr(base_model, "language_model"):
            text_model = base_model.language_model
        else:
            text_model = base_model

        from transformers.modeling_flash_attention_utils import (  # noqa: WPS433
            is_flash_attn_available,
        )

        if is_flash_attn_available():
            from transformers import modeling_flash_attention_utils
            from transformers.modeling_flash_attention_utils import (
                _flash_attention_forward,
            )
            from swift.sequence_parallel.ulysses import DistributedAttention

            _distributed_flash_attention = DistributedAttention(
                _flash_attention_forward, self
            )
            modeling_flash_attention_utils._flash_attention_forward_origin = (
                _flash_attention_forward
            )

            def flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                *args,
                **kwargs,
            ):
                if self.world_size == 1:
                    return _flash_attention_forward(
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        q_len,
                        *args,
                        **kwargs,
                    )
                return _distributed_flash_attention(
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    *args,
                    **kwargs,
                )

            modeling_flash_attention_utils._flash_attention_forward = (
                flash_attention_forward
            )

        # also drop into the shared attention function registry — this is the
        # real hook tf 5.5 uses for per-module attention dispatch.
        try:
            from transformers.integrations.flash_attention import (
                flash_attention_forward as _tf_flash_attention_forward,  # type: ignore  # noqa: WPS433
            )

            # Nothing to do here in tf 5.5: the dispatch uses
            # modeling_flash_attention_utils._flash_attention_forward which we
            # already swapped. Left as explicit import for discoverability.
            _ = _tf_flash_attention_forward
        except Exception:  # noqa: BLE001
            pass

    _ulysses.SequenceParallel._prepare_flash_attn = _prepare_flash_attn
    _ulysses.SequenceParallel._tf55_patched = True  # type: ignore[attr-defined]

    if os.environ.get("SWIFT_SP_PATCH_VERBOSE", "") == "1":
        print("[swift_sp_patch] patched swift.sequence_parallel.ulysses._prepare_flash_attn for transformers 5.5.x",
              flush=True)


apply()
