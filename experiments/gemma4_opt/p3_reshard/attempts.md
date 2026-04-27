# P3 reshard_after_forward=false sweep — attempts timeline

> Goal: measure step-time / peak-mem trade-off when switching FSDP2 from
> ZeRO-3 (reshard=true) to ZeRO-2 (reshard=false) at the P2 peak.
>
> Hypothesis: native OOMs (full model unsharded = 52 GB / rank > 80 GB ceiling
> after activations + opt state); offload may or may not fit.

| run_run_20260426_205154_gbs4_no_ac_zero2_native | native mode | FAILED | OOM (Tried to allocate 2.92 GiB. GPU 1 has a total capacity of 79.18 GiB of which 1.40 GiB is free) |
| run_run_20260426_205309_gbs4_no_ac_zero2_offload | offload mode | SUCCESS | see report.json |
