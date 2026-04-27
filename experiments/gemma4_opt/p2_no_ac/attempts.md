# P2 NO_AC sweep — attempts timeline

> Goal: measure step-time / peak-mem trade-off when disabling FSDP2's wrap-level activation_checkpointing at the P1 peak (GBS=4 native).
>
> Hypothesis: ~90 GB extra activations from 30 gemma4 decoder layers (no recompute) → OOM on 80 GB without offload.

| run_run_20260426_203201_gbs4_no_ac_native | native mode | SUCCESS | see report.json |
| run_run_20260426_203358_gbs4_no_ac_offload | offload mode | SUCCESS | see report.json |
