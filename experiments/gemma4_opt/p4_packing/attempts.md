# P4 packing sweep — attempts timeline

> Goal: enable --packing true at the P2 peak. Expected ~5× tokens/s/GPU.
> Auto-escalate AC=off → AC=on → offload on OOM.

| run_run_20260426_211751_gbs4_pack_no_ac_native | NO_AC=true | FAILED | exit=1 |
| run_run_20260426_211846_gbs4_pack_ac_on_native | NO_AC=false | FAILED | exit=1 |
| run_run_20260426_211941_gbs4_pack_ac_on_offload | NO_AC=false FSDP_OFFLOAD=offload | FAILED | exit=1 |
| run_run_20260426_212104_gbs4_pack_no_ac_native | NO_AC=true | FAILED | exit=137 |
| run_run_20260426_212108_gbs4_pack_ac_on_native | NO_AC=false | SUCCESS | see report.json |
