# P8 GBS resweep — attempts timeline

> Goal: verify P5 peak (GBS=4 native + AC=off + packing + Liger) is still
> peak after the full optimized stack is in place.

| run_run_20260426_230916_gbs4_native_full_stack | MBS=1 GAS=1 | SUCCESS | see report.json |
| run_run_20260426_231338_gbs8_offload_full_stack | MBS=1 GAS=2 FSDP_OFFLOAD=offload | SUCCESS | see report.json |
| run_run_20260426_232908_gbs64_offload_full_stack | MBS=1 GAS=16 FSDP_OFFLOAD=offload | SUCCESS | see report.json |
