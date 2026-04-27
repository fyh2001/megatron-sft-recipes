# P5 Liger gemma4 dispatch — attempts timeline

> Goal: write Liger gemma4 dispatch (RMSNorm + GeGLU + fused linear-CE) and
> measure step / mem / MFU vs P4 (no Liger).  Loss should be bit-comparable.
> Pre-recon: gemma4 RMSNorm = output*weight (offset=0), MLP = GeGLU.

| run_run_20260426_221100_gbs4_pack_no_ac_native_liger | FAILED | exit=1 |
| run_run_20260426_223741_gbs4_pack_no_ac_native_liger | SUCCESS | see report.json |
| run_run_20260426_224506_gbs4_pack_no_ac_native_liger | SUCCESS | see report.json |
