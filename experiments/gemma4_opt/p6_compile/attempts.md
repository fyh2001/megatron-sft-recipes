# P6 torch.compile sweep — attempts timeline

> Goal: smoke test torch.compile at P5 peak.  Predicted +5-15% step time.
> Known risk: torch 2.10 + Inductor + TF32 API bug from qwen3.5 era may break
> compile.  If broken, document & skip to P7.

| run_run_20260426_225900_gbs4_pack_no_ac_native_liger_compile | FAILED | compile error: torch/_dynamo/eval_frame.py |
| run_run_20260426_230227_gbs4_pack_no_ac_native_liger_compile | FAILED | compile error: torch/_dynamo/eval_frame.py |
