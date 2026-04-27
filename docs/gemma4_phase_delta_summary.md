# Gemma-4-26B-A4B-it 优化 Phase Delta 精简表

> 每期一行的增量表。详见 [gemma4_optimization_walkthrough.md](gemma4_optimization_walkthrough.md)。
>
> 数字全部实测，不用预估。

## FSDP2 主线

| Phase | 轴 | steady step | peak mem | peak pwr | tokens/s/GPU | **真 MFU** | full-epoch wall | vs P0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **P0b** | FSDP2 baseline (math SDPA, AC=on no-pack SP=2 MBS=1 GAS=1 GBS=4) | 1881 ms | 76.6 GiB | — | 8710 | — | 73 min | 1.0× |
| **P0c** | DS prod baseline (offload opt+param, GBS=64 GAS=16) | 86180 ms (opt) | 43.6 GiB | — | 1521 | — | 422 min | 0.17× |
| **P0d** forward align | FSDP2 = single-GPU bitwise; DS bf16 mode 1.5% diff | — | — | — | — | — | — | — |
| **P0g** train align | run_13 step 1 = bit-identical to DS (Δloss 0%); 13 patches stack reproducible | — | 76.6 GiB | — | — | — | — | — |
| **P1 peak** | mem_eff SDPA + GQA repeat_kv + GBS=4 native + truncation=**right** | 1763 ms | 65.1 GiB | — | 9295 | TBD | 145 min | 2.91× vs P0c |
| P1 alt-delete | 同上 truncation=delete (P0b-equiv) | 1758 ms | 65.1 GiB | — | 9319 | TBD | 138 min | — |
| P1 alt-offload | GBS=64 offload (DS-equivalent config) | 45530 ms | 76.6 GiB | — | 5758 | — | 223 min | 1.89× vs DS prod |
| **P2 ★** | + NO_AC (activation_checkpointing=false) | **1656 ms** | **64.9 GiB** | — | **9893** | TBD | **137 min** | **3.08× vs P0c** / +6.4% vs P1 |
| P3 native | + reshard=false (ZeRO-2) | OOM | 77.8 GiB | — | — | — | — | ❌ 不采纳 |
| P3 offload | + reshard=false + offload | 17505 ms | 68.1 GiB | — | 936 | — | 1648 min | ❌ -90% throughput |
| **P4 ★★** | + packing (AC=off + native) | 3192 ms | 64.9 GiB | — | 5131 | TBD | 43 min | 9.81× vs DS prod / 3.42× vs P0b |
| **P5 ★★★** | + Liger RMSNorm+GeGLU (FLCE off) | **2781 ms** | **64.9 GiB** | — | **5891** | TBD | **37.5 min** | **11.3× vs DS prod** / +14.79% vs P4 |
| P5 alt | + Liger FLCE (loss reduction bug, predictions OK) | 2785 ms | 64.9 GiB | — | 5884 | TBD | — | ❌ loss 膨胀 145% |
| P6 | torch.compile | — | — | — | — | — | — | ❌ blocked: PyTorch 2.10 InductorError TF32 bug |
| P7 | MoE 调优 | — | — | — | — | — | — | ❌ blocked: gemma4 + swift FSDP2 没暴露 router_dtype / capacity_factor |
| **P8 confirm** | GBS resweep on full stack: GBS=4 native still global peak | 2796 ms | 64.9 GiB | — | 5861 | TBD | 37.8 min | confirms P5 peak |
| P8 alt | GBS=8 offload | 24460 ms | 76.6 GiB | — | 1340 | — | 165 min | 4.4× slower than peak |
| P8 alt | GBS=64 offload | 62471 ms | 76.6 GiB | — | 4196 | — | 53 min | 1.4× slower than peak |
| P4 | +packing | — | — | — | — | — | — | — |
| P5 | +Liger gemma4 dispatch | — | — | — | — | — | — | — |
| P6 | +compile / FP8 | — | — | — | — | — | — | — |
| P7 | +MoE 调优 | — | — | — | — | — | — | — |
| P8 | final（GBS 复核后） | — | — | — | — | — | — | — |

## DeepSpeed（单次 baseline，不参与轴扫）

| 配置 | 来源 | step (opt-step) | tokens/s/GPU | peak mem | 备注 |
|---|---|---:|---:|---:|---|
| 生产线上 exact 配置 | 用户 `gemma4_sft_0423.log` 启动命令 | — | — | — | DS ZeRO-3 + offload(opt+param) + SP=2 + GBS=64 + AC=on |

