# P1 GBS/MBS sweep — attempts timeline

> Goal: find peak FSDP2 throughput by sweeping MBS × GAS on FSDP2 baseline preset (truncation=delete, lr=1e-5, warmup=0.1, freeze_vit=true, USE_LIGER=true, mem_efficient SDPA + GQA repeat_kv patches always on).
>
> Configuration discovery: GAS=1 fits without offload; GAS≥2 needs `fsdp offload` (PyTorch FSDP2 no_sync mode keeps unsharded grads on each rank between micros, +45 GiB above GAS=1 baseline).

## Final 7 sweep configurations (post-cleanup)

| Run dir | MBS | GAS | GBS | mode | STATUS | tokens/s/GPU |
|---|---:|---:|---:|---|---|---:|
| `run_20260426_184752_mbs1_ga1_gbs4` | 1 | 1 | 4 | native | ✅ SUCCESS | **9319** ★ peak |
| `run_20260426_185825_mbs1_ga2_gbs8` | 1 | 2 | 8 | offload | ✅ SUCCESS | 1720 |
| `run_20260426_191012_mbs1_ga4_gbs16` | 1 | 4 | 16 | offload | ✅ SUCCESS | 2839 |
| `run_20260426_192311_mbs1_ga8_gbs32` | 1 | 8 | 32 | offload | ✅ SUCCESS | 4284 |
| `run_20260426_193957_mbs1_ga16_gbs64` | 1 | 16 | 64 | offload | ✅ SUCCESS | 5758 |
| `run_20260426_200409_mbs2_ga1_gbs8` | 2 | 1 | 8 | native | ❌ FAILED | swift Ulysses VLM dict-mask bug (not OOM) |
| `run_20260426_200513_mbs2_ga2_gbs16` | 2 | 2 | 16 | offload | ❌ FAILED | same MBS≥2 bug |

## Pre-cleanup failed attempts (script iteration)

| Cause | Affected runs | Fix |
|---|---|---|
| sitecustomize stdout 污染 bench `_cls_json=$(...)` 捕获 → fsdp_override.json 被注入 banner 损坏 | `run_20260426_184601` 系列 | sitecustomize prints redirect 到 stderr |
| swift Ulysses SP `no_sync` mode 在 GAS≥2 下保留 unsharded grads → 78 GB peak OOM | `run_20260426_185028` GBS=8/32 系列 | sweep 脚本 GAS≥2 自动加 `fsdp offload` + `--optim adamw_torch` |
| 上一轮 sweep `kill 1640007` 触发的 SIGKILL 级联 | `185201/185248/185414/185504/185501/185527/185553/185605/185645/185750` 系列 | 改进 inter-iteration cleanup（pkill -9 python + GPU mem poll）|
| 同一 `MASTER_PORT=29555` 在 TIME_WAIT 复用导致 `EADDRINUSE` | `185501/185527` 系列 | 每个 iteration 用 random port `29500 + (RANDOM % 200)` |
| `head -1` SIGPIPE → set -o pipefail → sweep 提前退出 | sweep 退出在 mbs1_ga2_gbs8 完成后 | cleanup 循环用 `awk 'NR==1'` 避免 SIGPIPE |

## MBS≥2 swift Ulysses VLM bug (新发现)

`/home/ubuntu/fyh/ms-swift-fork/swift/sequence_parallel/ulysses.py:705-714`：

```python
attention_mask = self.pad(attention_mask, padding_value=0)  # 705
...
def _do_pad(tensor):  # 490
    length = tensor.shape[dim]  # AttributeError when tensor is dict
```

gemma4 VLM 的 `attention_mask` 是 dict（含 image-related keys），MBS=1 时 swift 走 single-sample 路径直接传 dict（OK），MBS≥2 触发 batched collation → `pad_and_split_inputs` → `pad()` → `_do_pad()` 假设 tensor，崩溃。

P0g/P1 的 sweep 不需要 MBS≥2（peak GBS=4 用 MBS=1）。如果 P2/P3/P7 需要，可以走 swift fork 修复 + upstream PR。
| run_run_20260426_201142_mbs1_ga1_gbs4_truncRIGHT | MBS=1 GAS=1 GBS=4 | SUCCESS | see report.json |
