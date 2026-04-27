# P1 GBS/MBS sweep summary

Root: `/home/ubuntu/fyh/megatron_output/gemma4_opt/p1_gbs_sweep`  ·  total runs: 7  ·  ok: 5

## Per-config results

| MBS | GAS | GBS | mode | step (ms) | tokens/s/GPU | peak mem (GiB) | active TFLOPS/GPU | real MFU% | loss step1 | full-epoch wall* |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 4 | native | 1758 | **9319** ★ | 65.1 | 212 | 21.5% | 1.0803 | 138 min |
| 1 | 2 | 8 | offload | 19046 | **1720** | 76.5 | 39 | 4.0% | 1.7373 | 747 min |
| 2 | 1 | 8 | native | — | — | — | — | — | — | **FAILED — exit=1** |
| 1 | 4 | 16 | offload | 23083 | **2839** | 76.6 | 65 | 6.5% | 1.4718 | 452 min |
| 2 | 2 | 16 | offload | — | — | — | — | — | — | **FAILED — exit=1** |
| 1 | 8 | 32 | offload | 30594 | **4284** | 76.6 | 98 | 9.9% | 1.5264 | 300 min |
| 1 | 16 | 64 | offload | 45530 | **5758** | 76.6 | 131 | 13.3% | 1.5269 | 223 min |

_*full-epoch wall ≈ (18819 / GBS) × step_ms; theoretical, doesn't include data loader overhead. ★ = peak tokens/s/GPU._

## Peak

- **Best throughput**: MBS=1 GAS=1 GBS=4 (native)
- tokens/s/GPU = **9319**
- step = 1758 ms · peak mem = 65.1 GiB · active MFU = 21.5%
