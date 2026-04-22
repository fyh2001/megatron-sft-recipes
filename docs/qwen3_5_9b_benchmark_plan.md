# Qwen3.5-9B × 三后端性能对比测试计划

> 目标：在 8×H100 上用三套后端（FSDP2+compile / DeepSpeed ZeRO-3 / Megatron TP=2）对 Qwen3.5-9B 做 `GBS=384 × MAX_LEN=16384` 的实战 SFT 性能对比；产出 throughput / MFU / DCGM TC active / nsys profile，并写成 [`docs/qwen3_5_9b_benchmark_report.md`](./qwen3_5_9b_benchmark_report.md)。

## 前置事实（已核对）

- **模型**：`Qwen/Qwen3.5-9B`，9.65 B params，`Qwen3_5ForConditionalGeneration` VLM；text backbone 是 hybrid：
  - `text_config`：32 层 / `hidden=4096` / Q=16 heads / KV=4 / `head_dim=256` / FFN=12288 / vocab=248320
  - `layer_types`：24 层 `linear_attention` (Gated DeltaNet) + 8 层 `full_attention` (Gated Attention)，每 4 层一次 full，`full_attention_interval=4`
  - **长上下文**：`max_position_embeddings=262144` → seq=16384 完全在原生长度内，不需 YaRN 外推
- **Megatron 支持**：mcore_bridge 1.1.2 在 `mm_gpts` 路径登记了 `qwen3_5`（`mcore_bridge/model/mm_gpts/qwen3_5_gdn.py` 继承 `Qwen3NextBridge` 接入 GDN），ms-swift 4.1.2 把 `Qwen/Qwen3.5-9B` 注册为 `MLLMModelType.qwen3_5`。**硬约束**：`context_parallel_size == 1`。
- **DeepSpeed**：容器内 `deepspeed 0.18.9` + `accelerate 1.13.0`（`DeepSpeedPlugin`）+ `torch 2.10.0+cu129`，走 `accelerate launch --config_file <ds.yaml>` 复用 [`scripts/fsdp/train.py`](../scripts/fsdp/train.py)。
- **transformers 5.5.4** 原生支持 `qwen3_5` / `qwen3_5_text` / `qwen3_5_moe`。
- **本机**：容器 `fsdp_sft` 在跑；数据 `/home/ubuntu/perf_opt/data/train.jsonl` 存在；Qwen3.5-9B 权重**尚未下载**（约 19 GB bf16）；`dcgm-exporter-fast` 已起（端口 9500，1s interval）。

## 关键设计决定

- **训练模式**：text-only SFT，FSDP/DS 用 `--freeze_vision`，Megatron 用 `--freeze_vit true --freeze_aligner true`。
- **Batch / Seq**（用户确认的实战范围）：
  - `MAX_LEN=16384`（相比 Qwen2.5-7B 报告的 4096 是 4×）
  - `GBS=384`（MBS=1 GAS=48），tokens/step = **6,291,456**（约 6.3 M）
- **`GRAD_CKPT` 策略**：
  - 默认 **off**（和上一份 Qwen2.5-7B 报告一致）
  - 每个后端先试 off，**OOM 才退 on**，报告里明确标注是哪一组被迫开启；同一组不混用 on/off
  - 预期：FSDP/DS 边缘（~55-70 GB/卡），Megatron TP=2+SP 宽裕
- **框架三选**：
  - **FSDP2+compile**：Accelerate FSDP2 + `torch.compile(dynamic=True)` + bf16 + flash-attn-2 + `--pad_to_max`。主推数。
  - **DeepSpeed ZeRO-3**：Accelerate + DS stage-3（权重/梯度/优化器全切），不开 `torch.compile`（DS 0.18 + compile 已知不稳），bf16 + flash-attn-2。这是 DS 的公平上限。
  - **Megatron (ms-swift)**：`TP=2 PP=1` + sequence_parallel + packing + distributed_optimizer + 此前全套 fuse flag。
- **Run 矩阵**：

```
                          GBS=384 × seq=16384 × 15 步
FSDP2+compile            fsdp/         (主)
DeepSpeed ZeRO-3         deepspeed/    (主)
Megatron TP=2 PP=1       megatron/     (主)
```

- **nsys profile** 单独在 **GBS=8 × seq=16384 × 10 步 × synthetic** 的短 run 上各采一份（原因：GBS=384 单步 60-100 s，profile 文件会爆炸到几十 GB，无法分析）。包括 FSDP compile=true / compile=false 两份 + DeepSpeed + Megatron，共 4 份。
- **DCGM TC active**：在 3 组主 bench 中**并行后台**跑 `dcgm_scrape.py`，15 步 × ~80 s ≈ 20 分钟稳态，配合 1Hz 采样仍有 1000+ 样本，足够出稳定均值 / p95。
- **MFU 口径**：`report.py` 的 `6N·tok/s/peak`（Chinchilla）对 hybrid 模型是高估（24 层 linear attn FLOPs ∝ S 不是 S²）。报告里给三个口径：
  - 6N 口径（跨后端可直接比）
  - Precise 口径：只对 8 层 full_attention 加 `12·L_full·H·S` 项
  - DCGM TC active 硬件实测作为第三重交叉
- **参数一致性**（三组全一致）：`MAX_LEN=16384 MBS=1 GAS=48 GBS=384 lr=1e-5 warmup_ratio=0.1 bf16 seed=42 attn=flash_attention_2 GRAD_CKPT=false TOTAL_STEPS=15 WARMUP_BENCH=5 freeze_vision=true`；只有并行策略/compile/ZeRO stage 这三项变化。
- **步数收敛解释**：GBS=384 × seq=16384 下单步 60-100 s，已是上一份 7B × GBS=8 × seq=4096 的 346 ms/step 的 50-80 倍。每步内部就跑了 48 个 microbatch 的 fwd+bwd（8 GPU × MBS=1 × GAS=48），**单步本身就是一个小聚合统计量**；warmup=5 足够吃掉 compile 冷启 / mcore-bridge init / L2 warm，measure=10 步的 median/p95 已远比上一份 30 步稳定。50 步是此前 7B micro-bench 的惯性，这里没必要。

## 资源与耗时估算

| 阶段 | GPU 时长 | 备注 |
|---|---|---|
| 阶段 0 下模型 + 打补丁 | ~10 min | 19 GB 下载 + 5 个文件改动 |
| 阶段 1 三框架 smoke (GBS=8 seq=16384) | ~5 min | 5 步验 OOM 不翻车 |
| 阶段 2 主 bench × 3 | ~60-75 min | 15 步 × ~60-100 s × 3 framework，含 FSDP compile 冷启 + mcore init |
| 阶段 3 nsys × 4 | ~15 min | GBS=8 × 10 步 × 4 runs |
| 阶段 4 报告 | ~15 min | |
| **总计** | **~2 小时** | |

## 实施步骤

### 阶段 0 · 准备（~10 分钟）

- [ ] `docker exec fsdp_sft python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3.5-9B')"` → 约 19 GB
- [ ] 补丁 A · [`scripts/fsdp/train.py`](../scripts/fsdp/train.py) 第 374 行：`AutoModelForCausalLM.from_pretrained(...)` → 先试 causal，捕 `ValueError` 后退到 `AutoModelForImageTextToText.from_pretrained(...)`；保留 `--freeze_vision` 语义。
- [ ] 补丁 B · [`scripts/benchmark/bench_fsdp.sh`](../scripts/benchmark/bench_fsdp.sh)：新增 `FREEZE_VISION=true` 时附加 `--freeze_vision` 给 `accelerate launch`。
- [ ] 补丁 C · [`scripts/benchmark/bench_megatron.sh`](../scripts/benchmark/bench_megatron.sh)：`megatron sft` 分支在 `FREEZE_VIT=true` 时追加 `--freeze_vit true --freeze_aligner true`。
- [ ] **新** [`scripts/benchmark/bench_deepspeed.sh`](../scripts/benchmark/bench_deepspeed.sh)：~90% 复用 bench_fsdp.sh 的外壳，改用 `scripts/fsdp/accelerate_ds_zero3.yaml`，强制 `COMPILE=false`，输出目录 `deepspeed/`（可被 `BENCH_OUTPUT` env 覆盖）。
- [ ] **新** [`scripts/fsdp/accelerate_ds_zero3.yaml`](../scripts/fsdp/accelerate_ds_zero3.yaml)：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_processes: 8
num_machines: 1
mixed_precision: bf16
deepspeed_config:
  zero_stage: 3
  zero3_init_flag: true
  offload_optimizer_device: none
  offload_param_device: none
  gradient_accumulation_steps: auto
  gradient_clipping: 1.0
  bf16:
    enabled: true
```

- [ ] [`scripts/benchmark/fsdp_diag.py`](../scripts/benchmark/fsdp_diag.py) 校验 shard：期望 `allocated ≈ 2.4 GB/卡`（19 GB / 8）。

### 阶段 1 · Smoke（~5 分钟）

- [ ] 三框架各跑 5 步 `GBS=8 × seq=16384 SYNTHETIC=true COMPILE=false GRAD_CKPT=false` 验不 OOM；记录峰值显存。若 FSDP 或 DS 爆，切那一组到 `GRAD_CKPT=true` 并在报告中注明。

### 阶段 2 · 主 Bench（~60-75 分钟）

每组都配一个后台 DCGM scraper：

```bash
# scraper 启动 (每组 bench 前)
docker exec -d fsdp_sft bash -c "python scripts/benchmark/dcgm_scrape.py \
  megatron_output/benchmark/<outdir>/dcgm_tc.tsv http://localhost:9500/metrics"
```

- [ ] **FSDP compile=true × GBS=384 × seq=16384**：

```bash
MODEL=/home/ubuntu/.cache/modelscope/models/Qwen/Qwen3___5-9B \
MBS=1 GAS=48 MAX_LEN=16384 TOTAL_STEPS=15 WARMUP_BENCH=5 \
COMPILE=true GRAD_CKPT=false SYNTHETIC=false PAD_TO_MAX=true \
FREEZE_VISION=true ATTN_IMPL=flash_attention_2 \
bash scripts/benchmark/bench_fsdp.sh
```

→ `megatron_output/benchmark/fsdp/`

- [ ] **DeepSpeed ZeRO-3 × GBS=384 × seq=16384**：

```bash
MODEL=/home/ubuntu/.cache/modelscope/models/Qwen/Qwen3___5-9B \
MBS=1 GAS=48 MAX_LEN=16384 TOTAL_STEPS=15 WARMUP_BENCH=5 \
GRAD_CKPT=false FREEZE_VISION=true ATTN_IMPL=flash_attention_2 \
bash scripts/benchmark/bench_deepspeed.sh
```

→ `megatron_output/benchmark/deepspeed/`

- [ ] **Megatron TP=2 PP=1 × GBS=384 × seq=16384**：

```bash
USE_MEGATRON_BACKEND=true \
MODEL=/home/ubuntu/.cache/modelscope/models/Qwen/Qwen3___5-9B \
TP=2 PP=1 MBS=1 GBS=384 MAX_LEN=16384 TOTAL_STEPS=15 WARMUP_BENCH=5 \
RECOMPUTE=none FREEZE_VIT=true \
bash scripts/benchmark/bench_megatron.sh
```

→ `megatron_output/benchmark/megatron/`

每组结束：`kill` 对应 scraper；收集 `bench.jsonl` / `train.log` / `report.json` / `gpu_metrics.jsonl` / `dcgm_tc.tsv`。

### 阶段 3 · nsys profile（只在 GBS=8 × seq=16384 × 10 步，~15 分钟）

- [ ] **FSDP compile=true**：`PROFILE=true PROFILE_START_STEP=5 PROFILE_END_STEP=8 TOTAL_STEPS=10 WARMUP_BENCH=2 MBS=1 GAS=1 MAX_LEN=16384 COMPILE=true SYNTHETIC=true PAD_TO_MAX=true` → `fsdp/profile.nsys-rep`
- [ ] **FSDP compile=false**：同上 `COMPILE=false` → `fsdp_no_compile/profile.nsys-rep`
- [ ] **DeepSpeed**：同 FSDP 参数，复用 bench_deepspeed.sh 的 `PROFILE=true` 包装 → `deepspeed/profile.nsys-rep`
- [ ] **Megatron**：`PROFILE=true PROFILE_DELAY=130 PROFILE_DURATION=6 TOTAL_STEPS=15 MBS=1 GBS=8 MAX_LEN=16384 RECOMPUTE=none` → `megatron/profile.nsys-rep`
- [ ] 四份都过 [`scripts/benchmark/nsys_analyze.py`](../scripts/benchmark/nsys_analyze.py) → `benchmark/profile_analysis/{fsdp,fsdp_no_compile,deepspeed,megatron}_nsys.txt`

### 阶段 4 · 写报告 [`docs/qwen3_5_9b_benchmark_report.md`](./qwen3_5_9b_benchmark_report.md)

结构：

- **摘要表**：FSDP vs DeepSpeed vs Megatron，step_time / throughput / MFU(6N) / peak mem / 稳态功耗 / DCGM TC active，两两 delta
- **§1 测试配置**：对齐表 + 各自默认差异表 + 口径声明（6N MFU 对 hybrid 的高估说明）
- **§2 核心结果**：3 组在 GBS=384 × seq=16384 下的 throughput / MFU 对比
- **§3 nsys kernel 拆解**：GBS=8 × seq=16384 下每框架每步每 rank 的 GPU 活跃时间分类（GEMM / NCCL / flash-attn / linear-attn (GDN kernels) / elementwise / FSDP-split-cat / other），wall-clock vs 活跃累计（overlap 率）
- **§4 DCGM TC active 硬件交叉验证**：3 组在 GBS=384 稳态下的 TC active / DRAM active / GR engine active 均值、median、p95，配合功耗
- **§5 Hybrid 架构 (24 linear + 8 full attn) 对 MFU 口径的影响**：三种口径并排
- **§6 与 Qwen2.5-7B 结果跨代对比**：关键指标 delta
- **§7 选型建议**：9B VLM × 长序列 × 实战 GBS 下的三框架推荐矩阵
- **§8 可复现命令**：完整 3 个 bash 片段 + nsys 片段

## 可能踩到的坑（已知的）

1. **seq=16384 × MBS=1 OOM 风险**：FSDP/DS 侧活动显存 ~55-70 GB/卡是估算，实际可能溢出；smoke 阶段就能发现，必要时开 `GRAD_CKPT=true` 该组，在报告明确标注。
2. **AutoModel auto_class 不匹配**：train.py 只认 `AutoModelForCausalLM`；阶段 0 的补丁 A 必须做。
3. **`torch.compile` × Gated DeltaNet**：linear attn 的 `causal_conv1d` / 递归 state kernel 可能不被 inductor fuse，甚至触发 graph break；nsys 那组 FSDP compile=false 即是兜底。GBS=384 主 bench 若 compile 翻车，回退 `COMPILE=false` 重跑并报告里说清。
4. **DeepSpeed ZeRO-3 × VLM**：`zero3_init_flag=true` 需要 `transformers.integrations.deepspeed.HfDeepSpeedConfig` 在模型加载前 monkey-patch；走 accelerate 路径通常自动接管。若 `Qwen3_5ForConditionalGeneration` 子模块嵌套让 DS 的 param 切分进入 infinite recursion（VLM 类少见但不零），回退 `zero3_init_flag=false` 并手动 `deepspeed.init_distributed()`。
5. **Megatron VLM 权重转换**：首次 `qwen3_5` 会触发 mcore-bridge 的 hf→mcore ckpt 重建（~5-10 min，之后缓存）。首次运行前提前在非 bench 环境预热一次更好。
6. **GBS=384 dataloader 供给**：`MBS=1 GAS=48` 单步内连跑 48 个 micro-batch，`dataloader_num_workers=4` 可能不够，观察 step time 方差；必要时提到 8-12。
7. **DCGM 共享采样污染**：主 bench 期间避免其他容器跑 GPU 任务。
8. **packing 与 16k seq 的交互**：Megatron 的 `--packing true` + `MAX_LEN=16384` 下，真实 jsonl 的 18k 条多轮对话里大部分单条 << 16k，packing 会拼成接近 16k 的 sample，平均包装率应 >90%；若低于 50% 说明数据偏短，改走 `--packing false` 会公平一点但和上一份报告不一致，先保 packing=true。

## 输出交付物

1. `megatron_output/benchmark/fsdp/{bench.jsonl,report.json,train.log,gpu_metrics.jsonl,dcgm_tc.tsv,accelerate_config.rendered.yaml}`
2. `megatron_output/benchmark/deepspeed/{bench.jsonl,report.json,train.log,gpu_metrics.jsonl,dcgm_tc.tsv,accelerate_ds_zero3.rendered.yaml}`
3. `megatron_output/benchmark/megatron/{train.log,report.json,gpu_metrics.jsonl,dcgm_tc.tsv}`
4. `megatron_output/benchmark/{fsdp,fsdp_no_compile,deepspeed,megatron}/profile.nsys-rep`
5. `megatron_output/benchmark/profile_analysis/{fsdp,fsdp_no_compile,deepspeed,megatron}_nsys.txt`
6. [`docs/qwen3_5_9b_benchmark_report.md`](./qwen3_5_9b_benchmark_report.md)
7. 代码补丁：
   - [`scripts/fsdp/train.py`](../scripts/fsdp/train.py)（`AutoModelForImageTextToText` 回退）
   - [`scripts/benchmark/bench_fsdp.sh`](../scripts/benchmark/bench_fsdp.sh)（`FREEZE_VISION` 开关）
   - [`scripts/benchmark/bench_megatron.sh`](../scripts/benchmark/bench_megatron.sh)（`FREEZE_VIT` 开关）
   - **新** [`scripts/benchmark/bench_deepspeed.sh`](../scripts/benchmark/bench_deepspeed.sh)
   - **新** [`scripts/fsdp/accelerate_ds_zero3.yaml`](../scripts/fsdp/accelerate_ds_zero3.yaml)
