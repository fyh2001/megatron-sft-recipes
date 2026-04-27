# Gemma-4-26B-A4B-it SFT 优化项目 — 阅读索引

> 这是 **gemma-4-26B-A4B-it on 8×H100 80GB FSDP2 SFT 优化** 项目所有产出物的导航索引。最终成果：**vs DS prod baseline 11.2× faster**（422 min → 38 min full-epoch wall）。
>
> 不知道从哪开始？看下面的"按场景"读法 → 选最贴近你需求的那一条线。

---

## 按场景的快速入口

### 🟢 我只想知道"最终配置 + 数字"（5 分钟）

1. 读 [`gemma4_phase_delta_summary.md`](gemma4_phase_delta_summary.md) — 一行一期的 cumulative 表
2. 读 [`gemma4_optimization_walkthrough.md` §0.4 全局默认](gemma4_optimization_walkthrough.md#04-p1-onwards-全局默认已锁定不再变动) — 锁定的 12 项参数
3. 读 [`gemma4_optimization_walkthrough.md` §8.5 最终生产配置](gemma4_optimization_walkthrough.md#85-最终生产配置) — 复制粘贴即可跑的 `swift sft` 命令

**关键数字**（P5★ vs DS prod）：
- Step time: 86,180 ms → 2,781 ms
- Real tokens/sec/H100: 267 → 2,940
- Full-epoch wall: 422 min → 38 min（**11.2× 加速**）
- Peak mem: 43.6 → 64.9 GiB（仍在 80 GB 内）

### 🟡 我要复制配置在我自己的环境跑 / 改 hyperparam（30 分钟）

1. [`scripts/gemma4_opt/p5_liger.sh`](../scripts/gemma4_opt/p5_liger.sh) — P5★ peak 启动脚本（粘贴可跑）
2. [`scripts/gemma4_opt/_sdp_preamble/sitecustomize.py`](../scripts/gemma4_opt/_sdp_preamble/sitecustomize.py) — 5 个 monkey-patch（必须通过 PYTHONPATH 加载）
3. [`scripts/benchmark/liger_gemma4_patch.py`](../scripts/benchmark/liger_gemma4_patch.py) — Liger gemma4 dispatch 实现
4. 容器要求确认：
   - `docker exec fsdp_sft md5sum /usr/local/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py`
   - 期望 `39ebf386a992fea9eac0883f459ac658`（gemma4 modeling 4 处 patch）

### 🟣 我手上是干净机器，要从零搭一遍（90 分钟）

读 [`gemma4_setup_from_scratch.md`](gemma4_setup_from_scratch.md) — 12 步从硬件 checklist → docker pull → 模型下载 → patch 应用 → smoke test → P5 peak 完整流程，含 5 个最容易撞到的坑 + 一键 setup 脚本。

### ⚫ 我想自己踩一遍坑、自己写补丁、不要现成答案（6-10 小时）

读 [`gemma4_diy_journey.md`](gemma4_diy_journey.md) — 把项目反向拆成 **10 个 puzzle quest**，每关给"启动状态 + naive 命令 + 错误现象 + 渐进式 hint"，答案折叠在 `<details>` 里。
配套：`bash scripts/gemma4_opt/diy_reset.sh`（重置容器到"未踩坑"状态）/ `bash scripts/gemma4_opt/diy_restore.sh`（一键恢复）。

### 🟠 我要看每期具体踩过哪些坑、为什么这么定（2 小时）

主线读 [`gemma4_optimization_walkthrough.md`](gemma4_optimization_walkthrough.md)（≈800 行），按 9 期顺序：
- §0 环境 + 方法学 + 锁定参数
- §0g FSDP2 ↔ DS 训练曲线对齐（**最难的一期**，13 次 attempt 解 OOM 链）
- §1-§5 优化 phase（每期一段）
- §6 torch.compile blocked / §7 MoE blocked
- §8 GBS 复核 + final synthesis
- §9 累计改善 + 上游 PR 候选 + TL;DR

辅读：
- [`gemma4_debug_log.md`](gemma4_debug_log.md) — 集中错误复盘（按期分章节）
- [`gemma4_baseline_summary.md`](gemma4_baseline_summary.md) — 双后端 baseline 一览（Phase 0 产物）

### 🔴 我要审计某次 run 的原始数据（10 min/run）

每个 run 目录下固定 5 个文件：
```
experiments/gemma4_opt/p<N>_<axis>/run_<TIMESTAMP>_<LABEL>/
├── cmd.sh            ← 该次启动的完整命令（粘贴可复跑）
├── stdout.log        ← 完整 stdout + stderr（不截断）
├── STATUS            ← 一行: SUCCESS / FAILED + 原因摘要
├── fsdp_override.json← FSDP2 配置副本
└── report.json       ← bench 解析后的数字（仅 SUCCESS）
+ logging.jsonl       ← swift trainer 的 per-step 日志（symlink）
+ dcgm_tc.tsv         ← DCGM tensor-core active 10Hz
+ gpu_metrics.jsonl   ← gpu_monitor.py 1Hz 全卡 power/util
```

每期 run 的时间线索引在 `experiments/gemma4_opt/p<N>_<axis>/attempts.md`。

---

## 主要文档（按重要性排序）

### 1. [`gemma4_phase_delta_summary.md`](gemma4_phase_delta_summary.md) — 一行一期的 cumulative 表 ⭐⭐⭐
最快速度看到所有 phase 的关键数字（step ms、peak mem、tokens/s/GPU、full-epoch wall、speedup）。如果你只读一份，读这个。

### 2. [`gemma4_optimization_walkthrough.md`](gemma4_optimization_walkthrough.md) — 主 walkthrough ⭐⭐⭐
完整的 9 期优化叙事 + 每期的：
- 启动命令（粘贴可跑）
- 4 段式 debug 记录（现象 → 定位 → 修改 → 验证）
- 实测数据（step / mem / MFU / loss 头部 / NaN / wall）
- 分析 + 下期预告

参考样式来自 Qwen3.5 的 [`fsdp2_optimization_walkthrough.md`](fsdp2_optimization_walkthrough.md)（如果你之前看过那个，结构一致）。

### 3. [`gemma4_optimization_plan.md`](gemma4_optimization_plan.md) — 项目计划（开工时写的）⭐⭐
9 期路线图原始版本，含每期的目标、扫盘矩阵、风险预案。**和实测对比可以看到哪些预案命中了**（packing 真的是最大头 / torch.compile 真的崩 / MoE knob 真没暴露）。

### 4. [`gemma4_debug_log.md`](gemma4_debug_log.md) — 错误复盘集中册 ⭐
按期分章节的所有 4 段式 debug 汇总。如果你后人想"跟随我们的定位思路重现调查"，这是入口。

### 5. [`gemma4_baseline_summary.md`](gemma4_baseline_summary.md) — Phase 0 双后端 baseline 一览 ⭐
项目早期产物，就单独 P0 阶段写的。

### 6. [`gemma4_setup_from_scratch.md`](gemma4_setup_from_scratch.md) — 从零搭建复现环境 ⭐⭐
12 步从干净机器到 P5 peak 跑通：硬件 checklist → docker → 模型下载 → ms-swift fork install → modeling patch → smoke test → DS baseline → P5 peak → 全套 phase。含一键 setup 脚本。

### 7. **本文** [`gemma4_README.md`](gemma4_README.md) — 你正在读

---

## 各 phase 的 `attempts.md`（时间线索引）

每期目录下都有，按时间顺序排所有 run（成功 + 失败），格式：`run_NN/ · 时间戳 · STATUS · 一句话结论`。

| Phase | 链接 | 关键内容 |
|---|---|---|
| P0a/b/c | （没单独 attempts.md，run 较少）| FSDP2 baseline + DS prod baseline |
| **P0g** | [`p0_train_align/attempts.md`](../experiments/gemma4_opt/p0_train_align/attempts.md) | **13 次 attempt 解 OOM 链** — 项目最难的一期 |
| P1 | [`p1_gbs_sweep/attempts.md`](../experiments/gemma4_opt/p1_gbs_sweep/attempts.md) | 7 个配置点的 GBS/MBS 二维扫盘 |
| P2 | [`p2_no_ac/attempts.md`](../experiments/gemma4_opt/p2_no_ac/attempts.md) | AC=off native vs offload |
| P3 | [`p3_reshard/attempts.md`](../experiments/gemma4_opt/p3_reshard/attempts.md) | ZeRO-2 模式（不采纳）|
| P4 | [`p4_packing/attempts.md`](../experiments/gemma4_opt/p4_packing/attempts.md) | packing —— 最大单期增益 |
| P5 | [`p5_liger/attempts.md`](../experiments/gemma4_opt/p5_liger/attempts.md) | Liger gemma4 dispatch + FLCE bug |
| P6 | [`p6_compile/attempts.md`](../experiments/gemma4_opt/p6_compile/attempts.md) | torch.compile blocked |
| P8 | [`p8_gbs_resweep/attempts.md`](../experiments/gemma4_opt/p8_gbs_resweep/attempts.md) | 完整 stack 上 GBS 复核 |

---

## 脚本（按用途分类）

### 启动 phase run（粘贴可跑）
- [`p0_baseline_fsdp2.sh`](../scripts/gemma4_opt/p0_baseline_fsdp2.sh) — FSDP2 baseline
- [`p0_baseline_ds_prod.sh`](../scripts/gemma4_opt/p0_baseline_ds_prod.sh) — DS prod baseline
- [`p0_train_align_fsdp2_gbs64.sh`](../scripts/gemma4_opt/p0_train_align_fsdp2_gbs64.sh) — P0g align
- [`p1_gbs_sweep.sh`](../scripts/gemma4_opt/p1_gbs_sweep.sh) — P1 GBS sweep
- [`p2_no_ac.sh`](../scripts/gemma4_opt/p2_no_ac.sh)
- [`p3_reshard.sh`](../scripts/gemma4_opt/p3_reshard.sh)
- [`p4_packing.sh`](../scripts/gemma4_opt/p4_packing.sh)
- [`p5_liger.sh`](../scripts/gemma4_opt/p5_liger.sh)
- [`p6_compile.sh`](../scripts/gemma4_opt/p6_compile.sh)
- [`p8_gbs_resweep.sh`](../scripts/gemma4_opt/p8_gbs_resweep.sh)

### 共享 patches（核心知识资产）
- [`_sdp_preamble/sitecustomize.py`](../scripts/gemma4_opt/_sdp_preamble/sitecustomize.py) — 5 个 monkey-patch（PYTHONPATH 注入）
- [`liger_gemma4_patch.py`](../scripts/benchmark/liger_gemma4_patch.py) — Liger gemma4 dispatch（PR-ready）
- [`gemma4_modeling_compat.patch`](../scripts/gemma4_opt/gemma4_modeling_compat.patch) — modeling_gemma4 4 处 compat patch（已 apply 到容器，md5 校验）

### Bench 工具链
- [`scripts/benchmark/bench_swift_sp_v2.sh`](../scripts/benchmark/bench_swift_sp_v2.sh) — swift sft + SP bench wrapper
- [`scripts/benchmark/report_swift_sp.py`](../scripts/benchmark/report_swift_sp.py) — bench 输出 → report.json
- [`scripts/benchmark/extract_loss_curve.py`](../scripts/benchmark/extract_loss_curve.py) — logging.jsonl → loss_curve.tsv
- [`scripts/benchmark/compare_loss_curves.py`](../scripts/benchmark/compare_loss_curves.py) — 双 jsonl 逐步差异对比（P0g align 用）
- [`scripts/benchmark/gpu_monitor.py`](../scripts/benchmark/gpu_monitor.py) — 1Hz nvidia-smi 采集
- [`scripts/benchmark/dcgm_scrape.py`](../scripts/benchmark/dcgm_scrape.py) — 10Hz DCGM tensor-core active
- [`scripts/benchmark/run_with_nsys.sh`](../scripts/benchmark/run_with_nsys.sh) — per-rank nsys 采样
- [`scripts/benchmark/nsys_classify.py`](../scripts/benchmark/nsys_classify.py) — nsys kernel 分类（GEMM/NCCL/FA/...）

### Smoke tests（一次性 recon 用）
- `sdp_mem_efficient_smoke.py` — 验证 mem_efficient SDPA 在 head_dim=512 工作
- `sdp_with_mask_smoke.py` — 不同 mask 类型下 mem_eff 的支持
- `sdp_gqa_smoke.py` — `enable_gqa=True` kwarg 与 mem_eff 不兼容验证
- `dist_backend_smoke.py` — `cpu:gloo,cuda:nccl` mixed backend 验证

### 汇总工具
- [`build_p1_summary.py`](../scripts/gemma4_opt/build_p1_summary.py) — P1 sweep 报告聚合

---

## "5 件套 patch" 全集（**最重要的可移植知识资产**）

如果你想把这套搬到别的项目（其他 VLM、其他 size）：

```python
# scripts/gemma4_opt/_sdp_preamble/sitecustomize.py 里 5 个 patch：

(1) torch.backends.cuda.enable_mem_efficient_sdp(True) + flash/math 关
    解决: gemma4 global head_dim=512 fallback SDPA 时不撞 O(N²) attn matrix

(2) monkey-patch transformers.integrations.sdpa_attention.use_gqa_in_sdpa → False
    解决: mem_eff backend 不接 enable_gqa=True kwarg，强制走 repeat_kv 路径

(3) monkey-patch dist.init_process_group(nccl) → cpu:gloo,cuda:nccl
    解决: FSDP2 + CPU offload 下 clip_grad_norm 的 NCCL/CPU all_reduce 冲突

(4) swift.template.Gemma4Template.support_padding_free = True
    解决: swift 默认拒绝 VLM template 用 packing/padding_free，文本-only 训练强解锁

(5) liger_kernel.transformers.monkey_patch.MODEL_TYPE_TO_APPLY_LIGER_FN['gemma4']
    = liger_gemma4_patch._apply_liger_kernel_to_gemma4
    解决: Liger main 没 gemma4 dispatch，自家写一份 (RMSNorm + GeGLU)
```

```bash
# 对应的 swift cli flag 5 件套（最终生产配置）：
--packing true                  # P4 token 密度 5×
--use_liger_kernel true         # P5 触发 patch (5)
--truncation_strategy right     # 与 DS prod 一致
--gradient_accumulation_steps 1 # GAS=1，避开 FSDP2 no_sync OOM
# fsdp_override.json:
#   "fsdp": "full_shard auto_wrap"  (NOT offload, native FSDP2 = peak)
#   "activation_checkpointing": false  (P2 锁，gemma4 + mem_eff 下省不到)
#   "transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer"]
#   "cpu_ram_efficient_loading": false  (gemma4 VLM root params bug workaround)
```

---

## 上游 PR 候选

| 项目 | 改动 | 状态 |
|---|---|---|
| [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) | 加 `apply_liger_kernel_to_gemma4` dispatch | code 已写，需先修 FLCE reduction bug |
| [modelscope/ms-swift](https://github.com/modelscope/ms-swift) | `Gemma4Template.support_padding_free = True` | 一行改动 |
| [modelscope/ms-swift](https://github.com/modelscope/ms-swift) | `swift/sequence_parallel/ulysses.py` 修 dict-attention_mask 的 `pad()` 路径 | 中等改动，解锁 MBS≥2 |

---

## 下次工作能从哪开始（"future work" 索引）

详见 walkthrough §9 + 上面"未来方向"表。优先级建议：

1. **修 Liger FLCE bug**（小工作量，再省 8 GB 显存 + 3-5% 速度）
2. **修 swift Ulysses VLM dict-mask**（中工作量，解锁 MBS≥2）
3. **等 PyTorch 2.11 修 inductor TF32 bug 后重试 torch.compile**（被动等待）
4. **多机扩展（multi-node）** —— 跨 8/16 GPU 测 linear scaling

---

## 一些常见疑问

**Q: 文档里"GBS=4"和 bench report 里"gbs=8"为啥不一样？**  
A: bench `report_swift_sp.py` 计算 `gbs = MBS × NPROC × GAS`（不除 SP）。Ulysses SP=2 让 SP-pair 内两 rank 处理同一 sample，所以**真实 unique sample count = bench_gbs / SP**。文档统一用真实 GBS=4。

**Q: "tokens_per_sec_per_gpu" 在 bench report.json 里是 1521 (DS) / 5891 (P5)，跟我表里 267/2940 对不上？**  
A: bench 用 padded 公式（`MAX_LEN × bench_gbs / step_s / NPROC`），假设每 micro 都填满 16k。对 non-packing run 虚高 5×。文档里我用的是 `mean(rank0.tokens_this_step) × DP / step_s`，**真实非 padding token 吞吐**。详见 `gemma4_optimization_walkthrough.md` §0.4 末尾说明。

**Q: 为啥 P0c (DS) wall 422 min 这么慢？**  
A: ZeRO-3 stage offload(opt+param) 的每参数都 D2H/H2D 一次，每个 micro forward+backward 几十 GB IO，加上 GAS=16 累积 → step 86 sec。FSDP2 native 2.78 sec/step（**31× 快**）。详见 walkthrough §8.4。

**Q: 11.2× 加速这么夸张是真实的吗？**  
A: 是。P0c (DS prod 422 min) 实测 full epoch，P5★ (38 min) 是 30 步 bench × `tokens_this_step / step_s` 投影。两边都用 right truncation 覆盖完整 18819 sample。差异主要来自 (1) DS offload IO 太重 (2) packing 让 token 密度 5.8× (3) Liger fusion +14.7%。
