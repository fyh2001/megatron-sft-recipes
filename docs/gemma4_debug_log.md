# Gemma-4-26B-A4B-it 优化项目错误复盘集中册

> 所有 phase 过程中遇到的错误都汇到这里，方便后人快速定位同类问题。
>
> 格式：**现象 → 定位 → 修改 → 验证** 四段式。每条指向具体 `run_NN_<label>/stdout.log` 行号和修改的 `file:line`。
>
> 配套文档：[gemma4_optimization_walkthrough.md](gemma4_optimization_walkthrough.md)

---

## Phase 0

### P0-1 `Could not find the transformer layer class Gemma4AudioLayer in the model`

**现象**（[run_20260424_184213_first_try/stdout.log](../megatron_output/gemma4_opt/p0_baseline_fsdp2/run_20260424_184213_first_try/stdout.log)）：

所有 8 个 rank 初始化阶段集体抛：
```
[rank*]: ValueError: Could not find the transformer layer class Gemma4AudioLayer in the model.
```
训练连一步都没跑，swift CLI 启动后 ~45s 崩掉。

**定位**：
1. `grep "Could not find the transformer" /usr/local/lib/python3.12/site-packages/` 定位到 `accelerate/utils/dataclasses.py:2059`
2. 读 accelerate 源码发现 FSDP2 的 `auto_wrap_policy=TRANSFORMER_BASED_WRAP` 在初始化 wrap 策略时：
   ```python
   # accelerate/utils/dataclasses.py:2050-2060
   no_split_modules = getattr(model, "_no_split_modules", None)
   default_transformer_cls_names_to_wrap = list(no_split_modules) ...
   for layer_class in self.transformer_cls_names_to_wrap:
       transformer_cls = get_module_class_from_name(model, layer_class)
       if transformer_cls is None:
           raise ValueError(f"Could not find the transformer layer class {layer_class} in the model.")
   ```
3. 读 `transformers/models/gemma4/modeling_gemma4.py:1439` 找到 `Gemma4PreTrainedModel._no_split_modules = ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer", "Gemma4AudioLayer"]`
4. 但 **gemma-4-26B-A4B-it 没有音频 encoder**（只有 E2B/E4B 有）——`Gemma4AudioLayer` 类存在于代码中但 model 实例里没有这类的子模块
5. accelerate 的检查是严格的："每个类都必须存在于 model"，对 26B A4B 这种纯文本+视觉的 model 就会假阳性 fail

**修改**：
1. [scripts/benchmark/bench_swift_sp_v2.sh](../scripts/benchmark/bench_swift_sp_v2.sh) 新增 `FSDP_TRANSFORMER_CLS_NAMES` 环境变量支持，接受 comma-separated 类名，覆盖 `_no_split_modules` 的默认值，渲染到 `fsdp_override.json` 的 `transformer_cls_names_to_wrap` 字段
2. [scripts/gemma4_opt/p0_baseline_fsdp2.sh](../scripts/gemma4_opt/p0_baseline_fsdp2.sh) 加 `FSDP_TRANSFORMER_CLS_NAMES=Gemma4TextDecoderLayer,Gemma4VisionEncoderLayer`（去掉 `Gemma4AudioLayer`）

**验证**：见 Phase 0 第二次 run（run_20260424_1846xx_after_fix），应该能顺利过 wrap 初始化进入训练 loop。

**推广意义**：**任何 `_no_split_modules` 列表里包含当前 model 未实例化的类的场景**都会踩这个坑。Gemma4 之后所有轴（P1-P8）的 FSDP2 run 都需要带 `FSDP_TRANSFORMER_CLS_NAMES=Gemma4TextDecoderLayer,Gemma4VisionEncoderLayer` 这个 env。考虑把它烧进 `p0_baseline_fsdp2.sh` + 后续所有 gemma4 启动脚本。

---

## Phase 1：GBS/MBS 扫盘

_(空)_

---

## Phase 2：NO_AC

_(空)_

---

## Phase 3：reshard=false

_(空)_

---

## Phase 4：packing

_(空)_

---

## Phase 5：Liger gemma4 dispatch

_(空)_

---

## Phase 6：torch.compile / FP8

_(空)_

---

## Phase 7：MoE 调优

_(空)_

---

## Phase 8：GBS 复核

_(空)_

---

## 通用坑（每期都可能踩）

### C1. nsys profile 直接 wrap torchrun 抓不到 kernel

**现象**：`nsys profile python -m torch.distributed.run ...` 运行完 trace 文件存下来了，但 `nsys stats --report cuda_gpu_kern_sum` 报 `SKIPPED: xxx.sqlite does not contain CUDA kernel data`，或 kernel 数量 ≤ 5。

**根因**：torchrun 的 elastic agent 用 `subprocess.Popen` fork workers，workers 被 re-parent 到 init。nsys 从父进程视角看父进程先死，于是停止采样，workers 的 CUPTI 事件没落到 trace。

**修正**：用 per-rank wrapper，把 nsys 注入每个 worker 的 entry。`--no-python` 让 torchrun 把 wrapper 脚本当子进程执行，wrapper 里只 profile `LOCAL_RANK=0` 的那个。脚本见 [scripts/benchmark/run_with_nsys.sh](../scripts/benchmark/run_with_nsys.sh)。

### C2. DCGM PROF metrics 和 nsys 同时抓会互相干扰

**现象**：profile 跑完一切正常，但 DCGM 那边 `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` 读出来全 0；或 nsys 抓出来 kernel 名字正常但 TC active 信号异常。

**根因**：DCGM 和 nsys 都要独占 CUPTI profile counter slot。同时开会抢。

**修正**：profile 前 `docker stop dcgm-exporter dcgm-exporter-fast`，profile 后再 `docker start` + 重建 `-fast` 容器。

### C3. `nsys stats --format csv` 输出含 stderr / info 行污染

**现象**：CSV 解析器读出 0 行数据。

**根因**：`nsys stats` 默认把 "Generating SQLite file..." 和 "Processing..." 这类 info 往 stdout 打，不是 stderr。`2>/dev/null` 不管用。

**修正**：用 grep 过滤：`nsys stats ... 2>/dev/null | grep -E '^Time|^[0-9]' > kernels.csv`。
