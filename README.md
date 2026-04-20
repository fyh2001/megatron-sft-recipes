# ms-swift Megatron SFT 流水线

用 [ms-swift](https://github.com/modelscope/ms-swift) 的 **Megatron 后端（Mcore-Bridge 模式）** 在
单机 8 卡 A100/H100 上对 **Qwen2.5 / Gemma 4** 系列模型做 SFT。

## 方案特点

- **数据格式零展开**：你的 `messages + per-message loss` 数据直接映射成 ms-swift 原生格式
  （19009 条对话，223MB），per-message loss 精确生效，相当于免费拿到"自研方案 B"的效果
- **不需要独立权重转换**：`--model Qwen/Qwen2.5-7B-Instruct` 框架自动下载 + bridge 成 mcore
- **切新模型只改 1 个参数**：`--model ...`，Qwen2.5/3、Gemma 4、DeepSeek 等上百个模型通用
- **uv 统一管理**：Mac 本地 + H100 容器都用同一份 `pyproject.toml` + `uv.lock`

## 目录结构

```
megatron/
├── pyproject.toml           uv 项目配置（公共 + gpu 可选依赖组）
├── uv.lock                  uv 锁文件
├── .python-version          固定 Python 3.11
├── scripts/
│   ├── _common.sh           公共环境变量
│   ├── 01_setup_env.sh      H100 拉镜像 + 启容器 + 装 uv 环境
│   ├── 02_convert_data.py   Arrow -> jsonl（Mac 本地已执行完）
│   ├── 03_sft_qwen25_7b.sh  Qwen2.5-7B 全参 SFT
│   ├── 04_sft_gemma4_e4b.sh Gemma 4 E4B (~4.5B) SFT
│   ├── 05_sft_gemma4_31b.sh Gemma 4 31B Dense SFT
│   └── 06_infer.sh          推理验证
├── sft-data/                02 步产出（已就绪）
│   ├── train.jsonl          18819 条对话，223MB，76021 个 loss=true 的 assistant turn
│   └── valid.jsonl          190 条对话，2.7MB
└── README.md
```

## 已完成的工作（Mac 本地）

- 初始化 uv 项目（`pyproject.toml`、`.python-version`、`uv.lock`、`.venv/`）
  —— Mac 端仍用 venv 跑 02 步数据转换；GPU 容器端不再用 venv（见下文）
- 执行 `02_convert_data.py` 完成数据转换：
  - 原始 Arrow 19901 条 → 跳过 892 条（无任何 `loss=1` 的 assistant turn）→ 保留 19009 条
  - 按 1% 切分：`train.jsonl` 18819 条 / `valid.jsonl` 190 条
  - 字段映射：原 `loss: 1.0` → `loss: true`；`loss: 0.0` → `loss: false`；`loss: None` → 省略（走命令行默认）

## H100 机器上的执行步骤

**默认工作路径**：`/home/ubuntu/perf_opt`（宿主机和容器挂载同名，容器内外路径一致）。
如要改其他路径，设环境变量 `HOST_MOUNT=/your/path` 即可。

### 0. 把工作区同步到 H100

```bash
# 在 Mac 上
rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='.ruff_cache' \
    /Users/huangye/Desktop/megatron/ ubuntu@h100-host:/home/ubuntu/perf_opt/
```

**注意**：`.venv` 不要同步（平台/架构不同）。`sft-data/*.jsonl` 226MB 需要同步。

### 1. 启动容器 + 安装依赖

```bash
# 在 GPU 宿主机
cd /home/ubuntu/perf_opt
bash scripts/01_setup_env.sh
```

默认拉 NGC PyTorch 镜像。如果国内机器拉 `nvcr.io` 慢，可切到阿里云 modelscope 镜像：

```bash
# 国内机器推荐：阿里云杭州（modelscope 镜像自带 ms-swift 生态，首次装更快）
BASE_IMAGE=modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.10.1 \
    bash scripts/01_setup_env.sh

# 北京（备用）
BASE_IMAGE=modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.10.1 \
    bash scripts/01_setup_env.sh

# 海外机器但想避开 nvcr.io：阿里云美西
BASE_IMAGE=modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.10.1 \
    bash scripts/01_setup_env.sh
```

不管选哪个镜像，脚本会完成：
1. 拉镜像
2. 启动容器 `swift_sft`，把 `/home/ubuntu/perf_opt` 挂到容器里同名路径
3. 装 uv
4. `uv pip install --system --inexact -e ".[gpu]"` —— **直接装进容器自带的系统 Python**
   - `--system` 让 uv 把镜像预装的 torch / TE / apex / flash_attn 识别为已满足，
     避免被 megatron-core 的 `torch>=2.6` 约束触发一次 torch 重装，导致 ABI 失配。
   - `--inexact` 保留系统已装包不动（阿里云镜像的 ms-swift 3.10.1 会被升级到 ≥4.1.0；
     mcore-bridge 等缺失包会被补装）。
5. 自动验证所有关键依赖

> **为什么不用 venv？** `uv venv --system-site-packages + uv pip install --inexact` 在 venv 里做依赖解析时并不会把系统 `dist-packages` 里的 torch 视作已安装，任何声明 `torch>=...` 的依赖（例如 megatron-core）都会让 uv 去 pypi 拉一份新 torch 进 venv，覆盖掉镜像原装 torch 的同时，让为旧 torch 编译的 TE / apex / flash_attn 全线 ABI 失配。改用 `--system` 就绕过这个坑。

首次安装耗时：
- NGC 镜像：3-5 分钟（ms-swift 生态全新装）
- 阿里云 modelscope 镜像：1-2 分钟（只升级 ms-swift + 补 mcore-bridge）

### 2. 进入容器做训练

```bash
docker exec -it swift_sft bash
# 业务包已在系统 Python 里，不需要 venv，直接跑：

# ---- Qwen2.5-7B SFT ----
bash /home/ubuntu/perf_opt/megatron-sft-recipes/scripts/03_sft_qwen25_7b.sh

# ---- Gemma 4 E4B SFT（换模型只改 --model）----
bash /home/ubuntu/perf_opt/megatron-sft-recipes/scripts/04_sft_gemma4_e4b.sh

# ---- Gemma 4 31B Dense SFT ----
bash /home/ubuntu/perf_opt/megatron-sft-recipes/scripts/05_sft_gemma4_31b.sh
```

### 3. 推理验证

```bash
# 自动找最新 checkpoint
OUTPUT_SUB=qwen25_7b bash scripts/06_infer.sh

# 指定 checkpoint + prompt
CKPT_DIR=/home/ubuntu/perf_opt/megatron_output/qwen25_7b/v0-xxx/checkpoint-500 \
    PROMPT="Explain entropy in one sentence." \
    bash scripts/06_infer.sh

# 想要交互式对话
swift infer --model /home/ubuntu/perf_opt/megatron_output/qwen25_7b/v0-xxx/checkpoint-500 --stream true
```

## 训练参数速查

所有 SFT 脚本都支持环境变量覆盖：

```bash
MBS=2 \                            # micro batch size
GBS=64 \                           # global batch size
LR=2e-5 MIN_LR=2e-6 \              # 学习率
EPOCHS=3 \
MAX_LEN=8192 \                     # 上下文长度
TP=4 PP=2 \                        # 并行度
RECOMPUTE=full \                   # none/selective/full
OUTPUT_DIR=/home/ubuntu/perf_opt/custom_out \
    bash scripts/03_sft_qwen25_7b.sh
```

## 不同尺寸的推荐配置（8×80GB）

| 模型 | TP | PP | MBS | GBS | Recompute | 备注 |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5/1.5/3B | 1 | 1 | 4-8 | 64 | none | 单机余力大 |
| **Qwen2.5-7B** | 2 | 1 | 1 | 32 | selective | 03 脚本默认 |
| Qwen2.5-14B | 2 | 1 | 1 | 32 | selective | |
| Qwen2.5-32B | 4 | 2 | 1 | 16 | full | |
| Qwen2.5-72B | 8 | 1 | 1 | 8 | full + offload | 极限，建议 LoRA |
| **Gemma 4 E4B** | 1 | 1 | 2 | 32 | selective | 04 脚本默认 |
| Gemma 4 E2B | 1 | 1 | 4 | 64 | none | |
| **Gemma 4 31B** | 4 | 2 | 1 | 16 | full | 05 脚本默认 |
| Gemma 4 26B MoE | 1 | 1 | 1 | 32 | selective + EP=4 | 需加 `--expert_model_parallel_size 4` |

## OOM 排查优先级

按以下顺序逐项尝试：

1. `MBS=1`（最小）
2. `RECOMPUTE=selective` → `full`
3. `MAX_LEN` 缩短（4096 → 2048）
4. `TP` 加大（2 → 4 → 8）
5. `PP` 加大（1 → 2）
6. 加上 `--optimizer_cpu_offload true`（ZeRO-Offload 优化器到 CPU）
7. 改成 LoRA：`TRAIN_TYPE=lora LORA_RANK=16 ...`（需要脚本里加对应参数）

## 常见问题

**Q：容器里 `uv pip install` 报 "could not build wheels for apex" / "could not build wheels for flash-attn"**
A：说明 uv 在往 venv 或别处装，没识别到系统 Python 里镜像已预装的版本。必须用 `--system`：
```bash
cd /home/ubuntu/perf_opt/megatron-sft-recipes
uv pip install --system --inexact -e ".[gpu]"
```
如果确实在 venv 路线里踩到这个坑，清掉 venv 重新走 01 脚本即可（`rm -rf .venv && bash scripts/01_setup_env.sh`）。

**Q：训练开始时卡在 "downloading model"**
A：`MODELSCOPE_CACHE` / `HF_HOME` 指向了非挂载卷，每次重启都要重下。`_common.sh`
默认把两个 cache 都放在 `/home/ubuntu/perf_opt/.cache/`，多机时请改成共享存储（NFS 等）。

**Q：Gemma 4 报 model not found**
A：需要 ms-swift ≥ 4.1.0。01 脚本已按 pyproject.toml 的 gpu 组装，正常情况会装 4.1.0+。
验证：`python -c "import swift; print(swift.__version__)"`。

**Q：per-message `loss` 字段生效了吗？**
A：ms-swift 文档明确 per-message `loss` 优先级高于 `--loss_scale`。你可以开一个 iter
后看训练日志里 `num_tokens_to_loss` 的统计量是否约等于 76021（train 集里 loss=true 的
assistant turn 数）× 平均 token 数。
