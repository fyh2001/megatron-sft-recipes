# `fangyaohua/gemma4-e4b-it-sft` — Gemma-4-E4B-it text-only SFT runtime image

只含依赖（PyTorch / CUDA / ms-swift / modelscope）。业务代码（sitecustomize.py、
swift gemma.py patch、启动脚本）由 `entrypoint.sh` 在容器启动时按 `CODE_REPO` /
`CODE_REF` 自动拉取，**镜像稳定，业务代码版本独立**。

数值与 DeepSpeed ZeRO-3 baseline 对齐：mean Δloss +0.044%，比 DS3 快 1.36x。

## TL;DR

```bash
# 1. pull 镜像（~15 GB compressed）
docker pull fangyaohua/gemma4-e4b-it-sft:runtime-260506-u22.04-cu12.9.1-py3.12-t2.10.0-v0.19.0-m1.35.4-s4.1.2-r1

# 2. 下载 launcher 脚本
wget https://raw.githubusercontent.com/fyh2001/megatron-sft-recipes/v1.0/scripts/gemma4_E4B_opt/sft_fsdp.sh
chmod +x sft_fsdp.sh

# 3. 编辑 sft_fsdp.sh 顶部 [USER CONFIG]，填三项：
#    MODEL_HOST_DIR / DATA_HOST_PATH / OUTPUT_HOST_DIR

# 4. 启动
bash sft_fsdp.sh                  # 后台启动训练，2 epoch ~9h on 8x H100
                                # 想前台跑就把 RUN_IN_BACKGROUND=false
```

## 镜像设计

```
docker pull image
  ↓
docker run image
  ↓
entrypoint.sh
  ├─ 1. git clone $CODE_REPO -b $CODE_REF /workspace/code
  ├─ 2. apply swift gemma.py buffer-fix patch (idempotent)
  ├─ 3. export PYTHONPATH=/workspace/code/scripts/gemma4_opt/_sdp_preamble:...
  └─ 4. exec 用户命令
```

**镜像不变，代码改 → git push 即可生效**，不需要 rebuild。

## 关键 env vars

| env | 默认 | 作用 |
|---|---|---|
| `CODE_REPO` | `https://github.com/fyh2001/megatron-sft-recipes.git` | 业务代码 git URL |
| `CODE_REF` | `v1.0` | git ref（tag/branch/commit）|
| `CODE_DIR` | `/workspace/code` | 容器内 clone 目标 |

`CODE_REF=v1.0` 是经过 800-step 全跑验证的稳定版。`CODE_REF=main` 是最新但
未充分测试。私有仓库需要 token：`CODE_REPO=https://USER:TOKEN@gitlab.example.com/...`。

## 完整使用文档

详见仓库 [`docs/fsdp_user_guide.md`](https://github.com/fyh2001/megatron-sft-recipes/blob/v1.0/docs/fsdp_user_guide.md)，包含：
- 全新机器从 0 准备（7 步独立命令）
- v5 vs DS3 baseline 完整性能对比（loss / grad_norm percentile + 实测图）
- 故障排查
- v5 是怎么做的（5 个关键修复 + 调试时间线）

## 镜像 tag 规则

```
fangyaohua/gemma4-e4b-it-sft:<KIND>-<DATE>-<BASE_VERSIONS>-<RELEASE>
```

- `<KIND>` = `runtime`（仅依赖，业务代码 git 拉）
- `<DATE>` = build 日期
- `<BASE_VERSIONS>` = base image 各组件版本
- `<RELEASE>` = 这个 KIND 的 release 序号（`r1`、`r2`...）；**只在依赖升级时 bump**

业务代码版本由 git tag 控制（`v1.0`、`v1.1`...），跟镜像解耦。
