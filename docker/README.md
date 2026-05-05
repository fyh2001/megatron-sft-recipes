# `fangyaohua/gemma4-e4b-it-sft` — Gemma-4-E4B-it Text-only SFT (FSDP2 + DS3-equivalent)

Production-ready training image for `google/gemma-4-E4B-it` text-only SFT,
matching the loss / grad_norm / convergence of a DeepSpeed ZeRO-3 baseline
to within +0.10% mean Δloss while running on PyTorch FSDP2.

## On a fresh machine: `bash docker/prepare.sh`

If you've never run this image before, the easiest path is:

```bash
git clone <this-repo>
cd <this-repo>

# Auto-checks driver / docker / nvidia-container-toolkit, pulls the image,
# downloads the 28 GB model. Re-runnable; skips already-done steps.
bash docker/prepare.sh

# Then start your training run
DATASET_PATH=/path/to/your/sft.jsonl bash docker/prepare.sh --skip-model
# (the second invocation skips re-download but runs the 5-step smoke test)
```

The script prints `[ ok ]` for each passed check and `[FAIL]` with a
copy-pastable install command for missing pieces. See
[Prerequisites](#prerequisites-for-a-fresh-machine) below for what it expects.

## Already-prepared machine: TL;DR

```bash
# pull
docker pull fangyaohua/gemma4-e4b-it-sft:260505-u22.04-cu12.9.1-py3.12-t2.10.0-v0.19.0-m1.35.4-s4.1.2-v1

# run a full 2-epoch training (8x H100, ~9 hours)
docker run --rm -it --gpus all --shm-size=16g --ipc=host \
  -v $HOME/.cache/modelscope:/root/.cache/modelscope \
  -v /path/to/your/data:/data \
  -v $(pwd)/runs:/runs \
  -e MODEL=/root/.cache/modelscope/models/google/gemma-4-E4B-it \
  -e DATASET_PATH=/data/SFT.jsonl \
  -e OUT_ROOT=/runs \
  fangyaohua/gemma4-e4b-it-sft:260505-...-v1 \
  bash /opt/megatron-sft-recipes/scripts/gemma4_E4B_opt/fsdp2_offload_E4B_text_only_2ep_a3_pf_fp32master.sh
```

## Prerequisites for a fresh machine

Five things must be in place. `prepare.sh` checks all five and tells you
the install command for any missing one.

### 1. Linux + 8x GPU (80 GB each)

Ubuntu 22.04 / 24.04 verified. v5 config tuned for **8x H100 80GB**;
8x A100 80GB also works (~1.5x slower).
RTX 4090 / lower memory cards will OOM (peak 70 GiB needed).

### 2. NVIDIA driver

```bash
# Verify
nvidia-smi          # should print all 8 GPUs

# Install on Ubuntu (one-shot; reboot after)
sudo apt install -y nvidia-driver-550-server
sudo reboot
```

### 3. Docker (≥ 20.10)

```bash
# Verify
docker version

# Install
curl -fsSL https://get.docker.com | sudo bash
sudo usermod -aG docker $USER
newgrp docker        # or log out / log in
```

### 4. `nvidia-container-toolkit` (lets docker see GPUs)

```bash
# Verify
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi -L

# Install
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 5. Model weights (~28 GB)

`prepare.sh` does this for you, or manually:

```bash
# inside the image (fastest, no separate modelscope CLI install on host)
docker run --rm \
  -v $HOME/.cache/modelscope:/root/.cache/modelscope \
  fangyaohua/gemma4-e4b-it-sft:260505-...-v1 \
  modelscope download \
    --model google/gemma-4-E4B-it \
    --local_dir /root/.cache/modelscope/models/google/gemma-4-E4B-it
```

### 6. Training data (your own)

A single `.jsonl` file in messages format (one record per line, with
`messages: [{"role": "user/assistant", "content": "..."}, ...]`).
Mount it via `-v /your/path.jsonl:/data/sft.jsonl:ro`.

## What's in the image

| Path inside image | What |
|---|---|
| `/opt/ms-swift/` | ms-swift 4.2.0.dev0 (editable install) — patched `swift/model/models/gemma.py` to fix Gemma4 buffer-init bug (loss → 27 without this patch) |
| `/opt/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble/sitecustomize.py` | All v5 runtime monkey-patches (KV-share detach, fp32 NCCL reduce-scatter, fp32 cross-micro grad accum, swift act-offload compat, …) |
| `/opt/megatron-sft-recipes/scripts/gemma4_E4B_opt/fsdp2_offload_E4B_text_only_2ep_a3_pf_fp32master.sh` | Full 2-epoch production launch script (v5) |
| `/opt/megatron-sft-recipes/scripts/gemma4_E4B_opt/bench_variant.sh` | Short-run bench harness (smoke tests + variant comparison) |
| `/opt/megatron-sft-recipes/docs/gemma4_E4B_alt_offload_log.md` | Full debugging timeline (§1-26): every bug, fix, and performance trade-off explained |

What's **not** in the image (mount or download at runtime):
- Model weights (`google/gemma-4-E4B-it`, ~28 GB bf16) — mount `~/.cache/modelscope`
- Training data (`*.jsonl`) — mount `/path/to/data:/data`
- Run outputs (`experiments/`) — mount `/runs`

## Required mounts and environment variables

| docker arg | Required | Default in image | Purpose |
|---|---|---|---|
| `--gpus all` | yes | — | Need 8x GPU (H100 80GB recommended) |
| `--shm-size=16g` | yes | — | FSDP2 NCCL shared-memory transport |
| `--ipc=host` | recommended | — | Avoid spurious shm errors with dataloader workers |
| `-v $HOME/.cache/modelscope:/root/.cache/modelscope` | yes | — | Mount your modelscope cache containing the model |
| `-v /path/to/data:/data` | yes | — | Where your `.jsonl` dataset lives |
| `-v $(pwd)/runs:/runs` | yes | — | Where checkpoints + logs go |
| `-e MODEL=...` | yes | `/home/ubuntu/.cache/modelscope/models/google/gemma-4-E4B-it` | Override to point at your mounted model |
| `-e DATASET_PATH=...` | yes | `/home/ubuntu/fyh/megatron-sft-recipes/sft-data/SFT_0424_2.jsonl` | Override to your dataset |
| `-e OUT_ROOT=...` | yes | `/opt/megatron-sft-recipes/experiments/...` | Override so checkpoints are written under your mount |

## Verifying it's working

After `docker run`, the first stderr lines should include these patch banners
(rank 0 only) — if they're missing the v5 stack didn't load:

```
[gemma4 sdp_preamble] (1/3) backend prefs: flash=False mem_eff=True math=False
[gemma4 sdp_preamble] (8/8) patched Gemma4TextModel.forward + Gemma4TextAttention.forward to route shared_kv_states ...
[gemma4 sdp_preamble] (16/16) patched torch FSDP2 FSDPParam.to_accumulated_grad_if_needed: ...
[gemma4 sdp_preamble] (21) patched FSDP2 foreach_reduce to force reduce_dtype=fp32 ...
[gemma4 sdp_preamble] (21b) patched FSDP2 FSDPParam.init_dtype_attrs: keep reduce_dtype=fp32 ...
[gemma4 sdp_preamble] (29) patched swift activation_cpu_offload: wrap gradient_checkpointing_disable ...
[gemma4 sdp_preamble] (30) patched swift AsyncDoubleBufferGroupOffloadHandler.tensor_pop ...
```

Step 1 should reach **loss ≈ 2.225, grad_norm ≈ 10.29** (matches DS3 baseline
to 4 decimals). If you see `loss > 5` something is wrong — most likely the
swift Gemma4Loader buffer-fix patch did not get applied at build time.

## Performance expectations (8x H100 80GB)

| Metric | Value |
|---|---|
| Mean step time (s/it) | 40.2 |
| Peak GPU memory | 69.6 GiB / 80 GiB |
| TensorCore active | 11.2% (PCIe / NCCL bound, not compute bound) |
| Mean Δloss vs DS3 | +0.10% |
| 2-epoch wall time | ~9 hours |

Hardware that has been validated: 8x NVIDIA H100 80GB SXM (NVLink). Will
fit on 8x A100 80GB with the same config but ~1.5x slower step time.

## Smoke test (5 steps)

Quick way to confirm the image runs end-to-end before committing 9 hours:

```bash
docker run --rm --gpus all --shm-size=16g --ipc=host \
  -v $HOME/.cache/modelscope:/root/.cache/modelscope \
  -v /path/to/your_data:/data \
  -v $(pwd)/runs:/runs \
  -e MODEL=/root/.cache/modelscope/models/google/gemma-4-E4B-it \
  -e DATASET_PATH=/data/SFT.jsonl \
  -e OUT_ROOT=/runs/bench \
  -e LABEL=smoke -e MAX_STEPS=5 -e FULL_SCHED_STOP=1 \
  -e TORCH_DTYPE=float32 \
  -e EXTRA_ENV="GEMMA4_FSDP_WRAP_PLE=1 GEMMA4_KV_SHARE_DETACH=1 GEMMA4_FSDP_REDUCE_FP32_NCCL=1" \
  -e EXTRA_ARGS="--bf16 true --fp16 false --padding_free false --max_grad_norm 1.0" \
  fangyaohua/gemma4-e4b-it-sft:260505-...-v1 \
  bash /opt/megatron-sft-recipes/scripts/gemma4_E4B_opt/bench_variant.sh
```

Expected output (step 1 of `logging.jsonl`):
```json
{"loss": 2.2258, "grad_norm": 10.29, "memory(GiB)": ~69.6, "train_speed(s/it)": ~40}
```

## How v5 works (one-paragraph version)

The image preserves these key v5 design points:

1. **fp32 sharded master + bf16 mixed-precision compute** (`--torch_dtype float32 --bf16 true`) — without the fp32 master, FSDP2's bf16-only optimizer step accumulates rounding over 41 layers × 800 steps and loss diverges by 11% (§24-25 of the embedded debug log).
2. **fp32 NCCL reduce-scatter** (patch 21, env var `GEMMA4_FSDP_REDUCE_FP32_NCCL=1`) — bypasses a PyTorch 2.10 cublas bug that triggers when reduce_dtype != param_dtype.
3. **fp32 unsharded cross-micro grad accumulation** (patch 21b) — matches DS3's `contiguous_gradients=true` semantics. Costs 32 GB/rank of GPU memory which is paid for by `activation_cpu_offload`.
4. **KV-share path detach** (patch 8 with `GEMMA4_KV_SHARE_DETACH=1`) — Gemma-4 layers 24-41 read K/V from layers 22/23; FSDP2 + AC made the gradient flow back through this path twice, causing 1.29x grad-norm overestimation.
5. **swift activation_cpu_offload compat** (patches 29 + 30) — fixes ValueError on `Gemma4AudioModel` and a tensor-tuple assertion in `AsyncDoubleBufferGroupOffloadHandler.tensor_pop`.

Full story (and all attempts that *didn't* work) in `/opt/megatron-sft-recipes/docs/gemma4_E4B_alt_offload_log.md`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `loss > 5` at step 1 | Gemma4Loader buffer fix-up didn't apply | Re-build with the patch present in `docker/patches/`. Check `docker logs` for `[Gemma4Loader] manually loaded N buffer(s)` at startup. |
| OOM at micro 16 of step 1 | activation_cpu_offload didn't activate, or `--torch_dtype bfloat16` was used | Check `fsdp_override.json` has `"activation_cpu_offload": true` (the launch script writes this automatically). Make sure `--torch_dtype float32` (sharded fp32 master) is set. |
| `KeyError: 22` during forward | KV-share patch 8 not loaded — `PYTHONPATH` doesn't reach `_sdp_preamble/sitecustomize.py` | The launch script sets PYTHONPATH automatically; if you bypass it, add `-e PYTHONPATH=/opt/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble`. |
| `1.29x grad_norm vs DS3` | `GEMMA4_KV_SHARE_DETACH=1` not set | The launch script sets this by default. If running `bench_variant.sh` directly, pass it via `-e EXTRA_ENV="GEMMA4_KV_SHARE_DETACH=1 ..."`. |
| `Tried to allocate 1024 MiB ... 78 GiB in use` | Other processes contesting GPU | `nvidia-smi` to verify GPUs are clean before launching. |
| `dcgm_scrape.py fails: connection refused localhost:9500` | dcgm-exporter not in image (host-only service) | Already disabled by default (`DCGM_SCRAPE=0`). To re-enable, run with `--network=host -e DCGM_SCRAPE=1 -e DCGM_URL=http://localhost:9500/metrics`. |

## Image tag convention

```
fangyaohua/gemma4-e4b-it-sft:<DATE>-<BASE_VERSIONS>-<RELEASE>
                              260505-u22.04-cu12.9.1-...-v1
```

- `DATE` (yymmdd): build date — bump on rebuilds
- `BASE_VERSIONS`: pinned base image versions for reproducibility
- `RELEASE`: `v1`, `v2`, ... — bump when sitecustomize patches or scripts change
