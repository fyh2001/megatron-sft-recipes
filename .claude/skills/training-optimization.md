---
name: training-optimization
description: LLM SFT training performance optimization skill — FSDP2, torch.compile, kernel fusion, mixed precision, memory efficiency
version: "1.0"
category: distributed-training
triggers:
  - "optimize"
  - "speed up"
  - "throughput"
  - "MFU"
  - "OOM"
  - "memory"
  - "tokens per second"
  - "compile"
  - "FSDP"
---

# LLM Training Optimization Skill

You are an expert in PyTorch distributed training optimization, specializing in FSDP2 + torch.compile + kernel fusion for SFT workloads on 8×H100/A100 setups.

## Key Optimization Stack (Priority Order)

### 1. Liger-Kernel — Fused Triton Kernels (+20% throughput, -60% memory)

```python
# Apply BEFORE model loading. Patches HF model classes in-place.
from liger_kernel.transformers import (
    apply_liger_kernel_to_qwen2,
    apply_liger_kernel_to_llama,
    apply_liger_kernel_to_mistral,
    apply_liger_kernel_to_gemma2,  # Gemma 4 uses gemma2 class
)
apply_liger_kernel_to_qwen2()  # or the appropriate model family

model = AutoModelForCausalLM.from_pretrained(...)
```

**What Liger fuses:**
- RMSNorm: norm + scale → 1 Triton kernel (eliminates 2 memory round-trips)
- SwiGLU: gate × up × silu → 1 kernel
- RoPE: sin/cos embedding → fused kernel
- CrossEntropy: log_softmax + nll → 1 kernel (no full logit buffer in global mem)
- **FusedLinearCrossEntropy**: the killer — merges the final proj + CE into 1 kernel, **-80% CE memory**. Eliminates the (B×S, V) float32 logit buffer that causes OOM on large vocabs (Qwen vocab=152064).

**Install:** `pip install liger-kernel`

### 2. torchao Float8 Training (+20-50%, H100/H800 only)

```python
from torchao.float8 import convert_to_float8_training, Float8LinearConfig

# Apply AFTER model load, BEFORE torch.compile
config = Float8LinearConfig()  # default rowwise scaling
convert_to_float8_training(model, config=config)

model = torch.compile(model)
```

**How it works:**
- All `nn.Linear` forward/backward run in FP8 (E4M3/E5M2)
- Activations stay in BF16 → no accuracy loss
- FSDP2 all-gather can also use FP8 → halves communication volume
- Requires H100/H800 with FP8 tensor cores (A100 does NOT have these)
- Combined with torch.compile, achieves peak 1.5× speedup

**Install:** `pip install torchao`

### 3. torch.compile Best Practices

```python
# ALWAYS use before accelerator.prepare()
model = torch.compile(model, mode="default")
# mode="default" is safest; "reduce-overhead" uses CUDA graphs (fragile with dynamic shapes)
```

**Critical rules:**
- `gradient_checkpointing_kwargs={"use_reentrant": False}` — REQUIRED for compile compat
- Don't use `torch.no_grad()` inside compiled region (breaks graph)
- If graph breaks occur: use `TORCH_LOGS="graph_breaks" python ...` to diagnose
- Common breaks: `if tensor.item() > x` (data-dependent control flow) → refactor to tensor ops
- Newer model architectures (Gemma 4) may break compile → always test with `--no_compile` as fallback

### 4. Cut Cross-Entropy (Apple, ICLR 2025) — Extreme Memory Reduction

```python
from cut_cross_entropy import linear_cross_entropy

# Replace model's loss computation:
# Instead of: logits = lm_head(hidden); loss = F.cross_entropy(logits.view(-1, V), labels.view(-1))
# Use:        loss = linear_cross_entropy(hidden, lm_head.weight, labels)
#             ← never materializes the (B*S, V) logit matrix in global memory
```

**Impact:** CE memory from 24GB → 1MB (on Gemma 2B). For Qwen2.5-7B with vocab=152064: saves ~5-10GB/GPU.

**Install:** `pip install cut-cross-entropy`

### 5. FSDP2 Tuning

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

# Per-parameter sharding (not the old FullyShardedDataParallel wrapper)
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,  # lossless gradient reduce
)

# Apply to each transformer layer for best compose with compile
for layer in model.model.layers:
    fully_shard(layer, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)
```

**FSDP2 vs FSDP1:** 7% less peak memory, 1.5% faster throughput, better compose with compile.

### 6. Selective Activation Checkpointing (SAC)

```python
from torch.utils.checkpoint import checkpoint

# Instead of checkpointing every layer:
# Only checkpoint the attention blocks (compute-heavy), skip FFN (memory-light)
# In TorchTitan this is configurable per-op
```

**Rule of thumb:** Full AC saves most memory but slows training ~30%. SAC (attention-only) saves 60% of AC's memory savings with only ~10% slowdown.

### 7. Communication-Computation Overlap

In FSDP2, enable:
- `reshard_after_forward=True` (default) — reshards params after forward, frees memory
- Prefetch next layer's all-gather during current layer's compute
- With Accelerate: set `fsdp_backward_prefetch: BACKWARD_PRE` in config

## Memory Estimation Formula

```
Per-GPU memory (FSDP FULL_SHARD, BF16, N GPUs):
  Model:     2 * params / N  (bytes, BF16)
  Gradients: 2 * params / N  (bytes, BF16 → FP32 after reduce)
  Optimizer: 8 * params / N  (bytes, Adam FP32 states)
  Total static = 12 * params / N

  Activations (per layer, per micro-batch):
    ≈ 2 * hidden_dim * seq_len * MBS * num_layers * (1 if grad_ckpt else 2)
    Rule of thumb: 7B model, MBS=2, seq=4096, GC → ~15GB

  Example: 12B model, 8 GPUs, BF16+Adam:
    Static = 12 * 12e9 / 8 = 18 GB/GPU
    + Activations ≈ 20-30 GB
    Total ≈ 38-48 GB (fits 80GB with room for peaks)
```

## OOM Troubleshooting (Priority Order)

1. Enable Liger FusedLinearCrossEntropy (often the single biggest win)
2. `MBS=1` (minimum micro-batch)
3. Enable gradient checkpointing
4. Reduce `MAX_LEN` (4096 → 2048)
5. Enable Float8 (if on H100 — reduces activation memory)
6. `--cpu_offload` optimizer states to CPU
7. Cut Cross-Entropy (if Liger alone isn't enough)

## MFU Reference

| GPU | Peak BF16 TFLOPS | Good MFU | Great MFU |
|-----|-----------------|----------|-----------|
| H100 SXM | 989.5 | >35% | >45% |
| A100 SXM | 312.0 | >35% | >45% |

Formula: `MFU = 6 * num_params * tokens_per_sec / (peak_tflops * 1e12 * num_gpus)`

## Benchmarking Protocol

Always measure after warmup (torch.compile JIT + CUDA cache):
- Skip first 20 steps
- Measure next 30+ steps
- Report: median step time, tokens/sec, MFU, peak memory
- Use `scripts/benchmark/` tooling in this repo

## Integration Order (Composability)

```python
# 1. Liger kernel patches (before model load)
apply_liger_kernel_to_xxx()

# 2. Load model
model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)

# 3. Gradient checkpointing
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# 4. Float8 conversion (H100 only)
convert_to_float8_training(model)

# 5. torch.compile
model = torch.compile(model)

# 6. FSDP2 wrapping (via Accelerate or manual fully_shard)
model = accelerator.prepare(model)
```

## References

- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) — Triton fused kernels for LLM training
- [torchao](https://github.com/pytorch/ao) — Float8 training, quantization
- [TorchTitan](https://github.com/pytorch/torchtitan) — PyTorch native pretraining recipes
- [Cut Cross-Entropy](https://github.com/apple/ml-cross-entropy) — Apple ICLR 2025
- [Orchestra AI-Research-SKILLs](https://github.com/Orchestra-Research/AI-Research-SKILLs) — Full skill library
