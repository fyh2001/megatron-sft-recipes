# SFT Recipes

Multi-backend SFT training repo: Megatron (ms-swift) + FSDP2+compile (Accelerate).
Target: single-node 8×A100/H100 80GB.

## Project Structure

```
scripts/
├── _common.sh              # Shared env vars (Docker, paths, GPU count)
├── 02_convert_data.py      # Data conversion (Mac local)
├── megatron/               # ms-swift + mcore-bridge backend
├── fsdp/                   # Hand-written Accelerate + FSDP2 + torch.compile
│   └── train.py            # Core training script (~300 lines)
└── benchmark/              # Performance comparison tooling
```

## Key Technical Context

- Data format: jsonl with per-message `loss: true/false` field on assistant turns
- Per-message loss masking is critical — different assistant turns in the same conversation have different loss settings
- FSDP2 train.py uses incremental tokenization to build token-level loss masks
- All shell scripts use `: "${VAR:=default}"` pattern for env-var override
- Docker containers: `swift_sft` (Megatron), `fsdp_sft` (FSDP2)
- System Python install (no venv) to preserve Docker image preinstalled packages

## Optimization Stack (FSDP2 backend)

When optimizing or writing training code, apply in this order:
1. Liger-Kernel (fused Triton kernels: RMSNorm, SwiGLU, CrossEntropy, RoPE)
2. torchao Float8 (H100 only)
3. torch.compile (mode="default", use_reentrant=False for grad ckpt)
4. FSDP2 communication-computation overlap

See `.claude/skills/training-optimization.md` for detailed integration guide.

## Conventions

- Shell scripts are the entry points; Python scripts are called by shell scripts
- Parameters passed via env vars from shell → argparse in Python
- Chinese comments in shell scripts, English in Python
- Use `uv` for dependency management, ruff for linting
- No venv on GPU machines — `uv pip install --system --inexact`
