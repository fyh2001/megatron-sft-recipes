#!/usr/bin/env python3
"""Standalone determinism check on ms-swift BatchSamplerShard.

Verifies that given (rank, world_size, base_seed=42, total_samples=51557, batch_size=1):
- Two independent constructions produce identical iteration sequences
- set_epoch(0) -> shuffle A; set_epoch(1) -> shuffle B (different but deterministic)
- Result is purely a function of (rank, world_size, seed, epoch).

Conclusion: if HF Trainer calls set_epoch(epoch) the same way for both engines
(it does — Trainer.train() does it engine-agnostically), epoch 2 will see
identical shuffles in DS3 and FSDP2.
"""
import sys
sys.path.insert(0, "/opt/ms-swift")
from swift.dataloader.shard import BatchSamplerShard
from unittest.mock import patch


def _make_sampler(rank, world_size, seed=42):
    """Construct BatchSamplerShard with rank/world_size mocked (no torch.distributed)."""
    sampler = BatchSamplerShard.__new__(BatchSamplerShard)
    BatchSamplerShard.__init__(
        sampler,
        total_samples=51557,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        data_seed=seed,
    )
    # Override rank / world_size since we're not in a distributed setup
    sampler.__class__ = type("FakeBS", (BatchSamplerShard,), {
        "rank": rank,
        "world_size": world_size,
    })
    sampler.total_samples = 51557 // world_size
    return sampler


def epoch_indices(rank, world_size, epoch, seed=42, n_first=20):
    s = _make_sampler(rank, world_size, seed=seed)
    s.set_epoch(epoch)
    indices = []
    for i, batch in enumerate(s):
        if i >= n_first:
            break
        indices.append(batch[0])
    return indices


def total_indices(rank, world_size, epoch, seed=42):
    s = _make_sampler(rank, world_size, seed=seed)
    s.set_epoch(epoch)
    return [batch[0] for batch in s]


print("=" * 80)
print("Test 1: epoch 0 indices for rank 0..7 (expect identical to training run)")
print("=" * 80)
for rank in range(8):
    idx = epoch_indices(rank, 8, epoch=0, n_first=10)
    print(f"  rank {rank}: {idx}")

print()
print("=" * 80)
print("Test 2: epoch 1 indices for rank 0..7 (different shuffle, but deterministic)")
print("=" * 80)
for rank in range(8):
    idx = epoch_indices(rank, 8, epoch=1, n_first=10)
    print(f"  rank {rank}: {idx}")

print()
print("=" * 80)
print("Test 3: re-running construction yields identical indices?")
print("=" * 80)
for rank in [0, 3, 7]:
    a = epoch_indices(rank, 8, epoch=0, n_first=20)
    b = epoch_indices(rank, 8, epoch=0, n_first=20)
    print(f"  rank {rank} epoch 0 second run == first run? {a == b}")
    a = epoch_indices(rank, 8, epoch=1, n_first=20)
    b = epoch_indices(rank, 8, epoch=1, n_first=20)
    print(f"  rank {rank} epoch 1 second run == first run? {a == b}")

print()
print("=" * 80)
print("Test 4: epoch 0 vs epoch 1 — different shuffles?")
print("=" * 80)
for rank in [0, 3, 7]:
    a = epoch_indices(rank, 8, epoch=0, n_first=10)
    b = epoch_indices(rank, 8, epoch=1, n_first=10)
    n_overlap = len(set(a) & set(b))
    print(f"  rank {rank}: epoch0={a[:5]}... epoch1={b[:5]}... "
          f"overlap_in_first10 = {n_overlap}/10")

print()
print("=" * 80)
print("Test 5: full epoch coverage check — every dataset index appears across all ranks")
print("=" * 80)
ws = 8
for epoch in [0, 1]:
    seen = set()
    total = 0
    for rank in range(ws):
        idx = total_indices(rank, ws, epoch=epoch)
        seen.update(idx)
        total += len(idx)
    print(f"  epoch {epoch}: total batches yielded across 8 ranks = {total}, "
          f"unique indices = {len(seen)}, max = {max(seen) if seen else None}, "
          f"min = {min(seen) if seen else None}")
