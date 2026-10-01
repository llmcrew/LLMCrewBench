"""Deterministic, per-task randomness.

The suite must be identical for every model in a release and reproducible from
the master seed alone, so every generator gets its own stream derived from
``(master_seed, task_id)``. Never share one ``random.Random`` between tasks.
"""

from __future__ import annotations

import hashlib
import random

MASK = (1 << 64) - 1

#: Signed 64-bit ceiling. vLLM validates ``seed`` as a signed int64 and rejects
#: a plain 64-bit value, so the derived seed is folded into that range: the
#: stream stays deterministic, it just never overflows the wire format.
SEED_MAX = (1 << 63) - 1


def derived_seed(master_seed: int, *parts: str) -> int:
    """Stable 64-bit seed from the master seed and any task labels."""
    payload = "|".join([str(master_seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (SEED_MAX + 1)


def task_rng(master_seed: int, *parts: str) -> random.Random:
    return random.Random(derived_seed(master_seed, *parts))


def fingerprint(master_seed: int, payload: str) -> str:
    """SHA-256 of the master seed and a serialised fixture, for provenance."""
    return hashlib.sha256(f"{master_seed}\n{payload}".encode("utf-8")).hexdigest()


class Reader:
    """Reads shuffled values from a fixed pool, refilling deterministically."""

    def __init__(self, rng: random.Random, values: list[str]) -> None:
        self._rng = rng
        self._values = list(values)
        self._pool: list[str] = []

    def take(self) -> str:
        if not self._pool:
            self._pool = self._values[:]
            self._rng.shuffle(self._pool)
        return self._pool.pop()

    def take_many(self, n: int) -> list[str]:
        return [self.take() for _ in range(n)]
