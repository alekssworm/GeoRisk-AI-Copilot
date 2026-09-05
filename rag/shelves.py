"""Bounded working shelves with a permanent, searchable bottom shelf."""

from dataclasses import dataclass

from rag.config import (
    SHELF_HEAT_HALF_LIFE,
    SHELF_MIDDLE_CAPACITY,
    SHELF_MIN_SCORE,
    SHELF_RECENT_SLOTS,
    SHELF_TOP_CAPACITY,
)


@dataclass(frozen=True)
class ShelfPolicy:
    top_capacity: int = SHELF_TOP_CAPACITY
    middle_capacity: int = SHELF_MIDDLE_CAPACITY
    recent_slots: int = SHELF_RECENT_SLOTS
    heat_half_life: int = SHELF_HEAT_HALF_LIFE
    min_score: float = SHELF_MIN_SCORE

    def __post_init__(self):
        if self.top_capacity < 1 or self.middle_capacity < 1:
            raise ValueError("Shelf capacities must be positive.")
        if self.recent_slots < 0 or self.heat_half_life < 1:
            raise ValueError("Recent slots must be non-negative and half-life positive.")
        if not 0 < self.min_score <= 1:
            raise ValueError("Minimum retrieval score must be in (0, 1].")


@dataclass
class ShelfEntry:
    stable: bool = False
    hits: int = 0
    heat: float = 0.0
    last_used: int = 0


class ShelfManager:
    """Track popularity separately from immutable source text and citations.

    Entry positions match the append-only chunk list. The latest arrivals reserve
    part of shelf 1; the remaining places favor decaying retrieval frequency.
    Explicitly stable material always stays on shelf 3, even when retrieved.
    """

    def __init__(
        self,
        entries: list[ShelfEntry] | None = None,
        query_count: int = 0,
        policy: ShelfPolicy | None = None,
    ):
        self.entries = entries if entries is not None else []
        self.query_count = query_count
        self.policy = policy or ShelfPolicy()
        self.indices: dict[int, list[int]] = {}
        self.rebalance()

    def _heat(self, index: int) -> float:
        entry = self.entries[index]
        age = max(0, self.query_count - entry.last_used)
        return entry.heat * 0.5 ** (age / self.policy.heat_half_life)

    def rebalance(self) -> None:
        active = [i for i, entry in enumerate(self.entries) if not entry.stable]
        # Keep at least one popularity slot when the top shelf has multiple places.
        recent_count = min(self.policy.recent_slots, max(1, self.policy.top_capacity - 1))
        recent = active[-recent_count:][::-1] if recent_count else []
        recent_set = set(recent)
        ranked = sorted(
            (i for i in active if i not in recent_set),
            key=lambda i: (self._heat(i), self.entries[i].last_used, i),
            reverse=True,
        )
        top_free = self.policy.top_capacity - len(recent)
        top = recent + ranked[:top_free]
        middle = ranked[top_free : top_free + self.policy.middle_capacity]
        upper = set(top + middle)
        self.indices = {
            1: top,
            2: middle,
            3: [i for i in range(len(self.entries)) if i not in upper],
        }

    def add(self, count: int, *, stable: bool = False) -> None:
        self.entries.extend(ShelfEntry(stable=stable) for _ in range(count))
        self.rebalance()

    def record_hits(self, indices: list[int]) -> None:
        self.query_count += 1
        for index in set(indices):
            entry = self.entries[index]
            entry.heat = self._heat(index) + 1.0
            entry.hits += 1
            entry.last_used = self.query_count
        self.rebalance()

    def counts(self) -> dict[str, int]:
        return {str(shelf): len(indices) for shelf, indices in self.indices.items()}
