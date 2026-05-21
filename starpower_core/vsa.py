from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

FloatVector = NDArray[np.float64]


@dataclass(slots=True)
class VSAEngine:
    """Deterministic Vector Symbolic Architecture engine.

    The engine maps symbols into bipolar hypervectors, supports binding,
    bundling, similarity search, and cleanup memory. It is intentionally
    deterministic: the same symbol and dimension always produce the same vector.
    """

    dimensions: int = 4096
    cleanup_memory: dict[str, FloatVector] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dimensions < 128:
            raise ValueError("dimensions must be >= 128 for useful hypervectors")

    def encode(self, symbol: str) -> FloatVector:
        if not symbol or not symbol.strip():
            raise ValueError("symbol must be non-empty")
        digest = hashlib.sha256(symbol.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        vector = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=self.dimensions)
        self.cleanup_memory[symbol] = vector
        return vector

    def bind(self, *vectors: FloatVector) -> FloatVector:
        self._validate_vectors(vectors)
        result = np.ones(self.dimensions, dtype=np.float64)
        for vector in vectors:
            result *= vector
        return result

    def bundle(self, vectors: Iterable[FloatVector]) -> FloatVector:
        vector_list = list(vectors)
        self._validate_vectors(vector_list)
        summed = np.sum(np.stack(vector_list), axis=0)
        bundled = np.where(summed >= 0, 1.0, -1.0).astype(np.float64)
        return bundled

    def unbind(self, bound: FloatVector, known: FloatVector) -> FloatVector:
        self._validate_vectors([bound, known])
        return bound * known

    def similarity(self, left: FloatVector, right: FloatVector) -> float:
        self._validate_vectors([left, right])
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom == 0.0:
            return 0.0
        return float(np.dot(left, right) / denom)

    def cleanup(self, query: FloatVector, threshold: float = 0.15) -> tuple[str | None, float]:
        self._validate_vectors([query])
        best_symbol: str | None = None
        best_score = -1.0
        for symbol, vector in self.cleanup_memory.items():
            score = self.similarity(query, vector)
            if score > best_score:
                best_symbol = symbol
                best_score = score
        if best_score < threshold:
            return None, best_score
        return best_symbol, best_score

    def audit(self) -> dict[str, int | bool]:
        return {
            "dimensions": self.dimensions,
            "symbols": len(self.cleanup_memory),
            "healthy": self.dimensions >= 128,
        }

    def _validate_vectors(self, vectors: Iterable[FloatVector]) -> None:
        vectors_list = list(vectors)
        if not vectors_list:
            raise ValueError("at least one vector is required")
        for vector in vectors_list:
            if vector.shape != (self.dimensions,):
                raise ValueError(f"expected vector shape {(self.dimensions,)}, got {vector.shape}")
            if not np.isfinite(vector).all():
                raise ValueError("vector contains non-finite values")
