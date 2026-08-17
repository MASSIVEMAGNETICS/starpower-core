"""MRI-0: deterministic structured transformer initialization experiments.

This module repairs and quarantines the original MathRevolutionaryInitializer
prototype. It intentionally makes no performance claim. Its job is to provide
reproducible initial conditions that can be compared against conventional
baselines under identical experimental budgets.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class MRICfg:
    vocab_size: int = 6000
    d_model: int = 288
    n_heads: int = 8
    d_ff: int = 1152
    max_len: int = 512
    seed: int = 20250822
    julia_c_real: float = -0.74543
    julia_c_imag: float = 0.11301
    julia_iters: int = 64
    julia_escape: float = 2.0
    zipf_s: float = 1.07
    target_entropy: float = 2.5
    phi_scale: float = (1.0 + 5.0**0.5) / 2.0
    laplacian_strength: float = 0.15

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.d_model <= 0 or self.d_ff <= 0:
            raise ValueError("model dimensions must be positive")
        if self.max_len <= 0 or self.n_heads <= 0:
            raise ValueError("max_len and n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.julia_iters <= 0 or self.julia_escape <= 0:
            raise ValueError("Julia iteration settings must be positive")
        if self.zipf_s <= 0 or self.phi_scale <= 0:
            raise ValueError("zipf_s and phi_scale must be positive")
        if not 0.0 <= self.laplacian_strength <= 1.0:
            raise ValueError("laplacian_strength must be within [0, 1]")


@dataclass(frozen=True)
class MRIComponents:
    """Ablation switches for structured initialization."""

    fractal: bool = True
    chaos_embedding: bool = True
    prime_embedding: bool = True
    laplacian: bool = True
    entropy_rescale: bool = True
    zipf_scale: bool = True
    golden_layernorm: bool = True

    @classmethod
    def baseline(cls) -> MRIComponents:
        return cls(
            fractal=False,
            chaos_embedding=False,
            prime_embedding=False,
            laplacian=False,
            entropy_rescale=False,
            zipf_scale=False,
            golden_layernorm=False,
        )


class _DRand:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def normal(self, shape: tuple[int, ...], scale: float = 1.0) -> Array:
        return self.rng.normal(0.0, scale, size=shape).astype(np.float32)

    def uniform(
        self, shape: tuple[int, ...], low: float = -1.0, high: float = 1.0
    ) -> Array:
        return self.rng.uniform(low, high, size=shape).astype(np.float32)


def _n_primes(n: int) -> Array:
    if n <= 0:
        raise ValueError("n must be positive")
    limit = max(
        100,
        int(n * (math.log(max(n, 2)) + math.log(math.log(max(n, 3)))) + 16),
    )
    while True:
        sieve = np.ones(limit + 1, dtype=bool)
        sieve[:2] = False
        upper = int(math.sqrt(limit))
        for p in range(2, upper + 1):
            if sieve[p]:
                sieve[p * p :: p] = False
        primes = np.flatnonzero(sieve)
        if primes.size >= n:
            return primes[:n]
        limit *= 2


def _fan_std(in_dim: int, out_dim: int) -> float:
    return math.sqrt(2.0 / float(in_dim + out_dim))


def _rescale_std(x: Array, target_std: float) -> Array:
    y = x.astype(np.float32, copy=True)
    y -= np.float32(y.mean())
    std = float(y.std())
    if not math.isfinite(std) or std <= 1e-12:
        raise FloatingPointError("initializer produced degenerate variance")
    y *= np.float32(target_std / std)
    if not np.isfinite(y).all():
        raise FloatingPointError("initializer produced non-finite values")
    return y


class MathRevolutionaryInitializer:
    """Deterministic, fan-aware structured initializer with ablation controls."""

    VERSION = "MRI-0.2.2"

    def __init__(
        self,
        cfg: MRICfg | None = None,
        components: MRIComponents | None = None,
    ) -> None:
        self.cfg = cfg or MRICfg()
        self.components = components or MRIComponents()
        self._rng = _DRand(self.cfg.seed)

    def _julia_grid(self, h: int, w: int) -> Array:
        """Escape-time Julia field that stops updating escaped cells."""
        re = np.linspace(-1.6, 1.6, w, dtype=np.float32)
        im = np.linspace(-1.6, 1.6, h, dtype=np.float32)
        zr, zi = np.meshgrid(re, im)
        zr = zr.astype(np.float32, copy=True)
        zi = zi.astype(np.float32, copy=True)
        cr = np.float32(self.cfg.julia_c_real)
        ci = np.float32(self.cfg.julia_c_imag)
        escaped = np.zeros((h, w), dtype=bool)
        count = np.full((h, w), self.cfg.julia_iters, dtype=np.float32)
        escape_sq = np.float32(self.cfg.julia_escape**2)

        for step in range(1, self.cfg.julia_iters + 1):
            active = ~escaped
            if not active.any():
                break
            ar = zr[active]
            ai = zi[active]
            next_r = ar * ar - ai * ai + cr
            next_i = np.float32(2.0) * ar * ai + ci
            zr[active] = next_r
            zi[active] = next_i
            mag_sq = next_r * next_r + next_i * next_i
            newly_local = mag_sq > escape_sq
            if newly_local.any():
                active_indices = np.flatnonzero(active)
                escaped_indices = active_indices[newly_local]
                escaped.flat[escaped_indices] = True
                count.flat[escaped_indices] = np.float32(step)
                zr.flat[escaped_indices] = 0.0
                zi.flat[escaped_indices] = 0.0

        return count / np.float32(self.cfg.julia_iters)

    def chaos_vector(self, n: int, r: float = 3.99, x0: float = 0.7123) -> Array:
        if n <= 0:
            raise ValueError("n must be positive")
        x = np.empty(n, dtype=np.float32)
        value = np.float32(x0)
        for i in range(n):
            value = np.float32(r * value * (np.float32(1.0) - value))
            x[i] = value
        return _rescale_std(x, 1.0)

    @staticmethod
    def _informative_crop(field: Array, out_dim: int, in_dim: int) -> Array:
        """Select deterministic high-variance rows/columns for narrow heads."""
        if out_dim > field.shape[0] or in_dim > field.shape[1]:
            raise ValueError("requested crop exceeds sampled field")

        if out_dim < field.shape[0]:
            row_variance = np.var(field, axis=1)
            ranked_rows = np.argsort(-row_variance, kind="stable")[:out_dim]
            row_indices = np.sort(ranked_rows)
            cropped = field[row_indices, :]
        else:
            cropped = field

        if in_dim < cropped.shape[1]:
            column_variance = np.var(cropped, axis=0)
            ranked_columns = np.argsort(-column_variance, kind="stable")[:in_dim]
            column_indices = np.sort(ranked_columns)
            cropped = cropped[:, column_indices]

        return cropped.astype(np.float32, copy=True)

    def fractal_matrix(
        self,
        shape: tuple[int, int],
        target_std: float | None = None,
    ) -> Array:
        out_dim, in_dim = shape
        if out_dim <= 0 or in_dim <= 0:
            raise ValueError("matrix dimensions must be positive")
        if out_dim == 1 and in_dim == 1:
            raise FloatingPointError("1x1 fractal matrix cannot have non-zero variance")

        # Narrow Julia grids can quantize to a single escape-time value. Sample
        # a larger field first, randomize its spectral phase, then choose the
        # most informative deterministic rows/columns rather than blindly
        # taking the top-left corner. This guarantees narrow classifier and
        # regression heads inherit within-row structure when such structure
        # exists in the sampled field.
        sample_out = out_dim if out_dim >= 8 else max(8, min(max(in_dim, 8), 32))
        sample_in = in_dim if in_dim >= 8 else max(8, min(max(out_dim, 8), 32))
        base = _rescale_std(self._julia_grid(sample_out, sample_in), 1.0)
        spectrum = np.fft.rfft(base, axis=1)
        phase = np.exp(
            1j
            * self._rng.uniform(
                spectrum.shape,
                -math.pi,
                math.pi,
            ).astype(np.float64)
        )
        field = np.fft.irfft(
            np.abs(spectrum) * phase,
            n=sample_in,
            axis=1,
        ).astype(np.float32)
        out = self._informative_crop(field, out_dim, in_dim)
        return _rescale_std(out, target_std or _fan_std(in_dim, out_dim))

    def fourier_positional(self) -> Array:
        length, dim = self.cfg.max_len, self.cfg.d_model
        pos = np.arange(length, dtype=np.float32)[:, None]
        half = np.arange(dim // 2, dtype=np.float32)[None, :]
        rates = 1.0 / np.power(10000.0, (2.0 * half) / np.float32(dim))
        angles = pos * rates
        pe = np.zeros((length, dim), dtype=np.float32)
        pe[:, 0::2] = np.sin(angles)
        pe[:, 1::2] = np.cos(angles)
        return pe

    def laplacian_matrix(self, n: int, k: int = 3) -> Array:
        if n <= 1:
            raise ValueError("n must be greater than one")
        k = max(1, min(k, (n - 1) // 2 or 1))
        w = np.zeros((n, n), dtype=np.float32)
        rows = np.arange(n)
        for distance in range(1, k + 1):
            w[rows, (rows + distance) % n] = 1.0
            w[rows, (rows - distance) % n] = 1.0
        degree = np.diag(w.sum(axis=1))
        return _rescale_std(degree - w, 1.0)

    def entropy_band(self, x: Array) -> Array:
        y = _rescale_std(x, 1.0)
        hist, _ = np.histogram(y, bins=128, density=False)
        p = hist.astype(np.float64)
        p /= p.sum() + 1e-12
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
        scale = math.exp((self.cfg.target_entropy - entropy) * 0.25)
        return (y * np.float32(scale)).astype(np.float32)

    def xavier_matrix(self, shape: tuple[int, int]) -> Array:
        out_dim, in_dim = shape
        limit = math.sqrt(6.0 / float(in_dim + out_dim))
        return self._rng.uniform(shape, -limit, limit)

    def orthogonal_matrix(self, shape: tuple[int, int]) -> Array:
        out_dim, in_dim = shape
        raw = self._rng.normal(shape)
        if out_dim >= in_dim:
            q, _ = np.linalg.qr(raw, mode="reduced")
        else:
            q_t, _ = np.linalg.qr(raw.T, mode="reduced")
            q = q_t.T
        return _rescale_std(q.astype(np.float32), _fan_std(in_dim, out_dim))

    def _structured_matrix(
        self,
        shape: tuple[int, int],
        *,
        use_laplacian: bool,
    ) -> Array:
        out_dim, in_dim = shape
        target = _fan_std(in_dim, out_dim)
        w = (
            self.fractal_matrix(shape, target_std=target)
            if self.components.fractal
            else self.xavier_matrix(shape)
        )
        if use_laplacian and self.components.laplacian:
            lap = self.laplacian_matrix(in_dim)
            w = w + np.float32(self.cfg.laplacian_strength) * (w @ lap)
        if self.components.entropy_rescale:
            w = self.entropy_band(w)
        return _rescale_std(w, target)

    def token_embedding(self, vocab_size: int, dim: int) -> Array:
        target = 1.0 / math.sqrt(float(dim))
        if not (self.components.prime_embedding or self.components.chaos_embedding):
            return self._rng.normal((vocab_size, dim), scale=target)

        embedding = np.zeros((vocab_size, dim), dtype=np.float32)
        anchors = (
            _n_primes(vocab_size) % dim
            if self.components.prime_embedding
            else np.arange(vocab_size, dtype=np.int64) % dim
        )
        values = (
            self.chaos_vector(vocab_size)
            if self.components.chaos_embedding
            else self._rng.normal((vocab_size,))
        )
        embedding[np.arange(vocab_size), anchors] = values
        spectrum = np.fft.rfft(embedding, axis=1)
        lowpass = np.linspace(1.0, 0.6, spectrum.shape[1], dtype=np.float32)[None, :]
        embedding = np.fft.irfft(spectrum * lowpass, n=dim, axis=1).astype(
            np.float32
        )

        if self.components.zipf_scale:
            ranks = np.arange(1, vocab_size + 1, dtype=np.float32)
            probs = 1.0 / np.power(ranks, np.float32(self.cfg.zipf_s))
            probs /= probs.sum()
            scales = np.float32(0.5) + np.float32(1.5) * np.cumsum(probs)
            embedding *= scales[:, None]
        return _rescale_std(embedding, target)

    def init_linear(
        self,
        in_dim: int,
        out_dim: int,
        *,
        use_laplacian: bool = False,
    ) -> dict[str, Array]:
        return {
            "weight": self._structured_matrix(
                (out_dim, in_dim),
                use_laplacian=use_laplacian,
            ),
            "bias": np.zeros(out_dim, dtype=np.float32),
        }

    def init_attention(self) -> dict[str, Array]:
        dim = self.cfg.d_model
        result = {
            name: self._structured_matrix((dim, dim), use_laplacian=False)
            for name in ("Wq", "Wk", "Wv")
        }
        wo = self._structured_matrix((dim, dim), use_laplacian=False)
        if self.components.prime_embedding:
            mask = np.zeros_like(wo)
            columns = _n_primes(dim) % dim
            mask[np.arange(dim), columns] = np.float32(_fan_std(dim, dim))
            wo = _rescale_std(
                np.float32(0.85) * wo + np.float32(0.15) * mask,
                _fan_std(dim, dim),
            )
        result["Wo"] = wo
        for name in ("bq", "bk", "bv", "bo"):
            result[name] = np.zeros(dim, dtype=np.float32)
        result["n_heads"] = np.asarray(self.cfg.n_heads, dtype=np.int32)
        result["head_dim"] = np.asarray(dim // self.cfg.n_heads, dtype=np.int32)
        return result

    def init_mlp(self) -> dict[str, Array]:
        first = self.init_linear(
            self.cfg.d_model,
            self.cfg.d_ff,
            use_laplacian=True,
        )
        second = self.init_linear(
            self.cfg.d_ff,
            self.cfg.d_model,
            use_laplacian=False,
        )
        return {
            "fc1.W": first["weight"],
            "fc1.b": first["bias"],
            "fc2.W": second["weight"],
            "fc2.b": second["bias"],
        }

    def init_layernorm(self) -> dict[str, Array]:
        gamma_value = (
            1.0 / self.cfg.phi_scale if self.components.golden_layernorm else 1.0
        )
        return {
            "gamma": np.full(self.cfg.d_model, gamma_value, dtype=np.float32),
            "beta": np.zeros(self.cfg.d_model, dtype=np.float32),
            "eps": np.asarray(1e-5, dtype=np.float32),
        }

    def init_block(self) -> dict[str, Any]:
        return {
            "attn": self.init_attention(),
            "mlp": self.init_mlp(),
            "ln1": self.init_layernorm(),
            "ln2": self.init_layernorm(),
        }

    def init_transformer(self, n_blocks: int = 2) -> dict[str, Any]:
        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive")
        model = {
            "meta": {
                "version": self.VERSION,
                "seed": self.cfg.seed,
                "components": asdict(self.components),
            },
            "cfg": asdict(self.cfg),
            "embeddings": {
                "tok": self.token_embedding(self.cfg.vocab_size, self.cfg.d_model),
                "pos": self.fourier_positional(),
            },
            "blocks": {f"b{i}": self.init_block() for i in range(n_blocks)},
            "lm_head": self.init_linear(
                self.cfg.d_model,
                self.cfg.vocab_size,
                use_laplacian=True,
            ),
        }
        self.assert_finite(model)
        return model

    @staticmethod
    def iter_arrays(
        tree: Mapping[str, Any],
        prefix: str = "",
    ) -> Iterator[tuple[str, Array]]:
        for key, value in tree.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, np.ndarray):
                yield path, value
            elif isinstance(value, Mapping):
                yield from MathRevolutionaryInitializer.iter_arrays(value, path)

    @classmethod
    def assert_finite(cls, model: Mapping[str, Any]) -> None:
        for path, array in cls.iter_arrays(model):
            if not np.isfinite(array).all():
                raise FloatingPointError(f"non-finite values in {path}")

    @classmethod
    def flatten_arrays(cls, model: Mapping[str, Any]) -> dict[str, Array]:
        return {path: array for path, array in cls.iter_arrays(model)}

    @classmethod
    def save_npz(
        cls,
        model: Mapping[str, Any],
        path: str | os.PathLike[str],
    ) -> str:
        """Atomically serialize model arrays and return SHA-256 of final artifact."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_name(f".{destination.name}.tmp")
        arrays = cls.flatten_arrays(model)
        metadata = {
            "meta": model.get("meta", {}),
            "cfg": model.get("cfg", {}),
            "array_paths": sorted(arrays),
        }
        payload = {
            **arrays,
            "__metadata_json__": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
        try:
            with temp.open("wb") as handle:
                np.savez_compressed(handle, **payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, destination)
        finally:
            if temp.exists():
                temp.unlink()
        return hashlib.sha256(destination.read_bytes()).hexdigest()


def tensor_statistics(array: Array) -> dict[str, float | int | bool]:
    data = np.asarray(array)
    finite = np.isfinite(data)
    return {
        "size": int(data.size),
        "all_finite": bool(finite.all()),
        "mean": float(data.mean()) if data.size else 0.0,
        "std": float(data.std()) if data.size else 0.0,
        "max_abs": float(np.max(np.abs(data))) if data.size else 0.0,
    }
