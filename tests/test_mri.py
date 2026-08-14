from __future__ import annotations

import hashlib

import numpy as np

from starpower_core.research.mri import MathRevolutionaryInitializer, MRICfg, MRIComponents


def _cfg() -> MRICfg:
    return MRICfg(
        vocab_size=64,
        d_model=16,
        n_heads=4,
        d_ff=32,
        max_len=16,
        julia_iters=12,
        seed=7,
    )


def test_full_mri_is_deterministic_and_finite() -> None:
    first = MathRevolutionaryInitializer(_cfg()).init_transformer(n_blocks=1)
    second = MathRevolutionaryInitializer(_cfg()).init_transformer(n_blocks=1)
    a = MathRevolutionaryInitializer.flatten_arrays(first)
    b = MathRevolutionaryInitializer.flatten_arrays(second)
    assert a.keys() == b.keys()
    for key in a:
        assert np.array_equal(a[key], b[key]), key
        assert np.isfinite(a[key]).all(), key


def test_ablation_changes_weights() -> None:
    full = MathRevolutionaryInitializer(_cfg(), MRIComponents()).init_transformer(1)
    baseline = MathRevolutionaryInitializer(
        _cfg(), MRIComponents.baseline()
    ).init_transformer(1)
    full_w = MathRevolutionaryInitializer.flatten_arrays(full)["blocks.b0.attn.Wq"]
    base_w = MathRevolutionaryInitializer.flatten_arrays(baseline)["blocks.b0.attn.Wq"]
    assert not np.array_equal(full_w, base_w)


def test_linear_initialization_is_fan_aware() -> None:
    init = MathRevolutionaryInitializer(_cfg())
    weight = init.init_linear(16, 32)["weight"]
    expected = np.sqrt(2.0 / (16.0 + 32.0))
    assert np.isclose(weight.std(), expected, rtol=0.05)


def test_save_npz_is_atomic_and_hashes_artifact(tmp_path) -> None:
    model = MathRevolutionaryInitializer(_cfg()).init_transformer(1)
    target = tmp_path / "mri0.npz"
    digest = MathRevolutionaryInitializer.save_npz(model, target)
    assert target.exists()
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    with np.load(target, allow_pickle=False) as payload:
        assert "blocks.b0.attn.Wq" in payload.files
        assert "__metadata_json__" in payload.files


def test_bad_head_geometry_fails_closed() -> None:
    try:
        MRICfg(d_model=10, n_heads=3)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("invalid head geometry should fail")
