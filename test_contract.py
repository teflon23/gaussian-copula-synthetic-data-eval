"""Shared interface and reproducibility contract tests."""
import json
import math
import numbers
import numpy as np
import pytest
from data import make_dataset
from app import run_experiment


def test_dataset_shape_and_seed():
    x, y = make_dataset(seed=42, n_samples=96)
    x2, y2 = make_dataset(seed=42, n_samples=96)
    x3, _ = make_dataset(seed=43, n_samples=96)
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    assert len(x) == len(y) == 96
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(y, y2)
    assert not np.array_equal(x, x3), "Seed must affect generated data"


def test_invalid_dataset_size():
    with pytest.raises(ValueError):
        make_dataset(seed=42, n_samples=0)


def test_experiment_reproducibility_and_metrics():
    first = run_experiment(seed=42, n_samples=96)
    second = run_experiment(seed=42, n_samples=96)
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(second, sort_keys=True, allow_nan=False)
    assert first["n_samples"] == 96
    for field in ("metrics", "baseline_metrics"):
        values = first[field]
        assert isinstance(values, dict) and values
        for key, value in values.items():
            assert isinstance(key, str) and key
            assert isinstance(value, numbers.Real) and not isinstance(value, bool)
            assert math.isfinite(value)


def test_different_dataset_size():
    assert run_experiment(seed=17, n_samples=128)["n_samples"] == 128
