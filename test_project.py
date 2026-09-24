"""test_project.py — Pytest suite for the Gaussian-copula synthetic-data project.

Tests cover:
  * data.make_dataset interface (shape, dtype, seed sensitivity, edge case).
  * app.run_experiment interface (JSON-serializable, n_samples propagation,
    metric keys, finite values, determinism).
  * Independent metric-calculation checks against known fixtures.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from data import make_dataset
from app import run_experiment, _ks_continuous, _tv_discrete, _corr_mae


# ---------------------------------------------------------------------------
# data.make_dataset tests
# ---------------------------------------------------------------------------

class TestMakeDataset:
    def test_default_shape_and_label_values(self):
        X, y = make_dataset(seed=42, n_samples=256)
        assert X.shape == (256, 4)
        assert y.shape == (256,)
        assert set(np.unique(y)).issubset({0, 1})

    def test_different_seed_changes_X(self):
        X1, _ = make_dataset(seed=42, n_samples=256)
        X2, _ = make_dataset(seed=43, n_samples=256)
        assert not np.array_equal(X1, X2)

    def test_invalid_n_samples_raises(self):
        with pytest.raises(ValueError, match="n_samples must be >= 32"):
            make_dataset(seed=42, n_samples=16)

    def test_min_valid_n_samples(self):
        X, y = make_dataset(seed=42, n_samples=32)
        assert X.shape == (32, 4)
        assert y.shape == (32,)


# ---------------------------------------------------------------------------
# app.run_experiment interface tests
# ---------------------------------------------------------------------------

class TestRunExperiment:
    def test_returns_json_serializable_dict(self):
        result = run_experiment(seed=42, n_samples=256)
        # Must be JSON-serializable without error.
        s = json.dumps(result)
        parsed = json.loads(s)
        assert parsed["n_samples"] == 256
        assert isinstance(parsed["metrics"], dict)
        assert isinstance(parsed["baseline_metrics"], dict)

    def test_n_samples_propagated(self):
        result = run_experiment(seed=42, n_samples=64)
        assert result["n_samples"] == 64

    def test_metrics_nonempty_and_finite(self):
        result = run_experiment(seed=42, n_samples=256)
        assert len(result["metrics"]) >= 4
        for k, v in result["metrics"].items():
            assert isinstance(v, float), f"metric {k} is not float"
            assert math.isfinite(v), f"metric {k} is not finite"
        assert len(result["baseline_metrics"]) >= 1
        for k, v in result["baseline_metrics"].items():
            assert isinstance(v, float)
            assert math.isfinite(v)

    def test_deterministic_same_inputs(self):
        r1 = run_experiment(seed=42, n_samples=256)
        r2 = run_experiment(seed=42, n_samples=256)
        assert r1["n_samples"] == r2["n_samples"]
        for key in r1["metrics"]:
            assert r1["metrics"][key] == r2["metrics"][key], f"metric {key} not deterministic"
        for key in r1["baseline_metrics"]:
            assert r1["baseline_metrics"][key] == r2["baseline_metrics"][key]

    def test_different_n_samples_gives_different_metrics(self):
        r256 = run_experiment(seed=42, n_samples=256)
        r64 = run_experiment(seed=42, n_samples=64)
        # At least one metric should differ when sample size changes.
        assert r256["n_samples"] != r64["n_samples"]
        metrics_same = all(
            r256["metrics"][k] == r64["metrics"][k] for k in r256["metrics"]
        )
        assert not metrics_same, "metrics identical for different n_samples"

    def test_invalid_n_samples_raises(self):
        with pytest.raises(ValueError):
            run_experiment(seed=42, n_samples=16)


# ---------------------------------------------------------------------------
# Independent metric-calculation tests against known fixtures
# ---------------------------------------------------------------------------

class TestMetricCalculations:
    def test_ks_continuous_identical_distributions(self):
        """Two-sample KS on identical arrays should be 0."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 2))
        X = np.column_stack([X, rng.integers(0, 3, 100), rng.integers(0, 4, 100)])
        ks = _ks_continuous(X, X.copy())
        assert_allclose(ks, 0.0, atol=1e-12)

    def test_ks_continuous_shifted_distribution(self):
        """Shifting a continuous feature should increase KS statistic."""
        rng = np.random.default_rng(1)
        X_real = rng.standard_normal((200, 2))
        X_syn = X_real.copy()
        X_syn[:, 0] += 2.0  # large shift on first continuous feature
        ks = _ks_continuous(X_real, X_syn)
        assert ks > 0.3, f"KS too small after large shift: {ks}"

    def test_tv_discrete_identical_distributions(self):
        """TV distance on identical discrete columns should be 0."""
        n = 48
        X = np.column_stack([
            np.zeros(n), np.zeros(n),
            np.tile([0, 1, 2], n // 3).astype(np.float64),
            np.tile([0, 1, 2, 3], n // 4).astype(np.float64),
        ])
        tv = _tv_discrete(X, X.copy())
        assert_allclose(tv, 0.0, atol=1e-12)

    def test_tv_discrete_uniform_vs_point_mass(self):
        """TV between uniform over 3 levels and a point mass should be 1/3."""
        n = 300
        X_real = np.column_stack([
            np.zeros(n), np.zeros(n),
            np.tile([0, 1, 2], n // 3).astype(np.float64),  # uniform over 3
            np.zeros(n),
        ])
        X_syn = np.column_stack([
            np.zeros(n), np.zeros(n),
            np.ones(n),  # all level 1
            np.zeros(n),
        ])
        tv = _tv_discrete(X_real, X_syn)
        # TV for col 2: 0.5 * (|1/3-0| + |1/3-1| + |1/3-0|) = 0.5 * (1/3+2/3+1/3) = 0.5*4/3 = 2/3
        # TV for col 3: identical (all zeros) -> 0
        # Mean = (2/3 + 0) / 2 = 1/3
        assert_allclose(tv, 1.0 / 3.0, atol=1e-10)

    def test_corr_mae_identical_matrices(self):
        """Correlation MAE on identical matrices should be 0."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((100, 4))
        mae = _corr_mae(X, X.copy())
        assert_allclose(mae, 0.0, atol=1e-12)

    def test_corr_mae_known_correlation(self):
        """Construct data with a known correlation and verify MAE."""
        n = 500
        rng = np.random.default_rng(3)
        z = rng.standard_normal(n)
        # x0 and x1 perfectly correlated (x1 = x0), others independent.
        X_real = np.column_stack([z, z, rng.standard_normal(n), rng.standard_normal(n)])
        X_syn = X_real.copy()
        mae = _corr_mae(X_real, X_syn)
        assert_allclose(mae, 0.0, atol=1e-12)

    def test_corr_mae_different_correlation(self):
        """Perturb one feature to change correlation; MAE should be > 0."""
        n = 500
        rng = np.random.default_rng(4)
        z = rng.standard_normal(n)
        X_real = np.column_stack([z, z, rng.standard_normal(n), rng.standard_normal(n)])
        X_syn = np.column_stack([z, -z, rng.standard_normal(n), rng.standard_normal(n)])
        mae = _corr_mae(X_real, X_syn)
        # corr(x0,x1) goes from 1 to -1, so MAE includes |1-(-1)|/6 = 2/6 = 1/3 at minimum
        assert mae > 0.1, f"corr_mae too small: {mae}"


# ---------------------------------------------------------------------------
# Edge / integration tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_min_samples_run_experiment(self):
        """run_experiment should work at the minimum valid n_samples."""
        result = run_experiment(seed=42, n_samples=32)
        assert result["n_samples"] == 32
        assert len(result["metrics"]) >= 4
        for v in result["metrics"].values():
            assert math.isfinite(v)

    def test_label_balance_in_dataset(self):
        """With the default seed, both classes should be present."""
        X, y = make_dataset(seed=42, n_samples=256)
        assert y.sum() > 0
        assert (1 - y).sum() > 0

    def test_discrete_feature_ranges(self):
        """Discrete features should stay within their defined level ranges."""
        X, y = make_dataset(seed=42, n_samples=256)
        assert X[:, 2].min() >= 0 and X[:, 2].max() <= 2
        assert X[:, 3].min() >= 0 and X[:, 3].max() <= 3
