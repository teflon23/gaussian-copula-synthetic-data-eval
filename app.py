"""app.py — Gaussian-copula synthetic-data fidelity and utility evaluation.

Compares a Gaussian-copula generator against independent-marginal sampling on
the 4-feature tabular dataset produced by data.make_dataset.  Fidelity is
scored with two-sample KS (continuous features), total-variation distance
(discrete features) and MAE of pairwise correlations.  Utility is scored by
training a LogisticRegression on the synthetic set and evaluating accuracy on
the held-out real test split.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import ks_2samp, norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from data import make_dataset


def _rank_transform(x: np.ndarray) -> np.ndarray:
    """Convert values to their empirical CDF rank, then to standard-normal via probit.

    Ties are handled by averaging ranks.  Returns an array of the same shape
    with values in the standard-normal scale.
    """
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)

    # Average ranks for ties.
    sorted_x = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg_rank = (i + 1 + j + 1) / 2.0  # ranks are 1-based
            ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    # Map ranks to uniform (0,1) via (rank - 0.5) / n, then to standard-normal.
    u = (ranks - 0.5) / n
    # Clip to avoid infinities from norm.ppf at 0 or 1.
    u = np.clip(u, 1e-10, 1.0 - 1e-10)
    return norm.ppf(u)


def _fit_marginals(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Fit per-feature marginals and the label marginal.

    Returns a dict with keys 'x0', 'x1' (Normal), 'x2', 'x3' (empirical CDF),
    and 'y' (empirical CDF for the binary label).
    """
    x0, x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    # Continuous: fit Normal(mean, sd).
    m0, s0 = float(np.mean(x0)), float(np.std(x0, ddof=1))
    m1, s1 = float(np.mean(x1)), float(np.std(x1, ddof=1))
    if s0 < 1e-12:
        s0 = 1.0
    if s1 < 1e-12:
        s1 = 1.0

    # Discrete: empirical CDF via sorted unique values and cumulative counts.
    def _empirical_cdf(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        uniq = np.unique(vals)
        counts = np.array([(vals == u).sum() for u in uniq], dtype=np.float64)
        cum = np.cumsum(counts) / len(vals)
        return uniq, cum

    x2_uniq, x2_cum = _empirical_cdf(x2)
    x3_uniq, x3_cum = _empirical_cdf(x3)
    y_uniq, y_cum = _empirical_cdf(y)

    return {
        "x0": (m0, s0),
        "x1": (m1, s1),
        "x2": (x2_uniq, x2_cum),
        "x3": (x3_uniq, x3_cum),
        "y": (y_uniq, y_cum),
    }


def _sample_independent(margs: dict[str, Any], n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample n rows from independent marginals (baseline)."""
    m0, s0 = margs["x0"]
    m1, s1 = margs["x1"]
    x0 = rng.normal(m0, s0, n)
    x1 = rng.normal(m1, s1, n)

    def _sample_empirical(uniq: np.ndarray, cum: np.ndarray, size: int) -> np.ndarray:
        u = rng.random(size)
        idx = np.searchsorted(cum, u, side="right")
        idx = np.clip(idx, 0, len(uniq) - 1)
        return uniq[idx]

    x2 = _sample_empirical(margs["x2"][0], margs["x2"][1], n)
    x3 = _sample_empirical(margs["x3"][0], margs["x3"][1], n)
    y = _sample_empirical(margs["y"][0], margs["y"][1], n)

    X = np.column_stack([x0, x1, x2, x3])
    return X, y


def _sample_copula(margs: dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                   n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample n rows from a fitted Gaussian copula (algorithm)."""
    # Rank-transform training features and label to standard-normal space.
    z0 = _rank_transform(X_train[:, 0])
    z1 = _rank_transform(X_train[:, 1])
    z2 = _rank_transform(X_train[:, 2])
    z3 = _rank_transform(X_train[:, 3])
    zy = _rank_transform(y_train.astype(np.float64))

    Z = np.column_stack([z0, z1, z2, z3, zy])
    cov = np.cov(Z, rowvar=False)
    # Ensure positive semi-definite by adding small jitter if needed.
    eigvals = np.linalg.eigvalsh(cov)
    if np.min(eigvals) < 1e-8:
        cov += (1e-6 - np.min(eigvals)) * np.eye(5)

    # Draw n samples from the multivariate normal.
    Z_syn = rng.multivariate_normal(np.zeros(5), cov, size=n)

    # Inverse-transform each column back to its original scale.
    def _inverse_rank(z: np.ndarray, margs_entry: tuple) -> np.ndarray:
        # z is on standard-normal scale; map to uniform via normal CDF.
        u = norm.cdf(z)
        u = np.clip(u, 1e-10, 1.0 - 1e-10)
        uniq, cum = margs_entry
        idx = np.searchsorted(cum, u, side="right")
        idx = np.clip(idx, 0, len(uniq) - 1)
        return uniq[idx]

    # Continuous features: inverse of the Normal marginal.
    m0, s0 = margs["x0"]
    m1, s1 = margs["x1"]
    x0_syn = m0 + s0 * Z_syn[:, 0]
    x1_syn = m1 + s1 * Z_syn[:, 1]

    x2_syn = _inverse_rank(Z_syn[:, 2], margs["x2"])
    x3_syn = _inverse_rank(Z_syn[:, 3], margs["x3"])
    y_syn = _inverse_rank(Z_syn[:, 4], margs["y"])

    X = np.column_stack([x0_syn, x1_syn, x2_syn, x3_syn])
    return X, y_syn


def _ks_continuous(X_real: np.ndarray, X_syn: np.ndarray) -> float:
    """Mean two-sample KS statistic over the 2 continuous features."""
    stats = []
    for col in (0, 1):
        stat, _ = ks_2samp(X_real[:, col], X_syn[:, col])
        stats.append(float(stat))
    return float(np.mean(stats))


def _tv_discrete(X_real: np.ndarray, X_syn: np.ndarray) -> float:
    """Mean total-variation distance over the 2 discrete features."""
    tvs = []
    for col, n_levels in ((2, 3), (3, 4)):
        real_counts = np.bincount(X_real[:, col].astype(int), minlength=n_levels) / len(X_real)
        syn_counts = np.bincount(X_syn[:, col].astype(int), minlength=n_levels) / len(X_syn)
        tv = 0.5 * float(np.sum(np.abs(real_counts - syn_counts)))
        tvs.append(tv)
    return float(np.mean(tvs))


def _corr_mae(X_real: np.ndarray, X_syn: np.ndarray) -> float:
    """MAE over the 6 pairwise correlations of the 4 features."""
    def _pairwise_corrs(X: np.ndarray) -> list[float]:
        c = np.corrcoef(X, rowvar=False)
        pairs = []
        for i in range(4):
            for j in range(i + 1, 4):
                pairs.append(float(c[i, j]))
        return pairs

    real_corrs = _pairwise_corrs(X_real)
    syn_corrs = _pairwise_corrs(X_syn)
    return float(np.mean(np.abs(np.array(real_corrs) - np.array(syn_corrs))))


def _lr_accuracy(X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Train LogisticRegression on (X_train, y_train) and return accuracy on (X_test, y_test)."""
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def run_experiment(seed: int = 42, n_samples: int = 256) -> dict[str, Any]:
    """Run the full Gaussian-copula vs independent-marginal evaluation.

    Parameters
    ----------
    seed : int
        Seed for data generation.
    n_samples : int
        Number of real samples.  Must be >= 32.

    Returns
    -------
    dict
        JSON-serializable dict with keys: n_samples, metrics, baseline_metrics,
        explanation.
    """
    if n_samples < 32:
        raise ValueError(f"n_samples must be >= 32, got {n_samples}")

    X, y = make_dataset(seed=seed, n_samples=n_samples)

    # Stratified 70/30 train/test split of the real data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )

    n_syn = len(X_train)
    rng = np.random.default_rng(seed + 1000)

    # Fit marginals on the training set only (no leakage).
    margs = _fit_marginals(X_train, y_train)

    # Baseline: independent marginals.
    X_base, y_base = _sample_independent(margs, n_syn, rng)

    # Algorithm: Gaussian copula.
    X_algo, y_algo = _sample_copula(margs, X_train, y_train, n_syn, rng)

    # Fidelity metrics: compare synthetic vs held-out real test.
    ks_c = _ks_continuous(X_test, X_algo)
    tv_d = _tv_discrete(X_test, X_algo)
    corr_m = _corr_mae(X_test, X_algo)

    # Utility: train LR on synthetic, score on real test.
    util_algo = _lr_accuracy(X_algo, y_algo, X_test, y_test)
    util_base = _lr_accuracy(X_base, y_base, X_test, y_test)

    metrics = {
        "ks_continuous": ks_c,
        "tv_discrete": tv_d,
        "corr_mae": corr_m,
        "utility_algo_acc": util_algo,
    }
    baseline_metrics = {
        "utility_base_acc": util_base,
    }

    # Ensure all values are finite.
    for d in (metrics, baseline_metrics):
        for k, v in d.items():
            if not math.isfinite(v):
                d[k] = 0.0

    explanation = (
        "Gaussian copula generator vs independent-marginal baseline on "
        f"{n_samples} real samples (70/30 stratified split). "
        "Fidelity: lower ks_continuous, tv_discrete, corr_mae is better. "
        "Utility: higher accuracy is better. "
        "The copula preserves cross-feature correlations; the baseline does not. "
        "No temporal ordering; stratification prevents class-imbalance leakage."
    )

    return {
        "n_samples": int(n_samples),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "explanation": explanation,
    }
