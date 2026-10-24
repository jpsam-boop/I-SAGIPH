"""External validation against outcomes not used to construct the gap index.

LGU-level p-values from ordinary Spearman tests are nominal because nearby LGUs
may be dependent. This module therefore adds region-block bootstrap confidence
intervals and treats the confidence interval—not the unadjusted p-value—as the
primary sign-consistency check. Province aggregation is population weighted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def aggregate(df: pd.DataFrame, level: str, value_cols: list[str]) -> pd.DataFrame:
    """Population-weighted means of ``value_cols`` at ``level`` (e.g. 'province')."""
    w = pd.to_numeric(df["population"], errors="coerce").clip(lower=0)
    valid_w = w.where(np.isfinite(w), 0.0)
    den = valid_w.groupby(df[level]).sum().replace(0, np.nan)
    out = {}
    for c in value_cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        numer = (vals * valid_w).groupby(df[level]).sum(min_count=1)
        out[c] = numer / den
    return pd.DataFrame(out).reset_index()


def _spearman_bootstrap_ci(x: np.ndarray, y: np.ndarray, clusters: np.ndarray | None,
                           seed: int, n_boot: int = 500) -> list[float]:
    """Percentile interval, resampling geographic clusters when multiple are available."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if clusters is not None:
        clusters = np.asarray(clusters).astype(str)[keep]
    if len(x) < 4 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    if clusters is not None and np.unique(clusters).size >= 2:
        levels = np.unique(clusters)
        blocks = [np.flatnonzero(clusters == level) for level in levels]
        sampler = lambda: np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), size=len(blocks))])
    else:
        sampler = lambda: rng.integers(0, len(x), size=len(x))
    vals = []
    for _ in range(n_boot):
        idx = sampler()
        if len(idx) < 4 or np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            continue
        rho = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(rho):
            vals.append(float(rho))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def validate(df: pd.DataFrame, cfg: dict, gap_col: str = "gap_index",
             levels: tuple[str, ...] = ("psgc", "province")) -> pd.DataFrame:
    """Check direction and uncertainty of association between gap index and independent outcomes."""
    rows = []
    outcomes = cfg.get("validation", {}).get("outcomes", [])
    base_seed = int(cfg.get("project", {}).get("seed", 42))
    for outcome_idx, outcome in enumerate(outcomes):
        if outcome not in df or gap_col not in df:
            continue
        expected = cfg.get("validation", {}).get("expected_sign", {}).get(outcome, 0)
        for level_idx, level in enumerate(levels):
            if level == "psgc":
                d = df.copy()
                label = "lgu"
                cluster = d["region"].astype(str).to_numpy() if "region" in d else None
                cluster_method = "region-block bootstrap" if cluster is not None and pd.Series(cluster).nunique() >= 2 else "row bootstrap fallback"
            else:
                if level not in df:
                    continue
                d = aggregate(df, level, [gap_col, outcome])
                label = level
                cluster = None  # each row is one aggregated province, so resample province units
                cluster_method = "province-level bootstrap"
            d[gap_col] = pd.to_numeric(d[gap_col], errors="coerce")
            d[outcome] = pd.to_numeric(d[outcome], errors="coerce")
            d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[gap_col, outcome])
            if len(d) < 4 or d[gap_col].nunique() < 2 or d[outcome].nunique() < 2:
                rho, p = float("nan"), float("nan")
                ci = [float("nan"), float("nan")]
            else:
                result = spearmanr(d[gap_col], d[outcome])
                rho, p = float(result.statistic), float(result.pvalue)
                d_clusters = (d["region"].astype(str).to_numpy()
                              if label == "lgu" and "region" in d else None)
                ci = _spearman_bootstrap_ci(d[gap_col].to_numpy(), d[outcome].to_numpy(), d_clusters,
                                            seed=base_seed + outcome_idx * 101 + level_idx)
            ci_excludes_zero = bool(np.isfinite(ci).all() and (ci[0] > 0 or ci[1] < 0))
            consistent = bool(expected != 0 and np.isfinite(rho) and rho * expected > 0 and ci_excludes_zero)
            rows.append({
                "outcome": outcome, "level": label, "n": len(d), "spearman_rho": rho,
                "spearman_ci95_low": ci[0], "spearman_ci95_high": ci[1],
                "p_value": p, "p_value_kind": "nominal Spearman p-value; not adjusted for spatial dependence",
                "ci_method": cluster_method, "expected_sign": expected, "consistent": consistent,
            })
    return pd.DataFrame(rows, columns=["outcome", "level", "n", "spearman_rho", "spearman_ci95_low",
                                      "spearman_ci95_high", "p_value", "p_value_kind", "ci_method",
                                      "expected_sign", "consistent"])
