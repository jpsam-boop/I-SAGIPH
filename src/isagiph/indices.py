"""Composite indices: Health Vulnerability Score, Healthcare Capacity Score,
and the Healthcare Gap Index (percentile gap, bivariate class, bed gap).

Construction follows the OECD/JRC Handbook on Constructing Composite Indicators
(2008): treat outliers, transform skewed variables, normalise, weight within
domains, then aggregate domains with a geometric mean to limit compensability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def winsorize(s: pd.Series, lo: float, hi: float) -> pd.Series:
    ql, qh = s.quantile([lo, hi])
    return s.clip(ql, qh)


def normalize(s: pd.Series, method: str = "minmax") -> pd.Series:
    """Scale to 0-100. ``zscore`` maps standard scores through the normal CDF."""
    if method == "minmax":
        rng = s.max() - s.min()
        return pd.Series(50.0, index=s.index) if rng == 0 else 100 * (s - s.min()) / rng
    if method == "zscore":
        sd = s.std(ddof=0)
        z = (s - s.mean()) / sd if sd > 0 else s * 0
        return pd.Series(100 * norm.cdf(z), index=s.index)
    raise ValueError(f"unknown normalisation method: {method}")


def prepare_indicator(s: pd.Series, name: str, ncfg: dict, method: str | None = None) -> pd.Series:
    """Impute, log-transform (if listed), winsorise, and normalise one indicator."""
    s = s.astype(float).fillna(s.median())
    if name in ncfg.get("log_vars", []):
        s = np.log1p(s.clip(lower=0))
    s = winsorize(s, *ncfg["winsor"])
    return normalize(s, method or ncfg["method"])


def available_domains(df: pd.DataFrame, domains: dict[str, list[str]]) -> dict[str, list[str]]:
    """Keep only indicators present in ``df``; drop domains left empty."""
    out = {d: [c for c in cols if c in df.columns] for d, cols in domains.items()}
    return {d: cols for d, cols in out.items() if cols}


def composite(df: pd.DataFrame, domains: dict[str, list[str]], ncfg: dict,
              domain_weights: dict[str, float] | None = None,
              method: str | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Weighted geometric mean of domain scores (each = mean of its normalised indicators).

    Returns the composite score (0-100) and the domain-score table.
    """
    domains = available_domains(df, domains)
    dom = pd.DataFrame(index=df.index)
    for d, cols in domains.items():
        dom[d] = pd.concat([prepare_indicator(df[c], c, ncfg, method) for c in cols], axis=1).mean(axis=1)
    w = pd.Series({d: (domain_weights or {}).get(d, 1.0) for d in domains}, dtype=float)
    w = w / w.sum()
    floored = dom.clip(lower=ncfg.get("floor", 1.0))
    score = np.exp((np.log(floored) * w[dom.columns]).sum(axis=1))
    return score, dom


def varimax(L: np.ndarray, gamma: float = 1.0, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Varimax rotation of a loading matrix (Kaiser, 1958)."""
    p, k = L.shape
    if k < 2:
        return L
    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        Lr = L @ R
        u, s, vt = np.linalg.svd(L.T @ (Lr ** 3 - (gamma / p) * Lr @ np.diag(np.diag(Lr.T @ Lr))))
        R = u @ vt
        d_new = s.sum()
        if d and d_new / d < 1 + tol:
            break
        d = d_new
    return L @ R


def pca_weights(X: pd.DataFrame) -> pd.DataFrame:
    """Data-driven indicator weights from factor analysis (OECD & JRC, 2008; Nicoletti et al., 2000).

    1. PCA on the correlation matrix of the (normalised, direction-aligned) indicators.
    2. Keep factors with eigenvalue > 1 (Kaiser criterion; at least one).
    3. Varimax-rotate the loadings.
    4. Assign each indicator to the factor on which it loads most; its weight inside that
       factor is its squared loading scaled to sum to 1.
    5. Weight each factor by its share of the variance explained by the retained factors.
    Returns one row per indicator: factor, loading, weight_in_factor, factor_weight.
    """
    Z = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    C = np.corrcoef(Z.to_numpy(), rowvar=False) if X.shape[1] > 1 else np.ones((1, 1))
    C = np.nan_to_num(C)
    eigval, eigvec = np.linalg.eigh(C)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    k = max(1, int((eigval > 1).sum()))
    L = varimax(eigvec[:, :k] * np.sqrt(np.clip(eigval[:k], 0, None)))
    sq = L ** 2
    factor = sq.argmax(axis=1)
    out = pd.DataFrame({"indicator": X.columns, "factor": factor + 1,
                        "loading": L[np.arange(len(factor)), factor]})
    out["weight_in_factor"] = sq[np.arange(len(factor)), factor]
    out["weight_in_factor"] /= out.groupby("factor")["weight_in_factor"].transform("sum")
    fvar = pd.Series(sq.sum(axis=0), index=np.arange(1, k + 1))
    fvar = fvar[fvar.index.isin(out["factor"])]
    out["factor_weight"] = out["factor"].map(fvar / fvar.sum())
    out["variance_explained"] = float(eigval[:k].sum() / max(eigval.sum(), 1e-12))
    return out


def composite_pca(df: pd.DataFrame, domains: dict[str, list[str]], ncfg: dict,
                  method: str | None = None) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Index from factor-analysis weights: weighted mean inside each factor, then a
    variance-weighted geometric mean across factors (limits compensability)."""
    cols = [c for cols in available_domains(df, domains).values() for c in cols]
    X = pd.concat([prepare_indicator(df[c], c, ncfg, method) for c in cols], axis=1)
    X.columns = cols
    w = pca_weights(X)
    fac = pd.DataFrame(index=df.index)
    for f, g in w.groupby("factor"):
        fac[f"factor_{f}"] = (X[g["indicator"]] * g["weight_in_factor"].to_numpy()).sum(axis=1)
    fw = w.groupby("factor")["factor_weight"].first()
    floored = fac.clip(lower=ncfg.get("floor", 1.0))
    score = np.exp((np.log(floored) * fw.to_numpy()).sum(axis=1))
    w["overall_weight"] = w["weight_in_factor"] * w["factor_weight"]
    return score, fac, w


def pct_rank(s: pd.Series) -> pd.Series:
    """Percentile rank on a 0-100 scale."""
    return 100 * (s.rank(method="average") - 1) / max(len(s) - 1, 1)


def tertile(s: pd.Series) -> pd.Series:
    return pd.qcut(s.rank(method="first"), 3, labels=[1, 2, 3]).astype(int)


def gap_index(hvs: pd.Series, hcs: pd.Series) -> pd.DataFrame:
    """Healthcare Gap Index and its bivariate (3x3) classification.

    gap = pct(HVS) - pct(HCS), in [-100, 100]; positive means need exceeds capacity.
    Priority class = top vulnerability tertile AND bottom capacity tertile.
    """
    out = pd.DataFrame(index=hvs.index)
    out["hvs_pct"] = pct_rank(hvs)
    out["hcs_pct"] = pct_rank(hcs)
    out["gap_index"] = out["hvs_pct"] - out["hcs_pct"]
    out["v_tertile"] = tertile(hvs)
    out["c_tertile"] = tertile(hcs)
    out["bivariate_class"] = "V" + out["v_tertile"].astype(str) + "-C" + out["c_tertile"].astype(str)
    out["priority"] = (out["v_tertile"] == 3) & (out["c_tertile"] == 1)
    return out


def bed_gap(population: pd.Series, accessible_beds: pd.Series, beds_per_1000: float) -> pd.Series:
    """Absolute shortfall (positive) or surplus (negative) of accessible beds vs. the benchmark."""
    return population / 1000.0 * beds_per_1000 - accessible_beds


def build_indices(df: pd.DataFrame, cfg: dict, method: str | None = None,
                  hvs_weights: dict | None = None, hcs_weights: dict | None = None,
                  weighting: str | None = None) -> pd.DataFrame:
    """Compute the Health Need Index (column HVS), Hospital Capacity Index (HCS),
    the Healthcare Gap Index, and the bed gap for every LGU.

    weighting = "pca" (default, factor-analysis weights) or "equal" (equal/Dirichlet
    domain weights). Indicator weights used are attached as ``out.attrs["weights"]``.
    """
    ncfg = cfg["normalization"]
    weighting = weighting or cfg.get("weighting", "equal")
    # Domain scores are always reported (equal weights) so the dashboard can show profiles.
    hvs_eq, hvs_dom = composite(df, cfg["indicators"]["hvs"], ncfg, hvs_weights, method)
    hcs_eq, hcs_dom = composite(df, cfg["indicators"]["hcs"], ncfg, hcs_weights, method)
    weights = None
    if weighting == "pca":
        hvs, _, wv = composite_pca(df, cfg["indicators"]["hvs"], ncfg, method)
        hcs, _, wc = composite_pca(df, cfg["indicators"]["hcs"], ncfg, method)
        weights = pd.concat([wv.assign(index="Health Need Index"), wc.assign(index="Hospital Capacity Index")])
    else:
        hvs, hcs = hvs_eq, hcs_eq
    out = pd.concat([df[["psgc"]], hvs_dom.add_prefix("hvs_"), hcs_dom.add_prefix("hcs_")], axis=1)
    out["HVS"] = hvs
    out["HCS"] = hcs
    out = pd.concat([out, gap_index(hvs, hcs)], axis=1)
    out["bed_gap"] = bed_gap(df["population"], df["accessible_beds"], cfg["benchmarks"]["beds_per_1000"])
    out["bed_gap_shortfall"] = out["bed_gap"].clip(lower=0)
    goal = cfg["benchmarks"].get("goal_2040_per_1000")
    if goal:                                   # beds still needed to reach the 2040 goal (no LGU is "done" at 1:800)
        out["bed_gap_2040"] = bed_gap(df["population"], df["accessible_beds"], goal).clip(lower=0)
    out.attrs["weights"] = weights
    return out
