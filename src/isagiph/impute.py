"""Graph-augmented imputation of missing DOH disease rates.

The DOH FHSIS tables report "No Data" for whole provinces and cities (166-266 cities and municipalities,
depending on the disease). A regional median ignores everything we know about those places. This module
predicts the missing rates instead, combining two kinds of information:

1. The map. Cities and municipalities are linked to their k nearest neighbours to form a graph, and known
   (log) rates are spread across it with label propagation (Zhou et al., 2004): each place repeatedly takes
   a weighted average of its neighbours while known places stay anchored to their reported value. A
   province with no data therefore borrows from the reporting provinces around it, and from their
   neighbours in turn.
2. The place itself. Poverty, age structure, geographic isolation and population density, which relate to
   tuberculosis, pneumonia and hypertension.

A gradient-boosting model (Friedman, 2001) learns how to combine the propagated value with the place's own
characteristics. Because DOH data go missing a whole province at a time, the method is validated the same
way: whole provinces are hidden (grouped cross-validation by province), their rates predicted from
everything else, and the error compared with the regional median and with label propagation alone. For
each disease the method with the lowest held-out error is used, so the imputation can only improve on the
regional median, never make it worse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from .accessibility import to_local_km

K_NEIGHBOURS = 10
ALPHA = 0.85            # how much each step trusts neighbours vs the place's own anchor
STEPS = 60


def knn_graph(lat: np.ndarray, lon: np.ndarray, k: int = K_NEIGHBOURS) -> sparse.csr_matrix:
    """Row-normalised k-nearest-neighbour graph with Gaussian distance weights."""
    xy = to_local_km(lat, lon)
    d, idx = cKDTree(xy).query(xy, k=k + 1)
    d, idx = d[:, 1:], idx[:, 1:]
    bw = np.median(d[:, -1]) + 1e-9
    w = np.exp(-(d / bw) ** 2)
    n = len(xy)
    W = sparse.csr_matrix((w.ravel(), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    W = (W + W.T) / 2                                           # symmetric: neighbourhood is mutual
    rs = np.asarray(W.sum(axis=1)).ravel()
    return sparse.diags(1 / np.where(rs > 0, rs, 1)) @ W


def propagate(W: sparse.csr_matrix, y: np.ndarray, known: np.ndarray, prior: np.ndarray | None = None) -> np.ndarray:
    """Label propagation: F <- alpha * W F + (1 - alpha) * Y. Y holds the known values and, for unknown places,
    a prior (the regional median of known places), so propagation refines the regional baseline."""
    base = prior if prior is not None else np.full(len(y), np.nanmean(y[known]))
    y0 = np.where(known, y, base)
    F = y0.copy()
    for _ in range(STEPS):
        F = ALPHA * (W @ F) + (1 - ALPHA) * y0
        F[known] = y[known]                                      # reported places stay at their value
    return F


def regional_prior(lg: pd.DataFrame, y: np.ndarray, known: np.ndarray) -> np.ndarray:
    med = pd.Series(y[known]).groupby(lg["region"].to_numpy()[known]).median()
    return lg["region"].map(med).fillna(np.median(y[known])).to_numpy()


def propagate_oof(W, lg: pd.DataFrame, y: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Propagated value for every place computed *without its own province*, so a known place's feature is
    built the same way as a hidden place's (no leakage of its own value into its feature)."""
    prov = lg["province"].astype(str).to_numpy()
    F = np.empty(len(y))
    for g in np.unique(prov):
        k = known & (prov != g)
        sel = prov == g
        F[sel] = propagate(W, y, k, regional_prior(lg, y, k))[sel]
    return F


def _features(lg: pd.DataFrame) -> pd.DataFrame:
    pop = lg["population"].clip(lower=1)
    return pd.DataFrame({
        "poverty": lg["poverty_incidence"].astype(float),
        "pct_0_4": lg["pop_0_4"] / pop, "pct_60": lg["pop_60_plus"] / pop,
        "gida": lg["gida_share"].astype(float),
        "log_density": np.log1p(lg["population"] / lg["area_sqkm"].clip(lower=0.01)),
        "log_pop": np.log1p(lg["population"]), "lat": lg["lat"], "lon": lg["lon"]}).fillna(0)


def _fit_predict(Xtr, ytr, Xte, seed=0):
    m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, min_samples_leaf=15,
                                      l2_regularization=1.0, random_state=seed)
    return m.fit(Xtr, ytr).predict(Xte)


def evaluate(lg: pd.DataFrame, col: str, W, X: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> dict:
    """Hide whole provinces; compare regional median, label propagation alone, and the graph model."""
    obs = lg[f"{col}_imputed"].eq(0).to_numpy()
    y = np.log1p(lg[col].to_numpy(dtype=float))
    idx = np.where(obs)[0]
    groups = lg["province"].astype(str).to_numpy()[idx]
    err = {"regional_median": [], "label_propagation": [], "graph_model": []}
    for tr_i, te_i in GroupKFold(n_splits=n_splits).split(idx, groups=groups):
        tr, te = idx[tr_i], idx[te_i]
        known = np.zeros(len(lg), bool); known[tr] = True
        med = regional_prior(lg, y, known)[te]
        F = propagate_oof(W, lg, y, known)                        # test provinces: hidden; train: own province hidden
        Xg = X.assign(propagated=F, prior=regional_prior(lg, y, known))
        pred = _fit_predict(Xg.iloc[tr], y[tr], Xg.iloc[te], seed)
        for k, p in (("regional_median", med), ("label_propagation", F[te]), ("graph_model", pred)):
            err[k].append(np.abs(p - y[te]))
    return {k: float(np.mean(np.concatenate(v))) for k, v in err.items()}


def graph_impute(lg: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Replace regional-median fills with graph-model predictions where the held-out test shows a gain."""
    lg = lg.copy()
    W = knn_graph(lg["lat"].to_numpy(), lg["lon"].to_numpy())
    X = _features(lg)
    report = {}
    for col in cols:
        flag = f"{col}_imputed"
        if flag not in lg or lg[flag].sum() == 0:
            continue
        ev = evaluate(lg, col, W, X)
        y = np.log1p(lg[col].to_numpy(dtype=float))
        known = lg[flag].eq(0).to_numpy()
        best = min(ev, key=ev.get)                                 # lowest held-out error wins
        if best != "regional_median":
            F = propagate_oof(W, lg, y, known)
            if best == "graph_model":
                Xg = X.assign(propagated=F, prior=regional_prior(lg, y, known))
                fill = _fit_predict(Xg[known], y[known], Xg[~known])
            else:
                fill = propagate(W, y, known, regional_prior(lg, y, known))[~known]
            lg.loc[~known, col] = np.expm1(fill).clip(min=0)
        names = {"regional_median": "regional median", "label_propagation": "label propagation",
                 "graph_model": "graph model"}
        report[col] = {**{f"mae_log_{k}": v for k, v in ev.items()}, "n_missing": int((~known).sum()),
                       "method_used": names[best],
                       "error_reduction_vs_median": float(1 - ev[best] / ev["regional_median"])}
    return lg, report
