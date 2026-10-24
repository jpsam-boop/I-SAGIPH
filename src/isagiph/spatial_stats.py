"""Spatial statistics on LGU centroids.

Self-contained NumPy implementations of global Moran's I, local Moran's I
(LISA) with conditional permutation inference, Getis-Ord Gi*, Benjamini-Hochberg
FDR control, and spatially constrained clustering. Results can be cross-checked
against PySAL ``esda`` (see tests/test_spatial_stats.py). KNN weights are used
so that island municipalities always have neighbours.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import norm

from .accessibility import to_local_km


def knn_weights(lat: np.ndarray, lon: np.ndarray, k: int = 6) -> sparse.csr_matrix:
    """Binary k-nearest-neighbour weights (no self-links), as a sparse matrix."""
    xy = to_local_km(lat, lon)
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    n = len(xy)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    return sparse.csr_matrix((np.ones(n * k), (rows, cols)), shape=(n, n))


def row_standardize(W: sparse.csr_matrix) -> sparse.csr_matrix:
    rs = np.asarray(W.sum(axis=1)).ravel()
    rs[rs == 0] = 1
    return sparse.diags(1 / rs) @ W


def morans_i(x: np.ndarray, W: sparse.csr_matrix, permutations: int = 999,
             seed: int = 0) -> dict:
    """Global Moran's I with a permutation pseudo p-value (two-sided, folded)."""
    rng = np.random.default_rng(seed)
    Wr = row_standardize(W)
    z = np.asarray(x, float) - np.mean(x)
    n = len(z)

    def stat(v: np.ndarray) -> float:
        return float(n / Wr.sum() * (v @ (Wr @ v)) / (v @ v))

    I = stat(z)
    sims = np.array([stat(rng.permutation(z)) for _ in range(permutations)])
    ei = -1.0 / (n - 1)
    larger = np.sum(np.abs(sims - ei) >= abs(I - ei))
    return {"I": I, "EI": ei, "p_sim": (larger + 1) / (permutations + 1),
            "z_sim": (I - sims.mean()) / sims.std()}


def local_moran(x: np.ndarray, W: sparse.csr_matrix, permutations: int = 999,
                seed: int = 0) -> pd.DataFrame:
    """LISA with conditional permutation: for each i, neighbours are redrawn
    from the other n-1 observations while x_i is held fixed."""
    rng = np.random.default_rng(seed)
    Wr = row_standardize(W).tocsr()
    x = np.asarray(x, float)
    n = len(x)
    z = (x - x.mean()) / x.std()
    lag = Wr @ z
    Ii = z * lag

    k = int(np.diff(Wr.indptr).max())
    wvals = np.zeros((n, k))
    for i in range(n):
        v = Wr.data[Wr.indptr[i]:Wr.indptr[i + 1]]
        wvals[i, :len(v)] = v
    # Shared random k-subsets of the other n-1 observations (same scheme as esda):
    # index r >= i is shifted by one so observation i is never its own neighbour.
    draws = np.vstack([rng.choice(n - 1, size=k, replace=False) for _ in range(permutations)])
    larger = np.zeros(n)
    for i in range(n):
        idx = draws + (draws >= i)
        sim = z[i] * (z[idx] * wvals[i]).sum(axis=1)
        larger[i] = np.sum(sim >= Ii[i])
    p = (np.minimum(larger, permutations - larger) + 1) / (permutations + 1)
    quad = np.select([(z > 0) & (lag > 0), (z < 0) & (lag < 0), (z > 0) & (lag < 0), (z < 0) & (lag > 0)],
                     ["HH", "LL", "HL", "LH"], default="NS")
    return pd.DataFrame({"lisa_I": Ii, "lisa_p": p, "lisa_quadrant": quad})


def getis_ord_gi_star(x: np.ndarray, W: sparse.csr_matrix) -> pd.DataFrame:
    """Gi* hot-spot statistic (binary weights including self), analytical z and p."""
    x = np.asarray(x, float)
    n = len(x)
    Ws = (W > 0).astype(float) + sparse.identity(n, format="csr")
    xbar, s = x.mean(), np.sqrt((x ** 2).mean() - x.mean() ** 2)
    wsum = np.asarray(Ws.sum(axis=1)).ravel()
    w2sum = np.asarray(Ws.multiply(Ws).sum(axis=1)).ravel()
    num = Ws @ x - xbar * wsum
    den = s * np.sqrt((n * w2sum - wsum ** 2) / (n - 1))
    zg = num / den
    return pd.DataFrame({"gi_z": zg, "gi_p": 2 * (1 - norm.cdf(np.abs(zg)))})


def fdr_bh(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: boolean mask of discoveries at false discovery rate alpha."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    thresh = alpha * np.arange(1, m + 1) / m
    passed = p[order] <= thresh
    keep = np.zeros(m, bool)
    if passed.any():
        keep[order[: np.max(np.nonzero(passed)) + 1]] = True
    return keep


def spatial_clusters(features: pd.DataFrame, W: sparse.csr_matrix, n_clusters: int) -> np.ndarray:
    """Spatially constrained Ward clustering (contiguous groups of similar LGUs).

    A scikit-learn stand-in for SKATER/max-p; swap in ``spopt.region.Skater``
    when geopandas/spopt are installed for a tree-based alternative.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(features.to_numpy(float))
    conn = ((W + W.T) > 0).astype(int)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", connectivity=conn)
    return model.fit_predict(X)


def run_spatial_analysis(df: pd.DataFrame, value_col: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Global + local statistics on ``value_col``; returns per-LGU table and a summary."""
    sc = cfg["spatial"]
    seed = cfg["project"]["seed"]
    W = knn_weights(df["lat"].to_numpy(), df["lon"].to_numpy(), sc["knn_k"])
    x = df[value_col].to_numpy(float)
    glob = morans_i(x, W, sc["permutations"], seed)
    lisa = local_moran(x, W, sc["permutations"], seed)
    gi = getis_ord_gi_star(x, W)
    out = pd.concat([df[["psgc"]].reset_index(drop=True), lisa, gi], axis=1)
    out["lisa_sig"] = fdr_bh(out["lisa_p"], sc["alpha"])
    out["lisa_cluster"] = np.where(out["lisa_sig"], out["lisa_quadrant"], "Not significant")
    out["gi_sig"] = fdr_bh(out["gi_p"], sc["alpha"])
    out["hotspot"] = np.select([out["gi_sig"] & (out["gi_z"] > 0), out["gi_sig"] & (out["gi_z"] < 0)],
                               ["Hot spot", "Cold spot"], default="Not significant")
    out["region_cluster"] = spatial_clusters(df[["HVS", "HCS"]], W, sc["n_regions"])
    return out, glob
