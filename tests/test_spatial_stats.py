import numpy as np
import pytest

from isagiph.spatial_stats import fdr_bh, knn_weights, local_moran, morans_i


def grid(n=15):
    lat, lon = np.meshgrid(np.linspace(10, 12, n), np.linspace(122, 124, n))
    return lat.ravel(), lon.ravel()


def test_moran_detects_clustering():
    lat, lon = grid()
    W = knn_weights(lat, lon, 6)
    clustered = lat + 0.01 * np.random.default_rng(0).normal(size=lat.size)
    random = np.random.default_rng(1).normal(size=lat.size)
    assert morans_i(clustered, W, 199)["p_sim"] < 0.05
    assert morans_i(random, W, 199)["p_sim"] > 0.01


def test_fdr_is_stricter_than_raw():
    p = np.array([0.001, 0.01, 0.03, 0.04, 0.2])
    keep = fdr_bh(p, 0.05)
    assert keep[0] and not keep[-1] and keep.sum() <= (p < 0.05).sum()


def test_lisa_matches_esda_if_installed():
    esda = pytest.importorskip("esda")
    libpysal = pytest.importorskip("libpysal")
    lat, lon = grid(10)
    x = lat + np.random.default_rng(2).normal(scale=0.3, size=lat.size)
    ours = local_moran(x, knn_weights(lat, lon, 6), 999)
    w = libpysal.weights.KNN(np.column_stack([lon * 109.0, lat * 110.6]), k=6)
    w.transform = "r"
    ref = esda.Moran_Local(x, w, permutations=999, seed=0)
    assert np.corrcoef(ours.lisa_I, ref.Is)[0, 1] > 0.99
