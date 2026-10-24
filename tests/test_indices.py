import numpy as np
import pandas as pd

from isagiph.config import load_config
from isagiph.indices import composite, gap_index, normalize
from isagiph.smoothing import eb_rates


def test_eb_shrinks_small_populations_more():
    cases = pd.Series([1, 100, 50])
    pop = pd.Series([100, 10000, 10000])
    raw = cases / pop
    eb = eb_rates(cases, pop)
    prior = cases.sum() / pop.sum()
    assert abs(eb[0] - prior) < abs(raw[0] - prior)


def test_normalize_range():
    s = pd.Series([1.0, 5.0, 9.0])
    for m in ("minmax", "zscore"):
        n = normalize(s, m)
        assert n.between(0, 100).all() and n.is_monotonic_increasing


def test_composite_direction_and_geometric_mean():
    cfg = load_config()
    ncfg = {**cfg["normalization"], "log_vars": [], "winsor": [0, 1]}
    df = pd.DataFrame({"a": [0.1, 0.5, 0.9], "b": [0.1, 0.5, 0.9]})
    score, dom = composite(df, {"d1": ["a"], "d2": ["b"]}, ncfg)
    assert score.is_monotonic_increasing
    assert np.allclose(score, dom.clip(lower=ncfg["floor"]).prod(axis=1) ** 0.5)


def test_gap_index_bounds_and_priority():
    hvs = pd.Series(np.arange(30, dtype=float))
    hcs = pd.Series(np.arange(30, dtype=float)[::-1])
    g = gap_index(hvs, hcs)
    assert g.gap_index.between(-100, 100).all()
    assert g.loc[29, "priority"] and not g.loc[0, "priority"]


def test_place_name_keys_and_regions():
    from isagiph.build_real import key, region_codes
    assert key("CITY OF CALOOCAN") == key("Caloocan City")
    assert key("Sta. Cruz") == key("SANTA CRUZ")
    assert key("CEBU CITY (Capital)") == key("Cebu City")
    assert region_codes("REGION 4-A") == ["PH04"] and region_codes("Reg IV-B") == ["PH17"]
    assert region_codes("NIR") == ["PH06", "PH07"]


def test_pca_weights_sum_and_structure():
    from isagiph.indices import pca_weights
    rng = np.random.default_rng(0)
    f1, f2 = rng.normal(size=500), rng.normal(size=500)
    X = pd.DataFrame({"a": f1 + 0.1 * rng.normal(size=500), "b": f1 + 0.1 * rng.normal(size=500),
                      "c": f2 + 0.1 * rng.normal(size=500), "d": f2 + 0.1 * rng.normal(size=500)})
    w = pca_weights(X)
    assert w.factor.nunique() == 2                               # two latent factors recovered
    assert w.groupby("factor").weight_in_factor.sum().round(6).eq(1).all()
    assert np.isclose(w.drop_duplicates("factor").factor_weight.sum(), 1)
    assert w.set_index("indicator").factor["a"] == w.set_index("indicator").factor["b"]
