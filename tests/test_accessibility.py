import numpy as np
import pandas as pd

from isagiph.accessibility import e2sfca, fallback_travel_times, gaussian_decay
from isagiph.synthetic import make_demo


def test_gaussian_decay_bounds():
    w = gaussian_decay(np.array([0, 10, 30, 31]), np.array([30, 30, 30, 30]))
    assert w[0] == 1.0 and w[2] == 0.0 and w[3] == 0.0
    assert 0 < w[1] < 1


def test_e2sfca_hand_example():
    pts = pd.DataFrame({"point_id": ["a", "b"], "population": [1000, 1000]})
    fac = pd.DataFrame({"facility_id": ["h"], "beds": [100], "d0": [30]})
    tt = pd.DataFrame({"point_id": ["a", "b"], "facility_id": ["h", "h"], "minutes": [0, 0]})
    A = e2sfca(pts, fac, tt)
    assert np.allclose(A.to_numpy(), [0.05, 0.05])


def test_e2sfca_conserves_supply():
    """Sum of A_i * P_i equals total beds of facilities that serve anyone."""
    d = make_demo(n_lgus=120, seed=1)
    pts, fac = d["points"], d["facilities"].assign(d0=60.0)
    tt = fallback_travel_times(pts, fac, max_minutes=60)
    A = e2sfca(pts, fac, tt)
    served = tt.merge(fac, on="facility_id")
    served = served[gaussian_decay(served.minutes.to_numpy(), served.d0.to_numpy()) > 0]
    used = fac[fac.facility_id.isin(served.facility_id)].beds.sum()
    assert np.isclose((A.to_numpy() * pts.population.to_numpy()).sum(), used, rtol=1e-6)


def test_fallback_includes_colocated_pairs():
    pts = pd.DataFrame({"point_id": ["p"], "lat": [14.6], "lon": [121.0]})
    fac = pd.DataFrame({"facility_id": ["f"], "lat": [14.6], "lon": [121.0]})
    tt = fallback_travel_times(pts, fac, 30)
    assert len(tt) == 1 and tt.minutes.iloc[0] == 0
