import numpy as np
import pandas as pd

from isagiph.maplayers import assign_values, hex_grid


def test_hex_grid_follows_points_and_closes_rings():
    lat = np.array([14.6, 14.7, 10.3])
    lon = np.array([121.0, 121.1, 123.9])
    h = hex_grid(lat, lon, radius_km=5)
    assert len(h) > 0
    assert all(r[0] == r[-1] and len(r) == 7 for r in h.ring)
    # no hexagon far from any input point (e.g. in the sea between them)
    assert ((h.lat - 12.5).abs() > 1.0).all()


def test_idw_reproduces_value_at_lgu():
    lgus = pd.DataFrame({"lat": [14.0, 15.0], "lon": [121.0, 121.0], "v": [10.0, 90.0], "c": ["a", "b"]})
    hexes = pd.DataFrame({"lat": [14.0, 15.0], "lon": [121.0, 121.0]})
    num = assign_values(hexes, lgus, "v", categorical=False, k=2)
    cat = assign_values(hexes, lgus, "c", categorical=True, k=2)
    assert num.value[0] < 20 and num.value[1] > 80
    assert list(cat.value) == ["a", "b"]
