"""Map geometry helpers for the dashboard.

``hex_heatmap`` turns point data into a hexagon grid that traces the inhabited
land area (hexagons are kept only near population points). Each hexagon gets an
inverse-distance-weighted value from nearby LGUs (or the nearest LGU's class for
categorical layers), so the map reads as a smooth heat surface while every
hexagon stays hoverable. Pure NumPy/pandas so it can be unit-tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .accessibility import KM_PER_DEG_LAT, KM_PER_DEG_LON_EQ, to_local_km


def _to_latlon(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat = y / KM_PER_DEG_LAT
    lon = x / (KM_PER_DEG_LON_EQ * np.cos(np.radians(lat)))
    return lat, lon


def hex_grid(mask_lat, mask_lon, radius_km: float = 8.0, mask_km: float | None = None) -> pd.DataFrame:
    """Pointy-top hexagons (circumradius ``radius_km``) within ``mask_km`` of any mask point."""
    mask_km = mask_km or radius_km * 1.6
    pts = to_local_km(mask_lat, mask_lon)
    (xmin, ymin), (xmax, ymax) = pts.min(0) - mask_km, pts.max(0) + mask_km
    dx, dy = np.sqrt(3) * radius_km, 1.5 * radius_km
    rows = np.arange(int((ymax - ymin) / dy) + 2)
    cols = np.arange(int((xmax - xmin) / dx) + 2)
    cc, rr = np.meshgrid(cols, rows)
    cx = xmin + dx * (cc + 0.5 * (rr % 2))
    cy = ymin + dy * rr
    centers = np.column_stack([cx.ravel(), cy.ravel()])
    dist, _ = cKDTree(pts).query(centers)
    centers = centers[dist <= mask_km]
    ang = np.radians(30 + 60 * np.arange(6))
    vx = centers[:, [0]] + radius_km * np.cos(ang)
    vy = centers[:, [1]] + radius_km * np.sin(ang)
    vlat, vlon = _to_latlon(vx, vy)
    clat, clon = _to_latlon(centers[:, 0], centers[:, 1])
    rings = [[[round(float(lo), 5), round(float(la), 5)] for la, lo in zip(vlat[i], vlon[i])] +
             [[round(float(vlon[i, 0]), 5), round(float(vlat[i, 0]), 5)]] for i in range(len(centers))]
    return pd.DataFrame({"lat": clat, "lon": clon, "ring": rings})


def assign_values(hexes: pd.DataFrame, lgus: pd.DataFrame, col: str,
                  categorical: bool, k: int = 6, power: float = 2.0) -> pd.DataFrame:
    """Value per hexagon: IDW of the k nearest LGUs (numeric) or nearest LGU (categorical)."""
    tree = cKDTree(to_local_km(lgus["lat"], lgus["lon"]))
    hx = to_local_km(hexes["lat"], hexes["lon"])
    k = min(k, len(lgus))
    d, idx = tree.query(hx, k=k)
    d, idx = d.reshape(len(hx), k), idx.reshape(len(hx), k)
    out = hexes.copy()
    out["nearest"] = idx[:, 0]
    if categorical:
        out["value"] = lgus[col].to_numpy()[idx[:, 0]]
    else:
        vals = lgus[col].to_numpy(float)[idx]
        w = 1.0 / np.maximum(d, 0.5) ** power
        w = np.where(np.isnan(vals), 0, w)
        out["value"] = np.nansum(w * np.nan_to_num(vals), axis=1) / np.maximum(w.sum(axis=1), 1e-12)
    return out


def to_feature_collection(rings: list, props: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [r]}, "properties": p}
        for r, p in zip(rings, props)]}
