"""Synthetic demo dataset in the project's standard schema.

PURPOSE: test the full pipeline end-to-end and record the demo before the real
data (e.g. the DOH eFOI bed-capacity request) arrive. Nothing generated here is
real; outputs built from it must never be presented as findings.
The schema matches what ``clean.py`` produces from real sources.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Approximate regional centres (lat, lon) used only to scatter synthetic LGUs.
REGIONS = {
    "NCR": (14.60, 121.00), "CAR": (17.10, 120.90), "I": (16.60, 120.40), "II": (17.20, 121.80),
    "III": (15.30, 120.70), "IV-A": (14.10, 121.30), "MIMAROPA": (11.80, 120.20), "V": (13.40, 123.50),
    "VI": (10.90, 122.50), "VII": (10.10, 123.80), "VIII": (11.50, 125.00), "IX": (7.80, 122.70),
    "X": (8.20, 124.60), "XI": (7.20, 125.60), "XII": (6.50, 124.80), "XIII": (8.90, 125.70),
    "BARMM": (7.20, 124.20),
}


def make_demo(n_lgus: int = 600, pts_per_lgu: int = 4, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    names = list(REGIONS)
    region = rng.choice(names, n_lgus)
    centre = np.array([REGIONS[r] for r in region])
    spread = np.where(region == "NCR", 0.08, 0.7)
    lat = centre[:, 0] + rng.normal(0, spread)
    lon = centre[:, 1] + rng.normal(0, spread)
    remoteness = np.hypot(lat - centre[:, 0], lon - centre[:, 1]) / 0.7     # ~0 core, >1 periphery
    urban = (region == "NCR") | (remoteness < 0.25)
    pop = np.round(np.exp(rng.normal(np.where(urban, 12.0, 10.4), 0.6))).astype(int)

    psgc = [f"{i:010d}" for i in range(1, n_lgus + 1)]
    province = [f"{r}-P{int(abs(a * 3) % 4) + 1}" for r, a in zip(region, lat)]
    poverty = np.clip(0.08 + 0.12 * remoteness + rng.normal(0, 0.05, n_lgus)
                      + np.where(np.isin(region, ["BARMM", "IX", "VIII"]), 0.12, 0), 0.01, 0.8)
    lgus = pd.DataFrame({
        "psgc": psgc, "name": [f"Demo LGU {i}" for i in range(1, n_lgus + 1)],
        "province": province, "region": region, "lat": lat, "lon": lon,
        "population": pop,
        "pop_0_4": np.round(pop * np.clip(rng.normal(0.10 + 0.04 * poverty, 0.01), 0.05, 0.18)).astype(int),
        "pop_60_plus": np.round(pop * np.clip(rng.normal(0.09, 0.02), 0.03, 0.2)).astype(int),
        "poverty_incidence": poverty,
        "gida_share": np.clip((remoteness - 0.8) * 0.6 + rng.normal(0, 0.05, n_lgus), 0, 1),
        "cases_tb": rng.poisson(pop * (0.0045 + 0.004 * poverty)),
        "cases_dengue": rng.poisson(pop * np.exp(rng.normal(np.log(0.002), 0.6))),
        "n_primary_care": rng.poisson(np.maximum(pop / np.where(urban, 30000, 18000), 0.3)),
        "n_ambulance_units": rng.poisson(np.maximum(pop / 60000, 0.2)),
    })

    # Demand points (stand-ins for barangay centroids), population split by Dirichlet shares.
    rows = []
    for _, r in lgus.iterrows():
        shares = rng.dirichlet(np.ones(pts_per_lgu))
        for k in range(pts_per_lgu):
            rows.append((f"{r.psgc}-{k}", r.psgc, r.lat + rng.normal(0, 0.04),
                         r.lon + rng.normal(0, 0.04), int(r.population * shares[k])))
    points = pd.DataFrame(rows, columns=["point_id", "psgc", "lat", "lon", "population"])

    # Hospitals: likelier and bigger in populous, central LGUs.
    p_hosp = np.clip(0.35 + 0.4 * urban + 0.2 * (pop > 60000) - 0.2 * remoteness, 0.05, 0.95)
    fac = []
    for i, r in lgus.iterrows():
        for _ in range(rng.binomial(4 if urban[i] else 2, p_hosp[i])):
            level = int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1] if urban[i] else [0.85, 0.13, 0.02]))
            beds = int(rng.lognormal({1: 3.2, 2: 4.4, 3: 5.6}[level], 0.4))
            fac.append((f"F{len(fac) + 1:05d}", f"Demo Hospital {len(fac) + 1}", r.psgc,
                        r.lat + rng.normal(0, 0.02), r.lon + rng.normal(0, 0.02), level, beds,
                        int(level >= 2 and rng.random() < 0.7), int(rng.random() < 0.85)))
    facilities = pd.DataFrame(fac, columns=["facility_id", "name", "psgc", "lat", "lon",
                                            "level", "beds", "malasakit", "philhealth"])

    # Outcome NOT used in the indices: facility-based delivery rate, driven by
    # true distance to care and poverty plus noise (to test external validity).
    from scipy.spatial import cKDTree
    from .accessibility import to_local_km
    d_km, _ = cKDTree(to_local_km(facilities.lat, facilities.lon)).query(to_local_km(lgus.lat, lgus.lon))
    logit = 2.2 - 0.04 * d_km - 3.0 * poverty + rng.normal(0, 0.4, n_lgus)
    lgus["facility_delivery_rate"] = 1 / (1 + np.exp(-logit))
    return {"lgus": lgus, "points": points, "facilities": facilities}


def write_demo(out_dir, **kw) -> None:
    for name, df in make_demo(**kw).items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
