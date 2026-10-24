"""Standard table schemas shared by the real-data cleaner and the demo generator.

``validate`` fails loudly on missing columns, duplicate keys, or impossible
values, so bad inputs are caught before they silently distort the indices.
"""
from __future__ import annotations

import pandas as pd

SCHEMAS = {
    "lgus": {
        "key": "psgc",
        "required": ["psgc", "name", "province", "region", "lat", "lon", "population",
                     "pop_0_4", "pop_60_plus", "poverty_incidence", "gida_share", "n_primary_care"],
        # optional: cases_tb, cases_dengue (disease domain), n_ambulance_units, outcome columns
        "nonneg": ["population", "pop_0_4", "pop_60_plus", "n_primary_care"],
        "unit": ["poverty_incidence", "gida_share"],
    },
    "points": {"key": "point_id", "required": ["point_id", "psgc", "lat", "lon", "population"],
               "nonneg": ["population"], "unit": []},
    "facilities": {"key": "facility_id",
                   "required": ["facility_id", "name", "psgc", "lat", "lon", "level", "beds", "malasakit"],
                   "nonneg": ["beds"], "unit": []},
}

PH_BBOX = {"lat": (4.0, 21.5), "lon": (114.0, 127.5)}   # west edge includes Kalayaan


def validate(df: pd.DataFrame, table: str) -> pd.DataFrame:
    s = SCHEMAS[table]
    missing = [c for c in s["required"] if c not in df]
    if missing:
        raise ValueError(f"{table}: missing columns {missing}")
    if df[s["key"]].duplicated().any():
        raise ValueError(f"{table}: duplicate {s['key']} values")
    for c in s["nonneg"]:
        if (df[c] < 0).any():
            raise ValueError(f"{table}: negative values in {c}")
    for c in s["unit"]:
        if not df[c].between(0, 1).all():
            raise ValueError(f"{table}: {c} must be a proportion in [0, 1]")
    for c, (lo, hi) in PH_BBOX.items():
        bad = ~df[c].between(lo, hi)
        if bad.any():
            raise ValueError(f"{table}: {bad.sum()} rows with {c} outside the Philippines")
    if table == "facilities" and not df["level"].isin([1, 2, 3]).all():
        raise ValueError("facilities: level must be 1, 2 or 3")
    return df
