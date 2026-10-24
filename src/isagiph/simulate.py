"""What-if simulator: add hypothetical facilities and recompute access, indices and the gap."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .accessibility import fallback_travel_times, lgu_accessibility, to_local_km
from .indices import build_indices

ACCESS_COLS = ["accessible_beds", "beds_per_1000_all", "beds_per_1000_l2plus", "cov_any_30", "cov_l2plus_60",
               "cov_malasakit_60", "mean_minutes_any", "pct_no_hospital_in_range"]
PRESETS = {  # label: (level, beds, malasakit, ownership, is_primary_care)
    "Public hospital · Level 1 (50 beds)": (1, 50, 0, "Government", False),
    "Public hospital · Level 2 (150 beds)": (2, 150, 0, "Government", False),
    "Big public hospital · Level 3 (500 beds)": (3, 500, 0, "Government", False),
    "Public hospital with Malasakit Center · Level 2 (150 beds)": (2, 150, 1, "Government", False),
    "Private hospital · Level 2 (100 beds)": (2, 100, 0, "Private", False),
    "Infirmary (10 beds)": (1, 10, 0, "Government", False),
    "Rural Health Unit / primary care facility": (0, 0, 0, "Government", True),
}


def base_frame(res: pd.DataFrame) -> pd.DataFrame:
    """LGU features without access and index outputs (the inputs the simulator recomputes)."""
    keep = [c for c in res.columns if c in (
        "psgc", "name", "province", "region", "lat", "lon", "population", "pop_0_4", "pop_60_plus",
        "poverty_incidence", "gida_share", "n_primary_care", "pct_age_0_4", "pct_age_60_plus",
        "eb_rate_tb", "eb_rate_dengue", "primary_care_per_10k", "pop_density")]
    return res[keep].copy()


def run(res: pd.DataFrame, points: pd.DataFrame, facilities: pd.DataFrame, tt_base: pd.DataFrame,
        added: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return the LGU table recomputed with ``added`` facilities (columns lat, lon, level, beds, malasakit, primary)."""
    base = base_frame(res)
    fac = facilities.copy()
    tt = tt_base
    if len(added):
        hosp = added[~added["primary"]].copy()
        if len(hosp):
            hosp["facility_id"] = [f"SIM{i}" for i in range(len(hosp))]
            hosp["name"] = "Simulated facility"
            hosp["level"] = hosp["level"].clip(lower=1).astype(int)
            acc = cfg["accessibility"]
            max_min = acc["d0_minutes"]["level2plus"]
            tt_new = fallback_travel_times(points, hosp, max_min, acc["fallback_speed_kmh"], acc["circuity"])
            fac = pd.concat([fac, hosp[fac.columns.intersection(hosp.columns)]], ignore_index=True)
            tt = pd.concat([tt_base, tt_new], ignore_index=True)
        prim = added[added["primary"]]
        if len(prim):   # each new RHU adds one primary care facility to the nearest LGU
            from scipy.spatial import cKDTree
            _, idx = cKDTree(to_local_km(base["lat"], base["lon"])).query(to_local_km(prim["lat"], prim["lon"]))
            add = pd.Series(1, index=base.index[idx]).groupby(level=0).sum()
            base.loc[add.index, "n_primary_care"] += add
            base["primary_care_per_10k"] = base["n_primary_care"] / base["population"] * 1e4
    access = lgu_accessibility(points, fac, tt, cfg)
    df = base.merge(access, on="psgc", how="left")
    idx = build_indices(df, cfg)
    return df.merge(idx, on="psgc")


def compare(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    b = before.set_index("psgc")
    a = after.set_index("psgc")
    out = pd.DataFrame({"name": b["name"], "province": b["province"], "population": b["population"],
                        "HGI_before": b["gap_index"], "HGI_after": a["gap_index"],
                        "priority_before": b["priority"], "priority_after": a["priority"],
                        "cov30_before": b["cov_any_30"], "cov30_after": a["cov_any_30"],
                        "beds_before": b["beds_per_1000_all"], "beds_after": a["beds_per_1000_all"],
                        "shortfall_before": b["bed_gap_shortfall"], "shortfall_after": a["bed_gap_shortfall"]})
    out["dHGI"] = out["HGI_after"] - out["HGI_before"]
    return out.reset_index()
