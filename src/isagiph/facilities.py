"""Facility de-duplication and enrichment across NHFR, OSM, and the eFOI bed list."""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .accessibility import to_local_km

STOP = r"\b(HOSPITAL|MEDICAL|CENTER|CENTRE|INC|CORP|OF|THE|AND|DISTRICT|PROVINCIAL)\b"


def norm_name(s: str) -> str:
    """Uppercase, strip accents/punctuation/generic words for fuzzy matching."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(STOP, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def similarity(a: str, b: str) -> float:
    try:
        from rapidfuzz import fuzz
        return fuzz.token_set_ratio(a, b) / 100.0
    except ImportError:
        return SequenceMatcher(None, a, b).ratio()


def match_by_location_and_name(left: pd.DataFrame, right: pd.DataFrame,
                               max_km: float = 0.5, min_sim: float = 0.85) -> pd.DataFrame:
    """For each row in ``left``, the best ``right`` row within max_km and min name similarity."""
    tree = cKDTree(to_local_km(right["lat"], right["lon"]))
    cands = tree.query_ball_point(to_local_km(left["lat"], left["lon"]), r=max_km)
    ln = left["name"].map(norm_name).to_numpy()
    rn = right["name"].map(norm_name).to_numpy()
    out = []
    for i, js in enumerate(cands):
        best = max(((similarity(ln[i], rn[j]), j) for j in js), default=(0.0, None))
        out.append((i, best[1] if best[0] >= min_sim else None, best[0]))
    m = pd.DataFrame(out, columns=["left_idx", "right_idx", "score"])
    return m


def attach_beds(hospitals: pd.DataFrame, beds: pd.DataFrame, min_sim: float = 0.9) -> pd.DataFrame:
    """Join eFOI licensed beds/level to registry hospitals by municipality + fuzzy name.

    Unmatched hospitals keep NaN beds and are listed in the coverage report rather
    than silently imputed.
    """
    h = hospitals.copy()
    h["_n"] = h["name"].map(norm_name)
    b = beds.copy()
    b["_n"] = b["name"].map(norm_name)
    level, nbeds, score = [], [], []
    for _, r in h.iterrows():
        pool = b[b["municipality"].str.upper() == str(r.get("municipality", "")).upper()]
        pool = pool if len(pool) else b
        sims = pool["_n"].map(lambda x: similarity(r["_n"], x))
        if len(sims) and sims.max() >= min_sim:
            best = pool.loc[sims.idxmax()]
            level.append(best["level"]); nbeds.append(best["beds"]); score.append(sims.max())
        else:
            level.append(np.nan); nbeds.append(np.nan); score.append(sims.max() if len(sims) else 0)
    h["level"], h["beds"], h["bed_match_score"] = level, nbeds, score
    return h.drop(columns="_n")
