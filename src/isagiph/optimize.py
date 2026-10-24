"""Where should the next facilities go? Need-weighted maximal coverage siting.

This is the maximal covering location problem (Church & ReVelle, 1974): choose N sites for new
hospitals so that as many people as possible who are now more than 30 minutes from any hospital come
within 30 minutes. Each uncovered person is weighted by their area's Health Need Index, so the same
number of people counts more where need is higher.

The problem is NP-hard, but its objective (need-weighted people covered) is submodular, so the greedy
algorithm - repeatedly add the site that covers the most still-uncovered weighted people - is guaranteed
to reach at least 1 - 1/e (about 63%) of the best possible coverage (Nemhauser, Wolsey & Fisher, 1978).
A lazy priority queue makes it fast enough to run interactively for the whole country.

Candidate sites are the barangay centres of uncovered barangays. Current coverage uses the pipeline's
travel times (road times where computed, configured generalized-speed estimates elsewhere); a new site's
reach uses the same configured fallback speed and circuity. The suggestion is therefore conditional on those
transport assumptions and is not a routed door-to-door prediction for a future site. The chosen sites can be sent to the ISAGIP Simulator, which
re-runs the full access model and all three indices to verify the effect.
"""
from __future__ import annotations

import heapq

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .accessibility import to_local_km


def current_minutes(points: pd.DataFrame, tt: pd.DataFrame) -> pd.Series:
    """Minutes from each barangay to its nearest hospital (inf if none within the travel-time table)."""
    return tt.groupby("point_id")["minutes"].min().reindex(points["point_id"]).fillna(np.inf)


def best_sites(points: pd.DataFrame, need: pd.Series, minutes_now: pd.Series, n_sites: int = 5,
               threshold_min: float = 30.0, speed_kmh: float = 15.0, circuity: float = 1.4,
               within: pd.Series | None = None) -> pd.DataFrame:
    """Greedy need-weighted maximal coverage.

    points: point_id, lat, lon, population, psgc. need: Health Need Index (0-100) per psgc.
    minutes_now: current minutes to the nearest hospital, aligned to points. within: optional boolean mask of
    points that may host a new site (e.g. one region). Returns one row per chosen site, in the order chosen.
    """
    pts = points.reset_index(drop=True)
    uncovered = (minutes_now.to_numpy() > threshold_min)
    w = (pts["population"].to_numpy() * (pts["psgc"].map(need).fillna(need.median()).to_numpy() / 100.0)) * uncovered
    pop = pts["population"].to_numpy() * uncovered
    if not uncovered.any():
        return pd.DataFrame(columns=["rank", "lat", "lon", "psgc", "people_covered", "need_weighted"])
    xy = to_local_km(pts["lat"].to_numpy(), pts["lon"].to_numpy())
    radius_km = threshold_min / 60.0 * speed_kmh / circuity
    cand = np.where(uncovered & (within.to_numpy() if within is not None else True))[0]
    if len(cand) == 0:
        return pd.DataFrame(columns=["rank", "lat", "lon", "psgc", "people_covered", "need_weighted"])
    tree = cKDTree(xy[uncovered])
    unc_idx = np.where(uncovered)[0]
    reach = [unc_idx[np.asarray(tree.query_ball_point(xy[c], radius_km), dtype=int)] for c in cand]
    alive = w.copy()
    heap = [(-alive[r].sum(), j) for j, r in enumerate(reach)]           # lazy greedy: stale gains are upper bounds
    heapq.heapify(heap)
    picks = []
    while heap and len(picks) < n_sites:
        neg, j = heapq.heappop(heap)
        gain = alive[reach[j]].sum()
        if gain <= 0:                                                    # nothing left to cover here: drop it
            continue
        if heap and -heap[0][0] > gain + 1e-9:                          # gain went down: re-insert and retry
            heapq.heappush(heap, (-gain, j))
            continue
        r = reach[j]
        c = cand[j]
        picks.append({"rank": len(picks) + 1, "lat": float(pts.at[c, "lat"]), "lon": float(pts.at[c, "lon"]),
                      "psgc": pts.at[c, "psgc"], "people_covered": float(pop[r][alive[r] > 0].sum()),
                      "need_weighted": float(gain)})
        alive[r] = 0.0
    out = pd.DataFrame(picks)
    if len(out):
        out["cumulative_people"] = out["people_covered"].cumsum()
    return out
