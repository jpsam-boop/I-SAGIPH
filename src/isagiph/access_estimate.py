"""Transparent summaries of modeled hospital travel-time reach.

The result estimates the population outside a modeled hospital travel-time threshold
using the facility inventory and the project's configured transportation assumptions.
A supplied DOH reference figure is treated as an external comparison after estimation.
"""
from __future__ import annotations


import pandas as pd


def summarize_hospital_access(results: pd.DataFrame, cfg: dict, road_network_population_share: float) -> dict:
    """Return modeled 30-minute hospital-access counts and the associated reference-comparison context."""
    required = {"population", "cov_any_30"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Cannot summarize hospital access; missing columns: {sorted(missing)}")
    population = pd.to_numeric(results["population"], errors="coerce").fillna(0).clip(lower=0)
    share = pd.to_numeric(results["cov_any_30"], errors="coerce").fillna(0).clip(lower=0, upper=1)
    total = float(population.sum())
    within = float((population * share).sum())
    outside = max(0.0, total - within)
    acc = cfg["accessibility"]
    roads = cfg.get("roads", {})
    return {
        "estimate_type": "model-derived population-weighted hospital access estimate",
        "facility_scope": "licensed hospitals and included infirmaries in the analysis facility table",
        "threshold_minutes": 30,
        "population_total": int(round(total)),
        "population_within_threshold": int(round(within)),
        "population_outside_threshold": int(round(outside)),
        "share_within_threshold": within / total if total > 0 else None,
        "share_outside_threshold": outside / total if total > 0 else None,
        "road_network_population_share": float(road_network_population_share),
        "transport_assumptions": {
            "fallback_effective_speed_kmh": float(acc["fallback_speed_kmh"]),
            "straight_line_circuity": float(acc["circuity"]),
            "road_access_egress_speed_kmh": float(roads.get("access_speed_kmh", 15.0)),
            "road_class_speeds_kmh": roads.get("speeds_kmh") or "src/isagiph/roadtimes.py DEFAULT_SPEEDS",
        },
        "interpretation": (
            "This is an estimate under the configured transport assumptions, population allocation and licensed hospital/infirmary inventory. "
            "It estimates the number of residents modeled as outside the 30-minute hospital threshold; it is not a count of patients who failed to obtain care. "
            "Where the supplied DOH reference figure is available, compare it with this estimate as an external check after the travel-time calculation."
        ),
        "reference_comparison": {
            "role": "external comparison after estimation",
            "description": "Compare the modeled population beyond 30 minutes from a hospital with the supplied DOH reference figure, where definitions and geographic scope align.",
        },
    }
