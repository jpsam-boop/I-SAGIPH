"""End-to-end pipeline: standard tables -> accessibility -> indices -> spatial
statistics -> sensitivity -> validation -> outputs and report.

Usage:
    python -m isagiph.pipeline --demo            # synthetic data, runs anywhere
    python -m isagiph.pipeline                   # real data from data/interim/
    python -m isagiph.pipeline --demo --runs 100 # faster sensitivity analysis
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .accessibility import fallback_travel_times, lgu_accessibility
from .access_estimate import summarize_hospital_access
from .config import DEFAULT_CONFIG, load_config
from .indices import build_indices
from .report import write_report
from .schema import validate
from .sensitivity import run_sensitivity
from .smoothing import eb_rates_grouped
from .spatial_stats import run_spatial_analysis
from .synthetic import write_demo
from .validation import validate as validate_outcomes


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def _sha256(path: Path) -> str:
    """Stream a file hash so provenance tracking does not load large spatial files into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(config_path: Path, source_dir: Path, seed: int, manual_dir: Path | None = None) -> dict:
    """Record input hashes, including transport-source files, for reproducible audits."""
    files: dict[str, dict] = {}
    candidates = [source_dir / f"{name}.csv" for name in ("lgus", "points", "facilities")]
    candidates += [source_dir / name for name in ("travel_times.csv", "imputation.json")]
    # Road matrices and the raw source GeoPackages matter because route coverage changes access estimates.
    candidates += sorted(source_dir.glob("road_times_*.csv"))
    candidates += sorted(source_dir.glob("road_points_*.csv"))
    if manual_dir is not None and manual_dir.exists():
        candidates += sorted(manual_dir.glob("*.gpkg"))
    for path in [config_path, *candidates]:
        try:
            if path.is_file():
                key = str(path.resolve())
                files[key] = {"sha256": _sha256(path), "bytes": int(path.stat().st_size)}
        except OSError as exc:
            files[str(path)] = {"hash_error": f"{type(exc).__name__}: {exc}"}

    package_versions = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "streamlit", "geopandas", "libpysal", "esda"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": package_versions,
        "sha256_files": files,
        "note": "Input hashes are provenance aids; public source data may be revised independently of this code release.",
    }


def load_tables(cfg: dict, demo: bool) -> dict[str, pd.DataFrame]:
    src = cfg["paths"]["demo" if demo else "interim"]
    if demo and not (src / "lgus.csv").exists():
        write_demo(src, seed=cfg["project"]["seed"])
    t = {n: pd.read_csv(src / f"{n}.csv", dtype={"psgc": str}) for n in ("lgus", "points", "facilities")}
    for n, df in t.items():
        validate(df, n)
    return t


def lgu_features(lgus: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Indicators that do not depend on travel time."""
    df = lgus.copy()
    df["pct_age_0_4"] = df["pop_0_4"] / df["population"]
    df["pct_age_60_plus"] = df["pop_60_plus"] / df["population"]
    grp = cfg["disease"]["smoothing_group"]
    for d in ("tb", "dengue"):
        if f"cases_{d}" in df:
            df[f"raw_rate_{d}"] = df[f"cases_{d}"] / df["population"] * cfg["disease"]["per"]
            df[f"eb_rate_{d}"] = eb_rates_grouped(df, f"cases_{d}", "population", grp, cfg["disease"]["per"])
    df["primary_care_per_10k"] = df["n_primary_care"] / df["population"] * 1e4
    if "n_ambulance_units" in df:
        df["ambulance_per_100k"] = df["n_ambulance_units"] / df["population"] * 1e5
    return df


def travel_times(t: dict, cfg: dict, demo: bool) -> tuple[pd.DataFrame, str]:
    """Assemble route times using the configured transport assumptions and available route matrices."""
    interim = cfg["paths"]["interim"]
    acc = cfg["accessibility"]
    speed = float(acc["fallback_speed_kmh"])
    circuity = float(acc["circuity"])
    max_min = acc["d0_minutes"]["level2plus"] * max(cfg["sensitivity"]["d0_multipliers"])
    fallback = fallback_travel_times(t["points"], t["facilities"], max_min, speed, circuity)
    fallback["source"] = "straight"
    fallback_label = (f"straight-line fallback (effective-speed assumption {speed:g} km/h, "
                      f"circuity {circuity:g}; configured transport assumptions for geographic context)")

    if demo:
        return fallback, fallback_label

    # Optional full sparse matrix, if the user already prepared one.
    path = interim / "travel_times.csv"
    if path.exists():
        tt0 = pd.read_csv(path, dtype={"point_id": t["points"]["point_id"].dtype})
        if not {"point_id", "facility_id", "minutes"}.issubset(tt0.columns):
            raise ValueError(f"{path} must contain point_id, facility_id, minutes")
        tt0 = tt0[tt0["facility_id"].astype(str).isin(set(t["facilities"]["facility_id"].astype(str)))].copy()
        tt0["facility_id"] = tt0["facility_id"].astype(t["facilities"]["facility_id"].dtype)
        tt0["source"] = "road"
        return tt0, "precomputed travel-time matrix (used as supplied under its documented transport assumptions)"

    road_files = sorted(interim.glob("road_times_*.csv"))
    if not road_files:
        print(f"  NOTE: no road travel times found in {interim} (road_times_*.csv); every demand point uses the "
              "configured generalized straight-line fallback. Run `python -m isagiph.roadtimes --all` to compute road routes.", flush=True)
        return fallback, fallback_label

    pid_type = t["points"]["point_id"].dtype
    road = pd.concat([pd.read_csv(file, dtype={"point_id": pid_type}) for file in road_files], ignore_index=True)
    road = road[road["facility_id"].astype(str).isin(set(t["facilities"]["facility_id"].astype(str)))].copy()
    road["facility_id"] = road["facility_id"].astype(t["facilities"]["facility_id"].dtype)
    # Once a point belongs to a routed region, do not give it optimistic straight-line substitutes
    # just because the route search found no hospital within its cut-off.
    covered = set(road["point_id"])
    for file in sorted(interim.glob("road_points_*.csv")):
        covered |= set(pd.read_csv(file, dtype={"point_id": pid_type})["point_id"])
    covered &= set(t["points"]["point_id"])
    road = road[road["minutes"] <= max_min]
    tt = pd.concat([fallback[~fallback["point_id"].isin(covered)], road.assign(source="road")], ignore_index=True)
    tt.attrs["road_points"] = covered
    n_pts = len(t["points"])
    missing = [file.stem.replace("road_times_", "") for file in road_files
               if not (interim / file.name.replace("road_times_", "road_points_")).exists()]
    if missing:
        print(f"  NOTE: {len(missing)} region(s) have road_times files without matching road_points files;"
              " demand points without route pairs may use the configured fallback. Re-run `python -m isagiph.roadtimes --all --redo`.", flush=True)
    src = (f"road network for {len(covered):,} of {n_pts:,} barangays ({100 * len(covered) / n_pts:.0f}%, "
           f"{len(road_files)} regions); {fallback_label} elsewhere")
    return tt, src


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="use synthetic demo data")
    ap.add_argument("--runs", type=int, default=None, help="override sensitivity runs")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    cfg = load_config(a.config)
    np.random.seed(cfg["project"]["seed"])

    t = load_tables(cfg, a.demo)
    log(f"loaded {len(t['lgus']):,} LGUs, {len(t['points']):,} demand points, "
        f"{len(t['facilities']):,} hospitals", t0)
    base = lgu_features(t["lgus"], cfg)
    tt, tt_src = travel_times(t, cfg, a.demo)
    log(f"travel-time pairs: {len(tt):,} via {tt_src}", t0)

    access = {m: lgu_accessibility(t["points"], t["facilities"], tt, cfg, m)
              for m in cfg["sensitivity"]["d0_multipliers"]}
    log("E2SFCA accessibility computed for each catchment multiplier", t0)

    road_pts = tt.attrs.get("road_points") or (set(tt.loc[tt["source"].eq("road"), "point_id"]) if "source" in tt else set())
    pts_ = t["points"].assign(_road=t["points"]["point_id"].isin(road_pts) * t["points"]["population"])
    road_share = (pts_.groupby("psgc")["_road"].sum() / pts_.groupby("psgc")["population"].sum()).rename("road_share")
    base = base.merge(road_share.reset_index(), on="psgc", how="left").fillna({"road_share": 0.0})
    df = base.merge(access[1.0], on="psgc", how="left")
    idx = build_indices(df, cfg)
    weights = idx.attrs.get("weights")
    res = df.merge(idx, on="psgc")
    log("HVS, HCS, gap index and bed gap computed", t0)

    spat, glob = run_spatial_analysis(res, "gap_index", cfg)
    res = res.merge(spat, on="psgc")
    log(f"spatial statistics: Moran's I = {glob['I']:.3f} (p = {glob['p_sim']:.3f})", t0)

    sens = run_sensitivity(base, access, cfg, a.runs)
    res = res.merge(sens, on="psgc")
    log(f"sensitivity analysis ({a.runs or cfg['sensitivity']['n_runs']} runs) done", t0)

    valid = validate_outcomes(res, cfg)

    out = cfg["paths"]["processed"]
    res.to_csv(out / "lgu_results.csv", index=False)
    t["facilities"].to_csv(out / "facilities.csv", index=False)
    t["points"].to_csv(out / "points.csv", index=False)      # used to draw the hex heatmap
    src_dir = cfg["paths"]["demo" if a.demo else "interim"]
    import shutil
    for name in ("lgu_boundaries.geojson", "lgu_boundaries_lite.geojson", "ph_mask.geojson",   # map outlines
                 "imputation.json"):                                                           # model reports
        if (src_dir / name).exists():
            shutil.copy(src_dir / name, out / name)
    if (src_dir / "directory.csv").exists():
        import shutil
        shutil.copy(src_dir / "directory.csv", out / "directory.csv")   # facility finder
    from .ml import run_ml
    ml_metrics = run_ml(res, out, cfg["project"]["seed"], cfg.get("ml", {}))
    nested_ml = ml_metrics.get("nested_selection", {})
    if nested_ml.get("available"):
        log(f"machine learning: nested spatial-CV R² = {nested_ml.get('r2_log', float('nan')):.2f}; "
            f"MAE = {nested_ml.get('mae_beds', float('nan')):.3f} beds per 1,000", t0)
    else:
        log(f"machine learning: nested selection unavailable; top pooled outer-CV candidate = "
            f"{ml_metrics.get('best', 'unknown')} (exploratory R² = "
            f"{ml_metrics.get(ml_metrics.get('best', ''), {}).get('r2', float('nan')):.2f})", t0)
    valid.to_csv(out / "validation.csv", index=False)
    if weights is not None:
        weights.to_csv(out / "index_weights.csv", index=False)
    config_path = Path(a.config).expanduser().resolve() if a.config else DEFAULT_CONFIG.resolve()
    road_population_share = (float(pts_["_road"].sum() / pts_["population"].sum())
                             if float(pts_["population"].sum()) > 0 else 0.0)
    access_estimate = summarize_hospital_access(res, cfg, road_population_share)
    (out / "access_estimate.json").write_text(json.dumps(access_estimate, indent=2, default=float))
    meta = {"mode": "demo" if a.demo else "real", "travel_time_source": tt_src,
            "road_network_population_share": road_population_share,
            "hospital_access_estimate": access_estimate,
            "weighting": cfg.get("weighting", "equal"),
            "n_runs": a.runs or cfg["sensitivity"]["n_runs"], "global_moran": glob,
            "permutations": cfg["spatial"]["permutations"], "knn_k": cfg["spatial"]["knn_k"],
            "beds_per_1000": cfg["benchmarks"]["beds_per_1000"],
            "goal_2040": cfg["benchmarks"].get("goal_2040_per_1000", 4.0),
            "instability_threshold": cfg["sensitivity"]["instability_threshold"]}
    meta["provenance"] = _provenance(config_path, src_dir, cfg["project"]["seed"], cfg["paths"]["manual"])
    meta["ml_validation_summary"] = {
        "target": ml_metrics.get("target"),
        "target_definition": ml_metrics.get("target_definition"),
        "cv": ml_metrics.get("cv"),
        "best_single_model": ml_metrics.get("best"),
        "nested_selection": ml_metrics.get("nested_selection"),
        "baseline_comparison": ml_metrics.get("baseline_comparison"),
        "prediction_band": ml_metrics.get("prediction_band"),
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=float, allow_nan=True))
    write_report(res, glob, valid, meta, cfg["paths"]["reports"] / "findings.md")
    log(f"outputs written to {out} and reports/findings.md", t0)


if __name__ == "__main__":
    main()
