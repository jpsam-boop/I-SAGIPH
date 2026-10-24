"""Reusable calculations for interactive I-SAGIPH analytical workspaces.

These helpers operate on the project's existing outputs and keep statistical
calculations separate from Streamlit presentation for easier testing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t


def regression_metrics(reference, prediction) -> dict:
    """Calculate regression metrics in the original target units."""
    y = pd.to_numeric(pd.Series(reference), errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(pd.Series(prediction), errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y) & np.isfinite(p)
    y, p = y[keep], p[keep]
    if not len(y):
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "r2": np.nan}
    err = p - y
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum(err ** 2) / denom) if len(y) > 1 and denom > 0 else np.nan
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))), "bias": float(np.mean(err)), "r2": r2}


def relationship_statistics(df: pd.DataFrame, x_col: str, y_col: str,
                             region_col: str = "region", n_boot: int = 300,
                             seed: int = 42, weights_col: str | None = None) -> dict:
    """Summarise an exploratory relationship with region-block bootstrap CI for Spearman rho.

    Weighted least-squares slope is optionally estimated with population weights;
    the Spearman correlation and its interval remain unweighted rank statistics.
    """
    cols = [x_col, y_col] + ([region_col] if region_col in df else [])
    # Preserve row identifiers and population for linked chart tooltips. Deduplicate
    # so a population-weighted fit does not select the same column twice.
    for col in ("psgc", "name", "province", "population"):
        if col in df and col not in cols:
            cols.append(col)
    if weights_col and weights_col in df and weights_col not in cols:
        cols.append(weights_col)
    d = df[cols].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
    d = d[np.isfinite(d[x_col]) & np.isfinite(d[y_col])]
    if weights_col and weights_col in d:
        d[weights_col] = pd.to_numeric(d[weights_col], errors="coerce").fillna(0).clip(lower=0)
    result = {"n": int(len(d)), "rho": np.nan, "p_value": np.nan,
              "rho_ci_low": np.nan, "rho_ci_high": np.nan, "slope": np.nan,
              "intercept": np.nan, "bootstrap_method": "unavailable", "data": d}
    if len(d) < 4 or d[x_col].nunique() < 2 or d[y_col].nunique() < 2:
        return result
    rho, p = spearmanr(d[x_col], d[y_col])
    result["rho"], result["p_value"] = float(rho), float(p)

    x = d[x_col].to_numpy(float)
    y = d[y_col].to_numpy(float)
    if weights_col and weights_col in d and d[weights_col].sum() > 0:
        w = np.sqrt(d[weights_col].to_numpy(float))  # np.polyfit's w multiplies residuals
        slope, intercept = np.polyfit(x, y, 1, w=w)
    else:
        slope, intercept = np.polyfit(x, y, 1)
    result["slope"], result["intercept"] = float(slope), float(intercept)

    rng = np.random.default_rng(seed)
    if region_col in d and d[region_col].nunique() >= 2:
        groups = d[region_col].fillna("Unknown").astype(str).to_numpy()
        levels = np.unique(groups)
        blocks = [np.flatnonzero(groups == level) for level in levels]
        sample = lambda: np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        method = "region-block bootstrap"
    else:
        sample = lambda: rng.integers(0, len(d), len(d))
        method = "row bootstrap fallback (not spatially robust)"
    boot = []
    for _ in range(max(0, int(n_boot))):
        idx = sample()
        if len(idx) < 4 or np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            continue
        val = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(val):
            boot.append(float(val))
    if boot:
        result["rho_ci_low"], result["rho_ci_high"] = map(float, np.percentile(boot, [2.5, 97.5]))
    result["bootstrap_method"] = method
    return result



def adjusted_relationship_statistics(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    covariates: list[str] | tuple[str, ...] = (),
    region_col: str = "region",
    min_regions_for_interval: int = 8,
) -> dict:
    """Fit an exploratory standardized OLS association adjusted for selected covariates.

    The x/y coefficient is standardized (SD of Y per SD of X), conditional on the
    selected covariates. A region-cluster sandwich interval is returned only when
    enough distinct regional clusters are available. This is not a causal estimator
    and does not guarantee that within-region spatial dependence has been removed.

    Supported derived covariates are ``log_population`` and ``log_density``; they
    are created from the corresponding source columns without modifying the input.
    """
    from scipy.stats import t as student_t

    if x_col not in df or y_col not in df:
        return {"available": False, "reason": "Selected X or Y column is absent."}

    d = pd.DataFrame(index=df.index)
    d["__x"] = pd.to_numeric(df[x_col], errors="coerce")
    d["__y"] = pd.to_numeric(df[y_col], errors="coerce")
    used_covariates: list[str] = []
    for cov in list(dict.fromkeys(covariates)):
        if cov in {x_col, y_col, "__x", "__y"}:
            continue
        if cov == "log_population":
            if "population" not in df:
                continue
            values = np.log1p(pd.to_numeric(df["population"], errors="coerce").clip(lower=0))
        elif cov == "log_density":
            if "pop_density" not in df:
                continue
            values = np.log1p(pd.to_numeric(df["pop_density"], errors="coerce").clip(lower=0))
        elif cov in df:
            values = pd.to_numeric(df[cov], errors="coerce")
        else:
            continue
        name = f"__cov_{len(used_covariates)}"
        d[name] = values
        used_covariates.append(cov)

    if region_col in df:
        d["__region"] = df[region_col].fillna("Unknown region").astype(str)
    else:
        d["__region"] = "Unknown region"
    d = d.replace([np.inf, -np.inf], np.nan)
    required = ["__x", "__y"] + [c for c in d.columns if c.startswith("__cov_")]
    d = d.dropna(subset=required).copy()
    result = {
        "available": False,
        "n": int(len(d)),
        "n_regions": int(d["__region"].nunique()),
        "covariates": used_covariates,
        "standardized_beta": np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "p_value": np.nan,
        "inference_method": "not available",
        "inference_note": "",
    }
    n_parameters = 2 + len(used_covariates)  # intercept, X, and covariates
    if len(d) < max(10, n_parameters + 3):
        result["reason"] = "Too few complete observations for the selected adjustment set."
        return result
    if d["__x"].nunique() < 2 or d["__y"].nunique() < 2:
        result["reason"] = "X or Y has no variation in the complete-case sample."
        return result

    # Standardize all non-intercept columns so the X coefficient is comparable
    # across choices of outcome and predictor units.
    x = d["__x"].to_numpy(float)
    y = d["__y"].to_numpy(float)
    x_sd, y_sd = x.std(ddof=0), y.std(ddof=0)
    if not np.isfinite(x_sd) or not np.isfinite(y_sd) or x_sd <= 0 or y_sd <= 0:
        result["reason"] = "Unable to standardize a constant or invalid X/Y column."
        return result
    columns = [np.ones(len(d)), (x - x.mean()) / x_sd]
    kept_covariates: list[str] = []
    for i, cov in enumerate(used_covariates):
        raw = d[f"__cov_{i}"].to_numpy(float)
        sd = raw.std(ddof=0)
        if not np.isfinite(sd) or sd <= 0:
            continue
        columns.append((raw - raw.mean()) / sd)
        kept_covariates.append(cov)
    used_covariates = kept_covariates
    X = np.column_stack(columns)
    n, k = X.shape
    if n <= k + 1 or np.linalg.matrix_rank(X) < k:
        result["reason"] = "The adjusted design is underdetermined or collinear; remove covariates."
        result["covariates"] = used_covariates
        return result

    y_std = (y - y.mean()) / y_sd
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ X.T @ y_std
    residual = y_std - X @ beta
    result["n"] = int(n)
    result["covariates"] = used_covariates
    result["standardized_beta"] = float(beta[1])
    result["available"] = True

    regions = d["__region"].to_numpy(str)
    unique_regions = np.unique(regions)
    result["n_regions"] = int(len(unique_regions))
    if len(unique_regions) < int(min_regions_for_interval):
        result["inference_method"] = "coefficient only; region count below minimum"
        result["inference_note"] = (
            f"Coefficient is exploratory only. A region-cluster interval is suppressed because "
            f"{len(unique_regions)} regions are available; at least {min_regions_for_interval} are required. "
            "This threshold is a practical safeguard, not a universal statistical rule."
        )
        return result

    # Cluster-robust sandwich covariance at region level. A finite-cluster correction
    # is included; small numbers of clusters are still not treated as reliable.
    meat = np.zeros((k, k), dtype=float)
    for group in unique_regions:
        mask = regions == group
        score = X[mask].T @ residual[mask]
        meat += np.outer(score, score)
    g = len(unique_regions)
    correction = (g / (g - 1)) * ((n - 1) / max(1, n - k))
    covariance = correction * bread @ meat @ bread
    se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    if not np.isfinite(se) or se <= 0:
        result["inference_method"] = "region-cluster robust; standard error unavailable"
        result["inference_note"] = "The cluster-robust standard error could not be estimated reliably."
        return result
    dfree = g - 1
    critical = float(student_t.ppf(0.975, dfree))
    statistic = float(beta[1] / se)
    result.update({
        "ci95_low": float(beta[1] - critical * se),
        "ci95_high": float(beta[1] + critical * se),
        "p_value": float(2 * student_t.sf(abs(statistic), dfree)),
        "standard_error": se,
        "inference_method": "region-cluster robust sandwich (t reference, df = regions − 1)",
        "inference_note": (
            "Observational adjusted association; regional clustering is accounted for in the sandwich variance, "
            "but residual spatial dependence within regions, unmeasured confounding, measurement error, and model "
            "misspecification may remain. No causal interpretation is warranted."
        ),
    })
    return result

def disruption_summary(points: pd.DataFrame, tt_before: pd.DataFrame, tt_after: pd.DataFrame,
                       threshold_minutes: float = 30.0) -> tuple[pd.DataFrame, dict]:
    """Aggregate modeled reach loss and routing-record coverage after a disruption.

    ``tt_before`` and ``tt_after`` use point_id, facility_id, minutes. The road-time
    table is range-limited: a point with no valid travel-time record must not be
    silently interpreted as proof that it is geographically unreachable. Access
    shares retain the conservative all-population denominator, while separate route
    coverage fields show where the travel-time matrix does not have a usable record.
    """
    required = {"point_id", "psgc", "population"}
    if not required.issubset(points.columns):
        raise ValueError(f"points are missing required columns: {sorted(required - set(points.columns))}")
    for label, tt in (("before", tt_before), ("after", tt_after)):
        if not {"point_id", "minutes"}.issubset(tt.columns):
            raise ValueError(f"travel-time table ({label}) requires point_id and minutes")

    p = points[["point_id", "psgc", "population"]].copy()
    p["point_id"] = p["point_id"].astype(str)
    p["psgc"] = p["psgc"].astype(str)
    if p["point_id"].duplicated().any():
        raise ValueError("points must contain one row per unique point_id for route-coverage aggregation")
    p["population"] = pd.to_numeric(p["population"], errors="coerce").fillna(0).clip(lower=0)

    def _times_by_point(tt: pd.DataFrame) -> pd.Series:
        t = tt[["point_id", "minutes"]].copy()
        t["point_id"] = t["point_id"].astype(str)
        t["minutes"] = pd.to_numeric(t["minutes"], errors="coerce")
        t = t[np.isfinite(t["minutes"]) & t["minutes"].ge(0)]
        return t.groupby("point_id")["minutes"].min()

    before_min = _times_by_point(tt_before)
    after_min = _times_by_point(tt_after)
    p["minutes_before"] = p["point_id"].map(before_min).to_numpy(float)
    p["minutes_after"] = p["point_id"].map(after_min).to_numpy(float)
    p["has_route_record_before"] = p["point_id"].isin(before_min.index)
    p["has_route_record_after"] = p["point_id"].isin(after_min.index)
    before, after = p["minutes_before"].to_numpy(float), p["minutes_after"].to_numpy(float)
    p["minutes_added"] = np.where(np.isfinite(before) & np.isfinite(after), after - before,
                                  np.where(np.isfinite(before) & ~np.isfinite(after), np.inf, np.nan))
    threshold = float(threshold_minutes)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold_minutes must be a finite positive number")
    p["reachable_before"] = np.isfinite(before) & (before <= threshold)
    p["reachable_after"] = np.isfinite(after) & (after <= threshold)
    # This comparison only includes points known to be inside the threshold before;
    # absent after-routes mean no route is recorded within the supported matrix.
    p["loses_reach"] = p["reachable_before"] & ~p["reachable_after"]
    p["gains_reach"] = ~p["reachable_before"] & p["reachable_after"]
    p["population_losing_reach"] = p["population"] * p["loses_reach"]
    p["population_gaining_reach"] = p["population"] * p["gains_reach"]
    p["pop_reachable_before"] = p["population"] * p["reachable_before"]
    p["pop_reachable_after"] = p["population"] * p["reachable_after"]
    p["pop_route_before"] = p["population"] * p["has_route_record_before"]
    p["pop_route_after"] = p["population"] * p["has_route_record_after"]

    summary = p.groupby("psgc", dropna=False).agg(
        population=("population", "sum"),
        population_within_threshold_before=("pop_reachable_before", "sum"),
        population_within_threshold_after=("pop_reachable_after", "sum"),
        population_losing_reach=("population_losing_reach", "sum"),
        population_gaining_reach=("population_gaining_reach", "sum"),
        population_with_route_record_before=("pop_route_before", "sum"),
        population_with_route_record_after=("pop_route_after", "sum"),
    ).reset_index()
    den = summary["population"].replace(0, np.nan)
    summary["reachable_share_before"] = summary["population_within_threshold_before"] / den
    summary["reachable_share_after"] = summary["population_within_threshold_after"] / den
    summary["route_coverage_share_before"] = summary["population_with_route_record_before"] / den
    summary["route_coverage_share_after"] = summary["population_with_route_record_after"] / den
    details = {
        "threshold_minutes": threshold,
        "population": float(p["population"].sum()),
        "population_losing_reach": float(p.loc[p["loses_reach"], "population"].sum()),
        "population_gaining_reach": float(p.loc[p["gains_reach"], "population"].sum()),
        "population_within_before": float(p.loc[p["reachable_before"], "population"].sum()),
        "population_within_after": float(p.loc[p["reachable_after"], "population"].sum()),
        "population_with_route_record_before": float(p.loc[p["has_route_record_before"], "population"].sum()),
        "population_with_route_record_after": float(p.loc[p["has_route_record_after"], "population"].sum()),
        "population_without_route_record_before": float(p.loc[~p["has_route_record_before"], "population"].sum()),
        "population_without_route_record_after": float(p.loc[~p["has_route_record_after"], "population"].sum()),
        "route_coverage_share_before": float(p.loc[p["has_route_record_before"], "population"].sum() / p["population"].sum()) if p["population"].sum() else np.nan,
        "route_coverage_share_after": float(p.loc[p["has_route_record_after"], "population"].sum() / p["population"].sum()) if p["population"].sum() else np.nan,
        "demand_points_losing_reach": int(p["loses_reach"].sum()),
        "demand_points_no_route_before": int((~p["has_route_record_before"]).sum()),
        "demand_points_no_route_after": int((~p["has_route_record_after"]).sum()),
        "demand_points_no_longer_routed": int((np.isfinite(before) & ~np.isfinite(after)).sum()),
    }
    return summary, details
