"""Interactive health relationship and equity analysis for Philippine LGUs."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import BLUE, CFG, NAVY, RED, footer, page_header, require_data, section, show_table
from isagiph.analysis_tools import adjusted_relationship_statistics, relationship_statistics
from equity_panel import render_equity_panel

res, fac, meta, valid, weights, gj, pts = require_data()
page_header(
    "Analyze · Interactive",
    "Health Relationships",
    "Explore relationships between healthcare access and health outcomes, and investigate how accessibility is distributed across poverty, isolation, age, and regions. Relationships are observational; equity indicators describe unequal access rather than its causes.",
)
relationship_tab, equity_tab = st.tabs(["Relationships", "Equity & Fairness"])

st.info("**What this means:** the chart shows which measures tend to move together in the available data. A visible relationship can help identify areas for further investigation, but it does not show that one measure causes the other.")


def render_relationship_analysis(res: pd.DataFrame, meta: dict) -> None:
    X_LABELS = {
        "beds_per_1000_all": "Accessible beds per 1,000 residents",
        "beds_per_1000_l2plus": "Level 2+ accessible beds per 1,000 residents",
        "cov_any_30": "Population share within 30 minutes of any hospital",
        "cov_l2plus_60": "Population share within 60 minutes of a Level 2+ hospital",
        "cov_malasakit_60": "Population share within 60 minutes of a Malasakit Center",
        "primary_care_per_10k": "Licensed primary-care facilities per 10,000 residents",
        "mean_minutes_any": "Mean travel time to a reachable hospital (minutes)",
    }
    Y_LABELS = {
        "facility_delivery_rate": "Facility-based delivery rate",
        "rate_tb": "Tuberculosis rate per 100,000",
        "rate_pneumonia": "Pneumonia rate per 100,000",
        "rate_hypertension": "Hypertension rate per 100,000",
        "rate_dengue": "Dengue rate per 100,000",
        "eb_rate_tb": "Empirical-Bayes-smoothed TB rate",
        "eb_rate_dengue": "Empirical-Bayes-smoothed dengue rate",
    }

    x_options = {k: v for k, v in X_LABELS.items() if k in res.columns and pd.to_numeric(res[k], errors="coerce").notna().sum() >= 4}
    demo_mode = str(meta.get("mode", "real")).lower() == "demo"
    if demo_mode and "facility_delivery_rate" in res.columns:
        Y_LABELS["facility_delivery_rate"] = "Facility-based delivery rate (synthetic demo only)"
    y_options = {k: v for k, v in Y_LABELS.items()
                 if k in res.columns and pd.to_numeric(res[k], errors="coerce").notna().sum() >= 4
                 and (k != "facility_delivery_rate" or demo_mode)}
    if not x_options or not y_options:
        st.warning("This dataset does not currently contain enough comparable observations for the relationship analysis. Check data preparation and the reference years of the access/outcome indicators.")
        return

    section("Set up the analysis", "Choose two measures to compare")
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    x_col = c1.selectbox("Healthcare access measure (X)", list(x_options), format_func=lambda k: x_options[k], key="rel_x")
    default_y = "rate_tb" if "rate_tb" in y_options else next(iter(y_options))
    y_col = c2.selectbox("Health outcome (Y)", list(y_options), index=list(y_options).index(default_y), format_func=lambda k: y_options[k], key="rel_y")
    regions = ["All regions"] + sorted(res["region"].dropna().astype(str).unique().tolist()) if "region" in res else ["All regions"]
    region = c3.selectbox("Geographic scope", regions, key="rel_region")

    analysis = res.copy()
    if region != "All regions":
        analysis = analysis[analysis["region"].astype(str) == region].copy()
    # Empirical-Bayes-smoothed outcomes inherit the missingness/imputation status of
    # their underlying raw rate; otherwise an imputed raw rate could bypass the toggle.
    if y_col.startswith("eb_rate_"):
        flag_base = "rate_" + y_col.removeprefix("eb_rate_")
    else:
        flag_base = y_col
    flag_col = f"{flag_base}_imputed"
    observed_only = False
    if flag_col in analysis.columns:
        observed_only = st.toggle("Use observed outcome records only (exclude imputed values)", value=True, help="The selected outcome (or its underlying raw rate for empirical-Bayes smoothing) has an imputation flag. Excluding estimated values helps distinguish observed from imputed outcomes; the sample may become smaller.")
        if observed_only:
            analysis = analysis[pd.to_numeric(analysis[flag_col], errors="coerce").fillna(1).eq(0)].copy()

    weighted_fit = st.checkbox("Weight fitted regression line by population", value=False, help="This changes the fitted linear trend to give more weight to observations with larger populations. The Spearman correlation and confidence interval remain unweighted.")
    if y_col == "facility_delivery_rate" and demo_mode:
        st.warning("Synthetic demonstration outcome: facility-based delivery rate is generated for the demo and is not a real Philippine observation. Use it only to explore how the interface works; it is excluded in real-data mode.")

    with st.expander("How are these measures constructed?", expanded=False):
        st.markdown(f"- **X:** {x_options[x_col]}. Some measures are modelled accessibility estimates derived from facility supply and travel-time assumptions.\n- **Y:** {y_options[y_col]}. Check source dates and whether values are observed or imputed.\n- The correlation is Spearman's rank correlation. Its 95% interval uses region-block resampling when at least two regions are represented; single-region analyses use a row-bootstrap fallback.\n- The fitted line describes a simple linear trend, not a causal model. Population weighting affects the line only, not the reported rank correlation.")

    stats = relationship_statistics(analysis, x_col, y_col, region_col="region", n_boot=300,
                                    seed=int(CFG["project"].get("seed", 42)),
                                    weights_col="population" if weighted_fit else None)
    d = stats.pop("data")
    if stats["n"] < 4:
        st.warning(f"Only {stats['n']} usable LGUs match these choices. Select another measure, widen the geographic scope, or include imputed outcomes if justified.")
        return

    section("Results", "What relationship appears in this dataset?")
    ka, kb, kc, kd = st.columns(4)
    ka.metric("LGUs in analysis", f"{stats['n']:,}")
    kb.metric("Spearman ρ", f"{stats['rho']:+.3f}" if np.isfinite(stats["rho"]) else "N/A", help="Rank-based association; ranges from -1 to +1.")
    kc.metric("95% bootstrap interval", f"[{stats['rho_ci_low']:+.2f}, {stats['rho_ci_high']:+.2f}]" if np.isfinite(stats["rho_ci_low"]) else "N/A")
    kd.metric("Nominal p-value", f"{stats['p_value']:.3g}" if np.isfinite(stats["p_value"]) else "N/A", help="Exploratory nominal test. It does not adjust for all confounding, multiple comparisons, or spatial dependence.")
    st.caption(f"Interval method: {stats['bootstrap_method']}. A narrow-looking interval does not eliminate confounding, source bias, reporting differences, or geographic measurement error.")
    rho = float(stats.get("rho", np.nan))
    lo_ci, hi_ci = float(stats.get("rho_ci_low", np.nan)), float(stats.get("rho_ci_high", np.nan))
    if np.isfinite(rho):
        strength = "strong" if abs(rho) >= 0.7 else "moderate" if abs(rho) >= 0.4 else "weak" if abs(rho) >= 0.2 else "very weak"
        direction = "positive" if rho > 0 else "negative" if rho < 0 else "near-zero"
        direction_text = (f"A {strength} {direction} rank association appears in this sample. "
                          + ("Higher values of the access measure tend to occur with higher outcome values." if rho > 0 else
                             "Higher values of the access measure tend to occur with lower outcome values." if rho < 0 else
                             "The two measures show little monotonic association in this sample."))
        if np.isfinite(lo_ci) and np.isfinite(hi_ci) and lo_ci <= 0 <= hi_ci:
            direction_text += " The bootstrap interval includes zero, so the direction is uncertain under this resampling procedure."
        direction_text += " Use this as a prompt to check local data quality, reporting differences, and adjusted results—not as evidence that access caused the outcome."
        st.info("**What the result suggests:** " + direction_text)

    plot = d.copy()
    plot["LGU"] = plot["name"].astype(str) if "name" in plot else plot["psgc"].astype(str)
    if "province" not in plot:
        plot["province"] = ""
    if "region" not in plot:
        plot["region"] = ""
    if weighted_fit and "population" in plot and plot["population"].sum() > 0:
        w = np.sqrt(pd.to_numeric(plot["population"], errors="coerce").fillna(0).clip(lower=0).to_numpy())
        slope, intercept = np.polyfit(plot[x_col].to_numpy(float), plot[y_col].to_numpy(float), 1, w=w)
    else:
        slope, intercept = np.polyfit(plot[x_col].to_numpy(float), plot[y_col].to_numpy(float), 1)
    x_grid = np.linspace(float(plot[x_col].min()), float(plot[x_col].max()), 80)
    line = pd.DataFrame({x_col: x_grid, y_col: intercept + slope * x_grid})
    points_chart = alt.Chart(plot).mark_circle(size=58, opacity=0.72).encode(
        x=alt.X(f"{x_col}:Q", title=x_options[x_col]),
        y=alt.Y(f"{y_col}:Q", title=y_options[y_col]),
        color=alt.Color("region:N", title="Region", legend=alt.Legend(orient="bottom", columns=4)) if region == "All regions" else alt.value(BLUE),
        size=alt.Size("population:Q", title="Population", scale=alt.Scale(range=[24, 180]), legend=None) if "population" in plot else alt.value(58),
        tooltip=[alt.Tooltip("LGU:N", title="LGU"), alt.Tooltip("province:N", title="Province"),
                 alt.Tooltip("region:N", title="Region"), alt.Tooltip(f"{x_col}:Q", format=".4g", title=x_options[x_col]),
                 alt.Tooltip(f"{y_col}:Q", format=".4g", title=y_options[y_col]),
                 alt.Tooltip("population:Q", format=",.0f", title="Population")] if "population" in plot else [alt.Tooltip("LGU:N", title="LGU"), alt.Tooltip(f"{x_col}:Q"), alt.Tooltip(f"{y_col}:Q")]
    ).properties(height=470).interactive()
    line_chart = alt.Chart(line).mark_line(color=RED, strokeWidth=2.5).encode(x=f"{x_col}:Q", y=f"{y_col}:Q")
    st.altair_chart(points_chart + line_chart, use_container_width=True)
    st.caption("Each point is an LGU. Larger markers indicate larger populations; the red line is an unadjusted simple fitted trend. If a health outcome is also used in the Health Need Index, its relationship with any index-derived measure may be partly circular.")

    # Optional adjusted association is reported separately from Spearman's marginal
    # association. Region-cluster intervals are suppressed when there are too few groups.
    section("Adjustment", "Compare the marginal pattern with an adjusted association",
            "This is an observational, standardized linear association—not a causal effect. Complete cases may differ from the unadjusted sample.")
    COVARIATE_LABELS = {
        "poverty_incidence": "Poverty incidence",
        "gida_share": "GIDA share",
        "pct_age_0_4": "Share aged 0–4",
        "pct_age_60_plus": "Share aged 60+",
        "log_population": "Log population",
        "log_density": "Log population density",
    }
    covariate_candidates = []
    for cov in ("poverty_incidence", "gida_share", "pct_age_0_4", "pct_age_60_plus"):
        if cov in analysis.columns:
            covariate_candidates.append(cov)
    if "population" in analysis.columns:
        covariate_candidates.append("log_population")
    if "pop_density" in analysis.columns:
        covariate_candidates.append("log_density")
    default_covariates = [c for c in ("poverty_incidence", "gida_share", "pct_age_0_4", "pct_age_60_plus", "log_population", "log_density") if c in covariate_candidates and c != x_col and c != y_col]
    chosen_covariates = st.multiselect(
        "Context variables to adjust for", options=[c for c in covariate_candidates if c != x_col and c != y_col],
        default=default_covariates, format_func=lambda c: COVARIATE_LABELS.get(c, c), key="rel_adjust_covariates",
        help="Choose available socioeconomic, isolation, age-structure, and population context variables. Adjustment only addresses measured variables and does not establish causality.")
    adjusted = adjusted_relationship_statistics(analysis, x_col, y_col, chosen_covariates, region_col="region", min_regions_for_interval=8)
    if adjusted.get("available"):
        aa, ab, ac, ad = st.columns(4)
        aa.metric("Complete LGUs", f"{adjusted['n']:,}")
        ab.metric("Adjusted standardized β", f"{adjusted['standardized_beta']:+.3f}", help="Standard-deviation change in Y associated with a one-standard-deviation change in X, conditional on the selected covariates. Observational, not causal.")
        if np.isfinite(adjusted.get("ci95_low", np.nan)) and np.isfinite(adjusted.get("ci95_high", np.nan)):
            ac.metric("95% region-cluster interval", f"[{adjusted['ci95_low']:+.2f}, {adjusted['ci95_high']:+.2f}]")
            ad.metric("Cluster-adjusted p-value", f"{adjusted['p_value']:.3g}" if np.isfinite(adjusted.get("p_value", np.nan)) else "N/A")
        else:
            ac.metric("95% interval", "Not reported")
            ad.metric("Regions in sample", str(adjusted.get("n_regions", 0)))
            st.warning(adjusted.get("inference_note", "Too few geographic clusters for a stable region-cluster interval."))
        st.caption("Adjustment method: standardized ordinary least squares. " + str(adjusted.get("inference_method", "")) + ". " + str(adjusted.get("inference_note", "")))
        beta = float(adjusted.get("standardized_beta", np.nan))
        ci_lo, ci_hi = float(adjusted.get("ci95_low", np.nan)), float(adjusted.get("ci95_high", np.nan))
        if np.isfinite(beta):
            implication = ("After accounting for the selected measured context variables, the fitted association is positive." if beta > 0 else
                           "After accounting for the selected measured context variables, the fitted association is negative." if beta < 0 else
                           "After accounting for the selected measured context variables, the fitted association is close to zero.")
            if np.isfinite(ci_lo) and np.isfinite(ci_hi) and ci_lo <= 0 <= ci_hi:
                implication += " The interval includes zero, so the adjusted direction is not clearly distinguished from zero in this sample."
            implication += " Verify source years and geography, and check whether the pattern persists across regions; this remains an observational association."
            st.info("**How to use the adjusted result:** " + implication)
    else:
        st.info("Adjusted association not available: " + str(adjusted.get("reason", "the selected variables do not provide enough complete observations.")))
    st.caption("Outcome reference years are not carried as per-indicator fields into this analysis view. Before formal interpretation, verify that the selected access measure and outcome refer to compatible geographic definitions and reference periods. Repeatedly exploring many pairs also makes nominal p-values exploratory; adjust for multiplicity in formal studies.")

    # Selecting an LGU below links the aggregate relationship to a concrete, inspectable record.
    section("Inspect one area", "Place an LGU in the context of the relationship")
    local = plot.copy()
    local["_label"] = local["LGU"] + " — " + local["province"].fillna("").astype(str) + " · " + local["psgc"].astype(str)
    local = local.sort_values("_label")
    selected = st.selectbox("LGU", local["_label"].tolist(), key="rel_lgu")
    r = local.loc[local["_label"] == selected].iloc[0]
    pred_y = intercept + slope * float(r[x_col])
    cols = st.columns(4)
    cols[0].metric("Selected X", f"{r[x_col]:,.4g}")
    cols[1].metric("Selected Y", f"{r[y_col]:,.4g}")
    cols[2].metric("Fitted trend at X", f"{pred_y:,.4g}")
    difference = float(r[y_col]) - pred_y
    cols[3].metric("Difference from fitted trend", f"{difference:+,.4g}")
    st.write(f"**{r['LGU']}** · {r['province']}, {r['region']}")
    if difference > 0:
        st.caption("This LGU's outcome is above the simple fitted trend for its access value. This is a sample comparison, not proof of an unusual cause.")
    elif difference < 0:
        st.caption("This LGU's outcome is below the simple fitted trend for its access value. This is a sample comparison, not proof of an unusual cause.")
    else:
        st.caption("This LGU sits on the simple fitted trend for its access value. The trend is descriptive, not a causal expectation.")
    if flag_col in r.index and pd.notna(r[flag_col]):
        st.info("This outcome record is marked as " + ("imputed" if int(r[flag_col]) == 1 else "observed") + ".")

    index_inputs = set()
    for vals in CFG.get("indicators", {}).get("hvs", {}).values():
        index_inputs.update(vals)
    if y_col in index_inputs:
        st.warning("The selected health outcome is also used in the Health Need Index. Associations between this outcome and the HNI/HGI can therefore be partly circular. Interpret the relationship as descriptive, not independent validation.")
    elif y_col == "facility_delivery_rate" and demo_mode:
        st.warning("This relationship uses a synthetic demo outcome and provides no evidence about real facility-based delivery in the Philippines.")
    else:
        st.info("Before interpreting this relationship, confirm that the outcome is independent of the selected index and that the source year and geographic unit align with the accessibility data.")

    footer(meta)

with relationship_tab:
    render_relationship_analysis(res, meta)

with equity_tab:
    render_equity_panel(res)
