"""Interactive what-if analysis for hospital closures and bed-capacity disruptions."""
from pathlib import Path

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from common import CFG, NAVY, PROC, RED, footer, make_map, page_header, require_data, section, show_table
from isagiph.accessibility import lgu_accessibility
from isagiph.analysis_tools import disruption_summary
from isagiph.indices import build_indices

res, facilities_loaded, meta, valid, weights, gj, pts_loaded = require_data()
page_header(
    "Healthcare resilience · Interactive",
    "Crisis Impact",
    "Explore how healthcare access changes when a facility becomes unavailable or loses bed capacity. "
    "This is a scenario analysis using the project's facility and travel-time data—not a real-time outage feed or forecast.",
)

st.info("**What this means:** compare the baseline with the scenario to see which communities may lose the most estimated access if the selected facility or capacity becomes unavailable. This is a planning simulation, not a prediction of an actual outage.")

required_files = [PROC / "points.csv", PROC / "facilities.csv"]
cache_inputs = required_files + [PROC / "lgu_results.csv", Path(__file__).resolve().parents[2] / "config" / "config.yaml"]
if not all(p.exists() for p in required_files):
    st.info("Demand points and facility records are required. Run the data pipeline to create the processed inputs.")
    st.stop()


def _version_stamp() -> tuple:
    # Include routing inputs as well as processed point/facility records so cached
    # travel times are invalidated when a road matrix changes.
    candidates = list(cache_inputs)
    interim = Path(CFG["paths"]["interim"])
    candidates.extend([interim / "travel_times.csv"])
    candidates.extend(sorted(interim.glob("road_times_*.csv")))
    candidates.extend(sorted(interim.glob("road_points_*.csv")))
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in candidates if p.exists())


@st.cache_resource(show_spinner="Preparing the existing travel-time network…")
def network_inputs(mode: str, stamp: tuple):
    """Read the pipeline's existing inputs and compute travel times with the same method as the pipeline."""
    points = pd.read_csv(PROC / "points.csv", dtype={"psgc": str, "point_id": str})
    facilities = pd.read_csv(PROC / "facilities.csv", dtype={"psgc": str, "facility_id": str})
    if facilities.empty:
        raise ValueError("No hospital facility records are available.")
    from isagiph.pipeline import travel_times
    tt, source = travel_times({"points": points, "facilities": facilities}, CFG, demo=(mode == "demo"))
    tt = tt.copy()
    tt["facility_id"] = tt["facility_id"].astype(str)
    tt["point_id"] = tt["point_id"].astype(str)
    return points, facilities, tt, source


@st.cache_data(show_spinner="Recalculating healthcare access after the scenario…", max_entries=12)
def calculate_disruption(facility_id: str, scenario: str, loss_pct: int, threshold: int, mode: str, stamp: tuple):
    points, facilities, tt, source = network_inputs(mode, stamp)
    facilities = facilities.copy()
    selected = facilities[facilities["facility_id"].astype(str) == str(facility_id)]
    if selected.empty:
        raise ValueError("The selected facility is no longer present in the processed facility table. Re-run the data pipeline.")
    if scenario == "Facility unavailable (closure)":
        after_facilities = facilities[facilities["facility_id"].astype(str) != str(facility_id)].copy()
        after_tt = tt[tt["facility_id"].astype(str) != str(facility_id)].copy()
    elif scenario == "Reduced bed capacity":
        if "beds" not in facilities:
            raise ValueError("The facility dataset has no bed-capacity column; capacity-reduction scenarios cannot be evaluated.")
        bed_value = pd.to_numeric(selected["beds"], errors="coerce").iloc[0]
        if not np.isfinite(bed_value) or bed_value <= 0:
            raise ValueError("The selected facility has no usable licensed-bed value. Choose a facility with a positive bed count or use the closure scenario.")
        after_facilities = facilities.copy()
        ix = after_facilities["facility_id"].astype(str) == str(facility_id)
        after_facilities.loc[ix, "beds"] = pd.to_numeric(after_facilities.loc[ix, "beds"], errors="coerce") * (1 - int(loss_pct) / 100)
        after_tt = tt.copy()
    else:
        raise ValueError(f"Unknown disruption scenario: {scenario}")

    if after_facilities.empty:
        raise ValueError("The scenario removes every facility in the available dataset; no comparison can be computed.")
    after_access = lgu_accessibility(points, after_facilities, after_tt, CFG)
    impact, totals = disruption_summary(points, tt, after_tt, threshold_minutes=threshold)
    # Capacity totals use the licensed-bed field in the input directory. This is a
    # modeled supply measure; it must not be described as staffed/operational beds.
    before_beds = pd.to_numeric(facilities.get("beds", pd.Series(np.nan, index=facilities.index)), errors="coerce")
    after_beds = pd.to_numeric(after_facilities.get("beds", pd.Series(np.nan, index=after_facilities.index)), errors="coerce")
    totals.update({
        "facility_count_before": int(len(facilities)),
        "facility_count_after": int(len(after_facilities)),
        "licensed_beds_before": float(before_beds.sum(min_count=1)) if before_beds.notna().any() else np.nan,
        "licensed_beds_after": float(after_beds.sum(min_count=1)) if after_beds.notna().any() else np.nan,
        "facilities_missing_bed_count_before": int(before_beds.isna().sum()),
        "facilities_missing_bed_count_after": int(after_beds.isna().sum()),
    })
    if np.isfinite(totals["licensed_beds_before"]) and np.isfinite(totals["licensed_beds_after"]):
        totals["licensed_beds_removed"] = float(totals["licensed_beds_before"] - totals["licensed_beds_after"])
        totals["licensed_bed_supply_change_pct"] = (100 * totals["licensed_beds_removed"] / totals["licensed_beds_before"]
                                                     if totals["licensed_beds_before"] > 0 else np.nan)
    else:
        totals["licensed_beds_removed"] = np.nan
        totals["licensed_bed_supply_change_pct"] = np.nan

    # Recompute the composite indices using the same configured index method and the changed access fields.
    scenario_data = res.copy()
    access_cols = [c for c in after_access.columns if c != "psgc"]
    scenario_data = scenario_data.drop(columns=[c for c in access_cols if c in scenario_data.columns]).merge(after_access, on="psgc", how="left")
    after_indices = build_indices(scenario_data, CFG)
    baseline_indices = build_indices(res.copy(), CFG)
    scenario_data = scenario_data.drop(columns=[c for c in after_indices.columns if c != "psgc" and c in scenario_data.columns], errors="ignore").merge(after_indices, on="psgc", how="left")

    base_cols = [c for c in ["psgc", "name", "province", "region", "lat", "lon", "population", "gap_index",
                            "beds_per_1000_all", "cov_any_30", "cov_l2plus_60", "cov_malasakit_60", "accessible_beds"] if c in res.columns]
    impact_table = res[base_cols].copy()
    idx_base = baseline_indices[["psgc", "gap_index", "HVS", "HCS"]].rename(columns={"gap_index": "gap_before", "HVS": "HVS_before", "HCS": "HCS_before"})
    idx_after = after_indices[["psgc", "gap_index", "HVS", "HCS"]].rename(columns={"gap_index": "gap_after", "HVS": "HVS_after", "HCS": "HCS_after"})
    impact_table = impact_table.drop(columns=[c for c in ["gap_index"] if c in impact_table.columns]).merge(idx_base, on="psgc", how="left").merge(idx_after, on="psgc", how="left")
    if "beds_per_1000_all" in after_access:
        impact_table = impact_table.merge(after_access[["psgc", "beds_per_1000_all", "cov_any_30", "cov_l2plus_60", "cov_malasakit_60"]].rename(columns={
            "beds_per_1000_all": "beds_after", "cov_any_30": "cov_any_30_after", "cov_l2plus_60": "cov_l2plus_60_after", "cov_malasakit_60": "cov_malasakit_60_after"}), on="psgc", how="left")
    impact_table = impact_table.merge(impact[["psgc", "population_losing_reach", "population_gaining_reach", "reachable_share_before", "reachable_share_after"]], on="psgc", how="left")
    impact_table["delta_beds_per_1000"] = impact_table["beds_after"] - impact_table["beds_per_1000_all"]
    impact_table["delta_cov_any_30"] = impact_table["cov_any_30_after"] - impact_table["cov_any_30"]
    impact_table["delta_gap"] = impact_table["gap_after"] - impact_table["gap_before"]
    return impact_table, totals, source, selected.iloc[0].to_dict()


stamp = _version_stamp()
mode = str(meta.get("mode", "real"))
valid_facilities = facilities_loaded.copy()
for col in ("facility_id", "name", "lat", "lon"):
    if col not in valid_facilities.columns:
        st.error(f"Processed facilities are missing required field `{col}`.")
        st.stop()
valid_facilities["facility_id"] = valid_facilities["facility_id"].astype(str)
valid_facilities = valid_facilities.dropna(subset=["facility_id", "name", "lat", "lon"])
if valid_facilities.empty:
    st.warning("No mappable facilities are available for disruption analysis.")
    st.stop()
if "psgc" in valid_facilities and {"psgc", "name", "province"}.issubset(res.columns):
    lookup = res[["psgc", "name", "province", "region"]].rename(columns={"name": "lgu_name", "province": "lgu_province", "region": "lgu_region"})
    valid_facilities = valid_facilities.merge(lookup, on="psgc", how="left")
else:
    valid_facilities["lgu_name"], valid_facilities["lgu_province"], valid_facilities["lgu_region"] = "", "", ""
valid_facilities["_facility_label"] = (valid_facilities["name"].astype(str) + " — " + valid_facilities["lgu_name"].fillna("").astype(str)
                                       + ", " + valid_facilities["lgu_province"].fillna("").astype(str) + " · "
                                       + valid_facilities["facility_id"].astype(str))

with st.form("disruption_form"):
    s1, s2 = st.columns([1.6, 1])
    selected_label = s1.selectbox("Facility to examine", valid_facilities.sort_values("_facility_label")["_facility_label"].tolist())
    scenario = s2.radio("Hypothetical disruption", ["Facility unavailable (closure)", "Reduced bed capacity"], help="Closure removes the facility from the model; reduced capacity lowers the facility's licensed bed supply while its location remains available.")
    max_loss = s1.slider("Reduction in licensed beds (%)", 10, 90, 50, 10, disabled=(scenario != "Reduced bed capacity"))
    threshold = s2.select_slider("Travel-time access threshold", options=[15, 30, 45, 60, 75, 90], value=30, format_func=lambda x: f"{x} minutes")
    submitted = st.form_submit_button("Run disruption scenario", type="primary", use_container_width=True)
if submitted:
    st.session_state["disruption_scenario"] = (valid_facilities.loc[valid_facilities["_facility_label"] == selected_label, "facility_id"].iloc[0], scenario, int(max_loss), int(threshold), selected_label)

if "disruption_scenario" not in st.session_state:
    st.info("Choose a facility and scenario, then run the analysis. All estimates use the currently processed facility and travel-time inputs.")
    footer(meta)
    st.stop()

facility_id, active_scenario, active_loss, active_threshold, active_label = st.session_state["disruption_scenario"]
try:
    impact_table, totals, tt_source, facility = calculate_disruption(facility_id, active_scenario, active_loss, active_threshold, mode, stamp)
except Exception as exc:
    st.error(f"The disruption scenario could not be calculated: {exc}")
    st.caption("Check that processed points, facility records, and the same travel-time source used by the pipeline are available. Capacity scenarios need a valid licensed-bed count.")
    footer(meta)
    st.stop()

st.markdown(f"**Scenario:** {active_label} · {active_scenario}" + (f" · {active_loss}% bed-capacity reduction" if active_scenario == "Reduced bed capacity" else ""))
st.caption(f"Travel-time source: {tt_source}. The travel-time matrix only covers its configured routing/search range; locations outside that range should be interpreted as not reachable in the modeled matrix, not as having literally infinite travel time.")
if mode == "demo":
    st.warning("Synthetic demo data are being used. These results demonstrate the method only and are not estimates of real Philippine communities.")

pop_loss = float(impact_table["population_losing_reach"].fillna(0).sum())
old_reach = float(impact_table["reachable_share_before"].fillna(0).mul(impact_table["population"].fillna(0)).sum() / impact_table["population"].fillna(0).sum()) if impact_table["population"].fillna(0).sum() else np.nan
new_reach = float(impact_table["reachable_share_after"].fillna(0).mul(impact_table["population"].fillna(0)).sum() / impact_table["population"].fillna(0).sum()) if impact_table["population"].fillna(0).sum() else np.nan
k1, k2, k3, k4 = st.columns(4)
k1.metric(f"People losing {active_threshold}-minute access", f"{pop_loss:,.0f}", help="Population at demand points that had a route within the selected threshold before the scenario and no recorded route within that threshold afterward. It is a network-model estimate, not observed care denial.")
k2.metric("Modeled reachable share before", f"{old_reach:.1%}" if np.isfinite(old_reach) else "N/A")
k3.metric("Modeled reachable share after", f"{new_reach:.1%}" if np.isfinite(new_reach) else "N/A", delta=f"{(new_reach-old_reach)*100:+.2f} pp" if np.isfinite(new_reach) and np.isfinite(old_reach) else None, delta_color="inverse")
k4.metric("LGUs with reduced accessible beds", f"{int((impact_table['delta_beds_per_1000'] < -1e-8).sum()):,}")
st.caption("Population affected is aggregated from modeled demand points. It is not a count of people who would actually fail to receive care, and does not model referrals, service substitution, ambulance availability, facility-level service suitability, or travel behavior changes.")
if pop_loss > 0:
    st.info(f"**Planning implication:** about {pop_loss:,.0f} modeled residents cross the selected {active_threshold}-minute reach threshold in this scenario. Review the affected LGUs below to identify where alternative routes, referrals or surge capacity need local verification.")
else:
    st.info("**Planning implication:** no modeled residents cross the selected travel-time threshold in this scenario, but accessible-bed capacity or relative gap indices may still change. Check the capacity metrics and affected-LGU table before concluding that the disruption has no impact.")

section("Model coverage and supply", "What does the scenario know—and what remains uncertain?")
ca, cb, cc, cd = st.columns(4)
ca.metric("People with route record before", f"{totals.get('route_coverage_share_before', np.nan):.1%}" if np.isfinite(totals.get("route_coverage_share_before", np.nan)) else "N/A")
cb.metric("People with route record after", f"{totals.get('route_coverage_share_after', np.nan):.1%}" if np.isfinite(totals.get("route_coverage_share_after", np.nan)) else "N/A")
cc.metric("Licensed beds before", f"{totals.get('licensed_beds_before', np.nan):,.0f}" if np.isfinite(totals.get("licensed_beds_before", np.nan)) else "N/A")
cd.metric("Licensed beds after", f"{totals.get('licensed_beds_after', np.nan):,.0f}" if np.isfinite(totals.get("licensed_beds_after", np.nan)) else "N/A")
coverage_total = float(totals.get("population", 0) or 0)
missing_route_population = float(totals.get("population_without_route_record_after", np.nan))
if np.isfinite(missing_route_population):
    st.caption(f"After the scenario, {missing_route_population:,.0f} people ({missing_route_population / coverage_total:.1%} of modeled population when the denominator is non-zero) have no usable travel-time record within the configured search matrix." if coverage_total > 0 else "The population denominator is unavailable.")
if totals.get("facilities_missing_bed_count_before", 0) or totals.get("facilities_missing_bed_count_after", 0):
    st.warning(f"Licensed-bed totals are incomplete: bed counts are missing for {totals.get('facilities_missing_bed_count_before', 0)} facilities before and {totals.get('facilities_missing_bed_count_after', 0)} after the scenario. Totals can understate actual listed capacity.")
st.caption("A route record means at least one facility route is present within the configured search range; it is not a guarantee of service availability or proof that all missing routes are missing data. No route may mean no facility within the routing range. Bed totals represent licensed beds in the source, not staffed or operational capacity. Remaining hospitals are treated as geographically available even when their service mix, referral arrangements, and surge capacity are unknown.")

section("Geographic impact", "Which LGUs lose the most access?")
left, right = st.columns([1.2, 1])
with left:
    map_view = impact_table.copy()
    map_view = map_view.sort_values(["population_losing_reach", "delta_beds_per_1000"], ascending=[False, True])
    map_view = map_view[map_view["population_losing_reach"].fillna(0).gt(0)].head(250)
    if map_view.empty:
        st.info("No demand points crossed the selected travel-time threshold in this scenario. Capacity or gap indicators may still change.")
    else:
        m = make_map(location=[12.5, 122.0], zoom_start=5)
        m.fit_bounds([[4.4, 116.0], [21.3, 127.0]])
        for _, row in map_view.iterrows():
            if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
                continue
            popup = (f"<b>{row.get('name','')}</b><br>{row.get('province','')}<br>Population losing threshold access: "
                     f"{row.get('population_losing_reach',0):,.0f}<br>Accessible beds change: {row.get('delta_beds_per_1000',0):+.3f} per 1,000")
            folium.CircleMarker([float(row["lat"]), float(row["lon"])], radius=5 + min(5, np.sqrt(max(0, float(row.get("population_losing_reach", 0))) / 1000)),
                                color=RED, fill=True, fill_color=RED, fill_opacity=0.7, tooltip=str(row.get("name", "LGU")),
                                popup=folium.Popup(popup, max_width=300)).add_to(m)
        st_folium(m, height=420, use_container_width=True, returned_objects=[])
        st.caption("Larger markers represent more modeled residents crossing the chosen reach threshold. The map shows the largest estimated losses, capped at 250 markers; check route coverage before treating a blank area as unaffected.")
with right:
    chart = impact_table.sort_values("delta_beds_per_1000").head(20).copy()
    chart["label"] = chart["name"].astype(str) + " · " + chart["province"].fillna("").astype(str)
    st.altair_chart(alt.Chart(chart).mark_bar(color=RED, cornerRadiusEnd=3).encode(
        x=alt.X("delta_beds_per_1000:Q", title="Change in accessible beds per 1,000", scale=alt.Scale(zero=True)),
        y=alt.Y("label:N", sort="x", title=None),
        tooltip=[alt.Tooltip("name:N", title="LGU"), alt.Tooltip("province:N", title="Province"),
                 alt.Tooltip("delta_beds_per_1000:Q", format="+.3f", title="Change in beds/1,000"),
                 alt.Tooltip("population_losing_reach:Q", format=",.0f", title="People losing threshold access"),
                 alt.Tooltip("delta_gap:Q", format="+.1f", title="HGI change")]
    ).properties(height=500), use_container_width=True)
    st.caption("Bars extending further left have larger modeled losses in accessible beds per 1,000 residents. They help identify where local validation of routes, facility alternatives and service mix may be most important.")

section("Local results", "Review affected communities")
order = impact_table.sort_values(["population_losing_reach", "delta_beds_per_1000"], ascending=[False, True])
show_cols = [c for c in ["name", "province", "region", "population", "population_losing_reach", "reachable_share_before", "reachable_share_after", "delta_beds_per_1000", "delta_cov_any_30", "delta_gap"] if c in order]
show_table(order[show_cols].head(100).rename(columns={
    "name": "LGU", "province": "Province", "region": "Region", "population": "Population",
    "population_losing_reach": f"People losing {active_threshold}-min access", "reachable_share_before": "Reachable share before",
    "reachable_share_after": "Reachable share after", "delta_beds_per_1000": "Change in accessible beds/1,000",
    "delta_cov_any_30": "Change in 30-min hospital coverage", "delta_gap": "Change in HGI",
}).round(4), hide_index=True, use_container_width=True, height=400)
st.caption("The HGI is a relative, normalized index. Its scenario change is recomputed across the available LGU dataset, so it can shift when other LGUs' percentile positions change; it is not an absolute clinical outcome. Bed-count changes and reachability changes are reported separately because capacity loss does not necessarily remove geographic access.")
footer(meta)
