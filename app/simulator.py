"""ISAGIP Simulator: place hypothetical facilities on the map and see how access and the gap change."""
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from common import (show_table, CFG, ENRICH_COLS, GLOSSARY, cascade, PH_BOUNDS, PROC, add_legend, add_lgu_layer, footer, lgu_feature_collection,
                    load_lite_boundaries, make_map, options_help, page_header, require_data, scale_for, term)
from isagiph.accessibility import fallback_travel_times
from isagiph.simulate import PRESETS, compare, run

res, fac, meta, valid, weights, gj, pts = require_data()
page_header("ISAGIP simulator", "What if we built one here?",
            "Choose a facility type, then <b>click the map</b> (or pick a place below the map) to add it. Access, the three "
            "indices and the gap are recalculated for every city and municipality.")

with st.expander("How the simulator works", expanded=True):
    st.markdown("""
**What you can test**
1. Choose a hospital level or a primary-care facility, set its licensed bed capacity when applicable, and select the area to inspect.
2. Click the map to place a hypothetical facility. Use **Undo last** to remove the latest placement or **Clear all** to start again.
3. Compare the baseline with the scenario. The map, accessible-bed estimates, coverage, Health Need / Hospital Capacity / Healthcare Gap indices, and priority counts update from the scenario calculations.

**How the calculation works**
- Hospital beds are shared across demand points that can reach each facility, with travel-time weights based on the existing E2SFCA accessibility model.
- The configured hospital catchment is 30 minutes for Level 1 and 60 minutes for Level 2/3. A primary-care placement changes the primary-care input for the nearest city or municipality; it is not treated as a new hospital.
- The model then recomputes the index values across all LGUs. Because HNI/HCI/HGI are comparative scores, helping one area can slightly change other areas' relative ranks.
- The “Suggest the best sites” tool uses a greedy, need-weighted coverage algorithm to propose sites that may reach more people who are currently outside the chosen time threshold.

**How to read the result**
A smaller healthcare gap or more residents within the modeled travel-time threshold is a positive modeled change—not a promise that a real facility will be built or that patients will use it. The scenario does not model construction costs, land availability, staffing, queues, referral decisions, fares, private-hospital affordability, or service-specific capacity. Results depend on the facility inventory and travel-time data available in this run.
""")


@st.cache_resource(show_spinner="Preparing travel times…")
def base_inputs():
    from isagiph.pipeline import travel_times
    points = pd.read_csv(PROC / "points.csv", dtype={"psgc": str})
    facilities = pd.read_csv(PROC / "facilities.csv", dtype={"psgc": str})
    tt, _ = travel_times({"points": points, "facilities": facilities}, CFG, demo=False)   # same as the pipeline
    return points, facilities, tt


@st.cache_data(show_spinner="Recalculating access and indices…", max_entries=30)
def simulate(added_key: tuple) -> pd.DataFrame:
    points, facilities, tt = base_inputs()
    added = pd.DataFrame(list(added_key), columns=["lat", "lon", "level", "beds", "malasakit", "ownership", "primary"])
    return run(res, points, facilities, tt, added, CFG)


if "sim" not in st.session_state:
    st.session_state.sim = []
    st.session_state.sim_last = None
    for _k in ("last_clicked", "last_object_clicked"):     # ignore clicks made before this visit
        st.session_state[f"sim_seen_{_k}"] = None
if "sim_view" not in st.session_state:
    st.session_state.sim_view = None          # (lat, lon) of the last click; the map re-centres there
    st.session_state.sim_set = {"region": "Whole country", "mode": "Change in gap"}
PRESET_HELP = ("**Level 1**: " + GLOSSARY["Level 1"] + "\n\n**Level 2**: " + GLOSSARY["Level 2"] + "\n\n**Level 3**: "
               + GLOSSARY["Level 3"] + "\n\n**Malasakit Center**: " + GLOSSARY["Malasakit Center"] + "\n\n**Infirmary**: "
               + GLOSSARY["Infirmary"] + "\n\n**Rural Health Unit**: " + GLOSSARY["Primary care"])

with st.container(border=True):
    c1, c2 = st.columns([2, 1])
    preset = c1.selectbox("Facility to place", list(PRESETS), help=PRESET_HELP)
    lvl, beds0, mal, own, primary = PRESETS[preset]
    beds = c2.slider("Number of beds", 0, 1000, beds0, step=10, disabled=primary,
                     help="Licensed beds the new facility would have.")
    fragment = getattr(st, "fragment", lambda f: f)

    @fragment
    def area_controls() -> None:
        a1, a2, a3 = st.columns([1.2, 1.4, .8])
        regions = ["Whole country"] + sorted(res["region"].unique())
        reg = a1.selectbox("Area", regions, index=regions.index(st.session_state.sim_set["region"]),
                           help="The whole-country view is lighter; choose a region to work closer in.")
        modes = ["Change in gap", "Gap after", "Gap before"]
        mode = a2.radio("Show", modes, index=modes.index(st.session_state.sim_set["mode"]), horizontal=True)
        a3.write("")
        if a3.button("Apply", type="primary", use_container_width=True):
            if reg != st.session_state.sim_set["region"]:
                st.session_state.sim_view = None
            st.session_state.sim_set = {"region": reg, "mode": mode}
            st.rerun()

    area_controls()
    v1, v2 = st.columns(2)
    if v1.button("↶ Undo last", use_container_width=True, disabled=not st.session_state.sim):
        st.session_state.sim.pop()
    if v2.button("✕ Clear all", use_container_width=True, disabled=not st.session_state.sim):
        st.session_state.sim = []
        st.session_state.sim_view = None
region, view_mode = st.session_state.sim_set["region"], st.session_state.sim_set["mode"]

added_key = tuple(st.session_state.sim)
before = simulate(tuple())
after = simulate(added_key) if added_key else before
cmp_ = compare(before, after)

national = region == "Whole country"
view = res if national else res[res["region"] == region]
view_after = after.set_index("psgc").loc[view["psgc"]].reset_index()
sim_gj = load_lite_boundaries() if (national and load_lite_boundaries() is not None) else gj
if st.session_state.sim_view:                 # stay where the user last clicked
    lat0, lon0 = st.session_state.sim_view
    m = make_map(location=[lat0, lon0], zoom_start=9)
else:
    m = make_map()
if st.session_state.sim_view:
    pass                                      # keep the view centred on the last placement
elif national:
    m.fit_bounds(PH_BOUNDS)
else:
    m.fit_bounds([[view.lat.min() - 0.2, view.lon.min() - 0.2], [view.lat.max() + 0.2, view.lon.max() + 0.2]])
extra = [c for c in ["priority_freq", "poverty_imputed"] + ENRICH_COLS if c in res.columns and c not in view_after.columns]
card_src = view_after.merge(res[["psgc"] + extra], on="psgc", how="left") if extra else view_after
if view_mode == "Change in gap":
    d = cmp_.set_index("psgc")["dHGI"]
    lim = max(5.0, float(d.abs().quantile(0.995)))
    import matplotlib
    cm = matplotlib.colormaps["RdYlGn_r"]
    norm = matplotlib.colors.Normalize(-lim, lim)
    to_color = lambda v: "#d9d9d9" if abs(v) < 0.5 else matplotlib.colors.to_hex(cm(norm(v)))  # noqa: E731
    fc = lgu_feature_collection(card_src, "gap_index", to_color, 8, sim_gj, values=d)
    add_lgu_layer(m, fc, 0.5)
    add_legend(m, "dHGI", "RdYlGn")
else:
    src = card_src if view_mode == "Gap after" else view
    to_color, _ = scale_for(res["gap_index"], "Gap", "")
    add_lgu_layer(m, lgu_feature_collection(src, "gap_index", to_color, 8, sim_gj), 0.45)
    add_legend(m, "gap_index", "Gap")
if not national:                              # existing hospitals: only for a region (keeps the page light on phones)
    near = fac[fac["lat"].between(view.lat.min() - 0.5, view.lat.max() + 0.5)
               & fac["lon"].between(view.lon.min() - 0.5, view.lon.max() + 0.5)]
    for _, f in near.iterrows():
        folium.CircleMarker([f["lat"], f["lon"]], radius=2.5, color="#555", weight=0, fill=True, fill_opacity=0.6).add_to(m)
for i, (lat, lon, lv, bd, ml, ow, pr) in enumerate(st.session_state.sim):
    icon = "🏥" if not pr else "🩺"
    folium.Marker([lat, lon], tooltip=f"New facility #{i + 1}: {'RHU' if pr else f'Level {lv}, {bd} beds'}"
                  + (" + Malasakit" if ml else ""),
                  icon=folium.DivIcon(html=f"<div style='font-size:26px;transform:translate(-13px,-26px)'>{icon}</div>")
                  ).add_to(m)
def place(lat: float, lon: float, where: str = "") -> None:
    """Add one facility of the chosen type at (lat, lon) and redraw."""
    st.session_state.sim_view = (lat, lon)
    st.session_state.sim.append((round(lat, 5), round(lon, 5), int(lvl), int(0 if primary else beds),
                                 int(mal), own, bool(primary)))
    st.session_state.sim_msg = f"Added: {preset}" + (f" in {where}" if where else "")
    st.rerun()


# A click on empty map is reported as "last_clicked"; a click on a coloured area as "last_object_clicked".
# Both count as a placement; each is remembered separately so a facility is never added twice.
out = st_folium(m, height=560, use_container_width=True, returned_objects=["last_clicked", "last_object_clicked"],
                key="simmap") or {}
for key in ("last_clicked", "last_object_clicked"):
    c = out.get(key)
    if isinstance(c, dict) and c.get("lat") is not None and c.get("lng") is not None \
            and c != st.session_state.get(f"sim_seen_{key}"):
        st.session_state[f"sim_seen_{key}"] = c
        place(float(c["lat"]), float(c["lng"]))
if st.session_state.get("sim_msg"):
    st.success(st.session_state.pop("sim_msg"), icon="✅")


@st.cache_data(show_spinner="Searching for the best sites…", max_entries=20)
def suggest(n_sites: int, region: str) -> pd.DataFrame:
    from isagiph.optimize import best_sites, current_minutes
    points, facilities, tt = base_inputs()
    need = res.set_index("psgc")["HVS"]
    within = None if region == "Whole country" else points["psgc"].isin(res.loc[res["region"] == region, "psgc"])
    acc = CFG["accessibility"]
    picks = best_sites(points, need, current_minutes(points, tt), n_sites=n_sites,
                       speed_kmh=acc["fallback_speed_kmh"],
                       circuity=acc["circuity"], within=within)
    return picks.merge(res[["psgc", "name", "province", "region"]], on="psgc", how="left") if len(picks) else picks


with st.container(border=True):          # optimisation: where would new hospitals help most?
    st.markdown("**Suggest the best sites.** Finds the places where new hospitals would bring the most people, "
                "weighted by health need, within 30 minutes of care (maximal covering location model, greedy "
                "optimisation).")
    o1, o2 = st.columns([2, 1])
    n_opt = o1.slider("How many new hospitals?", 1, 20, 5,
                      help="The model picks them one at a time, each time choosing the site that adds the most "
                           "need-weighted people not yet within 30 minutes of a hospital.")
    o2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if o2.button("Find best sites", use_container_width=True):
        st.session_state.sim_suggest = (n_opt, region)
    if st.session_state.get("sim_suggest"):
        n_s, reg_s = st.session_state.sim_suggest
        sug = suggest(n_s, reg_s)
        if sug.empty:
            st.info("Everyone in this area is already within 30 minutes of a hospital.")
        else:
            show_table(sug[["rank", "name", "province", "people_covered", "cumulative_people"]].rename(columns={
                "rank": "Order", "people_covered": "People newly within 30 min",
                "cumulative_people": "Total so far"}).round(0), hide_index=True, use_container_width=True)
            if st.button(f"Add these {len(sug)} sites to the map as: {preset}", type="primary"):
                for _, r_ in sug.iterrows():
                    st.session_state.sim.append((round(r_["lat"], 5), round(r_["lon"], 5), int(lvl),
                                                 int(0 if primary else beds), int(mal), own, bool(primary)))
                st.session_state.sim_view = None
                st.session_state.sim_msg = f"Added {len(sug)} suggested sites. The impact below is from the full model."
                st.session_state.sim_suggest = None
                st.rerun()

with st.container(border=True):          # placing by choosing a place (reliable on phones)
    st.markdown("**Or add it by choosing a place** (puts the facility at the city or municipality centre)")
    pc = st.columns([1, 1, 1, .7])
    p_reg, p_prov, p_lgu = cascade(pc[:3], sorted(res["region"].unique()), res, "simplace", with_lgu=True, multi=False)
    pc[3].markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if pc[3].button("Add facility here", type="primary", use_container_width=True):
        hit = res[(res["region"] == p_reg) & (res["province"] == p_prov) & (res["name"] == p_lgu)]
        if len(hit):
            place(float(hit.iloc[0]["lat"]), float(hit.iloc[0]["lon"]), f"{p_lgu}, {p_prov}")
st.caption("Grey dots (shown when an area is chosen) are existing licensed hospitals and infirmaries. Existing "
           "barangays use road travel times where road data exist; facilities you place use straight-line estimates, so "
           "a facility across the sea will look more helpful than it really is.")

# ---------------------------------------------------------------- results
SIM_COLS = {"name": "City/municipality", "province": "Province", "population": "Population", "HGI_before": "Gap before",
            "HGI_after": "Gap after", "dHGI": "Change in gap"}
n_new = len(st.session_state.sim)
st.markdown(f"#### Impact of {n_new} new facilit{'y' if n_new == 1 else 'ies'}" if n_new else
            "#### Place a facility on the map to see its impact")
pop = cmp_["population"]
r30_b = (cmp_["cov30_before"] * pop).sum()
r30_a = (cmp_["cov30_after"] * pop).sum()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Residents within 30 min of a hospital", f"{r30_a / 1e6:,.2f}M", f"{(r30_a - r30_b):+,.0f} people")
k2.metric("Bed shortfall", f"{cmp_['shortfall_after'].sum():,.0f}",
          f"{cmp_['shortfall_after'].sum() - cmp_['shortfall_before'].sum():+,.0f} beds", delta_color="inverse")
k3.metric("Priority areas", f"{int(cmp_['priority_after'].sum())}",
          f"{int(cmp_['priority_after'].sum() - cmp_['priority_before'].sum()):+d}", delta_color="inverse")
k4.metric("People in priority areas", f"{(cmp_['priority_after'] * pop).sum() / 1e6:,.2f}M",
          f"{((cmp_['priority_after'].astype(int) - cmp_['priority_before'].astype(int)) * pop).sum():+,.0f}", delta_color="inverse")
if n_new:
    imp = cmp_[cmp_["dHGI"] < -0.5].sort_values("dHGI").head(15)
    left = cmp_[cmp_["priority_before"] & ~cmp_["priority_after"]]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Biggest gap reductions**")
        show_table(imp[["name", "province", "population", "HGI_before", "HGI_after", "dHGI"]].round(1).rename(columns=SIM_COLS),
                     hide_index=True, use_container_width=True)
    with c2:
        st.markdown(f"**Areas that stop being a {term('priority')} ({len(left)})**", unsafe_allow_html=True)
        show_table(left[["name", "province", "population", "HGI_before", "HGI_after"]].round(1).rename(columns=SIM_COLS),
                     hide_index=True, use_container_width=True)
    st.markdown("**Your facilities**")
    show_table(pd.DataFrame(st.session_state.sim, columns=["Latitude", "Longitude", "Level", "Beds", "Malasakit Center",
                                                             "Ownership", "Primary care"]), hide_index=True, use_container_width=True)
footer(meta)
