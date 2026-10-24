"""Facility Finder: search licensed facilities by name or place; filter by type, level, ownership,
services (e.g. dialysis) and Malasakit; see bed capacity and recent community reports."""
import folium
import re
import unicodedata
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from common import (GLOSSARY, PH_BOUNDS, SERVICE_HELP, add_facility_markers, add_map_control, cascade, community_store,
                    facility_legend, footer, load_directory, make_map, options_help, page_header, require_data, show_table)
from isagiph.build_real import SERVICE_KEYWORDS

res, fac, meta, valid, weights, gj, pts = require_data()
d = load_directory()
page_header("Facility finder", "Find a health facility",
            "Search by facility name, city, or province. Filter by type, ownership, or services such as dialysis.")
if d.empty:
    st.info("The facility directory is not available.")
    st.stop()

FDEF = {"q": "", "regions": [], "provinces": [], "kinds": [], "owners": [], "services": [], "levels": [],
        "only_mal": False, "beds": (0, int(d["beds"].fillna(0).max() or 0))}
if "finder_applied" not in st.session_state:
    st.session_state.finder_applied = dict(FDEF)
if "finder_query" not in st.session_state:
    st.session_state.finder_query = st.session_state.finder_applied["q"]
fragment = getattr(st, "fragment", lambda fn: fn)
geo = d[["lgu_region", "lgu_province"]].rename(columns={"lgu_region": "region", "lgu_province": "province"})


def _norm_search(value) -> str:
    """Normalize punctuation and spacing so autocomplete labels can be used as free-text searches."""
    value = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _search_suggestions(frame: pd.DataFrame, query: str, limit: int = 8) -> list[str]:
    qn = _norm_search(query)
    if len(qn) < 2:
        return []
    candidates: list[str] = []
    if {"name", "lgu", "lgu_province"} <= set(frame.columns):
        slim = frame[["name", "lgu", "lgu_province", "address"]].fillna("").astype(str)
        for name, lgu, province, address in slim.itertuples(index=False, name=None):
            label = f"{name} — {lgu}, {province}".strip(" —,")
            search_blob = _norm_search(f"{name} {lgu} {province} {address}")
            if qn in search_blob:
                candidates.append(label)
            location = f"{lgu}, {province}".strip(" ,")
            if location and qn in _norm_search(location):
                candidates.append(location)
            if province and qn in _norm_search(province):
                candidates.append(province)
    # Deduplicate and rank prefix matches before other substring matches.
    unique = list(dict.fromkeys(candidates))
    unique.sort(key=lambda value: (not _norm_search(value).startswith(qn), len(value), _norm_search(value)))
    return unique[:limit]


def _use_finder_suggestion() -> None:
    selected = st.session_state.get("finder_suggestion")
    if selected:
        st.session_state["finder_query"] = selected
    st.session_state["finder_suggestion"] = None


def _clear_finder_suggestion() -> None:
    st.session_state["finder_suggestion"] = None


def _reset_finder() -> None:
    """Reset applied filters and all visible widget state before the next render."""
    st.session_state.finder_applied = dict(FDEF)
    defaults = {
        "finder_query": "", "finder_suggestion": None, "finder_reg": [], "finder_prov": [],
        "finder_kinds": [], "finder_owners": [], "finder_services": [], "finder_levels": [],
        "finder_only_mal": False, "finder_beds_min": FDEF["beds"][0], "finder_beds_max": FDEF["beds"][1],
    }
    for key, value in defaults.items():
        st.session_state[key] = value


@fragment
def finder_controls() -> None:
    a = st.session_state.finder_applied
    with st.container(border=True):
        q = st.text_input("Search", placeholder="e.g. Philippine General Hospital, Tacloban, Sulu…",
                          key="finder_query", on_change=_clear_finder_suggestion,
                          help="Start typing a facility, city/municipality, or province to see suggestions. Choose a suggestion or keep a free-text query.")
        suggestions = _search_suggestions(d, q)
        if suggestions:
            st.selectbox("Search suggestions", suggestions, index=None,
                         placeholder="Choose a matching facility or place…", key="finder_suggestion",
                         on_change=_use_finder_suggestion,
                         help="Suggestions are drawn from the current facility directory. Selecting one fills the search box; press Apply to search.")
        regions, provinces = cascade(st.columns(2), sorted(geo["region"].dropna().unique()), geo, "finder")
        a1, a2, a3 = st.columns(3)
        kinds_ui = a1.multiselect("Type", ["Hospital", "Infirmary", "Primary care"], default=a["kinds"],
                                  key="finder_kinds", placeholder="All types",
                                  help=options_help("Kind of facility.", ["Hospital", "Infirmary", "Primary care"]))
        owner_ui = a2.multiselect("Ownership", ["Public", "Private"],
                                  default=["Public" if o == "Government" else o for o in a["owners"]],
                                  key="finder_owners", placeholder="Public and private",
                                  help=options_help("Who runs the facility.", ["Public", "Private"]))
        levels = a3.multiselect("Hospital level", ["Level 1", "Level 2", "Level 3"], default=a["levels"],
                                key="finder_levels", placeholder="Any level",
                                help=options_help("DOH hospital level (hospitals only).", ["Level 1", "Level 2", "Level 3"]))
        b1, b2 = st.columns([1.4, 1])
        services = b1.multiselect("Services", list(SERVICE_KEYWORDS), default=a["services"], key="finder_services",
                                  placeholder="Any services",
                                  help="Services listed in the DOH file.\n\n" + "\n\n".join(
                                      f"**{k}**: {v}" for k, v in SERVICE_HELP.items()))
        hi = FDEF["beds"][1]
        b2.markdown("**Number of beds (range)**")
        bed_min_col, bed_max_col = b2.columns(2)
        min_default = min(max(int(a["beds"][0]), 0), hi)
        max_default = min(max(int(a["beds"][1]), 0), hi)
        beds_min = bed_min_col.number_input("Minimum", min_value=0, max_value=hi, value=min_default, step=1,
                                             key="finder_beds_min", help="Lowest licensed bed count to include.")
        beds_max = bed_max_col.number_input("Maximum", min_value=0, max_value=hi, value=max_default, step=1,
                                             key="finder_beds_max", help="Highest licensed bed count to include.")
        only_mal = st.toggle("Malasakit Center only", value=a["only_mal"], key="finder_only_mal",
                             help=GLOSSARY["Malasakit Center"])
        apply_col, reset_col = st.columns([3, 1])
        apply_clicked = apply_col.button("Apply", type="primary", use_container_width=True)
        reset_col.button("Reset", use_container_width=True, on_click=_reset_finder,
                         help="Clear all search filters and restore the full facility directory.")
        if apply_clicked:
            if beds_min > beds_max:
                st.error("The minimum bed count must be less than or equal to the maximum.")
            else:
                owners = ["Government" if o == "Public" else o for o in owner_ui]
                st.session_state.finder_applied = dict(q=q, regions=regions, provinces=provinces, kinds=kinds_ui,
                                                       owners=owners, services=services, levels=levels,
                                                       only_mal=only_mal, beds=(int(beds_min), int(beds_max)))
                st.rerun()


finder_controls()
A = st.session_state.finder_applied
q, regions, provinces = A["q"], A["regions"], A["provinces"]
f = d.copy()
if q:
    ql = _norm_search(q)
    search_text = f[["name", "lgu", "lgu_province", "address"]].fillna("").astype(str).agg(" ".join, axis=1).map(_norm_search)
    f = f[search_text.str.contains(ql, regex=False)]
if regions:
    f = f[f["lgu_region"].isin(regions)]
if provinces:
    f = f[f["lgu_province"].isin(provinces)]
if A["kinds"]:
    f = f[f["kind"].isin(A["kinds"])]
if A["owners"]:
    f = f[f["ownership"].isin(A["owners"])]
for s in A["services"]:
    f = f[f["services"].str.contains(s, regex=False)]
if A["levels"]:
    f = f[f["level"].isin(A["levels"])]
if A["only_mal"]:
    f = f[f["malasakit"] == 1]
lo, hi = A["beds"]
if (lo, hi) != FDEF["beds"]:
    f = f[f["beds"].fillna(0).between(lo, hi)]

try:
    status = community_store().latest_status(24)
except Exception:          # community storage unavailable (e.g. read-only host)
    status = pd.DataFrame()
if not status.empty:
    f = f.merge(status[["facility_id", "latest_crowding", "latest_wait", "reports"]], on="facility_id", how="left")

st.markdown(f"**{len(f):,} facilities** match · {int((f['ownership'] == 'Government').sum()):,} public · "
            f"{int(f['malasakit'].sum()):,} with a Malasakit Center · {f['beds'].fillna(0).sum():,.0f} licensed beds")

c_map, c_list = st.columns([1.15, 1])
with c_map:
    m = make_map()
    if len(f) and (q or regions or provinces):
        m.fit_bounds([[f.lat.min() - 0.05, f.lon.min() - 0.05], [f.lat.max() + 0.05, f.lon.max() + 0.05]])
    else:
        m.fit_bounds(PH_BOUNDS)
    shown = f.head(2500)
    kinds_on_map = [k for k in ["Public hospital", "Private hospital", "Infirmary", "Primary care", "Malasakit Center"]]
    add_facility_markers(m, shown, kinds_on_map, clickable=True)
    add_map_control(m, "<div style='background:rgba(255,255,255,.95);border-radius:12px;padding:8px 12px;"
                       "font:12px Inter,sans-serif;box-shadow:0 6px 20px rgba(0,33,87,.15)'><b>Facility type</b>"
                    + facility_legend(kinds_on_map) + "</div>")
    st_folium(m, height=520, use_container_width=True, returned_objects=[])
    if len(f) > 2500:
        st.caption("Showing the first 2,500 on the map; narrow the search to see all.")
    st.caption("📍 Locations come from OpenStreetMap where the name matches; otherwise the pin is at the city/municipality "
               "centre (marked *approximate* in the details).")
with c_list:
    cols = ["name", "kind", "level", "ownership", "lgu", "lgu_province", "beds", "services"]
    if "latest_crowding" in f:
        cols.append("latest_crowding")
    display_f = f[cols].copy()
    display_f["ownership"] = display_f["ownership"].replace({"Government": "Public"})
    if "ownership_class" in display_f.columns:
        display_f["ownership_class"] = display_f["ownership_class"].replace({"Doh Retained": "DOH-retained", "DOH Retained": "DOH-retained"})
    show_table(display_f.rename(columns={"name": "Facility", "latest_crowding": "Reported now"}),
               hide_index=True, use_container_width=True, height=520)

if len(f):
    st.markdown("#### Facility details")
    labels = (f["name"] + " — " + f["lgu"].fillna("") + ", " + f["lgu_province"].fillna("")).tolist()
    pick = st.selectbox("Choose a facility", labels)
    r = f.iloc[labels.index(pick)]
    with st.container(border=True):
        c1, c2 = st.columns([1.3, 1])
        with c1:
            from urllib.parse import quote_plus
            destination = ", ".join(str(v) for v in [r.get("name", ""), r.get("lgu", ""), r.get("lgu_province", ""), "Philippines"] if str(v).strip() and str(v).lower() != "nan")
            map_url = "https://www.google.com/maps/search/?api=1&query=" + quote_plus(destination)
            st.markdown(f"### [{r['name']}]({map_url})")
            ownership_display = "Public" if r["ownership"] == "Government" else r["ownership"]
            ownership_class = str(r.get("ownership_class", "") or "")
            if "doh" in ownership_class.casefold() and "retain" in ownership_class.casefold():
                ownership_class = "DOH-retained"
            st.markdown(f"{r['kind']} · **{r['level']}** · {ownership_display}"
                        + (f" ({ownership_class})" if ownership_class else "")
                        + (" · 🟥 **Malasakit Center**" if r["malasakit"] else ""))
            address_text = r['address'] if isinstance(r.get('address'), str) and r['address'].strip() else "Address not listed in the source data"
            st.markdown(f"📍 **Address:** {address_text}, {r['lgu']}, {r['lgu_province']}"
                        + (" *(approximate map location: town centre)*" if str(r["geocode"]).startswith("lgu_") else ""))
            if isinstance(r.get("contact"), str) and r["contact"]:
                st.markdown(f"☎️ {r['contact']}")
            st.markdown("**Services listed:** " + (r["services"] if r["services"] else "none listed in the DOH file"))
        with c2:
            k1, k2 = st.columns(2)
            k1.metric("Licensed beds", f"{r['beds']:.0f}" if pd.notna(r["beds"]) else "—")
            k2.metric("Dialysis stations", f"{r['dialysis_stations']:.0f}" if pd.notna(r.get("dialysis_stations")) else "—")
            st.metric("Reported right now", r.get("latest_crowding") if isinstance(r.get("latest_crowding"), str)
                      else "No reports in 24 h")
            st.caption("Licensed bed capacity is the DOH-authorised number, not current occupancy. Real-time occupancy "
                       "is not published, so “reported right now” comes from community reports (unverified).")
            st.page_link("views/community.py", label="Report how busy this facility is", icon=":material/campaign:")
footer(meta)
