"""Map: the three consolidated indices plus absolute bed adequacy, with a plain-language legend, a profile
panel, and optional facility overlays. Filters only take effect when the user presses Apply."""
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from common import (GLOSSARY, LAYER_HELP, LAYERS, PH_BOUNDS, add_facility_markers, add_legend, add_lgu_layer,
                    cached_lgu_layer, cascade, facility_legend, footer, lgu_card, load_directory, make_map,
                    options_help, page_header, require_data, term)

res, fac, meta, valid, weights, gj, pts = require_data()
directory = load_directory()
page_header("Interactive map", "Explore the map",
            "Choose what to show, press Apply, then click or tap an area to see what drives its score.")

FACILITY_OPTIONS = ["Public hospital", "Malasakit Center", "Private hospital", "Infirmary", "Primary care"]
DEFAULTS = {"layer": list(LAYERS)[0], "found": None, "regions": [], "provinces": [], "overlays": [],
            "only_priority": False, "hide_unstable": False, "opacity": 0.45, "hex_km": 4}
if "map_applied" not in st.session_state:
    st.session_state.map_applied = dict(DEFAULTS)
lgu_labels = (res["name"] + ", " + res["province"]).tolist()
fragment = getattr(st, "fragment", lambda f: f)


@fragment
def controls() -> None:
    """Widgets update each other instantly (regions -> provinces) without redrawing the map."""
    with st.container(border=True):
        a = st.session_state.map_applied
        s1, s2 = st.columns([1.3, 1])
        layers = list(LAYERS)
        layer = s1.radio("Show", layers, index=layers.index(a["layer"]), horizontal=True,
                         help="\n\n".join(f"**{k}**: {v[2]}" for k, v in LAYERS.items()))
        found = s2.selectbox("Find a city or municipality", sorted(lgu_labels),
                             index=sorted(lgu_labels).index(a["found"]) if a["found"] else None,
                             placeholder="Type a name…", help="Zooms the map to the place and opens its profile.")
        regions, provinces = cascade(st.columns(2), sorted(res["region"].unique()), res, "map")
        g1, g2, g3 = st.columns([1.4, 1, 1])
        overlays = g1.multiselect("Show facilities", FACILITY_OPTIONS, default=a["overlays"], placeholder="None",
                                  disabled=directory.empty,
                                  help=options_help("Licensed facilities from the DOH lists.",
                                                    ["Hospital", "Malasakit Center", "Infirmary", "Primary care",
                                                     "Public", "Private", "Public hospital", "Private hospital"]))
        priority_switch, priority_label = g2.columns([0.22, 1])
        only_priority = priority_switch.toggle(" ", value=a["only_priority"], key="map_priority_switch",
                                               help=None, label_visibility="collapsed")
        priority_label.markdown(term("Priority areas only", "priority"), unsafe_allow_html=True)
        with g3.popover("⚙ Map settings", use_container_width=True):
            hex_km = st.slider("Hexagon size (km)", 2, 20, a["hex_km"]) if gj is None else 4
            opacity = st.slider("Color opacity", 0.15, 0.85, a["opacity"], 0.05,
                                help="Lower = more of the street map shows through.")
            hide_unstable = st.toggle("Hide unstable results", value=a["hide_unstable"], help=GLOSSARY["unstable"])
        if st.button("Apply", type="primary", use_container_width=True):
            st.session_state.map_applied = dict(layer=layer, found=found, regions=regions, provinces=provinces,
                                                overlays=overlays, only_priority=only_priority,
                                                hide_unstable=hide_unstable, opacity=opacity, hex_km=hex_km)
            st.rerun()


controls()
a = st.session_state.map_applied

view = res.copy()
if a["regions"]:
    view = view[view["region"].isin(a["regions"])]
if a["provinces"]:
    view = view[view["province"].isin(a["provinces"])]
if a["only_priority"]:
    view = view[view["priority"]]
if a["hide_unstable"]:
    view = view[~view["unstable"]]

col, cmap, desc = LAYERS[a["layer"]]
m = make_map()
target = res.iloc[lgu_labels.index(a["found"])] if a["found"] else None
if target is not None:                      # search: zoom to the chosen place and open its profile
    if target["psgc"] not in set(view["psgc"]):
        view = pd.concat([view, res[res["psgc"] == target["psgc"]]])
    m.fit_bounds([[target["lat"] - 0.12, target["lon"] - 0.12], [target["lat"] + 0.12, target["lon"] + 0.12]])
elif (a["regions"] or a["provinces"]) and len(view):
    m.fit_bounds([[view.lat.min() - 0.2, view.lon.min() - 0.2], [view.lat.max() + 0.2, view.lon.max() + 0.2]])
else:
    m.fit_bounds(PH_BOUNDS)
national = not (a["regions"] or a["provinces"] or target is not None)
if len(view):
    add_lgu_layer(m, cached_lgu_layer(tuple(view["psgc"]), col, cmap, float(a["hex_km"]), lite=national), a["opacity"],
                  placeholder=lgu_card(target) if target is not None else None)
else:
    st.info("No areas match the current filters.")
if a["overlays"] and not directory.empty:
    add_facility_markers(m, directory[directory["psgc"].isin(view["psgc"])], a["overlays"])
add_legend(m, col, cmap, facility_legend(a["overlays"]))
st_folium(m, height=640, use_container_width=True, returned_objects=[], key="mainmap")
with st.expander("What am I looking at?"):
    st.write(LAYER_HELP[a["layer"]])
    st.write("Badges in the profile run from 0 to 100. For need factors, red means more need; for capacity factors, "
             "red means less care within reach. Teal only means better than other areas, not enough care.")
footer(meta)
