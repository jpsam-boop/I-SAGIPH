"""I-SAGIPH: Interactive Software for Analyzing Geographic Inadequacy in Philippine Healthcare.

Run:  python -m streamlit run app/app.py
Multipage app: this entry file sets up branding, styling and the grouped top navigation;
each page lives in app/views/. Data come from data/processed/ (python -m isagiph.pipeline).
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ASSETS, inject_css  # noqa: E402

st.set_page_config(page_title="I-SAGIPH · Healthcare Gap Map", page_icon=str(ASSETS / "favicon.png"),
                   layout="wide", initial_sidebar_state="collapsed")
st.logo(str(ASSETS / "logo_wide.png"), icon_image=str(ASSETS / "logo_icon.png"), size="large")
inject_css()

V = Path(__file__).resolve().parent / "views"


def page(file, title, icon, path, **kw):
    return st.Page(V / file, title=title, icon=icon, url_path=path, **kw)


pages = {
    "": [page("home.py", "Home", ":material/home:", "home", default=True)],
    "Healthcare Access": [page("citizen.py", "Healthcare Access", ":material/volunteer_activism:", "citizen")],
    "Explore": [page("map_page.py", "Map", ":material/map:", "map"),
                page("profile.py", "LGU Profile", ":material/location_city:", "profile"),
                page("priority.py", "Priority List", ":material/format_list_numbered:", "priority"),
                page("simulator.py", "ISAGIP Simulator", ":material/add_location_alt:", "simulator")],
    "Analyze": [page("regression_explorer.py", "Regression Explorer", ":material/ssid_chart:", "insights"),
                page("relationship_analyzer.py", "Health Relationships", ":material/query_stats:", "health-relationships"),
                page("disruption.py", "Crisis Impact", ":material/hub:", "crisis-impact")],
    "Learn": [page("how_to_read.py", "How to Read", ":material/menu_book:", "how-to-read"),
              page("methods.py", "How It Works", ":material/science:", "methods"),
              page("about.py", "About", ":material/info:", "about")],
}
# Top-navigation section labels support Markdown. Remember the active section and render it bold.
_active_hint = st.session_state.get("_isagiph_active_nav_section", "")
_nav_pages = {
    (f"**{label}**" if label and label == _active_hint else label): entries
    for label, entries in pages.items()
}
navigation = st.navigation(_nav_pages, position="top")
_current_path = str(getattr(navigation, "url_path", "") or "")
_active_section = next((label for label, entries in pages.items()
                        if any(str(getattr(item, "url_path", "") or "") == _current_path for item in entries)), "")
if _active_section != _active_hint:
    st.session_state["_isagiph_active_nav_section"] = _active_section
    st.rerun()
navigation.run()
