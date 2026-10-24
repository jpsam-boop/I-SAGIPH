"""Citizen-facing service navigation: find care, understand assistance, prepare, and report barriers.

This page deliberately distinguishes the project's research directory from live service status.
It does not determine eligibility for medical assistance or provide medical advice.
"""
from urllib.parse import quote_plus
import re
import html

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from common import (
    PH_BOUNDS,
    add_facility_markers,
    add_map_control,
    community_store,
    facility_legend,
    footer,
    load_directory,
    make_map,
    page_header,
    require_data,
    _canonical_services,
)
from views.community import render as render_community
from views.board import render as render_board

# Official pages are linked directly so the app does not reproduce application rules that can change.
DOH_MALASAKIT_DOH = "https://doh.gov.ph/mpo/mc-directory/doh"
DOH_MALASAKIT_LGU = "https://doh.gov.ph/mpo/mc-directory/lgu"
DSWD_MALASAKIT = "https://crisis-intervention.dswd.gov.ph/malasakit-centers/"
DOH_MAIFIP_GUIDE = "https://doh.gov.ph/sites/default/files/Process-of-Availing-Medical-Assistance.pdf"
PHILHEALTH_FACILITIES = "https://www.philhealth.gov.ph/partners/providers/facilities/accredited/"
EMERGENCY_HOTLINES = "https://ehotlines.e.gov.ph/"

res, fac, meta, valid, weights, gj, pts = require_data()
d = load_directory()
page_header("Public services", "Healthcare Access")

if meta.get("mode") == "demo":
    st.info("Demo mode is using synthetic LGU results. The facility directory and Malasakit flags require real facility data; demo model results are not findings about Philippine communities.")

# Clickable process flow. A query parameter allows Home's Community card to open this step directly.
steps = ["Find Care", "Assistance", "Prepare", "Community"]
query_step = str(st.query_params.get("step", "")).casefold()
if "health_access_step" not in st.session_state:
    st.session_state.health_access_step = steps[3] if query_step == "community" else steps[0]
flow_cols = st.columns([1, .16, 1, .16, 1, .16, 1])
for i, step in enumerate(steps):
    col = flow_cols[i * 2]
    with col:
        is_active = st.session_state.health_access_step == step
        if st.button(("● " if is_active else "○ ") + step, key=f"health_access_{i}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.health_access_step = step
            st.query_params.pop("step", None)
            st.rerun()
    if i < len(steps) - 1:
        with flow_cols[i * 2 + 1]:
            st.markdown("<div style='text-align:center;color:#8291A8;font-size:1.3rem;padding-top:8px'>→</div>", unsafe_allow_html=True)

find_tab = st.session_state.health_access_step == steps[0]
help_tab = st.session_state.health_access_step == steps[1]
checklist_tab = st.session_state.health_access_step == steps[2]
community_tab = st.session_state.health_access_step == steps[3]

barrier_tab = facility_reports_tab = announcements_tab = False   # only set when the Community step is open

# The community experience is intentionally grouped under Para sa Mamamayan:
# citizen-submitted facility reports, public health announcements and access barriers.
if community_tab:
    community_choice = st.radio("Community resources", ["Access barriers", "Facility reports", "Health announcements"], horizontal=True, key="health_access_community_choice")
    barrier_tab = community_choice == "Access barriers"
    facility_reports_tab = community_choice == "Facility reports"
    announcements_tab = community_choice == "Health announcements"

if find_tab:
    st.subheader("Find a health facility")
    st.caption("Search the directory, select a suggestion, then open a facility pin or a detail card for its address and listed services.")
    if d.empty:
        st.warning(
            "The facility directory has not been generated or is unavailable in this run. Add the required source datasets "
            "and run `python -m isagiph.build_real` to build it. For current Malasakit Center listings, use the official "
            "directories in the Assistance step."
        )
    else:
        # Guard against older/partial directories while preserving the standard project's schema.
        for col, default in {
            "name": "", "kind": "", "ownership": "", "services": "", "address": "", "contact": "",
            "lgu": "", "lgu_province": "", "lgu_region": "", "malasakit": 0, "beds": float("nan"),
            "lat": float("nan"), "lon": float("nan"), "geocode": "unknown", "facility_id": "",
        }.items():
            if col not in d.columns:
                d[col] = default
        d = d.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all").copy()
        for col in ["services", "name", "lgu", "lgu_province", "lgu_region", "address"]:
            d[col] = d[col].fillna("").astype(str)
        d["display_ownership"] = d["ownership"].replace({"Government": "Public"}).fillna("Not listed")
        bed_values = pd.to_numeric(d["beds"], errors="coerce")
        max_beds = max(0, int(bed_values.max())) if bed_values.notna().any() else 0

        def reset_citizen_filters():
            st.session_state["citizen_facility_search"] = ""
            st.session_state["citizen_suggestion"] = ""
            st.session_state["citizen_region"] = "All regions"
            st.session_state["citizen_province"] = "All provinces"
            st.session_state["citizen_kind"] = "All types"
            st.session_state["citizen_ownership"] = "Public and private"
            st.session_state["citizen_malasakit_only"] = False
            st.session_state["citizen_beds_min"] = 0
            st.session_state["citizen_beds_max"] = max_beds
            st.session_state["citizen_facility_selected"] = ""

        def clear_citizen_suggestion():
            st.session_state["citizen_suggestion"] = ""

        defaults = {
            "citizen_facility_search": "", "citizen_suggestion": "", "citizen_region": "All regions",
            "citizen_province": "All provinces", "citizen_kind": "All types",
            "citizen_ownership": "Public and private", "citizen_malasakit_only": False,
            "citizen_beds_min": 0, "citizen_beds_max": max_beds,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        st.session_state["citizen_beds_max"] = min(int(st.session_state["citizen_beds_max"]), max_beds)

        with st.container(border=True):
            search = st.text_input(
                "Search facilities, places, or services", key="citizen_facility_search",
                placeholder="Try a hospital name, city, province, dialysis, or X-ray",
                on_change=clear_citizen_suggestion,
            )
            suggestion_map = {}
            if len(search.strip()) >= 2:
                query_norm = re.sub(r"[^a-z0-9]+", " ", search.casefold()).strip()
                for _, row in d.iterrows():
                    name = str(row["name"]).strip()
                    if name and query_norm in re.sub(r"[^a-z0-9]+", " ", name.casefold()):
                        suggestion_map[f"Facility: {name} — {row['lgu']}, {row['lgu_province']}"] = ("facility", name)
                    if len(suggestion_map) >= 6:
                        break
                locs = d[["lgu", "lgu_province"]].dropna().astype(str).drop_duplicates()
                for lgu, prov in locs.itertuples(index=False, name=None):
                    label = f"Place: {lgu}, {prov}".strip(" ,")
                    blob = re.sub(r"[^a-z0-9]+", " ", f"{lgu} {prov}".casefold()).strip()
                    if query_norm in blob:
                        suggestion_map[label] = ("place", (lgu, prov))
                    if len(suggestion_map) >= 8:
                        break
                services = ["Dialysis", "X-ray / imaging", "Birthing / maternity", "Pharmacy", "Laboratory", "Dental", "Ambulance", "Blood service", "HIV testing"]
                for service in services:
                    if query_norm in service.casefold():
                        suggestion_map[f"Service: {service}"] = ("service", service)
            c_sugg, c_clear = st.columns([3, 1])
            with c_sugg:
                st.selectbox("Suggestions", [""] + list(suggestion_map), key="citizen_suggestion",
                             format_func=lambda x: "Choose a matching facility or place…" if x == "" else x,
                             help="Suggestions come from the loaded facility directory. You can also keep typing your own search.")
            with c_clear:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                st.button("Reset filters", key="citizen_reset_filters", on_click=reset_citizen_filters, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            regions = sorted(x for x in d["lgu_region"].unique() if x)
            if st.session_state["citizen_region"] not in ["All regions", *regions]:
                st.session_state["citizen_region"] = "All regions"
            region = c1.selectbox("Region", ["All regions", *regions], key="citizen_region")
            region_rows = d[d["lgu_region"].eq(region)] if region != "All regions" else d
            provinces = sorted(x for x in region_rows["lgu_province"].unique() if x)
            if st.session_state["citizen_province"] not in ["All provinces", *provinces]:
                st.session_state["citizen_province"] = "All provinces"
            province = c2.selectbox("Province", ["All provinces", *provinces], key="citizen_province")
            kind = c3.selectbox("Facility type", ["All types", "Hospital", "Infirmary", "Primary care"], key="citizen_kind")
            c4, c5 = st.columns(2)
            ownership_ui = c4.selectbox("Ownership", ["Public and private", "Public", "Private"], key="citizen_ownership")
            only_malasakit = c5.toggle("Malasakit Center listed", key="citizen_malasakit_only",
                                       help="The flag may be based on a dated source. Confirm current status using the official directory.")
            st.markdown("**Number of beds (range)**")
            bed_min_col, bed_max_col = st.columns(2)
            beds_min = bed_min_col.number_input("Minimum beds", min_value=0, max_value=max_beds, step=1, key="citizen_beds_min")
            beds_max = bed_max_col.number_input("Maximum beds", min_value=0, max_value=max_beds, step=1, key="citizen_beds_max")
            if beds_min > beds_max:
                st.warning("Minimum beds is above maximum beds. Swap the values or adjust the range.")

        f = d.copy()
        choice = st.session_state.get("citizen_suggestion", "")
        chosen = suggestion_map.get(choice) if choice else None
        if chosen:
            kind_choice, value_choice = chosen
            if kind_choice == "facility":
                f = f[f["name"].eq(value_choice)]
            elif kind_choice == "place":
                loc, prov = value_choice
                f = f[f["lgu"].eq(loc) & f["lgu_province"].eq(prov)]
            elif kind_choice == "service":
                f = f[f["services"].str.contains(str(value_choice), case=False, regex=False)]
        elif search.strip():
            q = search.strip().casefold()
            fields = ["name", "lgu", "lgu_province", "lgu_region", "address", "services"]
            keep = pd.Series(False, index=f.index)
            for col in fields:
                keep |= f[col].fillna("").astype(str).str.casefold().str.contains(q, regex=False)
            f = f[keep]
        if region != "All regions":
            f = f[f["lgu_region"].eq(region)]
        if province != "All provinces":
            f = f[f["lgu_province"].eq(province)]
        if kind != "All types":
            f = f[f["kind"].eq(kind)]
        if ownership_ui != "Public and private":
            owner_key = "Government" if ownership_ui == "Public" else ownership_ui
            f = f[f["ownership"].eq(owner_key)]
        if only_malasakit:
            f = f[pd.to_numeric(f["malasakit"], errors="coerce").fillna(0).eq(1)]
        f = f[pd.to_numeric(f["beds"], errors="coerce").fillna(0).between(int(beds_min), int(beds_max))]
        f = f.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all").copy()

        public_n = int(f["ownership"].eq("Government").sum())
        malasakit_n = int(pd.to_numeric(f["malasakit"], errors="coerce").fillna(0).eq(1).sum())
        st.markdown(f"**{len(f):,} results** · {public_n:,} public facilities · {malasakit_n:,} listed with a Malasakit Center in the source data")
        if only_malasakit:
            st.warning("The Malasakit listing may be based on an older source. Confirm current operation through the official directory before travelling.")
        if beds_min > beds_max:
            st.info("No results shown until the bed range is valid.")
            f = f.iloc[0:0]

        cmap, clist = st.columns([1.05, 1])
        with cmap:
            m = make_map()
            coords = f.dropna(subset=["lat", "lon"])
            if not coords.empty:
                m.fit_bounds([[float(coords["lat"].min()) - 0.05, float(coords["lon"].min()) - 0.05],
                              [float(coords["lat"].max()) + 0.05, float(coords["lon"].max()) + 0.05]])
            else:
                m.fit_bounds(PH_BOUNDS)
            shown = f.head(1500)
            kinds_on_map = ["Public hospital", "Private hospital", "Infirmary", "Primary care", "Malasakit Center"]
            if not shown.empty:
                add_facility_markers(m, shown, kinds_on_map, clickable=True)
            add_map_control(
                m,
                "<div style='background:rgba(255,255,255,.95);border-radius:12px;padding:8px 12px;"
                "font:12px Inter,sans-serif;box-shadow:0 6px 20px rgba(0,33,87,.15)'><b>Facility type</b>"
                + facility_legend(kinds_on_map) + "<div style='margin-top:6px;color:#42516a'>Click a marker for the facility name, address, beds, and listed services.</div></div>",
            )
            st_folium(m, height=430, use_container_width=True, returned_objects=[])
            st.caption("Some pins may mark a city/municipality centre rather than a facility entrance. Listed services and bed counts are not real-time availability.")
        with clist:
            if f.empty:
                st.info("No facilities match these filters. Try a broader search or another province.")
            else:
                table_cols = [c for c in ["name", "kind", "display_ownership", "lgu", "lgu_province", "address", "beds", "malasakit"] if c in f.columns]
                visible = f[table_cols].copy().rename(columns={"name": "Facility", "kind": "Type", "display_ownership": "Ownership",
                                                              "lgu": "City / municipality", "lgu_province": "Province",
                                                              "address": "Address", "beds": "Licensed beds", "malasakit": "Malasakit flag"})
                visible = visible.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all")
                st.dataframe(visible.head(150), hide_index=True, use_container_width=True, height=430)
                if len(f) > 150:
                    st.caption("Showing the first 150 rows. Narrow the search to see fewer facilities.")

        if not f.empty:
            st.markdown("#### Facility details")
            f = f.reset_index(drop=True)
            labels = (f["name"] + " — " + f["lgu"] + ", " + f["lgu_province"] + " · " + f["kind"]).tolist()
            if st.session_state.get("citizen_facility_selected") not in labels:
                st.session_state["citizen_facility_selected"] = labels[0]
            selected = st.selectbox("Choose a facility", labels, key="citizen_facility_selected")
            r = f.iloc[labels.index(selected)]
            with st.container(border=True):
                left, right = st.columns([1.5, 1])
                with left:
                    place = ", ".join(x for x in [str(r["name"]), str(r["lgu"]), str(r["lgu_province"]), "Philippines"] if x and x.lower() != "nan")
                    map_url = "https://www.google.com/maps/search/?api=1&query=" + quote_plus(place)
                    st.markdown(f"### [{html.escape(str(r['name']))}]({map_url})")
                    owner_display = "Public" if r["ownership"] == "Government" else str(r["ownership"])
                    st.write(f"{r['kind']} · {owner_display} · {r['lgu']}, {r['lgu_province']}")
                    address = str(r.get("address", "")).strip()
                    st.write("Address: " + (address if address and address.casefold() != "nan" else "Not listed in the source data"))
                    contact = str(r.get("contact", "")).strip()
                    if contact and contact.casefold() != "nan":
                        st.write(f"Contact listed: {contact}")
                    st.write("Services listed: " + _canonical_services(r.get("services", "")))
                    if str(r.get("geocode", "")).startswith("lgu_"):
                        st.caption("Location may use a city/municipality centre or urban core instead of the exact facility entrance.")
                with right:
                    beds = pd.to_numeric(pd.Series([r.get("beds")]), errors="coerce").iloc[0]
                    st.metric("Licensed beds", f"{beds:,.0f}" if pd.notna(beds) else "Not listed")
                    st.caption("Licensed beds do not indicate staffed beds, vacancies, or live availability.")
                    if int(pd.to_numeric(pd.Series([r.get("malasakit")]), errors="coerce").fillna(0).iloc[0]) == 1:
                        st.success("This source lists the facility as Malasakit-linked. Verify current availability using the official directory.")
                        st.link_button("Check official Malasakit directory", DOH_MALASAKIT_DOH, use_container_width=True)
                    st.link_button("Open directions / location", map_url, use_container_width=True)
        st.caption("Facility records, locations, service lists, and assistance flags can be incomplete or outdated. Confirm details with the facility before travelling.")

if help_tab:
    st.subheader("Medical and financial assistance")
    st.write(
        "Malasakit Centers are one-stop shops in participating hospitals where eligible patients can ask about government "
        "medical and financial assistance involving agencies such as DOH, DSWD, PCSO, and PhilHealth. A directory entry "
        "does not guarantee eligibility, funding, or that a service is available today; the hospital's medical social "
        "service office assesses the applicable requirements."
    )
    st.warning("Before travelling, verify that the center is currently operating at the hospital and ask what documents apply to your case. Do not pay an online stranger who promises guaranteed government medical assistance.")
    a, b = st.columns(2)
    with a:
        st.markdown("#### Find a Malasakit Center")
        st.link_button("DOH hospitals with Malasakit Centers", DOH_MALASAKIT_DOH, use_container_width=True)
        st.link_button("LGU hospitals with Malasakit Centers", DOH_MALASAKIT_LGU, use_container_width=True)
        st.link_button("DSWD: About Malasakit Centers", DSWD_MALASAKIT, use_container_width=True)
    with b:
        st.markdown("#### Other official service information")
        st.link_button("DOH medical assistance guide (MAIFIP)", DOH_MAIFIP_GUIDE, use_container_width=True)
        st.link_button("PhilHealth accredited facilities", PHILHEALTH_FACILITIES, use_container_width=True)
        st.link_button("Official emergency hotlines", EMERGENCY_HOTLINES, use_container_width=True)
    st.markdown("#### If your hospital does not have a Malasakit Center")
    st.write(
        "Ask the hospital's medical social work/service office or your regional DOH Center for Health Development "
        "which assistance process currently applies. The correct route can depend on the facility, program, and case. "
        "Use the official DOH instructions above rather than assuming every application follows the same steps."
    )
    st.caption("Official references are external links. I-SAGIPH does not process assistance applications and does not decide eligibility.")

if checklist_tab:
    st.subheader("Visit-preparation checklist")
    st.write("These are items worth asking about or preparing where applicable—not a universal or guaranteed list of requirements. Confirm the current checklist with the hospital or assistance office before going.")
    with st.container(border=True):
        st.markdown("#### Ask whether you need to bring:")
        lcol, rcol = st.columns(2)
        items_left = [
            "Patient's valid ID (and representative's ID, if applicable)",
            "Hospital bill or statement of account / cost estimate",
            "Medical abstract, certificate, or discharge summary (as applicable)",
            "Doctor's prescription, laboratory request, or procedure request",
        ]
        items_right = [
            "PhilHealth details or proof of coverage, if relevant",
            "Barangay certificate of indigency or other social assessment document, if requested",
            "Previous assistance / guarantee letters, if any",
            "Contact details of the hospital's billing or medical social service office",
        ]
        with lcol:
            for i, item in enumerate(items_left):
                st.checkbox(item, key=f"citizen_check_left_{i}")
        with rcol:
            for i, item in enumerate(items_right):
                st.checkbox(item, key=f"citizen_check_right_{i}")
    st.info("Keep original documents safe and ask the facility whether photocopies or digital copies are accepted. Never upload IDs, bills, prescriptions, or patient information into a public community report.")

if barrier_tab:
    st.subheader("Tell us about a barrier to healthcare")
    st.write("Optional community reports can help highlight issues such as transport difficulty, unavailable services, long waits, or unclear assistance information. Reports are unverified signals for further checking—not official complaints and not emergency alerts.")
    st.warning("Privacy: do not submit patient names, diagnosis, phone numbers, addresses of individuals, government ID numbers, bills, or other identifying/medical information. Reports are stored in this I-SAGIPH instance; they are not automatically sent to a government office.")
    if res.empty or not {"name", "province", "region", "psgc"}.issubset(res.columns):
        st.info("LGU reference data are not available, so barrier reporting is disabled in this run.")
    else:
        lg = res[["psgc", "name", "province", "region"]].drop_duplicates().copy()
        lg["label"] = lg["name"].astype(str) + ", " + lg["province"].astype(str) + " · " + lg["region"].astype(str)
        lg = lg.sort_values("label")
        with st.form("citizen_barrier_form", clear_on_submit=True):
            area_label = st.selectbox("City / municipality where the issue occurs", lg["label"].tolist())
            category = st.selectbox("What is the barrier?", [
                "Travel or transport difficulty", "Needed service unavailable", "Long wait or overcrowding",
                "Cost or assistance information is unclear", "Disability access or physical accessibility",
                "Facility information may be incorrect", "Other access barrier",
            ])
            details = st.text_area("Brief description (optional)", max_chars=500,
                                   placeholder="Describe the barrier without names or other identifying details.")
            privacy_ok = st.checkbox("I have not included patient names, personal contact information, IDs, or private medical details.")
            submitted = st.form_submit_button("Submit anonymous barrier report", type="primary", use_container_width=True)
        if submitted:
            if not privacy_ok:
                st.error("Please confirm that the description contains no identifying or private medical information.")
            else:
                selected_lgu = lg.loc[lg["label"].eq(area_label)].iloc[0]
                community_store().add_barrier_report(
                    psgc=str(selected_lgu["psgc"]), lgu=str(selected_lgu["name"]),
                    province=str(selected_lgu["province"]), region=str(selected_lgu["region"]),
                    category=category, details=details.strip() or None, alias="Anonymous",
                )
                st.success("Report saved to this I-SAGIPH instance. It has not been sent to a government office and may require independent verification.")
                st.rerun()

    try:
        reports = community_store().barrier_reports()
    except Exception:
        reports = pd.DataFrame()
    if not reports.empty:
        st.markdown("#### Reported barriers in this installation")
        st.caption("Unverified community reports. The display aggregates categories and locations; individual descriptions are not published here.")
        counts = reports.groupby("category", dropna=False).size().sort_values(ascending=False).rename("Reports").reset_index()
        st.dataframe(counts, hide_index=True, use_container_width=True)
        by_area = reports.groupby(["lgu", "province", "category"], dropna=False).size().sort_values(ascending=False).head(10).rename("Reports").reset_index()
        if not by_area.empty:
            st.markdown("**Areas and barrier types reported most often**")
            st.dataframe(by_area, hide_index=True, use_container_width=True)
    else:
        st.caption("No access-barrier reports have been submitted in this installation yet.")

if facility_reports_tab:
    st.subheader("Community facility reports")
    st.caption("Recent public reports about crowding, waiting times, and affordability. Reports are unverified.")
    render_community(show_header=False, show_footer=False)

if announcements_tab:
    st.subheader("Community health announcements")
    st.caption("Health activities and announcements shared by local organisations. Confirm details with the organiser.")
    render_board(show_header=False, show_footer=False)

st.markdown("---")
st.caption("Information and reports support navigation and local analysis; they do not replace medical advice, official service confirmation, or an emergency response.")
footer(meta)
