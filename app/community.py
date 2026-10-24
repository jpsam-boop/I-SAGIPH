"""Community reports: residents report how busy a facility is, waiting time, affordability, and a photo."""
import pandas as pd
import streamlit as st

from common import cascade, GLOSSARY, community_store, footer, load_directory, page_header, require_data
from isagiph.community import AFFORD, CROWDING, WAIT

def render(show_header: bool = True, show_footer: bool = True):
    res, fac, meta, valid, weights, gj, pts = require_data()
    d = load_directory()
    store = community_store()
    if show_header:
        page_header("Community reports", "How is it at the facility right now?")
    st.warning("Reports are submitted by the public and **unverified**. In an emergency, call **911** or go to the nearest "
               "facility; do not rely on these reports.", icon="⚠️")

    tab_feed, tab_new = st.tabs(["Recent reports", "Submit a report"])
    with tab_new:
        if d.empty:
            st.info("The facility directory is not available.")
        else:
            geo = d[["lgu_region", "lgu_province"]].rename(columns={"lgu_region": "region", "lgu_province": "province"})
            st.markdown("**Where is the facility?** (narrows the list below)")
            n_reg, n_prov = cascade(st.columns(2), sorted(geo["region"].dropna().unique()), geo, "repnew")
            keep = pd.Series(True, index=d.index)
            if n_reg:
                keep &= d["lgu_region"].isin(n_reg)
            if n_prov:
                keep &= d["lgu_province"].isin(n_prov)
            dd = d[keep]
            labels = (dd["name"] + " — " + dd["lgu"].fillna("") + ", " + dd["lgu_province"].fillna("")).tolist()
            with st.form("report", clear_on_submit=True):
                pick = st.selectbox("Facility", labels, index=None, placeholder="Type to search a facility…",
                                    help="The hospital, infirmary or primary care facility you visited.")
                c1, c2 = st.columns(2)
                crowd = c1.radio("How crowded is it?", CROWDING,
                                 help="How full the waiting area or emergency room was when you were there.")
                wait = c2.radio("Waiting time to be seen", WAIT,
                                help="Time from arriving until a doctor or nurse attended to you.")
                aff = st.select_slider("Affordability", options=list(AFFORD), format_func=lambda k: AFFORD[k], value=3,
                                       help="How hard the cost of care was for you, after any PhilHealth or Malasakit help.")
                oop = st.number_input("Out-of-pocket cost you paid (₱, optional)", 0, 2_000_000, 0, step=100)
                note = st.text_area("What did you see? (optional)", max_chars=500,
                                    placeholder="e.g. ER full, patients on chairs; pharmacy out of insulin")
                photo = st.file_uploader("Photo (optional, JPG/PNG, max 5 MB)", type=["jpg", "jpeg", "png"])
                alias = st.text_input("Display name (optional)", max_chars=40, placeholder="Anonymous")
                ok = st.checkbox("My photo and note show **no faces, patients, or personal information**, and this "
                                 "report is honest to the best of my knowledge.")
                sent = st.form_submit_button("Submit report", type="primary")
            if sent:
                if not pick:
                    st.error("Choose a facility.")
                elif not ok:
                    st.error("Please confirm the privacy and honesty statement.")
                elif photo is not None and photo.size > 5 * 1024 * 1024:
                    st.error("Photo is larger than 5 MB.")
                else:
                    f = dd.iloc[labels.index(pick)]
                    ph = store.save_photo(photo.getvalue()) if photo is not None else None
                    store.add_report(facility_id=f["facility_id"], facility_name=f["name"], psgc=f["psgc"], crowding=crowd,
                                     wait=wait, affordability=int(aff), out_of_pocket=float(oop) if oop else None,
                                     note=note.strip() or None, photo=ph, alias=alias.strip() or "Anonymous")
                    st.success("Thank you! Your report is now visible to others.")

    with tab_feed:
        rep = store.reports()
        if rep.empty:
            st.info("No reports yet. Be the first to share how a facility is doing.")
        else:
            rep = rep.merge(d[["facility_id", "lgu", "lgu_province", "lgu_region"]], on="facility_id", how="left") \
                if not d.empty else rep
            c1, c1b, c2, c3 = st.columns([1, 1, .8, .8])
            rgeo = rep[["lgu_region", "lgu_province"]].rename(columns={"lgu_region": "region", "lgu_province": "province"})
            reg, prov = cascade([c1, c1b], sorted(rgeo["region"].dropna().unique()), rgeo, "repfeed")
            hours = c2.selectbox("Time window", [24, 72, 168, 720], format_func=lambda h: {24: "Last 24 hours",
                                 72: "Last 3 days", 168: "Last 7 days", 720: "Last 30 days"}[h], index=2,
                                 help="Only show reports sent within this period.")
            only_photo = c3.toggle("With photos only", help="Only show reports that include a photo.")
            rep = rep[rep["created_utc"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)]
            if reg:
                rep = rep[rep["lgu_region"].isin(reg)]
            if prov:
                rep = rep[rep["lgu_province"].isin(prov)]
            if only_photo:
                rep = rep[rep["photo"].notna()]
            st.markdown(f"**{len(rep)} reports** · {(rep['crowding'].str.startswith('Full')).sum()} say the facility is full")
            colour = {"Not crowded": "🟢", "Busy": "🟡", "Very crowded": "🟠", "Full – turning patients away": "🔴"}
            for _, r in rep.head(60).iterrows():
                with st.container(border=True):
                    a, b = st.columns([3, 1.2]) if isinstance(r["photo"], str) else (st.container(), None)
                    with a:
                        ago = pd.Timestamp.now(tz="UTC") - r["created_utc"]
                        when = f"{int(ago.total_seconds() // 3600)} h ago" if ago.total_seconds() < 86400 else f"{ago.days} d ago"
                        st.markdown(f"**{r['facility_name']}** · {r.get('lgu', '') or ''}, {r.get('lgu_province', '') or ''}  \n"
                                    f"{colour.get(r['crowding'], '⚪')} **{r['crowding']}** · ⏱ {r['wait']} · "
                                    f"💸 {AFFORD.get(int(r['affordability']), '')}"
                                    + (f" (paid ₱{r['out_of_pocket']:,.0f})" if pd.notna(r["out_of_pocket"]) else ""))
                        if isinstance(r["note"], str):
                            st.write(r["note"])
                        st.caption(f"{r['alias']} · {when} · unverified")
                        if st.button("Flag as inappropriate", key=f"flag{r['id']}"):
                            store.flag("reports", r["id"])
                            st.rerun()
                    if b is not None:
                        b.image(str(store.root / "photos" / r["photo"]), use_container_width=True)
    if show_footer:
        footer(meta)


if __name__ == "__main__":
    render()
