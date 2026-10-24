"""Resource board: one place for LGU health announcements (vaccination days, medical missions, advisories)."""
import datetime as dt

import pandas as pd
import streamlit as st

from common import GLOSSARY, cascade, community_store, footer, page_header, require_data
from isagiph.community import CATEGORIES

def render(show_header: bool = True, show_footer: bool = True):
    res, fac, meta, valid, weights, gj, pts = require_data()
    store = community_store()
    if show_header:
        page_header("Resource board", "Health announcements near you")
    ICON = {"Vaccination": "💉", "Medical mission / free check-up": "🩺", "Blood drive": "🩸", "Health advisory": "📢",
            "Nutrition program": "🥗", "Dental mission": "🦷", "Mental health support": "💬", "Other": "📌"}

    tab_board, tab_post = st.tabs(["Announcements", "Post an announcement"])
    lgus = res.assign(label=res["name"] + ", " + res["province"]).sort_values("label")
    with tab_post:
        st.markdown("**Where is the activity?**")
        b_reg, b_prov, b_lgu = cascade(st.columns(3), sorted(res["region"].unique()), res, "boardpost", with_lgu=True,
                                       multi=False)
        lgu = f"{b_lgu}, {b_prov}" if b_lgu else None
        with st.form("post", clear_on_submit=True):
            cat = st.selectbox("Category", CATEGORIES, help="What kind of activity or notice this is, for example a "
                               "vaccination drive, a free check-up, a blood drive, or a health advisory.")
            title = st.text_input("Title", max_chars=100, placeholder="Free flu vaccination for senior citizens")
            details = st.text_area("Details", max_chars=1000, placeholder="Who can join, what to bring, requirements…")
            d1, d2 = st.columns(2)
            start = d1.date_input("Start date", dt.date.today())
            end = d2.date_input("End date", dt.date.today())
            venue = st.text_input("Venue", max_chars=150)
            e1, e2 = st.columns(2)
            contact = e1.text_input("Contact (office phone or email)", max_chars=100)
            source = e2.text_input("Official source link (Facebook page, LGU website)", max_chars=300, placeholder="https://")
            poster = st.text_input("Posted by (office or organisation)", max_chars=80, placeholder="e.g. City Health Office")
            ok = st.checkbox("This announcement is accurate and comes from, or is confirmed by, the LGU or organiser.")
            sent = st.form_submit_button("Post announcement", type="primary")
        if sent:
            if not (lgu and title and ok):
                st.error("Fill in the city/municipality and title, and confirm the statement.")
            elif end < start:
                st.error("End date is before the start date.")
            elif source and not source.startswith(("http://", "https://")):
                st.error("The source link must start with http:// or https://")
            else:
                r = lgus[lgus["label"] == lgu].iloc[0]
                store.add_announcement(psgc=r["psgc"], lgu=r["name"], province=r["province"], region=r["region"], category=cat,
                                       title=title.strip(), details=details.strip(), start_date=str(start), end_date=str(end),
                                       venue=venue.strip(), contact=contact.strip(), source_url=source.strip(),
                                       posted_by=poster.strip() or "Unspecified")
                st.success("Posted! It now appears on the board.")
        st.caption("A production version would require verified LGU accounts. Here, posts show who posted them and are "
                   "marked unverified unless an official source link is given.")

    with tab_board:
        a = store.announcements()
        if a.empty:
            st.info("No announcements yet. LGU health offices and organisers can post in the next tab.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            reg = c1.multiselect("Region", sorted(a["region"].unique()), placeholder="All regions", help=GLOSSARY["Region"])
            prov = c2.multiselect("Province", sorted(a.loc[a["region"].isin(reg) if reg else slice(None), "province"].unique()),
                                  placeholder="All provinces", help=GLOSSARY["Province"])
            cats = c3.multiselect("Category", CATEGORIES, placeholder="All categories",
                                  help="Show only these kinds of activities or notices.")
            past = c4.toggle("Include past events", help="Also show announcements whose end date has passed.")
            if reg:
                a = a[a["region"].isin(reg)]
            if prov:
                a = a[a["province"].isin(prov)]
            if cats:
                a = a[a["category"].isin(cats)]
            if not past:
                a = a[pd.to_datetime(a["end_date"]).dt.date >= dt.date.today()]
            st.markdown(f"**{len(a)} announcements**")
            for _, r in a.iterrows():
                with st.container(border=True):
                    s, e = pd.to_datetime(r["start_date"]).date(), pd.to_datetime(r["end_date"]).date()
                    when = s.strftime("%b %d, %Y") + ("" if s == e else " – " + e.strftime("%b %d, %Y"))
                    status = "🟢 Ongoing" if s <= dt.date.today() <= e else ("🗓 Upcoming" if s > dt.date.today() else "Past")
                    st.markdown(f"{ICON.get(r['category'], '📌')} **{r['title']}**  \n"
                                f"{r['lgu']}, {r['province']} · {r['category']} · {status} · {when}")
                    if r["details"]:
                        st.write(r["details"])
                    meta_line = " · ".join(x for x in [f"📍 {r['venue']}" if r["venue"] else "",
                                                       f"☎️ {r['contact']}" if r["contact"] else ""] if x)
                    if meta_line:
                        st.markdown(meta_line)
                    src = f"[official source]({r['source_url']})" if r["source_url"] else "unverified"
                    st.caption(f"Posted by {r['posted_by']} · {src}")
                    if st.button("Flag", key=f"flagA{r['id']}"):
                        store.flag("announcements", r["id"])
                        st.rerun()
    if show_footer:
        footer(meta)


if __name__ == "__main__":
    render()
