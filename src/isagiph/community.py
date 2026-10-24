"""Community reports and the resource board, stored locally in SQLite (data/community/).

Reports are user-submitted and unverified. Photos are re-encoded to strip EXIF metadata
(including GPS location) and capped in size before saving.
"""
from __future__ import annotations

import datetime as dt
import io
import sqlite3
import uuid
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
  id TEXT PRIMARY KEY, created_utc TEXT NOT NULL, facility_id TEXT, facility_name TEXT, psgc TEXT,
  crowding TEXT, wait TEXT, affordability INTEGER, out_of_pocket REAL, note TEXT, photo TEXT, alias TEXT,
  flagged INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS announcements (
  id TEXT PRIMARY KEY, created_utc TEXT NOT NULL, psgc TEXT, lgu TEXT, province TEXT, region TEXT,
  category TEXT, title TEXT, details TEXT, start_date TEXT, end_date TEXT, venue TEXT, contact TEXT,
  source_url TEXT, posted_by TEXT, flagged INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS barrier_reports (
  id TEXT PRIMARY KEY, created_utc TEXT NOT NULL, psgc TEXT, lgu TEXT, province TEXT, region TEXT,
  category TEXT, details TEXT, alias TEXT DEFAULT 'Anonymous', flagged INTEGER DEFAULT 0);
"""
CROWDING = ["Not crowded", "Busy", "Very crowded", "Full – turning patients away"]
WAIT = ["Under 30 min", "30–60 min", "1–3 hours", "Over 3 hours"]
AFFORD = {1: "Free / fully covered", 2: "Mostly covered", 3: "Some out-of-pocket", 4: "Expensive", 5: "Very expensive"}
CATEGORIES = ["Vaccination", "Medical mission / free check-up", "Blood drive", "Health advisory", "Nutrition program",
              "Dental mission", "Mental health support", "Other"]


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "photos").mkdir(parents=True, exist_ok=True)
        self.db = self.root / "isagiph_community.db"
        with self._con() as c:
            c.executescript(SCHEMA)

    def _con(self):
        return sqlite3.connect(self.db)

    @staticmethod
    def _now() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    def save_photo(self, data: bytes, max_side: int = 1280) -> str:
        from PIL import Image, ImageOps

        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        im.thumbnail((max_side, max_side))
        name = f"{uuid.uuid4().hex}.jpg"
        im.save(self.root / "photos" / name, "JPEG", quality=82)   # re-encoded: no EXIF/GPS kept
        return name

    def add_report(self, **kw) -> None:
        row = {"id": uuid.uuid4().hex, "created_utc": self._now(), **kw}
        with self._con() as c:
            c.execute(f"INSERT INTO reports ({','.join(row)}) VALUES ({','.join('?' * len(row))})", list(row.values()))

    def reports(self) -> pd.DataFrame:
        with self._con() as c:
            df = pd.read_sql("SELECT * FROM reports WHERE flagged = 0 ORDER BY created_utc DESC", c)
        df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
        return df

    def latest_status(self, hours: int = 24) -> pd.DataFrame:
        r = self.reports()
        r = r[r["created_utc"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)]
        if r.empty:
            return r
        g = r.groupby("facility_id")
        return pd.DataFrame({"reports": g.size(), "latest_crowding": g["crowding"].first(),
                             "latest_wait": g["wait"].first(), "latest_utc": g["created_utc"].first()}).reset_index()

    def add_announcement(self, **kw) -> None:
        row = {"id": uuid.uuid4().hex, "created_utc": self._now(), **kw}
        with self._con() as c:
            c.execute(f"INSERT INTO announcements ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                      list(row.values()))

    def announcements(self) -> pd.DataFrame:
        with self._con() as c:
            return pd.read_sql("SELECT * FROM announcements WHERE flagged = 0 ORDER BY start_date", c)

    def add_barrier_report(self, **kw) -> None:
        """Store a citizen-submitted access barrier; reports are unverified and not an official complaint."""
        row = {"id": uuid.uuid4().hex, "created_utc": self._now(), "alias": "Anonymous", **kw}
        allowed = {"id", "created_utc", "psgc", "lgu", "province", "region", "category", "details", "alias"}
        row = {k: v for k, v in row.items() if k in allowed}
        with self._con() as c:
            c.execute(f"INSERT INTO barrier_reports ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                      list(row.values()))

    def barrier_reports(self) -> pd.DataFrame:
        """Return unflagged barrier reports for local aggregation and audit."""
        with self._con() as c:
            df = pd.read_sql("SELECT * FROM barrier_reports WHERE flagged = 0 ORDER BY created_utc DESC", c)
        if not df.empty:
            df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
        return df

    def flag(self, table: str, row_id: str) -> None:
        assert table in ("reports", "announcements", "barrier_reports")
        with self._con() as c:
            c.execute(f"UPDATE {table} SET flagged = 1 WHERE id = ?", (row_id,))


# --------------------------------------------------------------------------- demo samples
SAMPLE_TAG = "[Sample]"
SAMPLE_REPORTS = [  # (facility name contains, crowding, wait, affordability, out-of-pocket, note, hours ago)
    ("Eastern Visayas", "Very crowded", "1–3 hours", 2, None, "ER busy after the storm; patients waiting in the hallway.", 2),
    ("Bicol Medical Center", "Full – turning patients away", "Over 3 hours", 3, 1500, "No beds in the ward; told to try another hospital.", 5),
    ("Southern Philippines Medical Center", "Busy", "30–60 min", 1, None, "Malasakit desk helped cover the bill.", 9),
    ("Basilan Medical Center", "Busy", "1–3 hours", 2, 300, "Only one doctor on duty in the OPD this morning.", 20),
    ("Ospital ng Palawan", "Not crowded", "Under 30 min", 1, None, "Quick check-up; pharmacy had our medicines.", 30),
]
SAMPLE_POSTS = [  # (LGU name, province, category, title, details, venue, days from today, length in days)
    ("Tacloban City", "Leyte", "Vaccination", "Free flu vaccines for senior citizens",
     "Bring your senior citizen ID. First come, first served.", "City Health Office", 2, 3),
    ("Naga City", "Camarines Sur", "Medical mission / free check-up", "Free check-up and medicines",
     "General consultation, blood pressure and sugar screening, free maintenance medicines while supplies last.",
     "Barangay hall", 5, 1),
    ("Puerto Princesa City", "Palawan", "Blood drive", "Bloodletting activity",
     "Donors must be 18–65 years old and weigh at least 50 kg.", "City Coliseum", 7, 1),
    ("City of Isabela", "City of Isabela (not a province)", "Health advisory", "Dengue alert: search and destroy breeding sites",
     "Cases are rising. Cover water containers and seek care early for fever lasting two days or more.",
     "All barangays", 0, 14),
]


def seed_samples(store: Store, directory: pd.DataFrame, lgus: pd.DataFrame) -> tuple[int, int]:
    """Add clearly labelled sample reports and announcements (for demos). Returns counts added."""
    now = pd.Timestamp.now(tz="UTC")
    n_r = n_a = 0
    for key, crowd, wait, aff, oop, note, hours in SAMPLE_REPORTS:
        hit = directory[directory["name"].str.contains(key, case=False, regex=False)]
        if hit.empty:
            continue
        f = hit.iloc[0]
        row = {"id": uuid.uuid4().hex, "created_utc": (now - pd.Timedelta(hours=hours)).isoformat(timespec="seconds"),
               "facility_id": f["facility_id"], "facility_name": f["name"], "psgc": f["psgc"], "crowding": crowd,
               "wait": wait, "affordability": aff, "out_of_pocket": oop, "note": f"{SAMPLE_TAG} {note}",
               "photo": None, "alias": "Sample report"}
        with store._con() as c:
            c.execute(f"INSERT INTO reports ({','.join(row)}) VALUES ({','.join('?' * len(row))})", list(row.values()))
        n_r += 1
    today = dt.date.today()
    for name, prov, cat, title, details, venue, start, days in SAMPLE_POSTS:
        hit = lgus[(lgus["name"] == name) & (lgus["province"] == prov)]
        if hit.empty:
            hit = lgus[lgus["name"].str.contains(name.replace(" City", ""), case=False, regex=False)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        store.add_announcement(psgc=r["psgc"], lgu=r["name"], province=r["province"], region=r["region"], category=cat,
                               title=f"{SAMPLE_TAG} {title}", details=details,
                               start_date=str(today + dt.timedelta(days=start)),
                               end_date=str(today + dt.timedelta(days=start + days - 1)), venue=venue, contact="",
                               source_url="", posted_by="Sample post (demo data)")
        n_a += 1
    return n_r, n_a


def remove_samples(store: Store) -> None:
    with store._con() as c:
        c.execute("DELETE FROM reports WHERE alias = 'Sample report'")
        c.execute("DELETE FROM announcements WHERE posted_by = 'Sample post (demo data)'")


if __name__ == "__main__":
    import sys

    from .config import ROOT, load_config

    cfg = load_config()
    store = Store(ROOT / "data" / "community")
    if "--remove-samples" in sys.argv:
        remove_samples(store)
        print("Sample reports and announcements removed.")
    else:
        d = pd.read_csv(cfg["paths"]["processed"] / "directory.csv", dtype={"psgc": str})
        lg = pd.read_csv(cfg["paths"]["processed"] / "lgu_results.csv", dtype={"psgc": str})
        n_r, n_a = seed_samples(store, d, lg)
        print(f"Added {n_r} sample reports and {n_a} sample announcements, all labelled {SAMPLE_TAG}. "
              "Remove them with: python -m isagiph.community --remove-samples")
