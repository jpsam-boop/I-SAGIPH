"""Download automatable sources and print instructions for manual ones.

Usage:  python -m isagiph.ingest
Every download is logged with a SHA-256 checksum and timestamp in
data/raw/MANIFEST.csv so the exact inputs of each run are traceable.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import urllib.request
from pathlib import Path

import yaml

from .config import ROOT, load_config

HDX_API = "https://data.humdata.org/api/3/action/package_show?id={}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "isagiph/0.1 (research)"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as fh:
        fh.write(r.read())
    return dest


def hdx_resources(dataset_id: str) -> list[dict]:
    with urllib.request.urlopen(HDX_API.format(dataset_id), timeout=60) as r:
        return json.load(r)["result"]["resources"]


def main() -> None:
    cfg = load_config()
    sources = yaml.safe_load(open(ROOT / "config" / "sources.yaml"))["sources"]
    data_dir = ROOT / "data"
    log = data_dir / "raw" / "MANIFEST.csv"
    new_log = not log.exists()
    with open(log, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_log:
            w.writerow(["source", "file", "url", "sha256", "retrieved_utc"])
        for key, s in sources.items():
            mode = s["mode"]
            try:
                if mode == "url":
                    dest = fetch(s["url"], data_dir / s["file"])
                    w.writerow([key, dest.name, s["url"], sha256(dest), dt.datetime.utcnow().isoformat()])
                    print(f"[ok]     {key} -> {dest}")
                elif mode == "hdx":
                    for res in hdx_resources(s["hdx_id"]):
                        dest = data_dir / "raw" / key / Path(res["url"]).name
                        fetch(res["url"], dest)
                        w.writerow([key, dest.name, res["url"], sha256(dest), dt.datetime.utcnow().isoformat()])
                        print(f"[ok]     {key} -> {dest}")
                else:
                    present = (data_dir / s["file"]).exists()
                    print(f"[{'found' if present else 'manual'}] {key}: {s.get('page', '')}"
                          f"\n          place file at data/{s['file']}  {s.get('note', '')}")
            except Exception as exc:  # keep going; report every failure
                print(f"[fail]   {key}: {exc}")
    _ = cfg


if __name__ == "__main__":
    main()
