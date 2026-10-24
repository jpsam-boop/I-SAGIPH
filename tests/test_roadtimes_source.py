from pathlib import Path

import pytest

from isagiph import roadtimes


def _cfg(manual: Path, file: str = "", layer: str = ""):
    return {"paths": {"manual": manual}, "roads": {"file": file, "layer": layer}}


def test_auto_road_source_never_selects_health_facilities_gpkg(tmp_path, monkeypatch):
    manual = tmp_path / "manual"
    manual.mkdir()
    facilities = manual / "health_facilities.gpkg"
    roads = manual / "philippines_roads.gpkg"
    facilities.touch()
    roads.touch()

    # Mimic a facility points package and a valid OSM line roads package.
    monkeypatch.setattr(
        roadtimes, "_line_layer",
        lambda path: "roads" if Path(path).name == "philippines_roads.gpkg" else None,
    )
    monkeypatch.setattr(
        roadtimes, "_layer_fields",
        lambda path, layer: ["highway", "osm_id"] if Path(path).name == "philippines_roads.gpkg" else ["name"],
    )

    chosen_path, chosen_layer = roadtimes.find_roads_file(_cfg(manual))

    assert chosen_path == roads
    assert chosen_path != facilities
    assert chosen_layer == "roads"


def test_configured_roads_file_has_explicit_precedence(tmp_path, monkeypatch):
    manual = tmp_path / "manual"
    manual.mkdir()
    facilities = manual / "health_facilities.gpkg"
    roads = manual / "country_network.gpkg"
    facilities.touch()
    roads.touch()

    monkeypatch.setattr(roadtimes, "_line_layer", lambda path: "network_lines")

    chosen_path, chosen_layer = roadtimes.find_roads_file(
        _cfg(manual, file="country_network.gpkg")
    )
    assert chosen_path == roads
    assert chosen_layer == "network_lines"


def test_ambiguous_roads_sources_fail_instead_of_silently_picking_one(tmp_path, monkeypatch):
    manual = tmp_path / "manual"
    manual.mkdir()
    (manual / "roads_primary.gpkg").touch()
    (manual / "roads_secondary.gpkg").touch()
    (manual / "health_facilities.gpkg").touch()

    monkeypatch.setattr(roadtimes, "_line_layer", lambda path: "roads")
    monkeypatch.setattr(roadtimes, "_layer_fields", lambda path, layer: ["highway", "fclass"])

    with pytest.raises(ValueError, match="set roads.file"):
        roadtimes.find_roads_file(_cfg(manual))


def test_configured_missing_roads_file_fails_clearly(tmp_path):
    manual = tmp_path / "manual"
    manual.mkdir()
    (manual / "health_facilities.gpkg").touch()

    with pytest.raises(FileNotFoundError, match="Configured roads GeoPackage does not exist"):
        roadtimes.find_roads_file(_cfg(manual, file="philippines_roads.gpkg"))


def test_explicit_health_facilities_gpkg_is_rejected_for_road_routing(tmp_path):
    manual = tmp_path / "manual"
    manual.mkdir()
    facilities = manual / "health_facilities.gpkg"
    facilities.touch()
    with pytest.raises(ValueError, match="not road routing"):
        roadtimes.find_roads_file(_cfg(manual, file="health_facilities.gpkg"))
