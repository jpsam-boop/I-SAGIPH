from isagiph.community import Store


def test_barrier_report_is_saved_anonymously_and_listed(tmp_path):
    store = Store(tmp_path / "community")
    store.add_barrier_report(
        psgc="0100100000",
        lgu="Example Municipality",
        province="Example Province",
        region="Region I",
        category="Travel or transport difficulty",
        details="The nearest facility takes a long time to reach.",
    )

    reports = store.barrier_reports()
    assert len(reports) == 1
    assert reports.iloc[0]["lgu"] == "Example Municipality"
    assert reports.iloc[0]["alias"] == "Anonymous"
    assert str(reports.iloc[0]["created_utc"].tz) == "UTC"


def test_barrier_report_can_be_flagged(tmp_path):
    store = Store(tmp_path / "community")
    store.add_barrier_report(
        psgc="0100100000", lgu="Example Municipality", province="Example Province", region="Region I",
        category="Other access barrier", details="Test report",
    )
    report_id = store.barrier_reports().iloc[0]["id"]

    store.flag("barrier_reports", report_id)

    assert store.barrier_reports().empty
