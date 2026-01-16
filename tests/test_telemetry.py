from spectator.core.telemetry import TelemetrySnapshot, collect_basic_telemetry


def test_collect_basic_telemetry_fields_present() -> None:
    snapshot = collect_basic_telemetry()

    assert isinstance(snapshot, TelemetrySnapshot)
    assert isinstance(snapshot.ts, float)
    assert isinstance(snapshot.pid, int)
    assert isinstance(snapshot.platform, str)
    assert isinstance(snapshot.python, str)


def test_collect_basic_telemetry_ram_types() -> None:
    snapshot = collect_basic_telemetry()

    assert snapshot.ram_total_mb is None or isinstance(snapshot.ram_total_mb, int)
    assert snapshot.ram_avail_mb is None or isinstance(snapshot.ram_avail_mb, int)
