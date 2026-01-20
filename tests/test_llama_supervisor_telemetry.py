from __future__ import annotations

from types import SimpleNamespace

import llama_supervisor.telemetry as telemetry


def test_read_gpu_metrics_parses_nvidia_smi(monkeypatch) -> None:
    fake = SimpleNamespace(returncode=0, stdout="24000, 12000, 12000, 67\n")
    monkeypatch.setattr(telemetry.subprocess, "run", lambda *args, **kwargs: fake)

    metrics = telemetry._read_gpu_metrics(0)
    assert metrics is not None
    assert metrics["memory_total_mb"] == 24000
    assert metrics["memory_used_mb"] == 12000
    assert metrics["memory_free_mb"] == 12000
    assert metrics["utilization_gpu"] == 67.0
