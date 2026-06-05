from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from lct_activation.cli import bench_linear_main
from lct_activation.results import collect_result_rows


def test_bench_linear_emits_json_and_writes_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "linear_bench.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lct-bench-linear",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--in-features",
            "8",
            "--out-features",
            "8",
            "--steps",
            "1",
            "--warmup-steps",
            "0",
            "--output",
            str(output),
        ],
    )

    bench_linear_main()
    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output.read_text())

    assert stdout_payload == file_payload
    assert stdout_payload["device"] == "cpu"
    assert stdout_payload["in_features"] == 8
    assert stdout_payload["out_features"] == 8
    assert isinstance(stdout_payload["lct_over_dense"], float)


def test_result_summary_reads_saved_linear_benchmark_json(tmp_path: Path) -> None:
    artifact = tmp_path / "bench_linear.json"
    artifact.write_text(
        json.dumps(
            {
                "device": "cpu",
                "batch_size": 2,
                "in_features": 8,
                "out_features": 8,
                "mode": "forward",
                "normalization": "unitary",
                "direct_fourier_backend": "fft",
                "compiled": False,
                "dense_ms": 0.1,
                "lct_ms": 0.2,
                "lct_over_dense": 2.0,
            }
        )
    )

    rows = collect_result_rows([artifact])

    assert len(rows) == 1
    assert rows[0].name == "lct-bench-linear"
    assert rows[0].lct_over_dense == 2.0
    assert rows[0].note == "forward 8x8 fft unitary"
