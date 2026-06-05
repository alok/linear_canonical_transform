from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from lct_activation.cli import bench_linear_main, lct_main, quickstart_main
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


def test_lct_umbrella_dispatches_doctor_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lct",
            "doctor",
            "--result-dir",
            "paper/results",
            "--require-results",
            "--format",
            "json",
        ],
    )

    lct_main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert any(check["name"] == "spectral-frft" for check in payload["checks"])


def test_quickstart_emits_json_and_writes_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "quickstart.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lct-quickstart",
            "--format",
            "json",
            "--features",
            "8",
            "--property-length",
            "8",
            "--output",
            str(output),
        ],
    )

    quickstart_main()
    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output.read_text())

    assert stdout_payload == file_payload
    assert stdout_payload["ok"] is True
    assert stdout_payload["input_shape"] == [2, 8]
    assert stdout_payload["output_shape"] == [2, 8]
    assert stdout_payload["dense_equivalent_matches"] is True
    assert stdout_payload["spectral_frft"]["composition_error"] <= 1e-5


def test_lct_umbrella_dispatches_property_check(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lct",
            "check-properties",
            "--length",
            "8",
            "--first-angle-degrees",
            "30",
            "--second-angle-degrees",
            "-30",
            "--discretization",
            "spectral-frft",
        ],
    )

    lct_main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["discretization"] == "spectral-frft"
    assert payload["composition_error"] <= 1e-5


def test_lct_umbrella_dispatches_quickstart_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lct",
            "quickstart",
            "--format",
            "json",
            "--features",
            "8",
            "--property-length",
            "8",
        ],
    )

    lct_main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["dense_equivalent_matches"] is True


def test_lct_umbrella_without_subcommand_prints_help(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["lct"])

    lct_main()
    output = capsys.readouterr().out

    assert "Umbrella command for lct-activation tools." in output
    assert "check-properties" in output
    assert "quickstart" in output
    assert "Report finite-grid LCT property diagnostics." in output
    assert "summarize-results" in output
