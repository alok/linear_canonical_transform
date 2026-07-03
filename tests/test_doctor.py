from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from lct_activation import DoctorCheck, DoctorReport, format_doctor_text, run_doctor
from lct_activation.cli import doctor_main


def test_run_doctor_passes_core_checks_with_repo_results() -> None:
    report = run_doctor(result_dir=Path("paper/results"), require_results=True)
    checks = {check.name: check for check in report.checks}

    assert report.ok
    assert checks["layer-smoke"].status == "pass"
    assert checks["spectral-frft"].status == "pass"
    assert checks["compat-import"].status == "pass"
    assert checks["result-artifacts"].status == "pass"


def test_run_doctor_warns_for_missing_optional_results(tmp_path: Path) -> None:
    report = run_doctor(result_dir=tmp_path / "missing", require_results=False)
    result_check = next(check for check in report.checks if check.name == "result-artifacts")

    assert report.ok
    assert report.has_warnings
    assert result_check.status == "warn"


def test_run_doctor_fails_for_required_missing_results(tmp_path: Path) -> None:
    report = run_doctor(result_dir=tmp_path / "missing", require_results=True)
    result_check = next(check for check in report.checks if check.name == "result-artifacts")

    assert not report.ok
    assert result_check.status == "fail"


def test_doctor_text_summary_distinguishes_warnings() -> None:
    report = DoctorReport(
        (
            DoctorCheck("core", "pass", "core passed"),
            DoctorCheck("optional", "warn", "optional missing"),
        )
    )

    text = format_doctor_text(report)

    assert "[PASS] core: core passed" in text
    assert "[WARN] optional: optional missing" in text
    assert "summary: ok with warnings" in text


def test_doctor_cli_emits_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lct-doctor",
            "--result-dir",
            "paper/results",
            "--require-results",
            "--format",
            "json",
        ],
    )

    doctor_main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert any(check["name"] == "spectral-frft" for check in payload["checks"])


def test_doctor_cli_strict_exits_on_warnings(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["lct-doctor", "--result-dir", "does-not-exist", "--strict"])

    with pytest.raises(SystemExit) as exc:
        doctor_main()

    assert exc.value.code == 1
    assert "summary: ok with warnings" in capsys.readouterr().out
