from __future__ import annotations

import json
import importlib.util
from hashlib import sha256
from pathlib import Path

import pytest

_SMOKE_DIST_PATH = Path(__file__).resolve().parents[1] / "scripts" / "smoke_dist.py"
_SPEC = importlib.util.spec_from_file_location("smoke_dist", _SMOKE_DIST_PATH)
assert _SPEC is not None and _SPEC.loader is not None
smoke_dist = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke_dist)


def test_latest_wheel_picks_newest_lct_wheel(tmp_path: Path) -> None:
    old = tmp_path / "lct_activation-0.1.0-py3-none-any.whl"
    new = tmp_path / "lct_activation-0.2.0-py3-none-any.whl"
    old.write_text("old")
    new.write_text("new")

    assert smoke_dist._latest_wheel(tmp_path) == new


def test_latest_wheel_fails_when_missing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="No lct_activation wheel"):
        smoke_dist._latest_wheel(tmp_path)


def test_wheel_spec_includes_content_hash(tmp_path: Path) -> None:
    wheel = tmp_path / "lct_activation-0.1.0-py3-none-any.whl"
    wheel.write_text("wheel")

    spec = smoke_dist._wheel_spec(wheel.resolve())

    assert spec.startswith("lct-activation @ file://")
    assert spec.endswith(f"#sha256={sha256(b'wheel').hexdigest()}")


def test_smoke_dist_validates_isolated_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wheel = tmp_path / "lct_activation-0.1.0-py3-none-any.whl"
    wheel.write_text("wheel")

    def fake_run_isolated(wheel_spec: str, args: list[str]) -> str:
        assert wheel_spec.startswith("lct-activation @ file://")
        assert "#sha256=" in wheel_spec
        if args[:3] == ["lct", "quickstart", "--format"]:
            return json.dumps({"ok": True})
        if args[:2] == ["lct", "sweep-properties"]:
            return json.dumps([{"composition_error": 1e-7}])
        if args[:2] == ["lct", "assert-properties"]:
            return json.dumps({"ok": True})
        if args[0] == "lct-assert-properties":
            return json.dumps({"ok": True})
        if args[0] == "python":
            return json.dumps(
                {
                    "same_compat_class": True,
                    "rows": 1,
                    "composition_error": 1e-7,
                    "assessment_ok": True,
                }
            )
        raise AssertionError(f"unexpected args: {args}")

    monkeypatch.setattr(smoke_dist, "_run_isolated", fake_run_isolated)

    report = smoke_dist.smoke_dist(wheel)

    assert report["quickstart_ok"] is True
    assert report["sweep_rows"] == 1
    assert report["property_assert_ok"] is True
    assert report["direct_property_assert_ok"] is True
    assert report["compat_import_ok"] is True
    assert report["assessment_import_ok"] is True
