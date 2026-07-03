from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest


_VERIFY_RELEASE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_release.py"
_SPEC = importlib.util.spec_from_file_location("verify_release", _VERIFY_RELEASE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
verify_release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify_release)


def _metadata(*, version: str = "0.1.0", license_value: str = "Apache-2.0") -> bytes:
    return (
        "Metadata-Version: 2.4\n"
        "Name: lct-activation\n"
        f"Version: {version}\n"
        f"License-Expression: {license_value}\n"
        "Classifier: License :: OSI Approved :: Apache Software License\n"
        "Project-URL: Documentation, https://github.com/alok/linear_canonical_transform#readme\n"
        "Project-URL: Repository, https://github.com/alok/linear_canonical_transform\n"
        "Project-URL: Issues, https://github.com/alok/linear_canonical_transform/issues\n"
        "\n"
    ).encode()


def _write_wheel(path: Path, *, version: str = "0.1.0", license_value: str = "Apache-2.0") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"lct_activation-{version}.dist-info/METADATA", _metadata(version=version, license_value=license_value))
        archive.writestr(f"lct_activation-{version}.dist-info/licenses/LICENSE", "Apache License\n")


def _write_sdist(path: Path, *, version: str = "0.1.0", license_value: str = "Apache-2.0") -> None:
    metadata = _metadata(version=version, license_value=license_value)
    license_content = b"Apache License\n"
    with tarfile.open(path, "w:gz") as archive:
        pkg_info = tarfile.TarInfo(f"lct_activation-{version}/PKG-INFO")
        pkg_info.size = len(metadata)
        archive.addfile(pkg_info, io.BytesIO(metadata))
        license_info = tarfile.TarInfo(f"lct_activation-{version}/LICENSE")
        license_info.size = len(license_content)
        archive.addfile(license_info, io.BytesIO(license_content))


def test_verify_release_accepts_local_artifacts_and_pypi_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wheel = tmp_path / "lct_activation-0.1.0-py3-none-any.whl"
    sdist = tmp_path / "lct_activation-0.1.0.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)
    (tmp_path / "LICENSE").write_text("Apache License\nVersion 2.0\n")

    monkeypatch.setattr(
        verify_release,
        "_git_origin_url",
        lambda repo_root: "https://github.com/alok/linear_canonical_transform.git",
    )
    monkeypatch.setattr(verify_release, "_run_smoke_dist", lambda repo_root, wheel: {"quickstart_ok": True})
    monkeypatch.setattr(
        verify_release,
        "_check_pypi_version",
        lambda version, timeout: {
            "checked": True,
            "http_status": 404,
            "package_present": False,
            "current_version": version,
            "current_version_present": False,
            "version_available": True,
        },
    )

    report = verify_release.verify_release(
        repo_root=tmp_path,
        dist_dir=tmp_path,
        wheel=wheel,
        sdist=sdist,
        check_pypi=True,
        skip_smoke=False,
        allow_existing_version=False,
        pypi_timeout=1.0,
    )

    assert report["ok"] is True
    assert report["version"] == "0.1.0"
    assert report["wheel_metadata"]["license_ok"] is True
    assert report["sdist_metadata"]["project_urls_ok"] is True
    assert report["git"]["origin_ok"] is True
    assert report["pypi"]["version_available"] is True


def test_verify_release_fails_when_pypi_version_already_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wheel = tmp_path / "lct_activation-0.1.0-py3-none-any.whl"
    sdist = tmp_path / "lct_activation-0.1.0.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)
    (tmp_path / "LICENSE").write_text("Apache License\nVersion 2.0\n")

    monkeypatch.setattr(verify_release, "_git_origin_url", lambda repo_root: verify_release.EXPECTED_REPOSITORY_URL)
    monkeypatch.setattr(verify_release, "_run_smoke_dist", lambda repo_root, wheel: {"quickstart_ok": True})
    monkeypatch.setattr(
        verify_release,
        "_check_pypi_version",
        lambda version, timeout: {
            "checked": True,
            "http_status": 200,
            "package_present": True,
            "current_version": version,
            "current_version_present": True,
            "version_available": False,
        },
    )

    report = verify_release.verify_release(
        repo_root=tmp_path,
        dist_dir=tmp_path,
        wheel=wheel,
        sdist=sdist,
        check_pypi=True,
        skip_smoke=False,
        allow_existing_version=False,
        pypi_timeout=1.0,
    )

    assert report["ok"] is False
    assert any("PyPI does not show lct-activation 0.1.0 as uploadable" in failure for failure in report["failures"])


def test_verify_release_can_allow_existing_pypi_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wheel = tmp_path / "lct_activation-0.1.0-py3-none-any.whl"
    sdist = tmp_path / "lct_activation-0.1.0.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)
    (tmp_path / "LICENSE").write_text("Apache License\nVersion 2.0\n")

    monkeypatch.setattr(verify_release, "_git_origin_url", lambda repo_root: verify_release.EXPECTED_REPOSITORY_URL)
    monkeypatch.setattr(verify_release, "_run_smoke_dist", lambda repo_root, wheel: {"quickstart_ok": True})
    monkeypatch.setattr(
        verify_release,
        "_check_pypi_version",
        lambda version, timeout: {
            "checked": True,
            "http_status": 200,
            "package_present": True,
            "current_version": version,
            "current_version_present": True,
            "version_available": False,
        },
    )

    report = verify_release.verify_release(
        repo_root=tmp_path,
        dist_dir=tmp_path,
        wheel=wheel,
        sdist=sdist,
        check_pypi=True,
        skip_smoke=False,
        allow_existing_version=True,
        pypi_timeout=1.0,
    )

    assert report["ok"] is True


def test_allow_existing_version_does_not_hide_pypi_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wheel = tmp_path / "lct_activation-0.1.0-py3-none-any.whl"
    sdist = tmp_path / "lct_activation-0.1.0.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)
    (tmp_path / "LICENSE").write_text("Apache License\nVersion 2.0\n")

    monkeypatch.setattr(verify_release, "_git_origin_url", lambda repo_root: verify_release.EXPECTED_REPOSITORY_URL)
    monkeypatch.setattr(verify_release, "_run_smoke_dist", lambda repo_root, wheel: {"quickstart_ok": True})
    monkeypatch.setattr(
        verify_release,
        "_check_pypi_version",
        lambda version, timeout: {
            "checked": True,
            "http_status": None,
            "error": "network unavailable",
            "package_present": None,
            "current_version": version,
            "current_version_present": None,
            "version_available": False,
        },
    )

    report = verify_release.verify_release(
        repo_root=tmp_path,
        dist_dir=tmp_path,
        wheel=wheel,
        sdist=sdist,
        check_pypi=True,
        skip_smoke=False,
        allow_existing_version=True,
        pypi_timeout=1.0,
    )

    assert report["ok"] is False
    assert any("PyPI does not show lct-activation 0.1.0 as uploadable" in failure for failure in report["failures"])


def test_verify_release_fails_bad_wheel_license(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wheel = tmp_path / "lct_activation-0.1.0-py3-none-any.whl"
    sdist = tmp_path / "lct_activation-0.1.0.tar.gz"
    _write_wheel(wheel, license_value="MIT")
    _write_sdist(sdist)
    (tmp_path / "LICENSE").write_text("Apache License\nVersion 2.0\n")

    monkeypatch.setattr(verify_release, "_git_origin_url", lambda repo_root: verify_release.EXPECTED_REPOSITORY_URL)

    report = verify_release.verify_release(
        repo_root=tmp_path,
        dist_dir=tmp_path,
        wheel=wheel,
        sdist=sdist,
        check_pypi=False,
        skip_smoke=True,
        allow_existing_version=False,
        pypi_timeout=1.0,
    )

    assert report["ok"] is False
    assert any("wheel license is 'MIT'" in failure for failure in report["failures"])


def test_latest_dist_file_fails_when_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run `uv build` first"):
        verify_release._latest_dist_file(tmp_path, "lct_activation-*.whl", "wheel")
