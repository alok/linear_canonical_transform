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


def _write_sdist(
    path: Path,
    *,
    version: str = "0.1.0",
    license_value: str = "Apache-2.0",
    extra_entries: dict[str, bytes] | None = None,
) -> None:
    metadata = _metadata(version=version, license_value=license_value)
    license_content = b"Apache License\n"
    readme_content = b"# lct-activation\n"
    pyproject_content = b"[build-system]\nrequires = ['hatchling']\nbuild-backend = 'hatchling.build'\n"
    init_content = b"from .layers import LCTLinear\n"
    compat_content = b"from lct_activation import LCTLinear\n"
    typed_content = b""
    with tarfile.open(path, "w:gz") as archive:
        pkg_info = tarfile.TarInfo(f"lct_activation-{version}/PKG-INFO")
        pkg_info.size = len(metadata)
        archive.addfile(pkg_info, io.BytesIO(metadata))
        license_info = tarfile.TarInfo(f"lct_activation-{version}/LICENSE")
        license_info.size = len(license_content)
        archive.addfile(license_info, io.BytesIO(license_content))
        readme_info = tarfile.TarInfo(f"lct_activation-{version}/README.md")
        readme_info.size = len(readme_content)
        archive.addfile(readme_info, io.BytesIO(readme_content))
        pyproject_info = tarfile.TarInfo(f"lct_activation-{version}/pyproject.toml")
        pyproject_info.size = len(pyproject_content)
        archive.addfile(pyproject_info, io.BytesIO(pyproject_content))
        init_info = tarfile.TarInfo(f"lct_activation-{version}/src/lct_activation/__init__.py")
        init_info.size = len(init_content)
        archive.addfile(init_info, io.BytesIO(init_content))
        typed_info = tarfile.TarInfo(f"lct_activation-{version}/src/lct_activation/py.typed")
        typed_info.size = len(typed_content)
        archive.addfile(typed_info, io.BytesIO(typed_content))
        compat_init_info = tarfile.TarInfo(
            f"lct_activation-{version}/src/linear_canonical_transform/__init__.py"
        )
        compat_init_info.size = len(compat_content)
        archive.addfile(compat_init_info, io.BytesIO(compat_content))
        compat_typed_info = tarfile.TarInfo(
            f"lct_activation-{version}/src/linear_canonical_transform/py.typed"
        )
        compat_typed_info.size = len(typed_content)
        archive.addfile(compat_typed_info, io.BytesIO(typed_content))
        for relative_path, content in sorted((extra_entries or {}).items()):
            entry = tarfile.TarInfo(f"lct_activation-{version}/{relative_path}")
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))


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
        tag="v0.1.0",
    )

    assert report["ok"] is True
    assert report["version"] == "0.1.0"
    assert report["wheel_metadata"]["license_ok"] is True
    assert report["sdist_metadata"]["project_urls_ok"] is True
    assert report["sdist_content"]["content_ok"] is True
    assert report["git"]["origin_ok"] is True
    assert report["pypi"]["version_available"] is True
    assert report["tag"] == {
        "checked": True,
        "provided": "v0.1.0",
        "expected": "v0.1.0",
        "matches_version": True,
    }


def test_verify_release_fails_when_tag_does_not_match_artifact_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wheel = tmp_path / "lct_activation-0.1.0-py3-none-any.whl"
    sdist = tmp_path / "lct_activation-0.1.0.tar.gz"
    _write_wheel(wheel)
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
        tag="v0.1.1",
    )

    assert report["ok"] is False
    assert report["tag"]["matches_version"] is False
    assert any(
        "release tag is 'v0.1.1', expected 'v0.1.0' for artifact version '0.1.0'" in failure
        for failure in report["failures"]
    )


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


def test_verify_release_fails_unexpected_sdist_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wheel = tmp_path / "lct_activation-0.1.0-py3-none-any.whl"
    sdist = tmp_path / "lct_activation-0.1.0.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist, extra_entries={"site/package-lock.json": b"junk"})
    (tmp_path / "LICENSE").write_text("Apache License\nVersion 2.0\n")

    monkeypatch.setattr(verify_release, "_git_origin_url", lambda repo_root: verify_release.EXPECTED_REPOSITORY_URL)
    monkeypatch.setattr(verify_release, "_run_smoke_dist", lambda repo_root, wheel: {"quickstart_ok": True})

    report = verify_release.verify_release(
        repo_root=tmp_path,
        dist_dir=tmp_path,
        wheel=wheel,
        sdist=sdist,
        check_pypi=False,
        skip_smoke=False,
        allow_existing_version=False,
        pypi_timeout=1.0,
    )

    assert report["ok"] is False
    assert report["sdist_content"]["content_ok"] is False
    assert any("source distribution contains unexpected paths" in failure for failure in report["failures"])


def test_check_sdist_content_and_size_rejects_large_archive(tmp_path: Path) -> None:
    sdist = tmp_path / "lct_activation-0.1.0.tar.gz"
    _write_sdist(sdist)
    failures: list[str] = []

    original_limit = verify_release.MAX_SDIST_COMPRESSED_BYTES
    verify_release.MAX_SDIST_COMPRESSED_BYTES = 1
    try:
        report = verify_release._check_sdist_content_and_size(sdist, failures)
    finally:
        verify_release.MAX_SDIST_COMPRESSED_BYTES = original_limit

    assert report["compressed_size_ok"] is False
    assert any("source distribution is " in failure for failure in failures)


def test_latest_dist_file_fails_when_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run `uv build` first"):
        verify_release._latest_dist_file(tmp_path, "lct_activation-*.whl", "wheel")
