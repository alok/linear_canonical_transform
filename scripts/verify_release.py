from __future__ import annotations

import argparse
from email import policy
from email.message import Message
from email.parser import BytesParser
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request
import zipfile


PACKAGE_NAME = "lct-activation"
PACKAGE_STEM = "lct_activation"
EXPECTED_LICENSE = "Apache-2.0"
EXPECTED_LICENSE_CLASSIFIER = "License :: OSI Approved :: Apache Software License"
EXPECTED_REPOSITORY_URL = "https://github.com/alok/linear_canonical_transform"
EXPECTED_PROJECT_URLS = {
    "Documentation, https://github.com/alok/linear_canonical_transform#readme",
    "Repository, https://github.com/alok/linear_canonical_transform",
    "Issues, https://github.com/alok/linear_canonical_transform/issues",
}
PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
EXPECTED_SDIST_FILES = {".gitignore", "LICENSE", "PKG-INFO", "README.md", "pyproject.toml"}
MAX_SDIST_COMPRESSED_BYTES = 1_000_000
MAX_SDIST_UNPACKED_BYTES = 5_000_000


def _latest_dist_file(dist_dir: Path, pattern: str, label: str) -> Path:
    paths = sorted(dist_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not paths:
        raise ValueError(f"No {label} found in {dist_dir}; run `uv build` first.")
    return paths[-1]


def _artifact_path(provided: Path | None, dist_dir: Path, pattern: str, label: str) -> Path:
    path = provided if provided is not None else _latest_dist_file(dist_dir, pattern, label)
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _parse_metadata(content: bytes) -> Message:
    return BytesParser(policy=policy.default).parsebytes(content)


def _read_wheel(wheel: Path) -> tuple[Message, set[str]]:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_names = sorted(name for name in names if name.endswith(".dist-info/METADATA"))
        if len(metadata_names) != 1:
            raise ValueError(f"Expected exactly one wheel METADATA file in {wheel}; found {metadata_names}")
        return _parse_metadata(archive.read(metadata_names[0])), names


def _read_sdist(sdist: Path) -> tuple[Message, set[str]]:
    with tarfile.open(sdist, "r:*") as archive:
        members = archive.getmembers()
        names = {member.name for member in members}
        pkg_info_members = sorted(
            (member for member in members if member.name.endswith("/PKG-INFO")),
            key=lambda member: member.name.count("/"),
        )
        if not pkg_info_members:
            raise ValueError(f"No PKG-INFO found in source distribution {sdist}")
        file_obj = archive.extractfile(pkg_info_members[0])
        if file_obj is None:
            raise ValueError(f"Could not read {pkg_info_members[0].name} from {sdist}")
        return _parse_metadata(file_obj.read()), names


def _normalise_sdist_member(name: str) -> str:
    parts = Path(name).parts
    if len(parts) <= 1:
        return ""
    return Path(*parts[1:]).as_posix()


def _sdist_member_allowed(path: str) -> bool:
    if not path:
        return True
    if path in EXPECTED_SDIST_FILES or path == "src":
        return True
    if path == "src/lct_activation" or path.startswith("src/lct_activation/"):
        return "__pycache__" not in Path(path).parts and not path.endswith((".pyc", ".pyo"))
    if path == "src/linear_canonical_transform" or path.startswith("src/linear_canonical_transform/"):
        return "__pycache__" not in Path(path).parts and not path.endswith((".pyc", ".pyo"))
    return False


def _check_sdist_content_and_size(sdist: Path, failures: list[str]) -> dict[str, Any]:
    with tarfile.open(sdist, "r:*") as archive:
        members = archive.getmembers()

    root_entries = sorted({Path(member.name).parts[0] for member in members if member.name})
    relative_entries = sorted(
        {
            relative
            for member in members
            if (relative := _normalise_sdist_member(member.name))
        }
    )
    unexpected_entries = [path for path in relative_entries if not _sdist_member_allowed(path)]
    compressed_bytes = sdist.stat().st_size
    unpacked_bytes = sum(member.size for member in members if member.isfile())
    report = {
        "compressed_bytes": compressed_bytes,
        "max_compressed_bytes": MAX_SDIST_COMPRESSED_BYTES,
        "unpacked_bytes": unpacked_bytes,
        "max_unpacked_bytes": MAX_SDIST_UNPACKED_BYTES,
        "root_entries": root_entries,
        "unexpected_entries": unexpected_entries,
        "compressed_size_ok": compressed_bytes <= MAX_SDIST_COMPRESSED_BYTES,
        "unpacked_size_ok": unpacked_bytes <= MAX_SDIST_UNPACKED_BYTES,
        "content_ok": not unexpected_entries,
        "single_root_ok": len(root_entries) == 1,
    }

    if report["single_root_ok"] is not True:
        failures.append(f"source distribution should have exactly one root directory; found {root_entries}.")
    if report["content_ok"] is not True:
        failures.append(
            "source distribution contains unexpected paths: "
            f"{unexpected_entries[:20]}{'...' if len(unexpected_entries) > 20 else ''}."
        )
    if report["compressed_size_ok"] is not True:
        failures.append(
            f"source distribution is {compressed_bytes} bytes compressed; limit is {MAX_SDIST_COMPRESSED_BYTES}."
        )
    if report["unpacked_size_ok"] is not True:
        failures.append(
            f"source distribution expands to {unpacked_bytes} bytes; limit is {MAX_SDIST_UNPACKED_BYTES}."
        )

    return report


def _metadata_summary(metadata: Message) -> dict[str, Any]:
    return {
        "name": metadata.get("Name"),
        "version": metadata.get("Version"),
        "license": metadata.get("License-Expression") or metadata.get("License"),
        "classifiers": metadata.get_all("Classifier") or [],
        "project_urls": metadata.get_all("Project-URL") or [],
    }


def _has_license_file(names: set[str]) -> bool:
    return any(name == "LICENSE" or name.endswith("/LICENSE") for name in names)


def _check_metadata(label: str, summary: dict[str, Any], names: set[str], failures: list[str]) -> dict[str, Any]:
    classifiers = set(summary["classifiers"])
    project_urls = set(summary["project_urls"])
    report = {
        "name_ok": summary["name"] == PACKAGE_NAME,
        "license_ok": summary["license"] == EXPECTED_LICENSE,
        "license_classifier_ok": EXPECTED_LICENSE_CLASSIFIER in classifiers,
        "license_file_ok": _has_license_file(names),
        "project_urls_ok": EXPECTED_PROJECT_URLS <= project_urls,
    }

    if not report["name_ok"]:
        failures.append(f"{label} package name is {summary['name']!r}, expected {PACKAGE_NAME!r}.")
    if not summary["version"]:
        failures.append(f"{label} metadata is missing Version.")
    if not report["license_ok"]:
        failures.append(f"{label} license is {summary['license']!r}, expected {EXPECTED_LICENSE!r}.")
    if not report["license_classifier_ok"]:
        failures.append(f"{label} is missing classifier {EXPECTED_LICENSE_CLASSIFIER!r}.")
    if not report["license_file_ok"]:
        failures.append(f"{label} does not include a LICENSE file.")
    if not report["project_urls_ok"]:
        missing = sorted(EXPECTED_PROJECT_URLS - project_urls)
        failures.append(f"{label} is missing project URLs: {missing}.")

    return {**summary, **report}


def _normalise_repo_url(url: str) -> str:
    if url.endswith(".git"):
        return url[:-4]
    return url


def _git_origin_url(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"could not read git origin URL: {detail}")
    return completed.stdout.strip()


def _check_source_license(repo_root: Path, failures: list[str]) -> dict[str, Any]:
    license_path = repo_root / "LICENSE"
    report = {"path": str(license_path), "present": license_path.is_file(), "apache_2_text_ok": False}
    if not license_path.is_file():
        failures.append(f"source LICENSE file is missing at {license_path}.")
        return report

    text = license_path.read_text(encoding="utf-8", errors="replace")
    report["apache_2_text_ok"] = "Apache License" in text and "Version 2.0" in text
    if report["apache_2_text_ok"] is not True:
        failures.append(f"source LICENSE file does not look like Apache-2.0 text: {license_path}.")
    return report


def _run_smoke_dist(repo_root: Path, wheel: Path) -> dict[str, Any]:
    command = ["uv", "run", "python", "scripts/smoke_dist.py", "--wheel", str(wheel)]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0:
        raise RuntimeError(f"wheel smoke test failed with exit code {completed.returncode}: {' '.join(command)}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"wheel smoke test did not emit JSON: {exc}") from exc


def _check_pypi_version(version: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        PYPI_JSON_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": "lct-activation-release-check/0.1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            body = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {
                "checked": True,
                "url": PYPI_JSON_URL,
                "http_status": 404,
                "package_present": False,
                "current_version": version,
                "current_version_present": False,
                "version_available": True,
            }
        return {
            "checked": True,
            "url": PYPI_JSON_URL,
            "http_status": exc.code,
            "error": str(exc),
            "package_present": None,
            "current_version": version,
            "current_version_present": None,
            "version_available": False,
        }
    except urllib.error.URLError as exc:
        return {
            "checked": True,
            "url": PYPI_JSON_URL,
            "http_status": None,
            "error": str(exc),
            "package_present": None,
            "current_version": version,
            "current_version_present": None,
            "version_available": False,
        }

    if status != 200:
        return {
            "checked": True,
            "url": PYPI_JSON_URL,
            "http_status": status,
            "package_present": None,
            "current_version": version,
            "current_version_present": None,
            "version_available": False,
        }

    data = json.loads(body.decode("utf-8"))
    releases = data.get("releases") or {}
    current_version_present = version in releases
    return {
        "checked": True,
        "url": PYPI_JSON_URL,
        "http_status": status,
        "package_present": True,
        "latest_version": (data.get("info") or {}).get("version"),
        "current_version": version,
        "current_version_present": current_version_present,
        "version_available": not current_version_present,
    }


def verify_release(
    *,
    repo_root: Path,
    dist_dir: Path,
    wheel: Path | None,
    sdist: Path | None,
    check_pypi: bool,
    skip_smoke: bool,
    allow_existing_version: bool,
    pypi_timeout: float,
    tag: str | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    report: dict[str, Any] = {
        "ok": False,
        "package": PACKAGE_NAME,
        "expected_repository_url": EXPECTED_REPOSITORY_URL,
        "failures": failures,
    }

    try:
        wheel_path = _artifact_path(wheel, dist_dir, f"{PACKAGE_STEM}-*.whl", "wheel")
        sdist_path = _artifact_path(sdist, dist_dir, f"{PACKAGE_STEM}-*.tar.gz", "source distribution")
    except ValueError as exc:
        failures.append(str(exc))
        return report

    report["artifacts"] = {"wheel": str(wheel_path), "sdist": str(sdist_path)}

    try:
        wheel_metadata, wheel_files = _read_wheel(wheel_path)
        wheel_report = _check_metadata("wheel", _metadata_summary(wheel_metadata), wheel_files, failures)
        report["wheel_metadata"] = wheel_report
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        failures.append(f"could not read wheel metadata: {exc}")
        wheel_report = {}

    try:
        sdist_metadata, sdist_files = _read_sdist(sdist_path)
        sdist_report = _check_metadata("source distribution", _metadata_summary(sdist_metadata), sdist_files, failures)
        report["sdist_metadata"] = sdist_report
        report["sdist_content"] = _check_sdist_content_and_size(sdist_path, failures)
    except (OSError, ValueError, tarfile.TarError) as exc:
        failures.append(f"could not read source distribution metadata: {exc}")
        sdist_report = {}

    wheel_version = wheel_report.get("version")
    sdist_version = sdist_report.get("version")
    if wheel_version and sdist_version and wheel_version != sdist_version:
        failures.append(f"wheel version {wheel_version!r} does not match source distribution version {sdist_version!r}.")
    version = wheel_version or sdist_version
    if version:
        report["version"] = version

    if tag is None:
        report["tag"] = {"checked": False}
    elif version:
        expected_tag = f"v{version}"
        tag_matches_version = tag == expected_tag
        report["tag"] = {
            "checked": True,
            "provided": tag,
            "expected": expected_tag,
            "matches_version": tag_matches_version,
        }
        if not tag_matches_version:
            failures.append(f"release tag is {tag!r}, expected {expected_tag!r} for artifact version {version!r}.")
    else:
        report["tag"] = {
            "checked": True,
            "provided": tag,
            "expected": None,
            "matches_version": False,
        }
        failures.append(f"could not verify release tag {tag!r} because artifact metadata has no version.")

    report["source_license_file"] = _check_source_license(repo_root, failures)

    try:
        origin_url = _git_origin_url(repo_root)
        origin_ok = _normalise_repo_url(origin_url) == EXPECTED_REPOSITORY_URL
        report["git"] = {"origin_url": origin_url, "origin_ok": origin_ok}
        if not origin_ok:
            failures.append(f"git origin is {origin_url!r}, expected {EXPECTED_REPOSITORY_URL!r}.")
    except RuntimeError as exc:
        report["git"] = {"origin_url": None, "origin_ok": False}
        failures.append(str(exc))

    if skip_smoke:
        report["smoke"] = {"skipped": True}
    else:
        try:
            report["smoke"] = _run_smoke_dist(repo_root, wheel_path)
        except RuntimeError as exc:
            report["smoke"] = {"skipped": False, "error": str(exc)}
            failures.append(str(exc))

    if check_pypi:
        if version:
            pypi_report = _check_pypi_version(str(version), pypi_timeout)
            report["pypi"] = pypi_report
            version_already_uploaded = pypi_report.get("current_version_present") is True
            allowed_existing_version = allow_existing_version and version_already_uploaded
            if pypi_report.get("version_available") is not True and not allowed_existing_version:
                failures.append(
                    f"PyPI does not show {PACKAGE_NAME} {version} as uploadable; "
                    f"status report: {pypi_report}."
                )
        else:
            report["pypi"] = {"checked": False, "reason": "missing local version"}
            failures.append("could not check PyPI because local artifact metadata has no version.")
    else:
        report["pypi"] = {"checked": False}

    report["ok"] = not failures
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify local lct-activation release artifacts before publishing."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--wheel", type=Path, help="Wheel to verify. Defaults to the newest dist/lct_activation-*.whl.")
    parser.add_argument(
        "--sdist",
        type=Path,
        help="Source distribution to verify. Defaults to the newest dist/lct_activation-*.tar.gz.",
    )
    parser.add_argument("--check-pypi", action="store_true", help="Check PyPI for current-version availability.")
    parser.add_argument("--allow-existing-version", action="store_true", help="Do not fail if this version exists on PyPI.")
    parser.add_argument("--tag", help="Require this tag to equal v{artifact version}, for example v0.1.0.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the isolated wheel smoke test.")
    parser.add_argument("--pypi-timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    dist_dir = args.dist_dir if args.dist_dir.is_absolute() else repo_root / args.dist_dir
    wheel = args.wheel if args.wheel is None or args.wheel.is_absolute() else repo_root / args.wheel
    sdist = args.sdist if args.sdist is None or args.sdist.is_absolute() else repo_root / args.sdist
    report = verify_release(
        repo_root=repo_root,
        dist_dir=dist_dir,
        wheel=wheel,
        sdist=sdist,
        check_pypi=args.check_pypi,
        skip_smoke=args.skip_smoke,
        allow_existing_version=args.allow_existing_version,
        pypi_timeout=args.pypi_timeout,
        tag=args.tag,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
