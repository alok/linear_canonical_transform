from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _latest_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("lct_activation-*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        raise SystemExit(f"No lct_activation wheel found in {dist_dir}; run `uv build` first.")
    return wheels[-1]


def _wheel_spec(wheel: Path) -> str:
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    return f"lct-activation @ {wheel.as_uri()}#sha256={digest}"


def _run_isolated(wheel_spec: str, args: list[str]) -> str:
    command = [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--with",
        wheel_spec,
        *args,
    ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(f"isolated command failed with exit code {completed.returncode}: {' '.join(command)}")
    return completed.stdout


def smoke_dist(wheel: Path) -> dict[str, object]:
    wheel = wheel.resolve()
    wheel_spec = _wheel_spec(wheel)

    quickstart = json.loads(_run_isolated(wheel_spec, ["lct", "quickstart", "--format", "json"]))
    sweep = json.loads(
        _run_isolated(
            wheel_spec,
            [
                "lct",
                "sweep-properties",
                "--length",
                "8",
                "--angle-pair",
                "30",
                "-30",
                "--discretization",
                "spectral-frft",
                "--format",
                "json",
            ],
        )
    )
    assertion = json.loads(
        _run_isolated(
            wheel_spec,
            [
                "lct",
                "assert-properties",
                "--length",
                "8",
                "--first-angle-degrees",
                "30",
                "--second-angle-degrees",
                "-30",
            ],
        )
    )
    direct_assertion = json.loads(
        _run_isolated(
            wheel_spec,
            [
                "lct-assert-properties",
                "--length",
                "8",
                "--first-angle-degrees",
                "30",
                "--second-angle-degrees",
                "-30",
            ],
        )
    )
    metadata = json.loads(
        _run_isolated(
            wheel_spec,
            [
                "python",
                "-c",
                (
                    "import importlib.metadata as md, json;"
                    "metadata = md.metadata('lct-activation');"
                    "classifiers = metadata.get_all('Classifier') or [];"
                    "urls = metadata.get_all('Project-URL') or [];"
                    "files = [str(path) for path in (md.files('lct-activation') or [])];"
                    "license_value = metadata.get('License-Expression') or metadata.get('License');"
                    "print(json.dumps({"
                    "'license': license_value,"
                    "'classifier_ok': 'License :: OSI Approved :: Apache Software License' in classifiers,"
                    "'license_file_present': any(path.endswith('LICENSE') for path in files),"
                    "'repo_url_ok': 'Repository, https://github.com/alok/linear_canonical_transform' in urls"
                    "}))"
                ),
            ],
        )
    )
    imports = json.loads(
        _run_isolated(
            wheel_spec,
            [
                "python",
                "-c",
                (
                    "import json;"
                    "from lct_activation import LCTLinear, assess_property_report, property_report, property_sweep;"
                    "from linear_canonical_transform import LCTLinear as CompatLCTLinear;"
                    "rows = property_sweep(lengths=[8], angle_pairs_degrees=[(30.0, -30.0)], "
                    "discretizations=('spectral-frft',));"
                    "assessment = assess_property_report(property_report(8, "
                    "(0.8660254, 0.5, -0.5), (0.8660254, -0.5, 0.5), "
                    "discretization='spectral-frft'));"
                    "print(json.dumps({"
                    "'same_compat_class': LCTLinear is CompatLCTLinear,"
                    "'rows': len(rows),"
                    "'composition_error': rows[0].composition_error,"
                    "'assessment_ok': assessment.ok"
                    "}))"
                ),
            ],
        )
    )

    if quickstart.get("ok") is not True:
        raise SystemExit(f"lct quickstart failed from wheel: {quickstart}")
    if not sweep or sweep[0]["composition_error"] > 1e-5:
        raise SystemExit(f"property sweep failed from wheel: {sweep}")
    if assertion.get("ok") is not True:
        raise SystemExit(f"property assertion failed from wheel: {assertion}")
    if direct_assertion.get("ok") is not True:
        raise SystemExit(f"direct property assertion failed from wheel: {direct_assertion}")
    if (
        metadata.get("license") != "Apache-2.0"
        or metadata.get("classifier_ok") is not True
        or metadata.get("license_file_present") is not True
        or metadata.get("repo_url_ok") is not True
    ):
        raise SystemExit(f"license metadata smoke failed from wheel: {metadata}")
    if (
        imports.get("same_compat_class") is not True
        or imports.get("assessment_ok") is not True
        or imports.get("composition_error", 1.0) > 1e-5
    ):
        raise SystemExit(f"import smoke failed from wheel: {imports}")

    return {
        "wheel": str(wheel),
        "quickstart_ok": quickstart["ok"],
        "sweep_rows": len(sweep),
        "property_assert_ok": assertion["ok"],
        "direct_property_assert_ok": direct_assertion["ok"],
        "license_metadata_ok": True,
        "repo_url_ok": True,
        "spectral_composition_error": sweep[0]["composition_error"],
        "compat_import_ok": imports["same_compat_class"],
        "assessment_import_ok": imports["assessment_ok"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test a built lct-activation wheel in isolated uv runs.")
    parser.add_argument("--wheel", type=Path, help="Wheel to install. Defaults to the newest dist/lct_activation-*.whl.")
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wheel = args.wheel if args.wheel is not None else _latest_wheel(args.dist_dir)
    print(json.dumps(smoke_dist(wheel), indent=2))


if __name__ == "__main__":
    main()
