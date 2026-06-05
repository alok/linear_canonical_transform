from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _latest_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("lct_activation-*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        raise SystemExit(f"No lct_activation wheel found in {dist_dir}; run `uv build` first.")
    return wheels[-1]


def _run_isolated(wheel: Path, args: list[str]) -> str:
    command = [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--with",
        str(wheel),
        *args,
    ]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return completed.stdout


def smoke_dist(wheel: Path) -> dict[str, object]:
    wheel = wheel.resolve()

    quickstart = json.loads(_run_isolated(wheel, ["lct", "quickstart", "--format", "json"]))
    if quickstart.get("ok") is not True:
        raise SystemExit(f"lct quickstart failed from wheel: {quickstart}")

    sweep = json.loads(
        _run_isolated(
            wheel,
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
    if not sweep or sweep[0]["composition_error"] > 1e-5:
        raise SystemExit(f"property sweep failed from wheel: {sweep}")

    imports = json.loads(
        _run_isolated(
            wheel,
            [
                "python",
                "-c",
                (
                    "import json;"
                    "from lct_activation import LCTLinear, property_sweep;"
                    "from linear_canonical_transform import LCTLinear as CompatLCTLinear;"
                    "rows = property_sweep(lengths=[8], angle_pairs_degrees=[(30.0, -30.0)], "
                    "discretizations=('spectral-frft',));"
                    "print(json.dumps({"
                    "'same_compat_class': LCTLinear is CompatLCTLinear,"
                    "'rows': len(rows),"
                    "'composition_error': rows[0].composition_error"
                    "}))"
                ),
            ],
        )
    )
    if imports.get("same_compat_class") is not True or imports.get("composition_error", 1.0) > 1e-5:
        raise SystemExit(f"import smoke failed from wheel: {imports}")

    return {
        "wheel": str(wheel),
        "quickstart_ok": quickstart["ok"],
        "sweep_rows": len(sweep),
        "spectral_composition_error": sweep[0]["composition_error"],
        "compat_import_ok": imports["same_compat_class"],
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
