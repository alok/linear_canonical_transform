from __future__ import annotations

import json
from pathlib import Path

from lct_activation.results import collect_result_rows, format_markdown_table


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2))


def test_collect_result_rows_from_standard_tune_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "nanogpt_example.json"
    _write_json(
        artifact,
        {
            "device": "cpu",
            "results": [
                {
                    "name": "baseline",
                    "variant": "baseline",
                    "final_val_loss": 4.0,
                    "tokens_per_second": 1000.0,
                    "parameter_count": 10,
                },
                {
                    "name": "linear-frft30",
                    "variant": "linear",
                    "final_val_loss": 3.8,
                    "tokens_per_second": 800.0,
                    "parameter_count": 8,
                },
            ],
        },
    )

    rows = collect_result_rows([artifact])

    assert [row.name for row in rows] == ["linear-frft30", "baseline"]
    assert rows[0].artifact == "nanogpt_example.json"
    assert rows[0].section == "cpu"
    assert rows[0].final_val_loss == 3.8
    assert rows[0].params == 8


def test_collect_result_rows_from_modal_artifacts(tmp_path: Path) -> None:
    gpu_artifact = tmp_path / "modal_gpu_sweep.json"
    linux_artifact = tmp_path / "modal_linux_smoke.json"
    _write_json(
        gpu_artifact,
        {
            "bench_linear_compile_fft": {
                "device": "cuda",
                "mode": "forward",
                "direct_fourier_backend": "fft",
                "lct_over_dense": 2.5,
            },
            "bench_nanogpt_compile_fft": {
                "baseline_tokens_per_second": 100.0,
                "linear_tokens_per_second": 70.0,
                "linear_over_baseline": 0.7,
            },
            "tune_nanogpt_gpu": {
                "results": [
                    {
                        "name": "linear-frft15",
                        "final_val_loss": 3.7,
                        "tokens_per_second": 50.0,
                    }
                ]
            },
        },
    )
    _write_json(
        linux_artifact,
        {
            "bench_stdout": json.dumps(
                {
                    "device": "cpu",
                    "in_features": 512,
                    "out_features": 512,
                    "lct_over_dense": 0.83,
                }
            ),
            "test_stdout": "26 passed in 6.28s",
        },
    )

    rows = collect_result_rows([gpu_artifact, linux_artifact])
    names = {row.name for row in rows}

    assert {"forward", "nanogpt-linear", "linear-frft15", "lct-bench-linear", "pytest"} <= names
    assert next(row for row in rows if row.name == "nanogpt-linear").speed_ratio == 0.7
    assert next(row for row in rows if row.name == "lct-bench-linear").lct_over_dense == 0.83


def test_collect_result_rows_from_modal_linux_object_shape() -> None:
    rows = collect_result_rows([Path("paper/results/modal_linux_smoke.json")])
    bench = next(row for row in rows if row.name == "lct-bench-linear")

    assert bench.lct_over_dense == 0.8357
    assert bench.note == "cpu 512x512"


def test_format_markdown_table() -> None:
    rows = collect_result_rows(
        [
            Path("paper/results/nanogpt_param_efficiency_mps.json"),
        ]
    )

    table = format_markdown_table(rows)

    assert "| artifact | section | name |" in table
    assert "linear64-frft30" in table
    assert "77,255" in table
