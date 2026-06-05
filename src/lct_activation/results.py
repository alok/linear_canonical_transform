from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

__all__ = [
    "EvidenceRow",
    "collect_result_rows",
    "format_markdown_table",
    "summarize_results_main",
]


@dataclass(frozen=True)
class EvidenceRow:
    artifact: str
    section: str
    name: str
    final_val_loss: float | None = None
    tokens_per_second: float | None = None
    params: int | None = None
    speed_ratio: float | None = None
    lct_over_dense: float | None = None
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_jsonish_text(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return None
    return parsed if isinstance(parsed, dict) else None


def _row_from_trial(artifact: Path, section: str, item: dict[str, Any]) -> EvidenceRow:
    return EvidenceRow(
        artifact=artifact.name,
        section=section,
        name=str(item.get("name") or item.get("variant") or "result"),
        final_val_loss=_as_float(item.get("final_val_loss")),
        tokens_per_second=_as_float(item.get("tokens_per_second")),
        params=_as_int(item.get("params", item.get("parameter_count"))),
        note=str(item.get("variant") or ""),
    )


def _rows_from_result_array(artifact: Path, section: str, values: list[Any]) -> list[EvidenceRow]:
    rows = [_row_from_trial(artifact, section, item) for item in values if isinstance(item, dict)]
    return sorted(
        rows,
        key=lambda row: (
            float("inf") if row.final_val_loss is None else row.final_val_loss,
            row.name,
        ),
    )


def _rows_from_modal_gpu(artifact: Path, payload: dict[str, Any]) -> list[EvidenceRow]:
    rows: list[EvidenceRow] = []
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        nested_results = value.get("results")
        if isinstance(nested_results, list):
            rows.extend(_rows_from_result_array(artifact, key, nested_results))
            continue

        if "lct_over_dense" in value:
            rows.append(
                EvidenceRow(
                    artifact=artifact.name,
                    section=key,
                    name=str(value.get("mode") or "linear-bench"),
                    lct_over_dense=_as_float(value.get("lct_over_dense")),
                    note=f"{value.get('device', '')} {value.get('direct_fourier_backend', '')}".strip(),
                )
            )
        elif "linear_over_baseline" in value:
            rows.append(
                EvidenceRow(
                    artifact=artifact.name,
                    section=key,
                    name="nanogpt-linear",
                    tokens_per_second=_as_float(value.get("linear_tokens_per_second")),
                    speed_ratio=_as_float(value.get("linear_over_baseline")),
                    note=f"baseline_tps={value.get('baseline_tokens_per_second')}",
                )
            )
    return rows


def _rows_from_modal_linux(artifact: Path, payload: dict[str, Any]) -> list[EvidenceRow]:
    rows: list[EvidenceRow] = []
    bench = payload.get("bench_stdout")
    parsed = bench if isinstance(bench, dict) else _load_jsonish_text(bench) if isinstance(bench, str) else None
    if parsed is not None:
        rows.append(
            EvidenceRow(
                artifact=artifact.name,
                section="bench_stdout",
                name="lct-bench-linear",
                lct_over_dense=_as_float(parsed.get("lct_over_dense")),
                note=f"{parsed.get('device', '')} {parsed.get('in_features')}x{parsed.get('out_features')}",
            )
        )
    test_stdout = payload.get("test_stdout")
    if isinstance(test_stdout, str) and test_stdout.strip():
        rows.append(
            EvidenceRow(
                artifact=artifact.name,
                section="test_stdout",
                name="pytest",
                note=test_stdout.strip(),
            )
        )
    return rows


def _rows_from_artifact(path: Path) -> list[EvidenceRow]:
    payload = json.loads(path.read_text())

    if isinstance(payload, list):
        return _rows_from_result_array(path, "results", payload)

    if not isinstance(payload, dict):
        return []

    if path.name == "modal_gpu_sweep.json":
        return _rows_from_modal_gpu(path, payload)

    if path.name == "modal_linux_smoke.json":
        return _rows_from_modal_linux(path, payload)

    results = payload.get("results")
    if isinstance(results, list):
        section = str(payload.get("device") or "results")
        return _rows_from_result_array(path, section, results)

    return []


def collect_result_rows(paths: Iterable[Path]) -> list[EvidenceRow]:
    rows: list[EvidenceRow] = []
    for path in sorted(paths):
        if path.is_file() and path.suffix == ".json":
            rows.extend(_rows_from_artifact(path))
    return rows


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    return f"{value:.4f}"


def _markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|")


def format_markdown_table(rows: list[EvidenceRow]) -> str:
    headers = [
        "artifact",
        "section",
        "name",
        "final val loss",
        "tokens/s",
        "params",
        "speed ratio",
        "lct/dense",
        "note",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [
            row.artifact,
            row.section,
            row.name,
            _format_float(row.final_val_loss),
            _format_float(row.tokens_per_second),
            "" if row.params is None else f"{row.params:,}",
            _format_float(row.speed_ratio),
            _format_float(row.lct_over_dense),
            row.note,
        ]
        lines.append("| " + " | ".join(_markdown_escape(value) for value in values) + " |")
    return "\n".join(lines)


def _result_paths(result_dir: Path, explicit_paths: list[Path]) -> list[Path]:
    if explicit_paths:
        return explicit_paths
    return sorted(result_dir.glob("*.json"))


def parse_summarize_results_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LCT experiment JSON artifacts.")
    parser.add_argument("--result-dir", type=Path, default=Path("paper/results"))
    parser.add_argument("--path", type=Path, action="append", default=[])
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def summarize_results_main() -> None:
    args = parse_summarize_results_args()
    rows = collect_result_rows(_result_paths(args.result_dir, args.path))
    if args.format == "json":
        output = json.dumps([row.as_dict() for row in rows], indent=2)
    else:
        output = format_markdown_table(rows)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    else:
        try:
            print(output)
        except BrokenPipeError:
            pass
