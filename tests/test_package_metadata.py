from __future__ import annotations

import importlib.metadata
import importlib.resources


def test_pep561_markers_are_packaged() -> None:
    assert importlib.resources.files("lct_activation").joinpath("py.typed").is_file()
    assert importlib.resources.files("linear_canonical_transform").joinpath("py.typed").is_file()


def test_user_facing_scripts_are_registered() -> None:
    scripts = {
        entry.name
        for entry in importlib.metadata.entry_points(group="console_scripts")
        if entry.value.startswith("lct_activation.")
    }

    assert {
        "lct",
        "lct-bench-linear",
        "lct-bench-nanogpt",
        "lct-check-properties",
        "lct-doctor",
        "lct-summarize-results",
        "lct-train-nanogpt",
        "lct-tune-nanogpt",
    } <= scripts
