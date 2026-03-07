"""NanoGPT integration helpers for ``lct_activation``."""

from .nanogpt import (
    DEFAULT_LOCAL_NANOGPT_REPO,
    DEFAULT_UPSTREAM_NANOGPT_REPO,
    DEFAULT_UPSTREAM_NANOGPT_URL,
    NonlinearLCTActivation,
    build_local_nanogpt,
    build_upstream_nanogpt,
    ensure_upstream_nanogpt_repo,
    infer_nanogpt_repo_kind,
    load_local_nanogpt_definitions,
    patch_feedforward_class,
    patch_upstream_nanogpt,
    run_upstream_train,
)

__all__ = [
    "DEFAULT_LOCAL_NANOGPT_REPO",
    "DEFAULT_UPSTREAM_NANOGPT_REPO",
    "DEFAULT_UPSTREAM_NANOGPT_URL",
    "NonlinearLCTActivation",
    "build_local_nanogpt",
    "build_upstream_nanogpt",
    "ensure_upstream_nanogpt_repo",
    "infer_nanogpt_repo_kind",
    "load_local_nanogpt_definitions",
    "patch_feedforward_class",
    "patch_upstream_nanogpt",
    "run_upstream_train",
]
