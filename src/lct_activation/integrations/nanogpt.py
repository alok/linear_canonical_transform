from __future__ import annotations

import importlib
import math
import os
import runpy
import shutil
import string
import subprocess
import sys
import types
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias

import torch
from torch import Tensor, nn

DEFAULT_LOCAL_NANOGPT_REPO = Path("/Users/alokbeniwal/nanogpt")
DEFAULT_UPSTREAM_NANOGPT_REPO = Path("extern/nanoGPT")
DEFAULT_UPSTREAM_NANOGPT_URL = "https://github.com/karpathy/nanoGPT.git"

RepoKind: TypeAlias = Literal["local", "upstream"]
ActivationFactory: TypeAlias = Callable[[], nn.Module]

_LOCAL_STOP_MARKERS = (
    "\ngpt = GPT()",
    "\ngpt = GPT(",
)
_PATCH_ATTR = "__lct_activation_patched__"
_ORIGINAL_INIT_ATTR = "__lct_activation_original_init__"


class NonlinearLCTActivation(nn.Module):
    """LCT-style spectral activation used to replace NanoGPT's MLP nonlinearity.

    The Linear Canonical Transform is linear, so the actual nonlinearity here
    comes from applying a sigmoid gate to the imaginary response of a simple
    chirp-FFT-chirp pipeline and then mixing it back with the residual stream.
    This keeps the module lightweight while still injecting an LCT-flavoured
    transformation across the hidden dimension.
    """

    def __init__(
        self,
        pre_chirp: float = 0.35,
        post_chirp: float = -0.2,
        gate_scale: float = 1.0,
        gate_bias: float = 0.0,
        residual_mix: float = 0.25,
    ) -> None:
        super().__init__()
        residual_mix = min(max(float(residual_mix), 1e-4), 1.0 - 1e-4)
        self.pre_chirp = nn.Parameter(torch.tensor(float(pre_chirp)))
        self.post_chirp = nn.Parameter(torch.tensor(float(post_chirp)))
        self.gate_scale = nn.Parameter(torch.tensor(float(gate_scale)))
        self.gate_bias = nn.Parameter(torch.tensor(float(gate_bias)))
        self.residual_logit = nn.Parameter(
            torch.tensor(math.log(residual_mix / (1.0 - residual_mix)))
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] < 2:
            return x

        original_dtype = x.dtype
        work_dtype = torch.float64 if original_dtype == torch.float64 else torch.float32
        real_x = x.to(dtype=work_dtype)

        coords = torch.linspace(
            -1.0,
            1.0,
            real_x.shape[-1],
            device=real_x.device,
            dtype=work_dtype,
        )
        phase_scale = torch.tensor(
            math.pi,
            device=real_x.device,
            dtype=work_dtype,
        )
        pre_angle = phase_scale * self.pre_chirp.to(real_x.device, work_dtype) * coords.square()
        post_angle = phase_scale * self.post_chirp.to(real_x.device, work_dtype) * coords.square()

        pre_kernel = torch.polar(torch.ones_like(pre_angle), pre_angle)
        post_kernel = torch.polar(torch.ones_like(post_angle), post_angle)

        complex_dtype = torch.complex128 if work_dtype == torch.float64 else torch.complex64
        signal = real_x.to(dtype=complex_dtype)
        transformed = torch.fft.ifft(
            torch.fft.fft(signal * pre_kernel, dim=-1, norm="ortho") * post_kernel,
            dim=-1,
            norm="ortho",
        )

        gate = torch.sigmoid(
            self.gate_scale.to(real_x.device, work_dtype) * transformed.imag
            + self.gate_bias.to(real_x.device, work_dtype)
        )
        spectral_response = transformed.real * gate
        residual_mix = torch.sigmoid(self.residual_logit).to(real_x.device, work_dtype)
        mixed = residual_mix * real_x + (1.0 - residual_mix) * spectral_response
        return mixed.to(dtype=original_dtype)


def infer_nanogpt_repo_kind(repo_dir: Path | str) -> RepoKind:
    repo_dir = Path(repo_dir)
    if (repo_dir / "train.py").exists() and (repo_dir / "model.py").exists():
        return "upstream"
    if (repo_dir / "nanogpt" / "__init__.py").exists():
        return "local"
    raise FileNotFoundError(
        f"Could not infer NanoGPT repo kind from {repo_dir}. "
        "Expected either train.py/model.py or nanogpt/__init__.py."
    )


def ensure_upstream_nanogpt_repo(
    repo_dir: Path | str = DEFAULT_UPSTREAM_NANOGPT_REPO,
    *,
    clone_if_missing: bool = False,
    repo_url: str = DEFAULT_UPSTREAM_NANOGPT_URL,
) -> Path:
    repo_dir = Path(repo_dir)
    if repo_dir.exists():
        return repo_dir
    if not clone_if_missing:
        raise FileNotFoundError(
            f"NanoGPT repo {repo_dir} does not exist. Pass clone_if_missing=True to clone it."
        )
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
        check=True,
    )
    return repo_dir


def _add_repo_to_syspath(repo_dir: Path) -> None:
    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _replace_activation(module: nn.Module, activation: nn.Module) -> None:
    if isinstance(getattr(module, "gelu", None), nn.Module):
        module.gelu = activation
        return

    for attr in ("activation", "act", "nonlinearity"):
        current = getattr(module, attr, None)
        if isinstance(current, nn.Module):
            setattr(module, attr, activation)
            return

    net = getattr(module, "net", None)
    if isinstance(net, nn.Sequential):
        for idx, layer in enumerate(net):
            if isinstance(
                layer,
                (
                    nn.GELU,
                    nn.ReLU,
                    nn.SiLU,
                    nn.Tanh,
                    nn.LeakyReLU,
                ),
            ):
                net[idx] = activation
                return

    raise RuntimeError(f"Could not find a replaceable activation inside {type(module).__name__}.")


def patch_feedforward_class(
    feedforward_cls: type[nn.Module],
    activation_factory: ActivationFactory = NonlinearLCTActivation,
) -> type[nn.Module]:
    if getattr(feedforward_cls, _PATCH_ATTR, False):
        return feedforward_cls

    original_init = feedforward_cls.__init__

    @wraps(original_init)
    def patched_init(self: nn.Module, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        _replace_activation(self, activation_factory())

    setattr(feedforward_cls, _ORIGINAL_INIT_ATTR, original_init)
    feedforward_cls.__init__ = patched_init
    setattr(feedforward_cls, _PATCH_ATTR, True)
    return feedforward_cls


def patch_upstream_nanogpt(
    repo_dir: Path | str = DEFAULT_UPSTREAM_NANOGPT_REPO,
    activation_factory: ActivationFactory = NonlinearLCTActivation,
) -> Any:
    repo_dir = Path(repo_dir)
    if infer_nanogpt_repo_kind(repo_dir) != "upstream":
        raise ValueError(f"{repo_dir} is not an upstream NanoGPT checkout.")

    _add_repo_to_syspath(repo_dir)
    importlib.invalidate_caches()
    model_module = importlib.import_module("model")
    patch_feedforward_class(model_module.MLP, activation_factory)
    return model_module


def build_upstream_nanogpt(
    repo_dir: Path | str = DEFAULT_UPSTREAM_NANOGPT_REPO,
    *,
    use_lct: bool = False,
    activation_factory: ActivationFactory = NonlinearLCTActivation,
    block_size: int = 256,
    vocab_size: int = 50304,
    n_layer: int = 12,
    n_head: int = 12,
    n_embd: int = 768,
    dropout: float = 0.0,
    bias: bool = True,
    device: torch.device | None = None,
) -> tuple[nn.Module, Any]:
    repo_dir = Path(repo_dir)
    if infer_nanogpt_repo_kind(repo_dir) != "upstream":
        raise ValueError(f"{repo_dir} is not an upstream NanoGPT checkout.")

    if use_lct:
        model_module = patch_upstream_nanogpt(repo_dir, activation_factory)
    else:
        _add_repo_to_syspath(repo_dir)
        importlib.invalidate_caches()
        model_module = importlib.import_module("model")

    config = model_module.GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
    )
    model = model_module.GPT(config)
    if device is not None:
        model = model.to(device)
    return model, model_module


def _make_bidict_shim() -> types.ModuleType:
    module = types.ModuleType("bidict")

    class MiniBidict(dict):
        @property
        def inv(self) -> dict[str, int]:
            return {value: key for key, value in self.items()}

    module.bidict = MiniBidict
    return module


def _make_jaxtyping_shim() -> types.ModuleType:
    module = types.ModuleType("jaxtyping")

    class _TensorAlias:
        def __class_getitem__(cls, _item: Any) -> type[Tensor]:
            return Tensor

    module.Float = _TensorAlias
    module.Integer = _TensorAlias
    return module


def _make_tqdm_shim() -> types.ModuleType:
    module = types.ModuleType("tqdm")
    module.trange = range
    return module


def _make_einops_shim() -> dict[str, types.ModuleType]:
    einops = types.ModuleType("einops")
    layers = types.ModuleType("einops.layers")
    layers_torch = types.ModuleType("einops.layers.torch")

    def rearrange(tensor: Tensor, pattern: str, **sizes: int) -> Tensor:
        pattern = " ".join(pattern.split())
        if pattern == "t t2 -> 1 1 t t2":
            return tensor.unsqueeze(0).unsqueeze(0)
        if pattern == "b t (nh c) -> b nh t c":
            bsz, tsz, width = tensor.shape
            nh = sizes["nh"]
            if width % nh != 0:
                raise ValueError(f"Cannot split hidden size {width} into {nh} heads.")
            return tensor.view(bsz, tsz, nh, width // nh).permute(0, 2, 1, 3)
        if pattern == "b nh t c -> b t (nh c)":
            bsz, nh, tsz, csz = tensor.shape
            return tensor.permute(0, 2, 1, 3).reshape(bsz, tsz, nh * csz)
        if pattern == "b t vocab -> (b t) vocab":
            bsz, tsz, vocab = tensor.shape
            return tensor.reshape(bsz * tsz, vocab)
        if pattern == "b t -> (b t)":
            return tensor.reshape(-1)
        raise NotImplementedError(f"Unsupported rearrange pattern in shim: {pattern}")

    def einsum(*args: Any) -> Tensor:
        *operands, pattern = args
        lhs, rhs = pattern.split("->")
        input_terms = [term.strip().split() for term in lhs.split(",")]
        output_terms = rhs.strip().split()
        token_map: dict[str, str] = {}
        letters = iter(string.ascii_letters)

        def encode(tokens: list[str]) -> str:
            encoded: list[str] = []
            for token in tokens:
                if token not in token_map:
                    token_map[token] = next(letters)
                encoded.append(token_map[token])
            return "".join(encoded)

        equation = ",".join(encode(term) for term in input_terms)
        equation = f"{equation}->{encode(output_terms)}"
        return torch.einsum(equation, *operands)

    def _unsupported(*_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError("This local NanoGPT shim only implements the ops used by the benchmark.")

    class _IdentityLayer(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return x

    einops.einsum = einsum
    einops.pack = _unsupported
    einops.rearrange = rearrange
    einops.reduce = _unsupported
    einops.repeat = _unsupported
    einops.unpack = _unsupported
    einops.layers = layers
    layers.torch = layers_torch
    layers_torch.Rearrange = _IdentityLayer
    layers_torch.Reduce = _IdentityLayer
    return {
        "einops": einops,
        "einops.layers": layers,
        "einops.layers.torch": layers_torch,
    }


@contextmanager
def _local_nanogpt_import_shims() -> Any:
    injected: list[str] = []

    def maybe_inject(name: str, module: types.ModuleType) -> None:
        if name in sys.modules:
            return
        try:
            importlib.import_module(name)
        except ImportError:
            sys.modules[name] = module
            injected.append(name)

    maybe_inject("bidict", _make_bidict_shim())
    maybe_inject("jaxtyping", _make_jaxtyping_shim())
    maybe_inject("tyro", types.ModuleType("tyro"))
    maybe_inject("tqdm", _make_tqdm_shim())
    try:
        importlib.import_module("einops")
    except ImportError:
        for name, module in _make_einops_shim().items():
            if name not in sys.modules:
                sys.modules[name] = module
                injected.append(name)

    try:
        yield
    finally:
        for name in reversed(injected):
            sys.modules.pop(name, None)


@contextmanager
def _temporary_chdir(path: Path) -> Any:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _patched_urlretrieve(repo_dir: Path) -> Any:
    import urllib.request

    original = urllib.request.urlretrieve
    local_input = repo_dir / "input.txt"

    def local_first(url: str, filename: str | os.PathLike[str] | None = None, *args: Any, **kwargs: Any):
        target = Path(filename) if filename is not None else local_input
        if not target.is_absolute():
            target = repo_dir / target
        if target.exists():
            return str(target), None
        if local_input.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_input, target)
            return str(target), None
        return original(url, str(target), *args, **kwargs)

    urllib.request.urlretrieve = local_first
    try:
        yield
    finally:
        urllib.request.urlretrieve = original


def _slice_local_nanogpt_source(source: str) -> str:
    for marker in _LOCAL_STOP_MARKERS:
        idx = source.find(marker)
        if idx != -1:
            return source[:idx]
    raise RuntimeError("Could not find the local NanoGPT training-loop marker.")


def _patch_local_gpt_forward(namespace: dict[str, Any]) -> None:
    gpt_cls = namespace["GPT"]
    if getattr(gpt_cls, "__lct_forward_patched__", False):
        return

    functional = namespace["F"]
    rearrange = namespace["rearrange"]

    def forward(self: nn.Module, idxs: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        _batch_size, seq_len = idxs.shape
        positions = torch.arange(seq_len, device=idxs.device)
        x = self.tok_emb(idxs) + self.pos_emb(positions)

        x = self.blocks(x)
        x = self.final_ln(x)
        logits = self.lang_head(x)

        if targets is None:
            loss = None
        else:
            loss = functional.cross_entropy(
                rearrange(logits, "b t vocab -> (b t) vocab"),
                rearrange(targets, "b t -> (b t)"),
            )
        return logits, loss

    gpt_cls.forward = forward
    gpt_cls.__lct_forward_patched__ = True


def _patch_local_multihead_init(namespace: dict[str, Any]) -> None:
    multihead_cls = namespace["MultiHead"]
    if getattr(multihead_cls, "__lct_init_patched__", False):
        return

    original_init = multihead_cls.__init__

    @wraps(original_init)
    def patched_init(self: nn.Module, *args: Any, **kwargs: Any) -> None:
        head_size = kwargs.get("head_size")
        n_heads = kwargs.get("n_heads", namespace["args"].N_HEADS)

        if len(args) >= 1:
            head_size = args[0]
        if len(args) >= 2 or "embed_dim" in kwargs:
            result = original_init(self, *args, **kwargs)
            self.mask = self.mask.to(dtype=torch.bool)
            return result
        if len(args) >= 3:
            n_heads = args[2]

        if head_size is None:
            head_size = namespace["args"].HEAD_SIZE
        kwargs["embed_dim"] = int(head_size) * int(n_heads)
        result = original_init(self, *args, **kwargs)
        self.mask = self.mask.to(dtype=torch.bool)
        return result

    multihead_cls.__init__ = patched_init
    multihead_cls.__lct_init_patched__ = True


def load_local_nanogpt_definitions(
    repo_dir: Path | str = DEFAULT_LOCAL_NANOGPT_REPO,
    *,
    use_lct: bool = False,
    activation_factory: ActivationFactory = NonlinearLCTActivation,
) -> dict[str, Any]:
    repo_dir = Path(repo_dir)
    source_path = repo_dir / "nanogpt" / "__init__.py"
    if not source_path.exists():
        raise FileNotFoundError(f"Expected local NanoGPT module at {source_path}.")

    source = source_path.read_text()
    pre_training_source = _slice_local_nanogpt_source(source)
    module_name = "nanogpt_local_defs"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    module_globals = module.__dict__
    sys.modules[module_name] = module

    try:
        with (
            _temporary_chdir(repo_dir),
            _patched_urlretrieve(repo_dir),
            _local_nanogpt_import_shims(),
        ):
            exec(compile(pre_training_source, str(source_path), "exec"), module_globals)
    finally:
        sys.modules.pop(module_name, None)

    _patch_local_multihead_init(module_globals)
    _patch_local_gpt_forward(module_globals)
    if use_lct:
        patch_feedforward_class(module_globals["FeedForward"], activation_factory)
    return module_globals


def configure_local_nanogpt(
    namespace: dict[str, Any],
    *,
    batch_size: int | None = None,
    ctx_len: int | None = None,
    n_heads: int | None = None,
    embed_dim: int | None = None,
    n_layers: int | None = None,
    drop_frac: float | None = None,
) -> Any:
    args = namespace["args"]
    if batch_size is not None:
        args.BATCH_SIZE = int(batch_size)
    if ctx_len is not None:
        args.CTX_LEN = int(ctx_len)
    if n_heads is not None:
        args.N_HEADS = int(n_heads)
    if embed_dim is not None:
        args.EMBED_DIM = int(embed_dim)
    if n_layers is not None:
        args.N_LAYERS = int(n_layers)
    if drop_frac is not None:
        args.DROP_FRAC = float(drop_frac)

    if embed_dim is not None or n_heads is not None:
        if args.EMBED_DIM % args.N_HEADS != 0:
            raise ValueError(
                f"embed_dim={args.EMBED_DIM} must be divisible by n_heads={args.N_HEADS}."
            )
        args.HEAD_SIZE = args.EMBED_DIM // args.N_HEADS
    return args


def build_local_nanogpt(
    repo_dir: Path | str = DEFAULT_LOCAL_NANOGPT_REPO,
    *,
    use_lct: bool = False,
    activation_factory: ActivationFactory = NonlinearLCTActivation,
    batch_size: int = 8,
    ctx_len: int = 128,
    n_heads: int = 8,
    embed_dim: int = 512,
    n_layers: int = 3,
    drop_frac: float = 0.0,
    vocab_size: int | None = None,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    namespace = load_local_nanogpt_definitions(
        repo_dir,
        use_lct=use_lct,
        activation_factory=activation_factory,
    )
    args = configure_local_nanogpt(
        namespace,
        batch_size=batch_size,
        ctx_len=ctx_len,
        n_heads=n_heads,
        embed_dim=embed_dim,
        n_layers=n_layers,
        drop_frac=drop_frac,
    )
    GPT = namespace["GPT"]
    model = GPT(
        vocab_size=int(vocab_size or namespace.get("VOCAB_SIZE", 65)),
        embed_dim=args.EMBED_DIM,
        n_heads=args.N_HEADS,
    )
    if device is not None:
        model = model.to(device)
    return model, namespace


def run_upstream_train(
    repo_dir: Path | str = DEFAULT_UPSTREAM_NANOGPT_REPO,
    *,
    train_argv: list[str] | tuple[str, ...] = (),
    clone_if_missing: bool = False,
    repo_url: str = DEFAULT_UPSTREAM_NANOGPT_URL,
    activation_factory: ActivationFactory = NonlinearLCTActivation,
) -> None:
    repo_dir = ensure_upstream_nanogpt_repo(
        repo_dir,
        clone_if_missing=clone_if_missing,
        repo_url=repo_url,
    )
    if infer_nanogpt_repo_kind(repo_dir) != "upstream":
        raise ValueError(
            f"{repo_dir} does not look like an upstream NanoGPT checkout with train.py/model.py."
        )

    patch_upstream_nanogpt(repo_dir, activation_factory)
    train_py = repo_dir / "train.py"
    old_argv = sys.argv[:]
    with _temporary_chdir(repo_dir):
        try:
            sys.argv = [str(train_py), *train_argv]
            runpy.run_path(str(train_py), run_name="__main__")
        finally:
            sys.argv = old_argv


__all__ = [
    "DEFAULT_LOCAL_NANOGPT_REPO",
    "DEFAULT_UPSTREAM_NANOGPT_REPO",
    "DEFAULT_UPSTREAM_NANOGPT_URL",
    "NonlinearLCTActivation",
    "build_local_nanogpt",
    "build_upstream_nanogpt",
    "configure_local_nanogpt",
    "ensure_upstream_nanogpt_repo",
    "infer_nanogpt_repo_kind",
    "load_local_nanogpt_definitions",
    "patch_feedforward_class",
    "patch_upstream_nanogpt",
    "run_upstream_train",
]
