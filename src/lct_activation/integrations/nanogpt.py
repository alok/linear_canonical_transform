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

from lct_activation.layers import LCTActivation as CoreLCTActivation
from lct_activation.layers import LCTLinear
from lct_activation.functional import NormMode

DEFAULT_LOCAL_NANOGPT_REPO = Path("/Users/alokbeniwal/nanogpt")
DEFAULT_UPSTREAM_NANOGPT_REPO = Path("extern/nanoGPT")
DEFAULT_UPSTREAM_NANOGPT_URL = "https://github.com/karpathy/nanoGPT.git"

RepoKind: TypeAlias = Literal["local", "upstream"]
ModelVariant: TypeAlias = Literal["baseline", "activation", "linear", "hybrid"]
ActivationFactory: TypeAlias = Callable[[], nn.Module]
LinearFactory: TypeAlias = Callable[[nn.Linear], nn.Module]

_LOCAL_STOP_MARKERS = (
    "\ngpt = GPT()",
    "\ngpt = GPT(",
)
_PATCH_ATTR = "__lct_activation_patched__"
_PATCH_VARIANT_ATTR = "__lct_patch_variant__"
_ORIGINAL_INIT_ATTR = "__lct_activation_original_init__"


class NonlinearLCTActivation(nn.Module):
    """Lazy wrapper around the package's real `LCTActivation`.

    NanoGPT integration points often construct activations without passing the
    hidden width. This wrapper waits until the first forward pass, then
    materializes the real shape-aware activation module with the observed width.
    """

    def __init__(
        self,
        a: float = 0.0,
        b: float = 1.0,
        c: float = 0.0,
        bias_init: float = 0.1,
        inverse_after_nonlinearity: bool = False,
        residual_mix: float = 0.0,
        dense_threshold: int = 256,
        normalization: NormMode = "unitary",
        unitary_projection: bool = True,
        learnable_transform: bool = False,
        transform_parameterization: Literal["legacy", "symplectic"] | None = None,
    ) -> None:
        super().__init__()
        self.activation: CoreLCTActivation | None = None
        self.kwargs = {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "bias_init": float(bias_init),
            "inverse_after_nonlinearity": bool(inverse_after_nonlinearity),
            "residual_mix": float(residual_mix),
            "dense_threshold": int(dense_threshold),
            "normalization": normalization,
            "unitary_projection": bool(unitary_projection),
            "learnable_transform": bool(learnable_transform),
            "transform_parameterization": transform_parameterization,
        }

    def _ensure_activation(self, width: int, *, device: torch.device) -> CoreLCTActivation:
        if self.activation is None or self.activation.channels != width:
            self.activation = CoreLCTActivation(width, **self.kwargs)
        return self.activation.to(device=device)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] < 2 or torch.is_complex(x):
            return x

        activation = self._ensure_activation(x.shape[-1], device=x.device)
        return activation(x)


def make_lct_activation_factory(
    *,
    a: float = 0.0,
    b: float = 1.0,
    c: float = 0.0,
    bias_init: float = 0.1,
    inverse_after_nonlinearity: bool = False,
    residual_mix: float = 0.0,
    dense_threshold: int = 256,
    normalization: NormMode = "unitary",
    unitary_projection: bool = True,
    learnable_transform: bool = False,
    transform_parameterization: Literal["legacy", "symplectic"] | None = None,
) -> ActivationFactory:
    return lambda: NonlinearLCTActivation(
        a=a,
        b=b,
        c=c,
        bias_init=bias_init,
        inverse_after_nonlinearity=inverse_after_nonlinearity,
        residual_mix=residual_mix,
        dense_threshold=dense_threshold,
        normalization=normalization,
        unitary_projection=unitary_projection,
        learnable_transform=learnable_transform,
        transform_parameterization=transform_parameterization,
    )


def make_lct_linear_factory(
    *,
    a: float = 0.0,
    b: float = 1.0,
    c: float = 0.0,
    inverse_after_multiply: bool = True,
    dense_threshold: int = 32,
    learnable_transform: bool = False,
    transform_parameterization: Literal["legacy", "symplectic"] | None = None,
    normalization: NormMode = "unitary",
    unitary_projection: bool = False,
    use_triton_kernels: bool = True,
    direct_fourier_backend: Literal["fft", "conv", "auto"] = "fft",
) -> LinearFactory:
    def factory(linear: nn.Linear) -> nn.Module:
        replacement = LCTLinear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            a=a,
            b=b,
            c=c,
            inverse_after_multiply=inverse_after_multiply,
            dense_threshold=dense_threshold,
            learnable_transform=learnable_transform,
            transform_parameterization=transform_parameterization,
            normalization=normalization,
            unitary_projection=unitary_projection,
            use_triton_kernels=use_triton_kernels,
            direct_fourier_backend=direct_fourier_backend,
        )
        replacement = replacement.to(device=linear.weight.device, dtype=linear.weight.dtype)
        if linear.bias is not None and replacement.bias is not None:
            replacement.bias.data.copy_(linear.bias.data)
        return replacement

    return factory


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


class LowRankLinear(nn.Module):
    """Rank-``r`` factorized linear map: the dense control for LCTLinear.

    At LCTLinear's parameter budget (about ``padded/2*2 + out`` parameters for
    a d -> 4d up-projection), the matching factorized rank is ~1, which makes
    this the honest 'what else could you buy for the same parameters' control.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.down = nn.Linear(self.in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.down(x))


def make_lowrank_linear_factory(*, rank: int = 1) -> LinearFactory:
    def factory(linear: nn.Linear) -> nn.Module:
        replacement = LowRankLinear(
            linear.in_features,
            linear.out_features,
            rank,
            bias=linear.bias is not None,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        if linear.bias is not None and replacement.up.bias is not None:
            replacement.up.bias.data.copy_(linear.bias.data)
        return replacement

    return factory


def _replace_first_linear(module: nn.Module, linear_factory: LinearFactory) -> None:
    for attr in ("c_fc", "fc1", "fc", "proj_in"):
        current = getattr(module, attr, None)
        if isinstance(current, nn.Linear):
            setattr(module, attr, linear_factory(current))
            return

    net = getattr(module, "net", None)
    if isinstance(net, nn.Sequential):
        for idx, layer in enumerate(net):
            if isinstance(layer, nn.Linear):
                net[idx] = linear_factory(layer)
                return

    raise RuntimeError(f"Could not find a replaceable linear layer inside {type(module).__name__}.")


def _resolve_variant(use_lct: bool, variant: ModelVariant | None) -> ModelVariant:
    if variant is not None:
        return variant
    return "activation" if use_lct else "baseline"


def patch_feedforward_class(
    feedforward_cls: type[nn.Module],
    *,
    variant: ModelVariant = "activation",
    activation_factory: ActivationFactory = NonlinearLCTActivation,
    linear_factory: LinearFactory | None = None,
) -> type[nn.Module]:
    if variant == "baseline":
        return feedforward_cls

    if getattr(feedforward_cls, _PATCH_ATTR, False) and getattr(feedforward_cls, _PATCH_VARIANT_ATTR, None) == variant:
        return feedforward_cls

    original_init = getattr(feedforward_cls, _ORIGINAL_INIT_ATTR, feedforward_cls.__init__)

    @wraps(original_init)
    def patched_init(self: nn.Module, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if variant in {"linear", "hybrid"}:
            if linear_factory is None:
                raise RuntimeError("linear_factory is required for linear or hybrid NanoGPT patching.")
            _replace_first_linear(self, linear_factory)
        if variant in {"activation", "hybrid"}:
            _replace_activation(self, activation_factory())

    setattr(feedforward_cls, _ORIGINAL_INIT_ATTR, original_init)
    feedforward_cls.__init__ = patched_init
    setattr(feedforward_cls, _PATCH_ATTR, True)
    setattr(feedforward_cls, _PATCH_VARIANT_ATTR, variant)
    return feedforward_cls


def patch_upstream_nanogpt(
    repo_dir: Path | str = DEFAULT_UPSTREAM_NANOGPT_REPO,
    *,
    variant: ModelVariant = "activation",
    activation_factory: ActivationFactory = NonlinearLCTActivation,
    linear_factory: LinearFactory | None = None,
) -> Any:
    repo_dir = Path(repo_dir)
    if infer_nanogpt_repo_kind(repo_dir) != "upstream":
        raise ValueError(f"{repo_dir} is not an upstream NanoGPT checkout.")

    _add_repo_to_syspath(repo_dir)
    importlib.invalidate_caches()
    if "model" in sys.modules:
        model_module = importlib.reload(sys.modules["model"])
    else:
        model_module = importlib.import_module("model")
    patch_feedforward_class(
        model_module.MLP,
        variant=variant,
        activation_factory=activation_factory,
        linear_factory=linear_factory,
    )
    return model_module


def build_upstream_nanogpt(
    repo_dir: Path | str = DEFAULT_UPSTREAM_NANOGPT_REPO,
    *,
    use_lct: bool = False,
    variant: ModelVariant | None = None,
    activation_factory: ActivationFactory = NonlinearLCTActivation,
    linear_factory: LinearFactory | None = None,
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

    resolved_variant = _resolve_variant(use_lct, variant)
    if resolved_variant != "baseline":
        model_module = patch_upstream_nanogpt(
            repo_dir,
            variant=resolved_variant,
            activation_factory=activation_factory,
            linear_factory=linear_factory,
        )
    else:
        _add_repo_to_syspath(repo_dir)
        importlib.invalidate_caches()
        if "model" in sys.modules:
            model_module = importlib.reload(sys.modules["model"])
        else:
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
        ctx_len = namespace["args"].CTX_LEN

        if len(args) >= 1:
            head_size = args[0]
        if len(args) >= 2 or "embed_dim" in kwargs:
            result = original_init(self, *args, **kwargs)
            causal_mask = torch.triu(
                torch.ones(ctx_len, ctx_len, device=self.mask.device, dtype=torch.bool),
                diagonal=1,
            )
            self.mask = causal_mask.unsqueeze(0).unsqueeze(0)
            return result
        if len(args) >= 3:
            n_heads = args[2]

        if head_size is None:
            head_size = namespace["args"].HEAD_SIZE
        kwargs["embed_dim"] = int(head_size) * int(n_heads)
        result = original_init(self, *args, **kwargs)
        causal_mask = torch.triu(
            torch.ones(ctx_len, ctx_len, device=self.mask.device, dtype=torch.bool),
            diagonal=1,
        )
        self.mask = causal_mask.unsqueeze(0).unsqueeze(0)
        return result

    multihead_cls.__init__ = patched_init
    multihead_cls.__lct_init_patched__ = True


def _patch_local_multihead_scaling(namespace: dict[str, Any]) -> None:
    """Add the standard ``1/sqrt(head_dim)`` attention scaling.

    The local NanoGPT omits it; that quirk degrades wider models at constant
    lr (measured: a dim-212 baseline beats dim-256 by 0.22 nats). Opt-in so
    existing artifacts stay reproducible; applied identically to all variants.
    """

    multihead_cls = namespace["MultiHead"]
    if getattr(multihead_cls, "__lct_scaling_patched__", False):
        return

    original_forward = multihead_cls.forward

    @wraps(original_forward)
    def scaled_forward(self: nn.Module, x: Tensor) -> Tensor:
        # Mirrors the module's einsum forward exactly (including its
        # k-rows/q-columns orientation), adding only the logit scale.
        batch, seq, channels = x.shape
        n_heads = self.n_heads
        head_dim = channels // n_heads

        def split(tensor: Tensor) -> Tensor:
            return tensor.view(batch, seq, n_heads, head_dim).transpose(1, 2)

        k = split(self.key(x))
        q = split(self.query(x))
        v = split(self.value(x))

        weights = (k @ q.transpose(-2, -1)) / math.sqrt(head_dim)
        masked = weights.masked_fill(self.mask[..., :seq, :seq], float("-inf"))
        masked = self.dropout(masked.softmax(dim=-1))
        out = (masked @ v).transpose(1, 2).reshape(batch, seq, channels)
        return self.proj(out)

    multihead_cls.forward = scaled_forward
    multihead_cls.__lct_scaling_patched__ = True


def load_local_nanogpt_definitions(
    repo_dir: Path | str = DEFAULT_LOCAL_NANOGPT_REPO,
    *,
    use_lct: bool = False,
    variant: ModelVariant | None = None,
    activation_factory: ActivationFactory = NonlinearLCTActivation,
    linear_factory: LinearFactory | None = None,
    attention_scaling: bool = False,
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

    with (
        _temporary_chdir(repo_dir),
        _patched_urlretrieve(repo_dir),
        _local_nanogpt_import_shims(),
    ):
        exec(compile(pre_training_source, str(source_path), "exec"), module_globals)

    _patch_local_multihead_init(module_globals)
    _patch_local_gpt_forward(module_globals)
    if attention_scaling:
        _patch_local_multihead_scaling(module_globals)
    resolved_variant = _resolve_variant(use_lct, variant)
    if resolved_variant != "baseline":
        patch_feedforward_class(
            module_globals["FeedForward"],
            variant=resolved_variant,
            activation_factory=activation_factory,
            linear_factory=linear_factory,
        )
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
    variant: ModelVariant | None = None,
    activation_factory: ActivationFactory = NonlinearLCTActivation,
    linear_factory: LinearFactory | None = None,
    batch_size: int = 8,
    ctx_len: int = 128,
    n_heads: int = 8,
    embed_dim: int = 512,
    n_layers: int = 3,
    drop_frac: float = 0.0,
    vocab_size: int | None = None,
    device: torch.device | None = None,
    seed: int | None = None,
    attention_scaling: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    namespace = load_local_nanogpt_definitions(
        repo_dir,
        use_lct=use_lct,
        variant=variant,
        activation_factory=activation_factory,
        linear_factory=linear_factory,
        attention_scaling=attention_scaling,
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
    # The exec'd nanogpt source calls torch.manual_seed(1337) at import time,
    # clobbering any seed the caller set beforehand. Reseed here, after the
    # module load and immediately before model construction, so `seed`
    # actually controls the initialization.
    if seed is not None:
        torch.manual_seed(seed)
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
    variant: ModelVariant = "activation",
    activation_factory: ActivationFactory = NonlinearLCTActivation,
    linear_factory: LinearFactory | None = None,
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

    patch_upstream_nanogpt(
        repo_dir,
        variant=variant,
        activation_factory=activation_factory,
        linear_factory=linear_factory,
    )
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
    "ModelVariant",
    "NonlinearLCTActivation",
    "build_local_nanogpt",
    "build_upstream_nanogpt",
    "configure_local_nanogpt",
    "ensure_upstream_nanogpt_repo",
    "infer_nanogpt_repo_kind",
    "load_local_nanogpt_definitions",
    "make_lct_activation_factory",
    "make_lct_linear_factory",
    "patch_feedforward_class",
    "patch_upstream_nanogpt",
    "run_upstream_train",
]
