"""CPU-perf helpers shared across model wrappers.

These are best-effort: every call is wrapped so that a missing torch / GPU /
incompatible module degrades gracefully (returns the input untouched).
"""

from __future__ import annotations

import os
from typing import Any


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def configure_threads() -> None:
    """Honor ``MESHMIND_NUM_THREADS`` (defaults to physical CPU count).

    Diffusers/transformers leave torch threads at the default, which on Linux
    can be lower than the actual core count.  Pinning gives a small but
    consistent CPU win for SD-Turbo and ShapE.
    """
    try:
        import torch
    except ImportError:
        return
    raw = os.environ.get("MESHMIND_NUM_THREADS")
    if raw:
        try:
            n = max(1, int(raw))
        except ValueError:
            return
    else:
        n = os.cpu_count() or 1
    try:
        torch.set_num_threads(n)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, n // 2))
    except Exception:
        pass


def to_channels_last(module: Any) -> Any:
    """Switch a UNet/conv module to ``channels_last`` memory format on CPU.

    Returns the module untouched if torch isn't available or the conversion
    raises.  Channels-last gives diffusers UNet ~5-15% on CPU MKL-DNN.
    """
    try:
        import torch

        return module.to(memory_format=torch.channels_last)
    except Exception:
        return module


def enable_attention_slicing(pipe: Any) -> None:
    """Best-effort attention slicing — reduces memory + minor CPU speed win."""
    fn = getattr(pipe, "enable_attention_slicing", None)
    if fn is None:
        return
    try:
        fn()
    except Exception:
        pass


def maybe_compile(module: Any, *, mode: str = "reduce-overhead") -> Any:
    """Optionally JIT-compile via ``torch.compile`` when ``MESHMIND_COMPILE=1``.

    First call after compile is much slower (10-60 s on CPU) but subsequent
    inferences run 30-50% faster.  Off by default because the compile cost
    dominates for one-shot generations.
    """
    if not _env_bool("MESHMIND_COMPILE", False):
        return module
    try:
        import torch

        return torch.compile(module, mode=mode, fullgraph=False)
    except Exception:
        return module


def optimize_diffusers_pipe(pipe: Any) -> Any:
    """Apply CPU-friendly tweaks to a diffusers pipeline in one call.

    - Channels-last memory format on the UNet (or prior, for ShapE).
    - Attention slicing (memory + small speed win).
    - Optional ``torch.compile`` on the heavy module.

    Returns the pipe so this can be chained at load time.
    """
    enable_attention_slicing(pipe)
    for attr in ("unet", "prior", "shap_e_renderer"):
        sub = getattr(pipe, attr, None)
        if sub is None:
            continue
        try:
            setattr(pipe, attr, to_channels_last(sub))
        except Exception:
            pass
    heavy = getattr(pipe, "unet", None) or getattr(pipe, "prior", None)
    if heavy is not None:
        compiled = maybe_compile(heavy)
        if heavy is not compiled:
            if hasattr(pipe, "unet") and getattr(pipe, "unet", None) is not None:
                pipe.unet = compiled
            elif hasattr(pipe, "prior"):
                pipe.prior = compiled
    return pipe
