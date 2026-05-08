"""Tests for the ``meshmind.utils.perf`` helpers.

These verify graceful degradation when torch features aren't usable, which
is the common case on CPU-only CI without GPU / OneDNN extras.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock


def test_configure_threads_honors_env_var(monkeypatch) -> None:  # noqa: ANN001
    import torch

    from meshmind.utils import perf

    monkeypatch.setenv("MESHMIND_NUM_THREADS", "2")
    perf.configure_threads()
    assert torch.get_num_threads() == 2


def test_configure_threads_no_env_uses_cpu_count(monkeypatch) -> None:  # noqa: ANN001
    import torch

    from meshmind.utils import perf

    monkeypatch.delenv("MESHMIND_NUM_THREADS", raising=False)
    perf.configure_threads()
    expected = os.cpu_count() or 1
    assert torch.get_num_threads() == expected


def test_to_channels_last_returns_module(monkeypatch) -> None:  # noqa: ANN001
    import torch

    from meshmind.utils import perf

    module = torch.nn.Conv2d(3, 8, 3)
    out = perf.to_channels_last(module)
    assert out is module


def test_optimize_diffusers_pipe_handles_missing_attrs() -> None:
    from meshmind.utils import perf

    # Pipe without unet/prior — must be a no-op, not crash.
    fake_pipe = MagicMock(spec=[])
    out = perf.optimize_diffusers_pipe(fake_pipe)
    assert out is fake_pipe


def test_maybe_compile_off_by_default(monkeypatch) -> None:  # noqa: ANN001
    import torch

    from meshmind.utils import perf

    monkeypatch.delenv("MESHMIND_COMPILE", raising=False)
    module = torch.nn.Linear(4, 4)
    out = perf.maybe_compile(module)
    assert out is module


def test_enable_attention_slicing_no_method() -> None:
    from meshmind.utils import perf

    fake = MagicMock(spec=[])
    perf.enable_attention_slicing(fake)  # must not raise
