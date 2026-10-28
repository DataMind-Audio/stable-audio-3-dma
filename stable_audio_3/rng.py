"""Per-request random streams.

Every random draw of a generation — the initial noise, the pingpong re-noise, the SAME-L decoder's token
noise — goes through here. A `generator_scope(g)` makes `g` the generator for the current thread/task
(a contextvars.ContextVar); outside any scope the helpers fall back to torch's default generator, so code
that never asked for a scope behaves exactly as before.

Why: `torch.manual_seed` + the default generator is process-global. Two renders in two threads would
interleave their draws, and a preview decode mid-sampling already shifted the stream once (2026-10-27).
A fresh `torch.Generator(device).manual_seed(seed)` yields the exact stream the default generator did
after `torch.manual_seed(seed)` (verified bitwise on CUDA, 2026-10-28) — provided each draw keeps its
shape: normal sampling is not shape-stable (randn(32) ≠ 4 × randn(8)), so the call sites below draw the
same shapes they always did.
"""
import contextlib
import contextvars
import typing as tp

import torch

_GEN: contextvars.ContextVar = contextvars.ContextVar("sa3_generator", default=None)


def current_generator() -> tp.Optional[torch.Generator]:
    return _GEN.get()


@contextlib.contextmanager
def generator_scope(generator: tp.Optional[torch.Generator]):
    """Make `generator` the random source for this thread/task until the block ends (None = default RNG)."""
    token = _GEN.set(generator)
    try:
        yield generator
    finally:
        _GEN.reset(token)


def seeded(seed: int, device) -> torch.Generator:
    """A generator on `device` seeded like `torch.manual_seed(seed)` would seed the default one."""
    dev = torch.device(device)
    g = torch.Generator(device=dev if dev.type != "mps" else "cpu")
    g.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    return g


def randn(*size, dtype=None, device=None, generator: tp.Optional[torch.Generator] = None) -> torch.Tensor:
    g = generator if generator is not None else _GEN.get()
    if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
        size = tuple(size[0])
    if g is not None and torch.device(device or "cpu").type != g.device.type:
        # a CPU-scoped generator cannot draw on CUDA (or the reverse): draw where the generator lives, move after
        return torch.randn(*size, dtype=dtype, device=g.device, generator=g).to(device)
    return torch.randn(*size, dtype=dtype, device=device, generator=g)


def randn_like(t: torch.Tensor, generator: tp.Optional[torch.Generator] = None) -> torch.Tensor:
    """`torch.randn_like(t)` from the scoped generator (torch.randn_like itself takes none)."""
    return randn(t.shape, dtype=t.dtype, device=t.device, generator=generator)
