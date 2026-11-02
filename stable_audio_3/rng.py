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
import math
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


class Variation:
    """Two streams mixed into one, variance kept: every draw is  ε = cos(c·π/2)·ε_seed + sin(c·π/2)·ε_var, c the strength.
    c = 0 is the seed's stream exactly (bitwise), c = 1 the variation seed's exactly; between, a take that drifts away from
    the seed's by degrees while every draw keeps unit variance (cos² + sin² = 1), so the sampler sees noise of the same
    size. Scope it like a Generator (generator_scope); rng.randn / randn_like draw through it. The Studio's Chaos dial in
    NEW mode (2026-11-02)."""

    def __init__(self, base: torch.Generator, variant: torch.Generator, strength: float):
        self.base, self.variant = base, variant
        self.strength = float(min(1.0, max(0.0, strength)))
        self.device = base.device
        a = self.strength * math.pi / 2
        self.wa, self.wb = math.cos(a), math.sin(a)

    def randn(self, size, dtype=None, device=None) -> torch.Tensor:
        if self.strength <= 0.0:
            out = torch.randn(*size, dtype=dtype, device=self.device, generator=self.base)
        elif self.strength >= 1.0:
            out = torch.randn(*size, dtype=dtype, device=self.device, generator=self.variant)
        else:
            a = torch.randn(*size, dtype=dtype, device=self.device, generator=self.base)
            b = torch.randn(*size, dtype=dtype, device=self.device, generator=self.variant)
            out = a * self.wa + b * self.wb
        return out if device is None or torch.device(device).type == out.device.type else out.to(device)


def variation(seed: int, variation_seed: int, strength: float, device):
    """The random source for a take: the seed's stream alone at strength 0, else a Variation of it towards variation_seed's."""
    if strength is None or strength <= 0.0:
        return seeded(seed, device)
    return Variation(seeded(seed, device), seeded(variation_seed, device), strength)


def randn(*size, dtype=None, device=None, generator=None) -> torch.Tensor:
    g = generator if generator is not None else _GEN.get()
    if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
        size = tuple(size[0])
    if isinstance(g, Variation):
        return g.randn(size, dtype=dtype, device=device)
    if g is not None and torch.device(device or "cpu").type != g.device.type:
        # a CPU-scoped generator cannot draw on CUDA (or the reverse): draw where the generator lives, move after
        return torch.randn(*size, dtype=dtype, device=g.device, generator=g).to(device)
    return torch.randn(*size, dtype=dtype, device=device, generator=g)


def randn_like(t: torch.Tensor, generator: tp.Optional[torch.Generator] = None) -> torch.Tensor:
    """`torch.randn_like(t)` from the scoped generator (torch.randn_like itself takes none)."""
    return randn(t.shape, dtype=t.dtype, device=t.device, generator=generator)
