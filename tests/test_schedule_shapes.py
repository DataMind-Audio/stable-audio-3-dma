"""Every sampler must honour the schedule shape generate() actually passes.

`generate()` asks the distribution shift for a PER-ELEMENT schedule, so the
samplers receive `sigmas` of shape (batch_size, steps+1) — never the flat
(steps+1,) the docstrings lead with. Euler, DPM++ and Pingpong branch on
`sigmas.dim()`. RK4 did not: it zipped `t[:-1]` with `t[1:]` over the ROWS,
so for batch size 1 the loop ran zero iterations and the caller's initial
noise came back as the "sample" and was decoded. Every RK4 render the
Kollektiv player made since launch was static (found 2026-09-15, withdrawn
from the player 2026-09-28).

This pins two things: the samplers the player offers run every step for the
2-D schedule at batch 1 (the size the worker always uses) and at batch 2, and
RK4 refuses that shape loudly instead of returning noise. CPU-only; a dummy
model stands in for the DiT.
"""
import pytest
import torch

from stable_audio_3.inference import sampling

PER_ELEMENT_SAMPLERS = [
    sampling.sample_discrete_euler,
    sampling.sample_flow_dpmpp,
    sampling.sample_flow_pingpong,
]


def _drift_model(x, t, **extra_args):
    # A constant, non-zero velocity: any sampler that takes a step moves x.
    return torch.ones_like(x)


def _per_element_schedule(steps, batch):
    return torch.linspace(1.0, 0.0, steps + 1)[None, :].repeat(batch, 1)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sampler", PER_ELEMENT_SAMPLERS, ids=lambda s: s.__name__)
def test_per_element_schedule_runs_every_step(sampler, batch):
    steps = 4
    x = torch.randn(batch, 2, 16, generator=torch.Generator().manual_seed(0))
    seen = []
    out = sampler(_drift_model, x, _per_element_schedule(steps, batch),
                  callback=lambda info: seen.append(int(info["i"])), disable_tqdm=True)
    assert seen == list(range(steps)), (
        f"{sampler.__name__} ran {len(seen)} of {steps} steps for a "
        f"({batch}, {steps + 1}) schedule")
    assert out.shape == x.shape
    assert not torch.equal(out, x), f"{sampler.__name__} returned its input noise unchanged"


def test_rk4_refuses_a_per_element_schedule():
    """Before the guard this returned `x` untouched after zero iterations."""
    x = torch.randn(1, 2, 16, generator=torch.Generator().manual_seed(0))
    with pytest.raises(ValueError, match="per-element"):
        sampling.sample_rk4(_drift_model, x, _per_element_schedule(4, 1), disable_tqdm=True)


def test_rk4_still_samples_a_global_schedule():
    """The guard must not break RK4 for callers that pass the shape it documents."""
    steps = 4
    x = torch.randn(1, 2, 16, generator=torch.Generator().manual_seed(0))
    seen = []
    out = sampling.sample_rk4(_drift_model, x, torch.linspace(1.0, 0.0, steps + 1),
                              callback=lambda info: seen.append(int(info["i"])), disable_tqdm=True)
    assert seen == list(range(steps))
    assert not torch.equal(out, x)
