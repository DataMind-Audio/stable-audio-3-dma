"""The Kollektiv worker cancels a generation by RAISING from the per-step
callback: every sampler here invokes ``callback(...)`` bare, inside its loop,
so the exception unwinds the sampler and frees the GPU within one step. That
is a contract nobody upstream promised. This test pins it, so a future merge
that wraps the callback in try/except (or drops it from a sampler) fails
loudly here instead of silently turning "cancel" into "render anyway".
CPU-only; a dummy model stands in for the DiT."""
import pytest
import torch

from stable_audio_3.inference import sampling

SAMPLERS = [
    sampling.sample_discrete_euler,
    sampling.sample_rk4,
    sampling.sample_flow_dpmpp,
    sampling.sample_flow_pingpong,
]


class Stop(Exception):
    pass


def _dummy_model(x, t, **extra_args):
    # Any tensor of x's shape keeps every sampler's arithmetic well-defined.
    return torch.zeros_like(x)


def _run(sampler, callback, steps=4):
    x = torch.randn(1, 2, 16, generator=torch.Generator().manual_seed(0))
    sigmas = torch.linspace(1.0, 0.0, steps + 1)
    return sampler(_dummy_model, x, sigmas, callback=callback, disable_tqdm=True)


@pytest.mark.parametrize("sampler", SAMPLERS, ids=lambda s: s.__name__)
def test_callback_is_invoked_every_step(sampler):
    seen = []
    _run(sampler, lambda info: seen.append(int(info["i"])))
    assert seen, f"{sampler.__name__} never called its callback"
    assert seen == sorted(seen), "steps reported out of order"


@pytest.mark.parametrize("sampler", SAMPLERS, ids=lambda s: s.__name__)
def test_exception_raised_in_callback_propagates_out_of_the_sampler(sampler):
    calls = []

    def cancel_on_second_step(info):
        calls.append(int(info["i"]))
        if len(calls) == 2:
            raise Stop("cancelled")

    with pytest.raises(Stop):
        _run(sampler, cancel_on_second_step)
    assert len(calls) == 2, (
        f"{sampler.__name__} kept stepping after the callback raised — "
        "cancellation would no longer free the GPU"
    )
