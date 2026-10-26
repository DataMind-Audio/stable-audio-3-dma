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


# ---- truncate_to_sigma_max: a schedule that starts at the init's noise level and only falls ----------------------------
# The default path shifts linspace(sigma_max, 0) and pins the first point, so from sigma_max 0.8 the second step is back
# near the top of the shifted curve (≈ 0.98 with the medium model's LogSNR shift): a "part-way down" start is re-noised.
# The truncated path enters the shifted curve where it equals sigma_max, so the whole schedule is monotone from there.

def _logsnr_shift():
    from stable_audio_3.inference.distribution_shift import LogSNRShift
    return LogSNRShift(anchor_length=2000, anchor_logsnr=-6.2, rate=0.0, logsnr_end=2.0)


def test_default_schedule_renoises_after_a_low_sigma_max():
    sig = sampling.build_schedule(8, sigma_max=0.8, dist_shift=_logsnr_shift(), fallback_seq_len=2000)
    assert abs(float(sig[0]) - 0.8) < 1e-6 and float(sig[1]) > 0.9   # the trap the truncated form exists for


def test_truncated_schedule_is_monotone_from_sigma_max():
    shift = _logsnr_shift()
    for smax in (0.95, 0.8, 0.5):
        sig = sampling.build_schedule(8, sigma_max=smax, dist_shift=shift, fallback_seq_len=2000, truncate_to_sigma_max=True)
        assert sig.shape == (9,)
        assert abs(float(sig[0]) - smax) < 1e-6, (smax, sig)
        assert float(sig[-1]) == 0.0
        assert all(float(sig[i + 1]) < float(sig[i]) for i in range(8)), (smax, sig)
        # it is the same shifted curve, entered lower: every later point lies on the full curve's range
        full = sampling.build_schedule(8, sigma_max=1.0, dist_shift=shift, fallback_seq_len=2000)
        assert float(sig[1]) < float(full[1])


def test_truncated_schedule_per_element_gives_one_row_each():
    shift = _logsnr_shift()
    sig = sampling.build_schedule(4, sigma_max=0.8, dist_shift=shift, effective_seq_len=torch.tensor([2000, 4000]),
                                  fallback_seq_len=2000, truncate_to_sigma_max=True)
    assert sig.shape == (2, 5)
    assert torch.allclose(sig[:, 0], torch.tensor([0.8, 0.8]))
    assert bool((sig[:, 1:] < sig[:, :-1]).all())


def test_truncation_is_a_no_op_at_sigma_max_1_and_without_a_shift():
    shift = _logsnr_shift()
    a = sampling.build_schedule(6, sigma_max=1.0, dist_shift=shift, fallback_seq_len=2000)
    b = sampling.build_schedule(6, sigma_max=1.0, dist_shift=shift, fallback_seq_len=2000, truncate_to_sigma_max=True)
    assert torch.equal(a, b)
    c = sampling.build_schedule(6, sigma_max=0.7, truncate_to_sigma_max=True)
    assert torch.allclose(c, torch.linspace(0.7, 0, 7))
