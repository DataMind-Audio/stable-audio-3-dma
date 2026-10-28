"""Per-request random streams (stable_audio_3.rng): a scoped torch.Generator feeds every draw of a
generation, reproduces torch.manual_seed's stream, and keeps renders in different threads apart.
CPU only; the pingpong sampler runs against a dummy model."""
import threading

import pytest
import torch

from stable_audio_3 import rng
from stable_audio_3.inference import sampling


def test_outside_a_scope_the_default_generator_is_used():
    torch.manual_seed(3); a = torch.randn(5)
    torch.manual_seed(3); b = rng.randn(5)
    torch.manual_seed(3); c = rng.randn_like(torch.zeros(5))
    assert torch.equal(a, b) and torch.equal(a, c) and rng.current_generator() is None


def test_scoped_generator_reproduces_manual_seed_and_leaves_the_global_stream_alone():
    torch.manual_seed(42); ref = [torch.randn(4, 3) for _ in range(3)]
    torch.manual_seed(0); before = torch.get_rng_state().clone()
    with rng.generator_scope(rng.seeded(42, "cpu")) as g:
        assert rng.current_generator() is g
        got = [rng.randn(4, 3) for _ in range(3)]
    assert all(torch.equal(x, y) for x, y in zip(ref, got))
    assert torch.equal(torch.get_rng_state(), before), "a scoped draw advanced the global RNG"
    assert rng.current_generator() is None


def test_nested_scope_restores_the_outer_one():
    outer = rng.seeded(1, "cpu")
    with rng.generator_scope(outer):
        with rng.generator_scope(rng.seeded(2, "cpu")) as inner:
            assert rng.current_generator() is inner
        assert rng.current_generator() is outer


def _dummy(x, t, **kw):
    return torch.zeros_like(x)


def _pingpong(seed, steps=4):
    with rng.generator_scope(rng.seeded(seed, "cpu")):
        x = rng.randn(1, 2, 16)
        return sampling.sample_flow_pingpong(_dummy, x, torch.linspace(1.0, 0.0, steps + 1), disable_tqdm=True)


def test_pingpong_draws_from_the_scope_deterministically():
    a, b = _pingpong(7), _pingpong(7)
    assert torch.equal(a, b)
    assert not torch.equal(a, _pingpong(8))
    # and it matches the pre-scope behaviour: manual_seed + default-generator draws of the same shapes
    torch.manual_seed(7)
    x = torch.randn(1, 2, 16)
    legacy = sampling.sample_flow_pingpong(_dummy, x, torch.linspace(1.0, 0.0, 5), disable_tqdm=True)
    assert torch.equal(legacy, a)


def test_two_threads_with_their_own_scopes_equal_their_solo_runs():
    solo = {7: _pingpong(7), 9: _pingpong(9)}
    out = {}
    gate = threading.Barrier(2)

    def run(seed):
        with rng.generator_scope(rng.seeded(seed, "cpu")):
            x = rng.randn(1, 2, 16)
            gate.wait()          # both samplers step at once; the GIL interleaves their draws
            out[seed] = sampling.sample_flow_pingpong(_dummy, x, torch.linspace(1.0, 0.0, 5), disable_tqdm=True)
    th = [threading.Thread(target=run, args=(s,)) for s in (7, 9)]
    [t.start() for t in th]; [t.join() for t in th]
    assert torch.equal(out[7], solo[7]) and torch.equal(out[9], solo[9])


def test_generate_signature_accepts_a_generator():
    import inspect
    from stable_audio_3 import StableAudioModel
    assert "generator" in inspect.signature(StableAudioModel.generate).parameters
