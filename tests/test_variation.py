"""NEW's Chaos (stable_audio_3.rng.Variation): two seeded streams mixed, variance kept. CPU only."""
import math

import torch

from stable_audio_3 import rng


def draws(src, n=4, shape=(2, 3, 5)):
    with rng.generator_scope(src):
        return [rng.randn(shape) for _ in range(n)]


def test_strength_0_is_the_seed_s_stream_bitwise_and_1_the_variation_s():
    base = draws(rng.seeded(7, "cpu")); var = draws(rng.seeded(99, "cpu"))
    assert all(torch.equal(a, b) for a, b in zip(draws(rng.variation(7, 99, 0.0, "cpu")), base))
    assert all(torch.equal(a, b) for a, b in zip(draws(rng.variation(7, 99, 1.0, "cpu")), var))
    assert isinstance(rng.variation(7, 99, 0.0, "cpu"), torch.Generator)       # no mixing at 0: a plain generator
    assert isinstance(rng.variation(7, 99, 0.5, "cpu"), rng.Variation)


def test_between_the_draws_are_the_cosine_mix_of_both_streams():
    c = 0.3
    base = draws(rng.seeded(7, "cpu")); var = draws(rng.seeded(99, "cpu")); mix = draws(rng.variation(7, 99, c, "cpu"))
    wa, wb = math.cos(c * math.pi / 2), math.sin(c * math.pi / 2)
    assert all(torch.allclose(m, a * wa + b * wb) for m, a, b in zip(mix, base, var))
    assert all(not torch.equal(m, a) for m, a in zip(mix, base))


def test_the_mix_keeps_unit_variance_and_is_deterministic():
    big = torch.cat([t.flatten() for t in draws(rng.variation(7, 99, 0.5, "cpu"), n=1, shape=(400, 400))])
    assert abs(big.std().item() - 1.0) < 0.01 and abs(big.mean().item()) < 0.01
    a = draws(rng.variation(7, 99, 0.5, "cpu")); b = draws(rng.variation(7, 99, 0.5, "cpu"))
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert all(not torch.equal(x, y) for x, y in zip(a, draws(rng.variation(7, 100, 0.5, "cpu"))))   # another variation seed: another take


def test_randn_like_and_the_pingpong_draw_go_through_the_variation():
    with rng.generator_scope(rng.variation(7, 99, 0.5, "cpu")) as g:
        assert rng.current_generator() is g
        t = rng.randn_like(torch.zeros(3, 4))
    ref = draws(rng.variation(7, 99, 0.5, "cpu"), n=1, shape=(3, 4))[0]
    assert torch.equal(t, ref)
