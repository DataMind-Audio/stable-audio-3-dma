"""Regions (models/regions.py): the per-frame words and the guidance lane, on CPU.

The fast form must equal the per-set reference (the whole module once per word set, outputs mixed per token) for every
attention variant the code handles; memory tokens read the shared words unless the plan mixes them; a call the fast
form cannot do falls back to the reference; nothing changes without a plan; and on a real (tiny) DiT a guidance lane
gives every frame exactly the output its own scale would.
"""
import itertools
import math

import pytest
import torch

from stable_audio_3.models.dit import DiffusionTransformer
from stable_audio_3.models.regions import RegionPlan, current_plan, frame_times
from stable_audio_3.models.transformer import Attention


def _attn(diff, qk, fs, dctx):
    torch.manual_seed(0)
    attn = Attention(64, dim_heads=16, dim_context=dctx, differential=diff, qk_norm=qk, feat_scale=fs,
                     zero_init_output=False).eval()
    if fs:
        with torch.no_grad():
            attn.lambda_dc.normal_(0, 0.3)
            attn.lambda_hf.normal_(0, 0.3)
    attn._region_site = True
    return attn


def _plan(ctxs, W, tokens, memory=None):
    plan = RegionPlan([f"set {k}" for k in range(len(ctxs))], W, memory=memory)
    plan.contexts, plan.tokens = ctxs, tokens
    return plan


def _reference(attn, x, context, ctxs, W, P, memory=None):
    out = attn._attend(x, context).clone()
    per = [attn._attend(x[:1], ck) for ck in ctxs]
    out[:1, P:] = sum(o[:, P:] * W[k][None, :, None] for k, o in enumerate(per))
    if memory is not None and P > 0:
        out[:1, :P] = sum(o[:, :P] * memory[k] for k, o in enumerate(per))
    return out


@pytest.mark.parametrize("diff,qk,fs,dctx,P,B,neg", list(itertools.product(
    (False, True), ("none", "l2", "ln", "rms"), (False, True), (64, 32), (0, 3), (1, 2), (False, True))))
def test_fast_form_equals_the_per_set_reference(diff, qk, fs, dctx, P, B, neg):
    attn = _attn(diff, qk, fs, dctx)
    n_seq, mtok, K = 40, 7, 3
    x = torch.randn(B, n_seq + P, 64)
    ctxs = [torch.randn(1, mtok, dctx) for _ in range(K)]
    context = torch.cat([torch.randn(1, mtok, dctx), torch.zeros(1, mtok, dctx)])[:B]
    W = torch.rand(K, n_seq)
    W = W / W.sum(0, keepdim=True)
    if neg:
        W = torch.stack([torch.full((n_seq,), -0.5), torch.full((n_seq,), 1.5), torch.zeros(n_seq)])
    with torch.no_grad():
        ref = _reference(attn, x, context, ctxs, W, P)
        with _plan(ctxs, W, n_seq) as plan:
            got = attn(x, context=context)
            again = attn(x, context=context)         # keys and values from the cache
        assert plan.fallbacks == 0
    scale = ref.abs().max()
    assert float((got - ref).abs().max() / scale) < 1e-5
    assert float((again - ref).abs().max() / scale) < 1e-5


@pytest.mark.parametrize("diff", (False, True))
def test_memory_tokens_read_a_mix_when_asked(diff):
    attn = _attn(diff, "rms", False, 64)
    P, n_seq = 4, 30
    x = torch.randn(2, P + n_seq, 64)
    ctxs = [torch.randn(1, 7, 64) for _ in range(2)]
    context = torch.cat([torch.randn(1, 7, 64), torch.zeros(1, 7, 64)])
    W = torch.rand(2, n_seq)
    W = W / W.sum(0, keepdim=True)
    mem = [0.3, 0.7]
    with torch.no_grad():
        ref = _reference(attn, x, context, ctxs, W, P, memory=mem)
        with _plan(ctxs, W, n_seq, memory=mem):
            got = attn(x, context=context)
    assert float((got - ref).abs().max() / ref.abs().max()) < 1e-5


def test_unsupported_arguments_take_the_reference_path():
    attn = _attn(True, "rms", False, 64)
    x = torch.randn(2, 40, 64)
    ctxs = [torch.randn(1, 7, 64) for _ in range(2)]
    context = torch.cat([torch.randn(1, 7, 64), torch.zeros(1, 7, 64)])
    W = torch.rand(2, 40)
    with torch.no_grad():
        ref = _reference(attn, x, context, ctxs, W, 0)
        plan = _plan(ctxs, W, 40)
        with plan:
            # the module's own computation, told to ignore the argument the fast form cannot do
            got = plan.cross_attention(attn, x, context, {"padding_mask": torch.ones(2, 40, dtype=torch.bool)},
                                       lambda x_, c_, **k_: attn._attend(x_, c_))
    assert plan.fallbacks == 1
    assert float((got - ref).abs().max() / ref.abs().max()) < 1e-5


def test_nothing_changes_without_a_plan_or_off_site():
    attn = _attn(True, "rms", True, 64)
    x = torch.randn(2, 20, 64)
    context = torch.randn(2, 7, 64)
    with torch.no_grad():
        plain = attn._attend(x, context)
        assert current_plan() is None
        assert torch.equal(attn(x, context=context), plain)
        attn._region_site = False
        with _plan([torch.randn(1, 7, 64)], torch.ones(1, 20), 20):
            assert torch.equal(attn(x, context=context), plain)
        attn._region_site = True
        with _plan([torch.randn(1, 7, 64)], torch.ones(1, 20), None):     # no DiT call published the tokens
            assert torch.equal(attn(x, context=context), plain)
    assert current_plan() is None


def test_weights_and_lane_as_functions_of_time():
    ft = frame_times(4, 1, 2.0, "cpu")
    assert torch.allclose(ft, torch.tensor([0.25, 0.75, 1.25, 1.75]))
    plan = RegionPlan(["a", "b"], lambda t: torch.stack([(t < 1).float(), (t >= 1).float()]),
                      guidance=lambda t: 1 + t, guidance_max=3.0)
    plan.fps, plan.patch = 2.0, 1
    assert torch.equal(plan.token_weights(4, "cpu"), torch.tensor([[1., 1., 0., 0.], [0., 0., 1., 1.]]))
    assert torch.allclose(plan.lane(4, "cpu", torch.float32)[0, 0], torch.tensor([1.25, 1.75, 2.25, 2.75]))
    with pytest.raises(ValueError):
        RegionPlan(["a"], lambda t: t, guidance=lambda t: t)              # a lane function needs its maximum
    with pytest.raises(ValueError):
        RegionPlan(["a", "b"], torch.ones(3, 4))                         # weights must be [K, T]
    with pytest.raises(RuntimeError):
        RegionPlan(["a"], torch.ones(1, 4)).__enter__()                  # prepare() first


# ---- the guidance lane, through a real (tiny) DiffusionTransformer --------------------------------------------------
def _tiny_dit():
    torch.manual_seed(3)
    dit = DiffusionTransformer(io_channels=4, embed_dim=128, cond_token_dim=16, depth=2, num_heads=2,
                               diffusion_objective="rectified_flow", num_memory_tokens=3,
                               attn_kwargs={"differential": True, "qk_norm": "rms"}).eval()
    with torch.no_grad():                              # the zero-initialised convs would hide every difference
        dit.preprocess_conv.weight.normal_(0, 0.2)
        dit.postprocess_conv.weight.normal_(0, 0.2)
    return dit


def test_the_dit_marks_its_cross_attention_only():
    dit = _tiny_dit()
    sites = [m for m in dit.modules() if isinstance(m, Attention) and m._region_site]
    assert len(sites) == 2                             # one per block; self-attention is never a site


def test_guidance_lane_gives_each_frame_its_own_scale():
    dit = _tiny_dit()
    T = 50
    x, t = torch.randn(1, 4, T), torch.full((1,), 0.6)
    cond = torch.randn(1, 9, 16)
    with torch.no_grad():
        plain = {g: dit(x, t, cross_attn_cond=cond, cfg_scale=g) for g in (1.0, 2.0, 3.0)}
        same_words = [dit.to_cond_embed(cond)]          # one set = the prompt itself: only the lane differs
        for g in (2.0, 3.0):
            plan = RegionPlan(["prompt"], torch.ones(1, T), guidance=torch.full((T,), g))
            plan.contexts = same_words
            with plan:
                got = dit(x, t, cross_attn_cond=cond, cfg_scale=3.0)
            assert float((got - plain[g]).abs().max()) < 1e-4, g
        times = frame_times(T, 1, 10.0, "cpu")
        lane = 3.0 - 2.0 * torch.cos(math.pi * ((times - 2.5) / 2.0).clamp(-0.5, 0.5)) ** 2
        plan = RegionPlan(["prompt"], torch.ones(1, T), guidance=lane)
        plan.contexts = same_words
        with plan:
            got = dit(x, t, cross_attn_cond=cond, cfg_scale=3.0)
    expect = plain[1.0] + (lane - 1.0)[None, None, :] / 2.0 * (plain[3.0] - plain[1.0])
    assert float((got - expect).abs().max()) < 1e-4


def test_two_word_sets_in_a_dit_follow_their_frames():
    """Frames that weigh one set fully see exactly that set's words; the plan reaches every block."""
    dit = _tiny_dit()
    T = 40
    x, t = torch.randn(1, 4, T), torch.full((1,), 0.6)
    shared, a, b = torch.randn(1, 9, 16), torch.randn(1, 9, 16), torch.randn(1, 9, 16)
    with torch.no_grad():
        plan = RegionPlan(["a", "b"], torch.stack([torch.ones(T), torch.zeros(T)]), memory=[1.0, 0.0])
        plan.contexts = [dit.to_cond_embed(a), dit.to_cond_embed(b)]
        with plan:
            got = dit(x, t, cross_attn_cond=shared, cfg_scale=1.0)
        ref = dit(x, t, cross_attn_cond=a, cfg_scale=1.0)
    assert float((got - ref).abs().max() / ref.abs().max()) < 1e-5
