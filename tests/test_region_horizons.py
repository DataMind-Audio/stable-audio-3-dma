"""Horizons (models/regions.py, EXPERIMENTAL): frames before a horizon see nothing after it, on CPU.

The mask has the layout the module doc describes for each memory view; a masked self-attention call equals the plain
call on the visible prefix of the sequence; APG's projection and the norm threshold are taken per frame over its
prefix; nothing changes without horizons; and through a real (tiny) DiT with guidance, APG and the norm threshold on,
the frames before a horizon come out the same whatever the later frames and the later words are ('prefix' and
'hidden'), while the 'open' memory still lets the later frames through.
"""
import itertools

import pytest
import torch

from stable_audio_3.models.dit import DiffusionTransformer
from stable_audio_3.models.regions import RegionPlan
from stable_audio_3.models.transformer import Attention, RotaryEmbedding


def _plan(horizons, view="prefix", fps=2.0, tokens=None, sets=1, T=6):
    plan = RegionPlan([f"set {k}" for k in range(sets)], torch.ones(sets, T) / sets, horizons=horizons, memory_view=view)
    plan.fps, plan.patch, plan.tokens = fps, 1, tokens
    plan.contexts = [torch.zeros(1, 1, 64) for _ in range(sets)]      # no cross-attention in these calls
    return plan


def test_the_mask_layout_for_each_memory_view():
    # 6 frames at 0.25 … 2.75 s, horizons at 1 s and 2 s, 2 memory tokens in front
    frames =torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0],
                           [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    for view in ("prefix", "open", "hidden"):
        m = _plan([2.0, 1.0], view).self_mask(8, 6, "cpu")
        assert torch.equal(m[2:, 2:], frames), view
        assert bool(m[:2, :2].all()), view                                   # the memory always sees itself
        if view == "prefix":
            assert torch.equal(m[:2, 2:], torch.tensor([[1, 1, 0, 0, 0, 0]] * 2, dtype=torch.bool))
            assert bool(m[2:, :2].all())
        elif view == "open":
            assert bool(m[:2].all()) and bool(m[2:, :2].all())
        else:
            assert bool(m[:2].all())
            assert torch.equal(m[2:, :2], torch.tensor([[0, 0]] * 4 + [[1, 1]] * 2, dtype=torch.bool))
    assert _plan([1.0]).horizons == [1.0] and _plan([2.0, 1.0, 2.0]).horizons == [1.0, 2.0]
    with pytest.raises(ValueError):
        RegionPlan(["a"], torch.ones(1, 4), horizons=[1.0], memory_view="sideways")
    with pytest.raises(ValueError):
        RegionPlan(["a"], torch.ones(1, 4), horizons=[0.0])
    with pytest.raises(RuntimeError):
        RegionPlan(["a"], torch.ones(1, 4), horizons=[1.0]).self_mask(4, 4, "cpu")     # seconds need the fps


def _attn(diff, qk):
    torch.manual_seed(1)
    a = Attention(64, dim_heads=16, differential=diff, qk_norm=qk, zero_init_output=False).eval()
    a._region_self_site = True
    return a


@pytest.mark.parametrize("diff,qk,B", list(itertools.product((False, True), ("none", "rms", "l2"), (1, 2))))
def test_a_masked_call_equals_the_plain_call_on_the_visible_prefix(diff, qk, B):
    """'prefix': the memory and the frames before the first horizon are a prefix of the sequence — rotary positions
    included — so their rows must equal the plain module run on that prefix alone."""
    attn, rope = _attn(diff, qk), RotaryEmbedding(8)        # partial rotary on the 16-dim heads
    P, tseq = 3, 24                                    # fps 4: frames at 0.125 … 5.875 s; horizon 2.0 s = 8 frames
    x = torch.randn(B, P + tseq, 64)
    plan = _plan([2.0], "prefix", fps=4.0, tokens=tseq, T=tseq)
    with torch.no_grad():
        with plan:
            got = attn(x, rotary_pos_emb=rope.forward_from_seq_len(P + tseq))
        cut = P + 8
        ref = attn._attend(x[:, :cut], rotary_pos_emb=rope.forward_from_seq_len(cut))
        full = attn._attend(x, rotary_pos_emb=rope.forward_from_seq_len(P + tseq))
    assert float((got[:, :cut] - ref).abs().max() / ref.abs().max()) < 1e-5
    assert float((got[:, cut:] - full[:, cut:]).abs().max() / full.abs().max()) < 1e-5   # after it: the whole take


def test_apg_and_the_norm_threshold_per_horizon():
    dit = DiffusionTransformer(io_channels=4, embed_dim=64, cond_token_dim=16, depth=1, num_heads=2,
                               diffusion_objective="rectified_flow")
    torch.manual_seed(2)
    T, f = 30, 12                                      # fps 5: horizon 2.4 s = 12 frames
    v0, v1 = torch.randn(1, 4, T), torch.randn(1, 4, T)
    plan = _plan([2.4], fps=5.0, T=T)
    par, orth = plan.apg_project(v0, v1)
    rp, ro = dit.apg_project(v0[..., :f], v1[..., :f])
    gp, go = dit.apg_project(v0, v1)
    assert torch.allclose(orth[..., :f], ro, atol=1e-5) and torch.allclose(par[..., :f], rp, atol=1e-5)
    assert torch.allclose(orth[..., f:], go[..., f:], atol=1e-5) and torch.allclose(par[..., f:], gp[..., f:], atol=1e-5)
    pm = torch.ones(1, T, dtype=torch.bool)
    pm[:, -4:] = False
    _, orth_m = plan.apg_project(v0, v1, padding_mask=pm)
    _, go_m = dit.apg_project(v0, v1, padding_mask=pm)
    assert torch.allclose(orth_m[..., f:], go_m[..., f:], atol=1e-5) and torch.allclose(orth_m[..., :f], ro, atol=1e-5)
    d = torch.randn(1, 4, T) * 3
    got = plan.norm_threshold(d, 2.0)
    pre = d[..., :f] * torch.clamp(2.0 / d[..., :f].norm(), max=1.0)
    post = d * torch.clamp(2.0 / d.norm(), max=1.0)
    assert torch.allclose(got[..., :f], pre, atol=1e-5) and torch.allclose(got[..., f:], post[..., f:], atol=1e-5)
    plan_none = _plan([], T=T)                           # no horizon: the global projection
    assert torch.allclose(plan_none.apg_project(v0, v1)[1], go, atol=1e-5)


# ---- through a real (tiny) DiffusionTransformer ---------------------------------------------------------------------
def _tiny_dit():
    torch.manual_seed(3)
    dit = DiffusionTransformer(io_channels=4, embed_dim=128, cond_token_dim=16, depth=2, num_heads=2,
                               diffusion_objective="rectified_flow", num_memory_tokens=3,
                               attn_kwargs={"differential": True, "qk_norm": "rms"}).eval()
    with torch.no_grad():                              # the zero-initialised convs and outputs would hide every difference
        dit.preprocess_conv.weight.normal_(0, 0.2)
        dit.postprocess_conv.weight.normal_(0, 0.2)
        for m in dit.modules():
            if isinstance(m, Attention):
                m.to_out.weight.normal_(0, 0.05)
    return dit


def test_the_dit_marks_every_self_attention_as_a_horizon_site():
    dit = _tiny_dit()
    sites = [m for m in dit.modules() if isinstance(m, Attention) and m._region_self_site]
    assert len(sites) == 2 and not any(m._region_site for m in sites)


T, FPS, H = 40, 10.0, 2.0          # 4 s of frames; the horizon at 2 s: frames 0–19 before it


def _render(dit, x, a, b, horizons, view, pm=None, **cfg):
    """One DiT call under a two-set plan: set a up to 2.5 s, then b (the window after the horizon, as a join sends)."""
    times = (torch.arange(T) + 0.5) / FPS
    wb = ((times - 2.5) / 0.5).clamp(0, 1)
    plan = RegionPlan(["a", "b"], torch.stack([1 - wb, wb]), horizons=horizons, memory_view=view)
    plan.contexts, plan.fps, plan.patch = [dit.to_cond_embed(a), dit.to_cond_embed(b)], FPS, 1
    with torch.no_grad(), plan:
        return dit(x, torch.full((1,), 0.6), cross_attn_cond=torch.zeros(1, 9, 16), padding_mask=pm, **cfg)


@pytest.mark.parametrize("cfg", [dict(cfg_scale=1.0),
                                 dict(cfg_scale=3.0, apg_scale=1.0, cfg_norm_threshold=0.5),
                                 dict(cfg_scale=3.0, apg_scale=0.4)], ids=["no-cfg", "apg+threshold", "apg-blend"])
@pytest.mark.parametrize("padded", (False, True))
def test_frames_before_a_horizon_ignore_everything_after_it(cfg, padded):
    dit = _tiny_dit()
    torch.manual_seed(4)
    x, a, b = torch.randn(1, 4, T), torch.randn(1, 9, 16), torch.randn(1, 9, 16)
    x2 = x.clone()
    x2[..., 20:] = torch.randn(1, 4, T - 20)            # other later frames …
    b2 = torch.randn(1, 9, 16)                           # … and other later words
    pm = None
    if padded:
        pm = torch.ones(1, T, dtype=torch.bool)
        pm[:, -3:] = False
    for view in ("prefix", "hidden"):
        one = _render(dit, x, a, b, [H], view, pm, **cfg)
        two = _render(dit, x2, a, b2, [H], view, pm, **cfg)
        assert float((one[..., :20] - two[..., :20]).abs().max()) < 1e-5, view
        assert float((one[..., 20:] - two[..., 20:]).abs().max()) > 1e-3, view       # the later frames did change
    leak = {}
    for name, hz, view in (("open", [H], "open"), ("none", None, "prefix")):
        one = _render(dit, x, a, b, hz, view, pm, **cfg)
        two = _render(dit, x2, a, b2, hz, view, pm, **cfg)
        leak[name] = float((one[..., :20] - two[..., :20]).abs().max())
    assert leak["none"] > 1e-4, leak                    # without horizons the later frames reach the earlier ones
    assert leak["open"] > 1e-6, leak                    # the open memory is a two-hop path


def test_a_horizon_past_the_end_changes_nothing_but_the_kernel():
    """Every frame before a horizon at 1000 s sees every frame: the masked path must give the plain result."""
    dit = _tiny_dit()
    torch.manual_seed(5)
    x, a, b = torch.randn(1, 4, T), torch.randn(1, 9, 16), torch.randn(1, 9, 16)
    plain = _render(dit, x, a, b, None, "prefix", cfg_scale=3.0, apg_scale=1.0)
    far = _render(dit, x, a, b, [1000.0], "prefix", cfg_scale=3.0, apg_scale=1.0)
    assert float((far - plain).abs().max() / plain.abs().max()) < 1e-5
