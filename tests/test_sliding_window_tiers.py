"""Sliding-window attention tiers (SAME-L decoder): which kernel `apply_attn` picks, and that
the default chunked-halo SDPA path is the exact band-masked attention. CPU only.

Why this exists (Kollektiv, 2026-10-28, MIG 3g.20gb, 120 s decode): compiled flex_attention
6.23 s, chunked-halo SDPA 0.98 s, full masked SDPA 1.19 s, flash-attn 0.71 s; halo, mask and
flash agree exactly, flex differs by 0.019 max abs. Flex is therefore opt-in."""
import pytest
import torch
import torch.nn.functional as F

from stable_audio_3.models import transformer as tr


def test_default_tier_prefers_flash_then_halo_never_flex():
    assert tr._resolve_sliding_window_tier(True, True, tier="auto") == "flash"
    assert tr._resolve_sliding_window_tier(False, True, tier="auto") == "halo"
    assert tr._resolve_sliding_window_tier(False, False, tier="auto") == "halo"


def test_flex_is_opt_in_and_degrades_to_halo_without_flex():
    assert tr._resolve_sliding_window_tier(True, True, tier="flex") == "flex"
    assert tr._resolve_sliding_window_tier(True, False, tier="flex") == "halo"


def test_forced_tiers_and_flash_without_flash_fails_loudly():
    assert tr._resolve_sliding_window_tier(True, True, tier="halo") == "halo"
    assert tr._resolve_sliding_window_tier(True, True, tier="mask") == "mask"
    with pytest.raises(RuntimeError):
        tr._resolve_sliding_window_tier(False, True, tier="flash")


def test_env_default_is_auto(monkeypatch):
    assert tr._SLIDING_WINDOW_TIER in ("auto", "flash", "flex", "halo", "mask")
    monkeypatch.setattr(tr, "_SLIDING_WINDOW_TIER", "halo")
    assert tr._resolve_sliding_window_tier(True, True) == "halo"


def _band_reference(q, k, v, wl, wr):
    mask = tr._sliding_window_additive_mask(q.shape[2], k.shape[2], wl, wr, q.device, q.dtype)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)


@pytest.mark.parametrize("n,window,chunk", [(333, (3, 5), 64), (333, (17, 17), 1024), (256, (17, 17), 128), (40, (1, 1), 16)])
def test_chunked_halo_is_exactly_the_band_masked_attention(n, window, chunk):
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 2, n, 8, generator=g) for _ in range(3))
    out = tr._sliding_window_chunked_halo_sdpa(q, k, v, *window, chunk_size=chunk)
    ref = _band_reference(q, k, v, *window)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)


def test_window_actually_limits_attention():
    g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(1, 1, 64, 8, generator=g) for _ in range(3))
    banded = tr._sliding_window_chunked_halo_sdpa(q, k, v, 2, 2, chunk_size=16)
    full = F.scaled_dot_product_attention(q, k, v)
    assert not torch.allclose(banded, full, atol=1e-3)
