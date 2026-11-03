"""Regions — one diffusion pass whose words (and guidance) change per latent frame.

Why. A Studio track has sections, each with its own words. Rendering it as a whole-track pass plus one inpaint pass
per section makes a section ignore its words: with the rest of the track held as context, the context wins (probe 8,
2026-10-29: prompt effect 0.06 / 0.26 against 0.92 for a plain one-prompt render, at any depth from 0.5 to 1.0). In
SA3 the prompt enters ONLY through cross-attention (cross_attn_cond_ids = prompt + seconds_total; the inpaint mask and
the kept audio come in by input concat), so every latent frame can read its own section's words in the same pass.

What. A `RegionPlan` holds K word sets (prompts), per-frame weights over them (a [K, T] tensor or a function of the
frames' centre times in seconds), what the transformer's memory tokens read (default: the generate() prompt — the
shared words), and optionally a per-frame guidance lane. `with plan:` publishes it through a ContextVar, as the
adapter bank's Session does, so renders in other threads never see it. `generate(..., regions=plan)` encodes the sets
(`prepare`), projects them through the DiT's `to_cond_embed` and enters the plan around sampling.

The maths (the fast form, probe 10). In each DiT cross-attention module the queries are computed once; every set's
keys and values are computed once per render and cached (the words do not change between steps; the cache key carries
the adapter set acting on `to_kv`, so a LoRA interval still applies); one batched attention call covers the K sets;
the K outputs are mixed per token BEFORE `to_out`, which is exact because `to_out` is linear with no bias (feat_scale's
mean and high-frequency terms are linear in the mix too). Only the conditional batch row is mixed; the unconditional
row keeps its own context. The P memory tokens in front of the frames read the generate() context unless `memory`
gives them a mix of the sets. Measured on a MIG 3g.20gb slice: diffusion +5–8 % for 2 sets, +9–15 % for 4, +16–31 %
for 8; 0.017 s of text encoding per set; equal to the per-set reference (the whole module once per set, outputs mixed)
to 1.7e-3 per call in fp16 and 1e-5 in fp32 (tests/test_regions.py).

Guidance lane. With scale_phi = 0 the guided prediction is linear in the scale — APG's projection and the norm
threshold do not depend on it — so cond + (g − 1)·Δ gives each frame its own g at no cost. generate() runs CFG at the
lane's maximum so the branch is taken whenever any frame wants guidance; the DiT's CFG combine reads the lane.

Horizons (EXPERIMENTAL, 2026-11-03). A frame never reads a later section's words before that section's window (its
weight there is 0), but it does see the later frames: self-attention spans the take, so a section about to change
anticipates the next one (a hard cut at 55 s with strict guidance brought trap hats in at 48 s). `horizons` are
times in seconds (the Studio sends a blind join's window start): a frame before a horizon attends only to the frames
before it — the frames at or after the last horizon see the whole take, so a later section still hears what came
before. Every other path across time goes with it: APG's projection and the norm threshold are taken per frame over
the frames it may see (one scalar over the whole take otherwise), and the memory tokens, which attend to every frame,
follow `memory_view`: 'prefix' (default) — they see only the frames before the first horizon, so the frames before it
are sealed from everything after; 'open' — they see the whole take (a two-hop path from later frames remains);
'hidden' — they see the whole take, and frames before a horizon do not see them. Under a plan without horizons
nothing changes. The DiT's convolutions are kernel-1; the autoencoder's decode window still spans ~1–2 s of latent.
"""
import contextvars

import torch
from einops import rearrange

_PLAN: contextvars.ContextVar = contextvars.ContextVar("sa3_regions_plan", default=None)

# cross-attention arguments the fast form does not implement; a call that carries one takes the per-set reference path
UNSUPPORTED_KWARGS = ("rotary_pos_emb", "rotary_pos_emb_k", "flex_attention_block_mask", "flex_attention_score_mod",
                      "flash_attn_sliding_window", "padding_mask", "varlen_metadata")


def current_plan():
    return _PLAN.get()


def mark_region_sites(dit) -> int:
    """Flag the DiT's cross-attention modules as region sites and its self-attention modules as horizon sites. The
    conditioner's encoder and the autoencoder are not DiT submodules, so their attention is never touched. Returns the
    number of cross-attention sites."""
    from .transformer import TransformerBlock
    n = 0
    for m in dit.modules():
        if isinstance(m, TransformerBlock):
            m.self_attn._region_self_site = True
            if getattr(m, "cross_attn", None) is not None:
                m.cross_attn._region_site = True
                n += 1
    return n


MEMORY_VIEWS = ("prefix", "open", "hidden")


def frame_times(n, patch, fps, device):
    """Centre time in seconds of each of n tokens of `patch` latent frames."""
    return (torch.arange(n, device=device, dtype=torch.float32) * patch + patch / 2) / fps


def _heads(t, h):
    return rearrange(t, "b n (h d) -> b h n d", h=h)


def _qk_norm(attn, t, which):
    if attn.qk_norm == "l2":
        return torch.nn.functional.normalize(t, dim=-1, eps=attn.qk_norm_eps)
    if attn.qk_norm != "none":
        return (attn.q_norm if which == "q" else attn.k_norm)(t).to(t.dtype)
    return t


def _adapter_signature(linear):
    """Which bank adapters act on this projection at the current σ (None without a session)."""
    site = getattr(linear, "_site", None)
    if site is None:
        return None
    from .lora.bank import current_session
    sess = current_session()
    return None if sess is None else sess.active_for(site)


class RegionPlan:
    """K word sets mixed per latent frame, for one generate() call (batch size 1).

    prompts   K prompt strings, encoded by `prepare` the way generate() encodes its own prompt.
    weights   [K, T] per latent frame, or f(times [T] seconds) -> [K, T]. Not renormalised: columns usually sum to 1;
              a negative weight extrapolates (probe 10 used -0.5 / 1.5).
    memory    None: the memory tokens read generate()'s own prompt (the shared words). Or a [K] mix of the sets.
    guidance  None, [T], or f(times) -> [T]: the guidance scale per frame. Needs `guidance_max` with a function.
    horizons  EXPERIMENTAL: times in seconds; a frame before one attends only to the frames before it (module doc).
    memory_view  with horizons: 'prefix' | 'open' | 'hidden' — what the memory tokens see and who sees them.
    """

    def __init__(self, prompts, weights, memory=None, guidance=None, guidance_max=None, horizons=None,
                 memory_view="prefix"):
        if not prompts:
            raise ValueError("a RegionPlan needs at least one word set")
        if memory_view not in MEMORY_VIEWS:
            raise ValueError(f"memory_view must be one of {MEMORY_VIEWS}, got {memory_view!r}")
        self.horizons = sorted({float(h) for h in (horizons or [])})
        if any(h <= 0 for h in self.horizons):
            raise ValueError("horizons must be after the start of the take")
        self.memory_view = memory_view
        self.prompts = list(prompts)
        self.weights = weights if callable(weights) else torch.as_tensor(weights, dtype=torch.float32)
        if not callable(self.weights) and (self.weights.dim() != 2 or self.weights.shape[0] != len(self.prompts)):
            raise ValueError(f"weights must be [K={len(self.prompts)}, T], got {tuple(self.weights.shape)}")
        self.memory = None if memory is None else torch.as_tensor(memory, dtype=torch.float32)
        if self.memory is not None and tuple(self.memory.shape) != (len(self.prompts),):
            raise ValueError(f"memory must be [K={len(self.prompts)}], got {tuple(self.memory.shape)}")
        self.guidance = guidance if (guidance is None or callable(guidance)) else torch.as_tensor(guidance, dtype=torch.float32)
        if guidance is not None and guidance_max is None:
            if callable(guidance):
                raise ValueError("a guidance function needs guidance_max")
            guidance_max = float(self.guidance.max())
        self.guidance_max = None if guidance_max is None else float(guidance_max)
        self.contexts = None       # K projected contexts [1, n_ctx, dim] (prepare(), or set directly in tests)
        self.fps = None            # latent frames per second (prepare())
        self.patch = 1
        self.tokens = None         # tokens per sequence in the current DiT call (DiT.forward publishes it)
        self.fallbacks = 0         # calls that took the per-set reference path
        self._kv = {}
        self._cache = {}
        self._token = None

    # -- lifecycle -------------------------------------------------------------------------------------------------
    def __enter__(self):
        if self.contexts is None:
            raise RuntimeError("RegionPlan.prepare() has not run")
        self._token = _PLAN.set(self)
        return self

    def __exit__(self, *exc):
        _PLAN.reset(self._token)
        self._token = None
        self._kv.clear()
        self._cache.clear()
        return False

    def prepare(self, wrapper, device, seconds_total):
        """Encode every set's words as generate() encodes its prompt, and project them for the DiT's cross-attention.
        `wrapper` is the conditioned diffusion wrapper (model.model): its conditioner, cross_attn_cond_ids and DiT."""
        from .dit import DiffusionTransformer
        dit = next(m for m in wrapper.modules() if isinstance(m, DiffusionTransformer))
        dtype = next(dit.parameters()).dtype
        self.patch = int(getattr(dit, "patch_size", 1))
        pt = getattr(wrapper, "pretransform", None)
        self.fps = wrapper.sample_rate / (pt.downsampling_ratio if pt is not None else 1)
        ctxs = []
        for p in self.prompts:
            cond = wrapper.conditioner([{"prompt": p, "seconds_total": seconds_total}], device)
            parts = []
            for ck in wrapper.cross_attn_cond_ids:
                x = cond[ck][0]
                parts.append(x.unsqueeze(1) if x.dim() == 2 else x)
            raw = torch.cat(parts, dim=1).to(dtype)
            ctxs.append(dit.to_cond_embed(raw) if hasattr(dit, "to_cond_embed") else raw)
        self.contexts = ctxs
        return self

    # -- per-frame values ------------------------------------------------------------------------------------------
    def _frames_value(self, src, n, patch, device, what):
        if callable(src):
            if self.fps is None:
                raise RuntimeError(f"{what} is a function of time: prepare() (or fps) is needed")
            return src(frame_times(n, patch, self.fps, device)).to(device=device, dtype=torch.float32)
        v = src.to(device=device, dtype=torch.float32)
        if patch > 1 and v.shape[-1] == n * patch:
            v = v.reshape(*v.shape[:-1], n, patch).mean(-1)
        if v.shape[-1] != n:
            raise ValueError(f"{what} covers {v.shape[-1]} frames, the model sees {n}")
        return v

    def token_weights(self, tseq, device):
        """[K, tseq] float32 weights for the sequence tokens."""
        key = ("w", tseq, str(device))
        if key not in self._cache:
            self._cache[key] = self._frames_value(self.weights, tseq, self.patch, device, "weights")
        return self._cache[key]

    def lane(self, frames, device, dtype):
        """[1, 1, frames] guidance per latent frame, for the DiT's CFG combine."""
        key = ("g", frames, str(device), dtype)
        if key not in self._cache:
            self._cache[key] = self._frames_value(self.guidance, frames, 1, device, "guidance").to(dtype)[None, None, :]
        return self._cache[key]

    # -- horizons: who may see whom across time ---------------------------------------------------------------------
    def token_horizons(self, tseq, device):
        """(times [tseq], horizon [tseq]): each sequence token's centre time and the first horizon after it (inf when
        none: it sees the whole take)."""
        key = ("h", tseq, str(device))
        if key not in self._cache:
            if self.fps is None:
                raise RuntimeError("horizons are in seconds: prepare() (or fps) is needed")
            t = frame_times(tseq, self.patch, self.fps, device)
            hz = torch.full_like(t, float("inf"))
            for h in reversed(self.horizons):
                hz = torch.where(t < h, torch.full_like(t, h), hz)
            self._cache[key] = (t, hz)
        return self._cache[key]

    def self_mask(self, n, tseq, device):
        """bool [n, n], True = may attend: n = P memory (and prepend) tokens + tseq sequence tokens."""
        key = ("m", n, tseq, str(device))
        if key not in self._cache:
            P = n - tseq
            t, hz = self.token_horizons(tseq, device)
            m = torch.ones(n, n, dtype=torch.bool, device=device)
            m[P:, P:] = t[None, :] < hz[:, None]                 # a frame sees the frames before its horizon
            if P > 0 and self.memory_view == "prefix":
                m[:P, P:] = (t < self.horizons[0])[None, :]      # the memory sees the frames before the first horizon
            elif P > 0 and self.memory_view == "hidden":
                m[P:, :P] = ~torch.isfinite(hz)[:, None]         # frames before a horizon do not see the memory
            self._cache[key] = m
        return self._cache[key]

    def self_attention(self, attn, x, kw, attend):
        """One call of a horizon site. `attend(x, **kw)` is the module's own self-attention."""
        tseq, n = self.tokens, x.shape[1]
        if not self.horizons or tseq is None or tseq > n:
            return attend(x, **kw)                   # not a DiT sequence this plan was published for
        return attend(x, attn_mask=self.self_mask(n, tseq, x.device), **kw)

    def frame_extent(self, T, device):
        """long [T]: for each latent frame, how many latent frames from the start it may use — the frames before its
        token's horizon (a prefix of the take), or all T."""
        key = ("e", T, str(device))
        if key not in self._cache:
            p = max(1, int(self.patch))
            tseq = T // p
            t, hz = self.token_horizons(tseq, device)
            visible = torch.searchsorted(t.contiguous(), hz.contiguous())       # tokens with t < horizon
            ext = (visible * p).repeat_interleave(p)
            if ext.numel() < T:
                ext = torch.cat([ext, torch.full((T - ext.numel(),), T, dtype=ext.dtype, device=ext.device)])
            self._cache[key] = ext.clamp(min=1, max=T)
        return self._cache[key]

    def apg_project(self, v0, v1, padding_mask=None):
        """DiffusionTransformer.apg_project, each frame's projection taken over the frames it may see. Without
        horizons every frame sees the take and this is the global projection (up to summation order)."""
        dtype = v0.dtype
        v0, v1 = v0.float(), v1.float()
        idx = self.frame_extent(v0.shape[-1], v0.device) - 1
        m = padding_mask.unsqueeze(1).float() if padding_mask is not None else None
        v0m, v1m = (v0 * m, v1 * m) if m is not None else (v0, v1)
        dot = (v0m * v1m).sum(1).cumsum(-1)[:, idx]                       # [B, T]
        sq = (v1m * v1m).sum(1).cumsum(-1)[:, idx]
        parallel = (dot / sq.clamp(min=1e-16))[:, None, :] * v1m
        orthogonal = v0 - parallel
        if m is not None:
            orthogonal = orthogonal * m
        return parallel.to(dtype), orthogonal.to(dtype)

    def norm_threshold(self, diff, threshold, padding_mask=None):
        """The CFG norm threshold, each frame's norm taken over the frames it may see."""
        idx = self.frame_extent(diff.shape[-1], diff.device) - 1
        d = diff.float() * padding_mask.unsqueeze(1).float() if padding_mask is not None else diff.float()
        norm = (d * d).sum(1).cumsum(-1)[:, idx].sqrt()                  # [B, T]
        scale = torch.minimum(torch.ones_like(norm), threshold / norm)
        return diff * scale[:, None, :].to(diff.dtype)

    def _full_weights(self, w, n, P):
        """w [K, n-P] over the sequence tokens -> [K+1, n] with P memory tokens in front. They read the shared words
        (the last set: generate()'s own context) unless `memory` gives them a mix of the K sets."""
        if P == 0:
            return w
        wf = torch.zeros(w.shape[0] + 1, n, device=w.device, dtype=w.dtype)
        wf[:-1, P:] = w
        if self.memory is None:
            wf[-1, :P] = 1.0
        else:
            wf[:-1, :P] = self.memory.to(device=w.device, dtype=w.dtype)[:, None]
        return wf

    # -- the cross-attention call ----------------------------------------------------------------------------------
    def _kv_for(self, attn, ctxs):
        key = (id(attn), _adapter_signature(attn.to_kv))
        c = self._kv.get(key)
        if c is None:
            ks, kds, vs = [], [], []
            for ck in ctxs:
                if attn.differential:
                    k, kd, v = (_heads(t, attn.kv_heads) for t in attn.to_kv(ck).chunk(3, dim=-1))
                    kds.append(_qk_norm(attn, kd, "k"))
                else:
                    k, v = (_heads(t, attn.kv_heads) for t in attn.to_kv(ck).chunk(2, dim=-1))
                ks.append(_qk_norm(attn, k, "k"))
                vs.append(v)
            c = {"k": torch.cat(ks), "kd": torch.cat(kds) if kds else None, "v": torch.cat(vs)}
            self._kv[key] = c
        return c

    def _fast_row(self, attn, x0, w, ctxs):
        """The conditional row of one cross-attention call under the sets mixed per token by w [K', n]."""
        h = attn.num_heads
        if attn.differential:
            q, qd = (_qk_norm(attn, _heads(t, h), "q") for t in attn.to_q(x0).chunk(2, dim=-1))
        else:
            q, qd = _qk_norm(attn, _heads(attn.to_q(x0), h), "q"), None
        c = self._kv_for(attn, ctxs)
        K = c["k"].shape[0]
        a = attn.apply_attn(q.expand(K, -1, -1, -1).contiguous(), c["k"], c["v"], causal=False)
        if qd is not None:
            a = a - attn.apply_attn(qd.expand(K, -1, -1, -1).contiguous(), c["kd"], c["v"], causal=False)
        w = w.to(a.dtype)
        out = attn.to_out(rearrange((a * w[:, None, :, None]).sum(0, keepdim=True), "b h n d -> b n (h d)"))
        if getattr(attn, "feat_scale", False):
            means = attn.to_out(rearrange(a.mean(dim=2, keepdim=True), "b h n d -> b n (h d)"))   # [K', 1, dim]
            dc = (w[:, :, None] * means).sum(0, keepdim=True)                                       # [1, n, dim]
            out = out + attn.lambda_dc * dc + attn.lambda_hf * (out - dc)
        return out

    def cross_attention(self, attn, x, context, kw, attend):
        """One call of a region site. `attend(x, context, **kw)` is the module's own computation."""
        tseq, n = self.tokens, x.shape[1]
        if tseq is None or tseq > n:
            return attend(x, context, **kw)          # not a DiT sequence this plan was published for
        P = n - tseq
        wf = self._full_weights(self.token_weights(tseq, x.device), n, P)
        ctxs = list(self.contexts) + ([context[:1]] if P > 0 else [])
        if hasattr(attn, "to_q") and not any(kw.get(k) is not None for k in UNSUPPORTED_KWARGS):
            out0 = self._fast_row(attn, x[:1], wf, ctxs)
        else:
            self.fallbacks += 1
            wd = wf.to(x.dtype)
            out0 = None
            for k, ck in enumerate(ctxs):
                ok = attend(x[:1], ck, **kw) * wd[k][None, :, None]
                out0 = ok if out0 is None else out0 + ok
        if x.shape[0] == 1:
            return out0
        return torch.cat([out0, attend(x[1:], context[1:], **kw)], dim=0)
