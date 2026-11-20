"""Adapter bank — LoRA / DoRA adapters as resident, read-only tensors applied per request in activation space.

Why. The parametrize chain (`LoRAParametrization` registered on every adapted weight) keeps the adapter
STATE on the shared model: one `lora_strength` buffer per adapter, `forward_fn` swapped per step for the
σ interval and the layer filter, and a composition order fixed by list position. That is one render at a
time, in one order. It also rebuilds every adapted weight in fp32 on every forward (+0.12 s per step per
active adapter on a MIG 3g.20gb slice, measured 2026-10-28) and attaches in 27 s.

What. `AdapterBank.install()` swaps every plain `nn.Linear` of the DiT and the conditioner to `BankLinear`
(same parameters, same state_dict; a `__class__` swap, as torch's parametrize does). The base weights are
never touched. `AdapterBank.load()` turns a checkpoint into per-site tensors (A, B fp16-of-the-model, the
DoRA magnitude and `W·Aᵀ` in fp32). A `Session` is one request's ordered list of
`(adapter, strength, interval, layer_filter)` plus its own RNG; `with session:` publishes it through a
`contextvars.ContextVar`, so renders in different threads never see each other's adapters.

The maths (exact sequential DoRA, in the request's order). For one site and output row i, with
`d_k = (α_k/r_k) · B_k[i] A_k` and the active adapters k₁…k_J in request order, the chain computes
`v₀ = w; u_j = v_{j−1} + s_{k_j} d_{k_j}; v_j = m_{k_j} · u_j / ‖u_j‖`. Every `v_j` lies in
span{w, d₁…d_K}, so `v_J = c₀ w + Σ c_k d_k` with per-row scalars from the Gram matrix of (w, d₁…d_K):
‖w‖², ⟨w,d_k⟩ = (α/r)·(W Aₖᵀ)[i]·Bₖ[i], ⟨d_j,d_k⟩ = (α/r)²·Bⱼ[i]ᵀ(AⱼAₖᵀ)Bₖ[i]. The forward is then
`y = c₀ ⊙ (W x) + Σ_k (c_k α_k/r_k) ⊙ Bₖ(Aₖ x)`: the base GEMM once plus one low-rank GEMM pair with inner
dimension J·r. Verified against the chain to 2e-7 (fp32, all orders of up to four adapters, Linear and
Conv1d) and on the real model (fp32 2.6e-7; with fp16 GEMMs 3–4e-4, which is closer to the exact answer
than the chain's own fp16 rounding between adapters, 6–10e-4). An adapter at strength 0 is skipped, exactly
as `dora_forward` returns W unchanged; plain `lora` adapters get `c₀ = 1, c_k = s_k`.

Conditioner sites (the `seconds_total` embedder) ignore the interval and the layer filter, as the chain
does — `dit.py` only toggles adapters on the DiT — and run once per request, before sampling.
"""
import contextvars
import threading
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import _expand
from .utils import infer_global_rank, load_lora_checkpoint, prepare_dora_state_dict, resolve_adapter_type

_SESSION: contextvars.ContextVar = contextvars.ContextVar("sa3_adapter_session", default=None)
_KEY_SUFFIX = ".parametrizations.weight.0."
SUPPORTED_ADAPTER_TYPES = ("lora", "dora-rows")


def current_session():
    """The Session active in this thread/task, or None (the base model)."""
    return _SESSION.get()


@dataclass(eq=False)
class Site:
    id: int
    name: str            # module name as the chain's layer filter saw it (relative to its root)
    root: str            # "dit" or "cond"
    module: nn.Module
    w_sq: torch.Tensor   # (out,) fp32: ‖w_i‖²

    @property
    def conditioner(self) -> bool:
        return self.root != "dit"


def solve_device(dev: "torch.device | str") -> torch.device:
    """Where the bank's float64 coefficient solve runs for tensors living on `dev`: the same device, except
    Apple's MPS backend, which has no float64 — there the (rank-sized) operands go to the CPU and the fp32
    coefficients come back. A 2026-11-20 change for the Player on Apple Silicon; CUDA and CPU are untouched."""
    dev = torch.device(dev)
    return torch.device("cpu") if dev.type == "mps" else dev


@dataclass(eq=False)
class SiteTensors:
    A: torch.Tensor                 # (r, in) in the model's dtype
    B: torch.Tensor                 # (out, r) in the model's dtype
    mag: "torch.Tensor | None"      # (out,) fp32 for dora-rows, None for lora
    WA: "torch.Tensor | None"       # (out, r) fp32 = W·Aᵀ (dora-rows only)
    scaling: float
    kind: str

    def nbytes(self) -> int:
        n = self.A.numel() * self.A.element_size() + self.B.numel() * self.B.element_size()
        for t in (self.mag, self.WA):
            if t is not None:
                n += t.numel() * t.element_size()
        return n


@dataclass(eq=False)
class ResidentAdapter:
    key: str
    path: "str | None"
    adapter_type: str
    rank: int
    alpha: float
    scaling: float
    per_site: dict = field(default_factory=dict)    # site id -> SiteTensors
    nbytes: int = 0
    refcount: int = 0
    loaded_at: float = 0.0
    unplaced: list = field(default_factory=list)   # checkpoint keys that matched no installed site


@dataclass
class AdapterSpec:
    """One entry of a request's stack, in request order."""
    adapter: ResidentAdapter
    strength: float
    interval: tuple = (0.0, 1.0)
    layer_filter: str = ""


@dataclass(eq=False)
class Plan:
    c0: torch.Tensor      # (out,) fp32
    A_cat: torch.Tensor   # (J·r, in)
    B_cat: torch.Tensor   # (out, J·r), c_k·α/r folded in


class BankLinear(nn.Linear):
    """`nn.Linear` that applies the active Session's adapters in activation space. No session: plain linear."""

    def forward(self, x):
        sess = _SESSION.get()
        if sess is None:
            return F.linear(x, self.weight, self.bias)
        plan = sess.plan_for(self._site)
        if plan is None:
            return F.linear(x, self.weight, self.bias)
        base = F.linear(x, self.weight)
        low = F.linear(F.linear(x, plan.A_cat), plan.B_cat)
        y = base.float().mul_(plan.c0).add_(low)
        y = y.to(x.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y


def _filter_disables(name: str, layer_filter: str) -> bool:
    """`filter_lora_layers` semantics: comma-separated, lower-cased, bracket-expanded substrings; a match DISABLES."""
    filters = [x.lower().strip() for x in (layer_filter or "").split(",") if len(x.strip())]
    for f in filters:
        for expanded in _expand(f):
            if expanded in name:
                return True
    return False


class Session:
    """One request's adapter stack. Enter it around the whole `generate()` call (the conditioner included)."""

    def __init__(self, bank: "AdapterBank", specs: list, generator=None):
        self.bank = bank
        self.specs = list(specs)
        self.generator = generator
        self.sigma = None
        self._disabled = [frozenset(s.id for s in bank.sites if not s.conditioner and _filter_disables(s.name, sp.layer_filter))
                          for sp in self.specs]
        self._active_cache: dict = {}
        self._plans: dict = {}
        self._token = None
        self._entered = 0

    # -- lifecycle -------------------------------------------------------------------------------------------
    def __enter__(self):
        if self._entered == 0:
            for sp in self.specs:
                sp.adapter.refcount += 1
            self._token = _SESSION.set(self)
        self._entered += 1
        return self

    def __exit__(self, *exc):
        self._entered -= 1
        if self._entered == 0:
            _SESSION.reset(self._token)
            self._token = None
            for sp in self.specs:
                sp.adapter.refcount -= 1
        return False

    def set_sigma(self, sigma: float):
        """Called by the DiT once per forward with σ of the current step; invalidates the per-σ active sets."""
        if sigma != self.sigma:
            self.sigma = sigma
            self._active_cache.clear()

    # -- gating ----------------------------------------------------------------------------------------------
    def active_for(self, site: Site) -> tuple:
        """Indices (request order) of the adapters that act on `site` at the current σ."""
        key = site.id
        hit = self._active_cache.get(key)
        if hit is not None:
            return hit
        out = []
        for k, sp in enumerate(self.specs):
            if sp.strength <= 0 or site.id not in sp.adapter.per_site:
                continue
            if not site.conditioner:
                if site.id in self._disabled[k]:
                    continue
                if self.sigma is not None and not (sp.interval[0] <= self.sigma <= sp.interval[1]):
                    continue
            out.append(k)
        res = tuple(out)
        self._active_cache[key] = res
        return res

    def plan_for(self, site: Site):
        act = self.active_for(site)
        if not act:
            return None
        key = (site.id, act)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._build_plan(site, act)
            self._plans[key] = plan
        return plan

    # -- the maths -------------------------------------------------------------------------------------------
    def _build_plan(self, site: Site, act: tuple) -> Plan:
        tens = [self.specs[k].adapter.per_site[site.id] for k in act]
        strengths = [float(self.specs[k].strength) for k in act]
        keys = [self.specs[k].adapter.key for k in act]
        c = self.bank.coefficients(site, tens, strengths, keys)          # (out, J+1) fp32
        dtype = site.module.weight.dtype
        A_cat = torch.cat([t.A for t in tens], 0).to(dtype)
        B_cat = torch.cat([(t.B.float() * (c[:, j + 1] * t.scaling).unsqueeze(1)) for j, t in enumerate(tens)], 1).to(dtype)
        return Plan(c0=c[:, 0].contiguous(), A_cat=A_cat.contiguous(), B_cat=B_cat.contiguous())


class AdapterBank:
    def __init__(self):
        self.sites: list = []
        self._by_root_name: dict = {}        # (root, name) -> Site
        self.adapters: dict = {}             # key -> ResidentAdapter
        self._load_locks: dict = {}
        self._lock = threading.Lock()
        self._gram: dict = {}                # (site id, key_j, key_k) -> (r_j, r_k) fp64 AⱼAₖᵀ
        self.installed_roots: dict = {}

    # -- install ---------------------------------------------------------------------------------------------
    def install(self, roots: dict) -> int:
        """Swap every plain nn.Linear under each root to BankLinear. roots = {"dit": DiT, "cond": conditioner}.
        Modules that are already parametrized (the chain) are left alone — bank and chain do not mix."""
        n = 0
        for root_tag, root in roots.items():
            if root is None:
                continue
            self.installed_roots[root_tag] = root
            for name, mod in root.named_modules():
                if type(mod) is not nn.Linear:
                    continue
                with torch.no_grad():
                    w_sq = mod.weight.detach().float().pow(2).sum(1)
                site = Site(id=len(self.sites), name=name, root=root_tag, module=mod, w_sq=w_sq)
                mod.__class__ = BankLinear
                mod._site = site
                self.sites.append(site)
                self._by_root_name[(root_tag, name)] = site
                n += 1
        return n

    def site_for(self, name: str):
        for root_tag in self.installed_roots:
            s = self._by_root_name.get((root_tag, name))
            if s is not None:
                return s
        return None

    # -- load ------------------------------------------------------------------------------------------------
    def load(self, path: str, key: "str | None" = None) -> ResidentAdapter:
        """Load a checkpoint once (single-flight per path); later calls return the resident adapter."""
        key = key or path
        with self._lock:
            hit = self.adapters.get(key)
            if hit is not None:
                return hit
            lock = self._load_locks.setdefault(path, threading.Lock())
        with lock:
            with self._lock:
                hit = self.adapters.get(key)
                if hit is not None:
                    return hit
            state_dict, cfg = load_lora_checkpoint(path)
            return self.load_state(state_dict, cfg, key, path=path)

    def load_state(self, state_dict: dict, cfg: dict, key: str, path: "str | None" = None) -> ResidentAdapter:
        atype = resolve_adapter_type(cfg.get("adapter_type", "lora"), state_dict)
        if atype not in SUPPORTED_ADAPTER_TYPES:
            raise NotImplementedError(f"adapter bank: adapter_type {atype!r} is not supported (chain mode handles it)")
        rank = int(cfg.get("rank", infer_global_rank(state_dict)))
        alpha = float(cfg.get("alpha", rank))
        scaling = alpha / rank
        prepare_dora_state_dict(state_dict)
        ad = ResidentAdapter(key=key, path=path, adapter_type=atype, rank=rank, alpha=alpha, scaling=scaling, loaded_at=time.time())
        with torch.no_grad():
            for k, A in state_dict.items():
                if not k.endswith("lora_A"):
                    continue
                base = k[: -len(_KEY_SUFFIX + "lora_A")]
                site = self.site_for(base)
                if site is None:
                    ad.unplaced.append(k)
                    continue
                W = site.module.weight
                if A.dim() != 2:
                    raise NotImplementedError(f"adapter bank: {k} has {A.dim()} dims; only Linear sites are supported")
                B = state_dict[base + _KEY_SUFFIX + "lora_B"]
                if A.shape[1] != W.shape[1] or B.shape[0] != W.shape[0]:
                    raise ValueError(f"adapter bank: {base}: A {tuple(A.shape)} / B {tuple(B.shape)} vs weight {tuple(W.shape)}")
                A = A.to(device=W.device, dtype=W.dtype)
                B = B.to(device=W.device, dtype=W.dtype)
                mag = WA = None
                if atype == "dora-rows":
                    mag = state_dict[base + _KEY_SUFFIX + "magnitude"].reshape(-1).to(device=W.device, dtype=torch.float32)
                    WA = W.detach().float() @ A.float().T
                t = SiteTensors(A=A.contiguous(), B=B.contiguous(), mag=mag, WA=WA, scaling=scaling, kind=atype)
                ad.per_site[site.id] = t
                ad.nbytes += t.nbytes()
        if not ad.per_site:
            raise ValueError(f"adapter bank: {path or key}: no tensor matched an installed site (first key: {next(iter(state_dict), None)})")
        with self._lock:
            self.adapters[key] = ad
        return ad

    def remove(self, key: str) -> bool:
        """Free an adapter nobody is rendering with. Returns False if it is in use or unknown."""
        with self._lock:
            ad = self.adapters.get(key)
            if ad is None or ad.refcount > 0:
                return False
            del self.adapters[key]
            for gk in [g for g in self._gram if key in g[1:]]:
                del self._gram[gk]
        return True

    def resident_bytes(self) -> int:
        return sum(a.nbytes for a in self.adapters.values())

    # -- maths -----------------------------------------------------------------------------------------------
    def gram_AA(self, site_id: int, key_j: str, key_k: str, A_j: torch.Tensor, A_k: torch.Tensor) -> torch.Tensor:
        gk = (site_id, key_j, key_k)
        g = self._gram.get(gk)
        if g is None:
            wdev = solve_device(A_j.device)
            g = A_j.to(wdev, torch.float64) @ A_k.to(wdev, torch.float64).T
            self._gram[gk] = g
            if key_j != key_k:
                self._gram[(site_id, key_k, key_j)] = g.T
        return g

    def coefficients(self, site: Site, tens: list, strengths: list, keys: list) -> torch.Tensor:
        """(out, J+1) fp32: c₀ for the base row and c_k for each active adapter, sequential composition in list order."""
        dev = site.w_sq.device
        # The solve is float64 (the per-row norms of a DoRA stack cancel badly in fp32). MPS has no float64,
        # so there the small operands come to the CPU and the fp32 coefficients go back (solve_device);
        # on CUDA and CPU nothing moves and the arithmetic is unchanged.
        wdev = solve_device(dev)
        d64 = lambda x: x.to(wdev, torch.float64)  # noqa: E731
        J = len(tens)
        G = torch.zeros(site.w_sq.shape[0], J + 1, J + 1, dtype=torch.float64, device=wdev)
        G[:, 0, 0] = d64(site.w_sq)
        needs_gram = any(t.kind == "dora-rows" for t in tens)
        if needs_gram:
            for k, t in enumerate(tens):
                if t.WA is None:
                    with torch.no_grad():
                        t.WA = site.module.weight.detach().float() @ t.A.float().T
                G[:, 0, k + 1] = G[:, k + 1, 0] = t.scaling * (d64(t.WA) * d64(t.B)).sum(1)
            for j, tj in enumerate(tens):
                for k, tk in enumerate(tens):
                    AA = self.gram_AA(site.id, keys[j], keys[k], tj.A, tk.A)
                    G[:, j + 1, k + 1] = tj.scaling * tk.scaling * ((d64(tj.B) @ AA) * d64(tk.B)).sum(1)
        c = torch.zeros(site.w_sq.shape[0], J + 1, dtype=torch.float64, device=wdev)
        c[:, 0] = 1.0
        for k, (t, s) in enumerate(zip(tens, strengths)):
            if s == 0:
                continue
            c[:, k + 1] += s
            if t.kind == "dora-rows":
                n2 = torch.einsum("oi,oij,oj->o", c, G, c).clamp_min(0.0)
                c = c * (d64(t.mag) / (n2.sqrt() + 1e-12)).unsqueeze(1)
        return c.float().to(dev)

    # -- sessions --------------------------------------------------------------------------------------------
    def session(self, specs: list, generator=None) -> Session:
        return Session(self, specs, generator=generator)

    def summary(self) -> dict:
        return {"sites": len(self.sites), "adapters": sorted(self.adapters), "bytes": self.resident_bytes()}
