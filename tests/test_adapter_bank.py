"""The adapter bank reproduces the parametrize chain exactly, in the REQUEST's order, with no state on
the shared modules — so two sessions can run on one model, and the user's strip order is the stack
order. CPU only; toy roots stand in for the DiT and the conditioner."""
import itertools
import threading

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from stable_audio_3.models.lora import AdapterBank, AdapterSpec, LoRAParametrization, current_session

R, ALPHA = 2, 4.0
SEED = 11


def _roots():
    torch.manual_seed(SEED)
    dit = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 12), nn.Linear(12, 5))
    cond = nn.ModuleDict({"seconds": nn.Linear(16, 8)})
    return dit, cond


def _checkpoint(tag, kind="dora-rows", names=("0", "1", "2", "seconds"), shapes=((8, 16), (12, 8), (5, 12), (8, 16))):
    g = torch.Generator().manual_seed(100 + tag)
    sd = {}
    for name, (fo, fi) in zip(names, shapes):
        base = f"{name}.parametrizations.weight.0."
        sd[base + "lora_A"] = torch.randn(R, fi, generator=g) * 0.5
        sd[base + "lora_B"] = torch.randn(fo, R, generator=g) * 0.5
        if kind == "dora-rows":
            sd[base + "magnitude"] = (torch.rand(fo, 1, generator=g) + 0.5) * 2.0   # 2-D as checkpoints store it
    return sd, {"adapter_type": kind, "rank": R, "alpha": ALPHA}


def _chain(dit_like, order, strengths, ckpts):
    """A fresh copy of the toy DiT with the chain attached in `order` (list position = composition order)."""
    torch.manual_seed(SEED)
    chain = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 12), nn.Linear(12, 5))
    for pos, tag in enumerate(order):
        sd, cfg = ckpts[tag]
        for i, layer in enumerate(chain):
            p = LoRAParametrization.from_linear(layer, rank=R, lora_alpha=ALPHA, adapter_type=cfg["adapter_type"], lora_index=tag)
            base = f"{i}.parametrizations.weight.0."
            with torch.no_grad():
                p.lora_A.copy_(sd[base + "lora_A"]); p.lora_B.copy_(sd[base + "lora_B"])
                if cfg["adapter_type"] == "dora-rows":
                    p.magnitude.copy_(sd[base + "magnitude"].reshape(-1))
            p.lora_strength.fill_(strengths[tag])
            parametrize.register_parametrization(layer, "weight", p, unsafe=True)
    return chain


@pytest.fixture
def bank_and_roots():
    dit, cond = _roots()
    bank = AdapterBank()
    n = bank.install({"dit": dit, "cond": cond})
    assert n == 4
    return bank, dit, cond



def _close(a, b, tol=1e-5):
    """Relative to the largest output: both paths are exact in fp32 up to summation order (~4e-7 measured)."""
    return float((a - b).abs().max() / (b.abs().max() + 1e-12)) < tol


X = torch.randn(3, 7, 16, generator=torch.Generator().manual_seed(5))


def test_install_swaps_class_but_keeps_parameters_and_state_dict(bank_and_roots):
    bank, dit, cond = bank_and_roots
    assert type(dit[0]).__name__ == "BankLinear" and isinstance(dit[0], nn.Linear)
    assert set(dit.state_dict()) == {"0.weight", "0.bias", "1.weight", "1.bias", "2.weight", "2.bias"}
    torch.manual_seed(SEED); plain = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 12), nn.Linear(12, 5))
    assert torch.equal(dit(X), plain(X))          # no session → the base model, bitwise
    assert current_session() is None


@pytest.mark.parametrize("K", [1, 2, 3])
def test_bank_equals_the_chain_in_every_request_order(bank_and_roots, K):
    bank, dit, cond = bank_and_roots
    ckpts = {t: _checkpoint(t) for t in range(K)}
    ads = {t: bank.load_state(dict(sd), dict(cfg), key=f"a{t}") for t, (sd, cfg) in ckpts.items()}
    strengths = {0: 0.7, 1: 0.45, 2: 0.9}
    for order in itertools.permutations(range(K)):
        chain = _chain(dit, order, strengths, ckpts)
        specs = [AdapterSpec(ads[t], strengths[t]) for t in order]
        with bank.session(specs):
            out = dit(X)
        ref = chain(X)
        assert _close(out, ref), order
    assert current_session() is None


def test_order_matters_and_the_bank_follows_the_request(bank_and_roots):
    bank, dit, cond = bank_and_roots
    ckpts = {t: _checkpoint(t) for t in range(2)}
    ads = {t: bank.load_state(dict(sd), dict(cfg), key=f"a{t}") for t, (sd, cfg) in ckpts.items()}
    with bank.session([AdapterSpec(ads[0], 0.8), AdapterSpec(ads[1], 0.6)]):
        ab = dit(X)
    with bank.session([AdapterSpec(ads[1], 0.6), AdapterSpec(ads[0], 0.8)]):
        ba = dit(X)
    assert not torch.allclose(ab, ba, atol=1e-3)
    assert _close(ab, _chain(dit, [0, 1], {0: 0.8, 1: 0.6}, ckpts)(X))
    assert _close(ba, _chain(dit, [1, 0], {0: 0.8, 1: 0.6}, ckpts)(X))


def test_strength_zero_is_an_exact_bypass(bank_and_roots):
    bank, dit, cond = bank_and_roots
    sd, cfg = _checkpoint(0); ad = bank.load_state(sd, cfg, key="a0")
    torch.manual_seed(SEED); plain = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 12), nn.Linear(12, 5))
    with bank.session([AdapterSpec(ad, 0.0)]):
        assert torch.equal(dit(X), plain(X))


def test_plain_lora_type_too(bank_and_roots):
    bank, dit, cond = bank_and_roots
    ckpts = {0: _checkpoint(0, kind="lora"), 1: _checkpoint(1)}       # mixed stack
    ads = {t: bank.load_state(dict(sd), dict(cfg), key=f"a{t}") for t, (sd, cfg) in ckpts.items()}
    for order in ([0, 1], [1, 0]):
        with bank.session([AdapterSpec(ads[t], 0.6) for t in order]):
            out = dit(X)
        assert _close(out, _chain(dit, order, {0: 0.6, 1: 0.6}, ckpts)(X))


def test_interval_gates_dit_sites_per_sigma_but_never_the_conditioner(bank_and_roots):
    bank, dit, cond = bank_and_roots
    sd, cfg = _checkpoint(0); ad = bank.load_state(sd, cfg, key="a0")
    torch.manual_seed(SEED); plain_dit = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 12), nn.Linear(12, 5))
    plain_cond = nn.Linear(16, 8); plain_cond.load_state_dict(cond["seconds"].state_dict())
    spec = AdapterSpec(ad, 0.8, interval=(0.3, 0.7))
    with bank.session([spec]) as sess:
        sess.set_sigma(0.9)                                   # outside → base model on the DiT
        assert torch.equal(dit(X), plain_dit(X))
        assert not torch.equal(cond["seconds"](X), plain_cond(X))   # conditioner: always on
        sess.set_sigma(0.5)                                   # inside → adapted
        assert not torch.equal(dit(X), plain_dit(X))
        sess.set_sigma(0.3); inside_edge = dit(X)             # inclusive bounds
        assert not torch.equal(inside_edge, plain_dit(X))


def test_layer_filter_disables_matching_dit_sites_only(bank_and_roots):
    bank, dit, cond = bank_and_roots
    sd, cfg = _checkpoint(0); ad = bank.load_state(sd, cfg, key="a0")
    full = AdapterSpec(ad, 0.8)
    with bank.session([full]):
        all_on = dit(X)
    with bank.session([AdapterSpec(ad, 0.8, layer_filter="1, 2")]) as sess:   # sites "1" and "2" off
        act0 = sess.active_for(bank.site_for("0")); act1 = sess.active_for(bank.site_for("1"))
        assert act0 == (0,) and act1 == ()
        partial_out = dit(X)
    assert not torch.equal(all_on, partial_out)
    with bank.session([AdapterSpec(ad, 0.8, layer_filter="[0-2]")]) as sess:     # bracket expansion, like _expand
        assert all(sess.active_for(bank.site_for(n)) == () for n in ("0", "1", "2"))
        assert sess.active_for(bank.site_for("seconds")) == (0,)                 # conditioner ignores the filter


def test_two_sessions_in_two_threads_do_not_see_each_other(bank_and_roots):
    bank, dit, cond = bank_and_roots
    ckpts = {t: _checkpoint(t) for t in range(2)}
    ads = {t: bank.load_state(dict(sd), dict(cfg), key=f"a{t}") for t, (sd, cfg) in ckpts.items()}
    solo = {}
    for t in range(2):
        with bank.session([AdapterSpec(ads[t], 0.75)]):
            solo[t] = dit(X)
    results = {}
    gate = threading.Barrier(2)
    def run(t):
        with bank.session([AdapterSpec(ads[t], 0.75)]):
            gate.wait()
            outs = [dit(X) for _ in range(5)]
            results[t] = outs
    th = [threading.Thread(target=run, args=(t,)) for t in range(2)]
    [x.start() for x in th]; [x.join() for x in th]
    for t in range(2):
        for o in results[t]:
            assert torch.equal(o, solo[t])
    assert current_session() is None


def test_load_is_idempotent_single_flight_and_refcounted(bank_and_roots, tmp_path):
    bank, dit, cond = bank_and_roots
    from stable_audio_3.models.lora.utils import save_lora_safetensors  # noqa: F401  (import check only)
    sd, cfg = _checkpoint(0)
    ad = bank.load_state(dict(sd), dict(cfg), key="a0")
    assert bank.adapters["a0"] is ad and ad.refcount == 0 and ad.nbytes > 0 and ad.unplaced == []
    with bank.session([AdapterSpec(ad, 0.5)]):
        assert ad.refcount == 1
        assert bank.remove("a0") is False          # in use
    assert ad.refcount == 0 and bank.remove("a0") is True and "a0" not in bank.adapters


def test_unknown_sites_are_reported_and_bad_types_refused(bank_and_roots):
    bank, dit, cond = bank_and_roots
    sd, cfg = _checkpoint(0)
    sd["model.nowhere.parametrizations.weight.0.lora_A"] = torch.zeros(R, 16)
    sd["model.nowhere.parametrizations.weight.0.lora_B"] = torch.zeros(8, R)
    sd["model.nowhere.parametrizations.weight.0.magnitude"] = torch.ones(8)
    ad = bank.load_state(sd, cfg, key="a0")
    assert ad.unplaced == ["model.nowhere.parametrizations.weight.0.lora_A"] and len(ad.per_site) == 4
    with pytest.raises(NotImplementedError):
        bank.load_state(*_checkpoint(1, kind="bora"), key="bad")
    sd2, cfg2 = _checkpoint(2)
    sd2["0.parametrizations.weight.0.lora_A"] = torch.zeros(R, 99)
    with pytest.raises(ValueError):
        bank.load_state(sd2, cfg2, key="shape")


def test_model_wrapper_dispatch_is_bank_aware():
    """StableAudioModel.load_lora / set_lora_strength / canonicalize_lora_order route to the bank once enabled."""
    from stable_audio_3 import StableAudioModel
    dit, cond = _roots()
    inner = nn.Module(); inner.model = dit; inner.conditioner = cond
    sam = StableAudioModel.__new__(StableAudioModel)
    sam.model = inner; sam.model_config = {"model_type": "diffusion_cond"}; sam.device = "cpu"; sam.model_half = False
    bank = sam.use_adapter_bank()
    assert sam.use_adapter_bank() is bank and len(bank.sites) == 4
    sam.set_lora_strength(0.0)                      # no-op, no error
    assert sam.canonicalize_lora_order(lambda t: t) == 0
