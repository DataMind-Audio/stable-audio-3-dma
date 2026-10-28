"""`load_adapter_tensors` places a LoRA checkpoint's tensors exactly as
`nn.Module.load_state_dict(strict=False)` did, without its cost on a parametrized model
(26.9 s of a 27.1 s attach on the SA3 DiT, measured 2026-10-28). CPU only."""
from functools import partial

import pytest
import torch
import torch.nn as nn

from stable_audio_3.models.lora import LoRAParametrization, add_lora, load_adapter_tensors


def _model():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 6), nn.Linear(6, 4))
    cfg = {nn.Linear: {"weight": partial(LoRAParametrization.from_linear, rank=2, lora_alpha=4.0,
                                          adapter_type="dora-rows", lora_index=0)}}
    torch.manual_seed(1)
    add_lora(m, cfg)
    return m


def _checkpoint(dtype=torch.float32, seed=7):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for i, (fo, fi) in enumerate(((6, 8), (4, 6))):
        base = f"{i}.parametrizations.weight.0."
        sd[base + "lora_A"] = torch.randn(2, fi, generator=g).to(dtype)
        sd[base + "lora_B"] = torch.randn(fo, 2, generator=g).to(dtype)
        sd[base + "magnitude"] = torch.rand(fo, generator=g).to(dtype) + 0.5
    return sd


def test_copy_matches_load_state_dict_bitwise():
    a, b = _model(), _model()
    sd = _checkpoint()
    missing = load_adapter_tensors(sd, a)
    b.load_state_dict(sd, strict=False)
    assert missing == []
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb and torch.equal(pa, pb), na
    x = torch.randn(3, 8)
    assert torch.equal(a(x), b(x))


def test_dtype_follows_the_parameter_not_the_checkpoint():
    a = _model()
    sd = _checkpoint(dtype=torch.float16)
    assert load_adapter_tensors(sd, a) == []
    p = dict(a.named_parameters())["0.parametrizations.weight.0.lora_A"]
    assert p.dtype == torch.float32
    assert torch.equal(p, sd["0.parametrizations.weight.0.lora_A"].float())


def test_unplaced_keys_are_returned_and_the_rest_still_land():
    a = _model()
    sd = _checkpoint()
    sd["9.parametrizations.weight.0.lora_A"] = torch.zeros(2, 8)
    sd["conditioner.something.lora_B"] = torch.zeros(4, 2)
    assert load_adapter_tensors(sd, a) == ["9.parametrizations.weight.0.lora_A", "conditioner.something.lora_B"]
    assert torch.equal(dict(a.named_parameters())["1.parametrizations.weight.0.lora_B"], sd["1.parametrizations.weight.0.lora_B"])


def test_second_target_receives_what_the_first_lacks():
    dit, cond = _model(), _model()
    sd = _checkpoint()
    # keys for the conditioner are addressed relative to it, as the loader does
    cond_sd = {k: v for k, v in sd.items() if k.startswith("1.")}
    dit_sd = {k: v for k, v in sd.items() if k.startswith("0.")}
    merged = {**dit_sd, **{"zzz." + k: v for k, v in cond_sd.items()}}   # "zzz." keys fit neither
    assert load_adapter_tensors(dit_sd, dit, cond) == []
    assert load_adapter_tensors(merged, dit, cond) == sorted("zzz." + k for k in cond_sd)


def test_shape_mismatch_raises_instead_of_skipping():
    a = _model()
    sd = _checkpoint()
    sd["0.parametrizations.weight.0.lora_A"] = torch.zeros(3, 8)   # rank 3 into a rank-2 slot
    with pytest.raises(ValueError, match="lora_A"):
        load_adapter_tensors(sd, a)
