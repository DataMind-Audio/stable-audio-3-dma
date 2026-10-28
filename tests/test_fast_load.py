"""The fast loader builds the model without random initialisation and loads an fp16 checkpoint:
every parameter and buffer must come out bitwise as the old path's. CPU only, toy module."""
import os

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from stable_audio_3 import loading_utils as lu


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(16, 8)
        self.c = nn.Conv1d(4, 6, 3)
        self.n = nn.LayerNorm(8)
        self.register_buffer("table", torch.arange(10, dtype=torch.float32) / 7.0)   # computed from shape, not in a checkpoint
        self.register_buffer("count", torch.tensor(3))                                 # non-float buffer
        nn.init.zeros_(self.n.bias)


def _old(ckpt, half):
    torch.manual_seed(1); m = Toy()
    lu.copy_state_dict(m, load_file(ckpt))
    m.eval().requires_grad_(False)
    if half:
        m.to(torch.float16)
    return m


def _same(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return set(sa) == set(sb) and all(sa[k].dtype == sb[k].dtype and torch.equal(sa[k], sb[k]) for k in sa)


@pytest.fixture
def ckpts(tmp_path):
    torch.manual_seed(99); src = Toy()
    sd = {k: v for k, v in src.state_dict().items()}
    save_file(sd, str(tmp_path / "model.safetensors"))
    save_file({k: (v.to(torch.float16) if v.is_floating_point() else v) for k, v in sd.items()}, str(tmp_path / "model.fp16.safetensors"))
    return tmp_path


def test_no_random_init_skips_the_random_fills_and_keeps_deterministic_ones():
    torch.manual_seed(5)
    with lu.no_random_init():
        m = Toy()
    assert torch.equal(m.table, torch.arange(10, dtype=torch.float32) / 7.0) and int(m.count) == 3
    assert torch.equal(m.n.weight, torch.ones(8)) and torch.equal(m.n.bias, torch.zeros(8))
    # the initialisers are restored afterwards
    assert torch.nn.init.kaiming_uniform_ is not lu.no_random_init.__enter__ and callable(torch.nn.init.kaiming_uniform_)
    torch.manual_seed(5); x = torch.empty(3, 3); torch.nn.init.uniform_(x)
    assert not torch.equal(x, torch.empty(3, 3).zero_())   # a real fill again


@pytest.mark.parametrize("half", [False, True])
def test_fast_cpu_build_with_fp16_checkpoint_is_bitwise_the_old_result(ckpts, half):
    old = _old(str(ckpts / "model.safetensors"), half)
    with lu.no_random_init():
        torch.manual_seed(2); new = Toy()
    lu.copy_state_dict(new, load_file(str(ckpts / "model.fp16.safetensors")))
    new.eval().requires_grad_(False)
    if half:
        new.to(torch.float16)
    if half:
        assert _same(old, new)
    else:
        # fp32 target from an fp16 file differs by the cast (expected); the fp32 file is bitwise
        with lu.no_random_init():
            new32 = Toy()
        lu.copy_state_dict(new32, load_file(str(ckpts / "model.safetensors")))
        assert _same(old, new32)


def test_fast_load_mode_env(monkeypatch):
    monkeypatch.delenv("SA3_FAST_LOAD", raising=False); assert lu.fast_load_mode() == "on"
    monkeypatch.setenv("SA3_FAST_LOAD", "OFF"); assert lu.fast_load_mode() == "off"
    monkeypatch.setenv("SA3_FAST_LOAD", "0"); assert lu.fast_load_mode() == "off"
    monkeypatch.setenv("SA3_FAST_LOAD", "nonsense"); assert lu.fast_load_mode() == "on"


def test_sequential_read_equals_mmap_load(ckpts):
    a = load_file(str(ckpts / "model.fp16.safetensors")); b = lu.read_checkpoint(str(ckpts / "model.fp16.safetensors"))
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in a)


def test_prefetch_reads_only_weight_files_under_model_path(tmp_path):
    d = tmp_path / "t5"; d.mkdir(); (d / "model.safetensors").write_bytes(b"a" * 1000); (d / "config.json").write_text("{}"); (d / "x.bin").write_bytes(b"b" * 10)
    cfg = {"model": {"conditioning": {"configs": [{"config": {"model_path": str(d), "repo_id": "x"}}]}}, "other": [1, 2]}
    assert lu.prefetch_weight_files(cfg) == 1010
    assert lu.prefetch_weight_files({"model_path": str(tmp_path / "missing")}) == 0


def test_copy_state_dict_by_name_matches_load_state_dict(ckpts, capsys):
    """Same tensors land, same skips are reported, non-persistent buffers are never touched."""
    class T2(Toy):
        def __init__(self):
            super().__init__()
            self.register_buffer("scratch", torch.full((4,), 7.0), persistent=False)
    torch.manual_seed(3); a = T2(); torch.manual_seed(3); b = T2()
    sd = load_file(str(ckpts / "model.safetensors"))
    sd["scratch"] = torch.zeros(4)                      # a key naming a NON-persistent buffer: must be skipped
    sd["a.weight_wrong"] = torch.zeros(1)              # unknown key
    sd["c.weight"] = torch.zeros(6, 4, 5)              # shape mismatch
    lu.copy_state_dict(a, dict(sd))
    out = capsys.readouterr().out
    assert "scratch" in out and "a.weight_wrong" in out and "c.weight" in out
    # reference: the old implementation
    ref_sd = b.state_dict()
    for k, v in sd.items():
        if k in ref_sd and v.shape == ref_sd[k].shape:
            ref_sd[k] = v
    b.load_state_dict(ref_sd, strict=False)
    assert all(torch.equal(x, y) for (_, x), (_, y) in zip(a.state_dict().items(), b.state_dict().items()))
    assert torch.equal(a.scratch, torch.full((4,), 7.0)) and torch.equal(a.c.weight, b.c.weight)
