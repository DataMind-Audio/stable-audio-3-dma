"""Stacked DoRA adapters compose in list order, and DoRA composition is not
commutative. `canonicalize_lora_order` makes the stack a function of the
attached SET so the same blend renders identically regardless of attach
history. CPU-only; no checkpoint or base model needed."""
import itertools

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from stable_audio_3.models.lora import (
    LoRAParametrization,
    canonicalize_lora_order,
    set_lora_strength,
)

# lora_index TAG -> canonical key (a platform slug in production)
NAMES = {0: "koto", 1: "bass_guitar", 2: "wood"}
CANONICAL = ["bass_guitar", "koto", "wood"]
STRENGTHS = {0: 0.7, 1: 0.4, 2: 0.9}
KEY = NAMES.__getitem__

X = torch.randn(4, 16, generator=torch.Generator().manual_seed(7))


def _attach(layer, tag):
    """Attach a dora-rows adapter whose weights depend on its TAG only, so two
    layers attached in different orders hold identical adapter modules."""
    p = LoRAParametrization.from_linear(
        layer, rank=2, lora_alpha=4.0, adapter_type="dora-rows", lora_index=tag
    )
    g = torch.Generator().manual_seed(100 + tag)
    with torch.no_grad():
        p.lora_A.copy_(torch.randn(p.lora_A.shape, generator=g))
        p.lora_B.copy_(torch.randn(p.lora_B.shape, generator=g))
        # a trained magnitude has drifted from the base row norms
        p.magnitude.mul_(0.5 + torch.rand(p.magnitude.shape, generator=g))
    parametrize.register_parametrization(layer, "weight", p, unsafe=True)


def _stack(order):
    """A Linear with the same base weight every time, adapters attached in
    `order`, strengths set by tag."""
    torch.manual_seed(0)
    layer = nn.Linear(16, 8)
    for tag in order:
        _attach(layer, tag)
    for tag in order:
        set_lora_strength(layer, STRENGTHS[tag], lora_index=tag)
    return layer


def _order_of(layer):
    return [NAMES[p.lora_index] for p in layer.parametrizations.weight
            if isinstance(p, LoRAParametrization)]


def test_stacked_dora_is_order_dependent_before_canonicalization():
    ab, ba = _stack([0, 1]), _stack([1, 0])
    assert not torch.allclose(ab(X), ba(X)), (
        "DoRA composition came out commutative — if this holds, "
        "canonicalize_lora_order is unnecessary; re-check dora_forward"
    )


def test_canonicalize_makes_output_independent_of_attach_history():
    ab, ba = _stack([0, 1]), _stack([1, 0])
    canonicalize_lora_order(ab, KEY)
    canonicalize_lora_order(ba, KEY)
    assert _order_of(ab) == _order_of(ba) == ["bass_guitar", "koto"]
    assert torch.equal(ab(X), ba(X))  # bitwise, not allclose


def test_every_attach_permutation_agrees_after_canonicalization():
    outs = []
    for perm in itertools.permutations([0, 1, 2]):
        layer = _stack(list(perm))
        canonicalize_lora_order(layer, KEY)
        assert _order_of(layer) == CANONICAL, perm
        outs.append(layer(X))
    for out in outs[1:]:
        assert torch.equal(outs[0], out)


def test_tags_and_strengths_survive_reordering():
    layer = _stack([2, 0, 1])
    canonicalize_lora_order(layer, KEY)
    for p in layer.parametrizations.weight:
        # lora_strength is a float32 buffer; compare at float32 precision
        assert float(p.lora_strength) == pytest.approx(STRENGTHS[p.lora_index])
    # Addressing by tag still hits the right module after the move.
    set_lora_strength(layer, 0.0, lora_index=0)  # koto off
    assert [float(p.lora_strength) for p in layer.parametrizations.weight] == pytest.approx([
        STRENGTHS[1], 0.0, STRENGTHS[2]  # canonical: bass_guitar, koto, wood
    ])


def test_inactive_adapter_is_identity_so_its_position_is_irrelevant():
    two, three = _stack([0, 1]), _stack([2, 0, 1])
    set_lora_strength(three, 0.0, lora_index=2)
    canonicalize_lora_order(two, KEY)
    canonicalize_lora_order(three, KEY)
    assert torch.equal(two(X), three(X))


def test_layers_with_fewer_than_two_adapters_are_untouched():
    plain, single = nn.Linear(16, 8), _stack([1])
    assert canonicalize_lora_order(plain, KEY) == 0
    assert canonicalize_lora_order(single, KEY) == 0
    assert _order_of(single) == ["bass_guitar"]


def test_idempotent_and_reports_changed_layers():
    layer = _stack([2, 1, 0])
    assert canonicalize_lora_order(layer, KEY) == 1
    assert canonicalize_lora_order(layer, KEY) == 0
    assert _order_of(layer) == CANONICAL
