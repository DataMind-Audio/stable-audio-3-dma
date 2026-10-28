import json

import os

import torch
from safetensors import safe_open
from safetensors.torch import load as st_load_bytes, load_file
from stable_audio_3.factory import (
    create_autoencoder_from_config,
    create_diffusion_cond_from_config,
)


def copy_state_dict(model, state_dict):
    """Copy the checkpoint's tensors into the model's parameters and persistent buffers by name.

    Same contract as before (keys the model lacks or whose shape differs are reported and skipped;
    nothing else changes), without `nn.Module.load_state_dict(strict=False)`: that call scans every
    key for every module (its unexpected-keys check), which on this model (thousands of modules,
    thousands of keys) took 78–96 s for the full checkpoint and 27 s for a 507-tensor LoRA
    (2026-10-28, probes 4b and 7/7b). Direct `copy_` by name is the same assignment in seconds.
    """
    model_state_dict = model.state_dict()                 # the persistent set: what load_state_dict would accept
    state_dict = remap_state_dict_keys(state_dict, model_state_dict)
    targets = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))
    with torch.no_grad():
        for key in state_dict:
            dst = targets.get(key) if key in model_state_dict else None
            if dst is not None and state_dict[key].shape == dst.shape:
                dst.copy_(state_dict[key])
            else:
                print(
                    f"Key {key} not found in target state_dict or shape mismatch. Skipping."
                )


def load_autoencoder(config_path: str, ckpt_path: str, device: str = "cpu"):
    """Load only the autoencoder from a combined DiT+autoencoder checkpoint.

    Only pretransform tensors are read from disk, directly onto the target device.
    Standalone AE-only checkpoints (e.g. stabilityai/SAME-L / SAME-S) have no prefix.
    """

    with open(config_path) as f:
        config = json.load(f)

    autoencoder = create_autoencoder_from_config(config["model"], config["sample_rate"])

    # Full DiT checkpoints nest the AE under pretransform.model.*;
    # standalone AE-only checkpoints have no prefix.
    nested_prefix = "pretransform.model."
    with safe_open(ckpt_path, framework="pt", device=device) as f:
        all_keys = list(f.keys())
    if any(k.startswith(nested_prefix) for k in all_keys):
        effective_prefix = nested_prefix
    else:
        effective_prefix = ""  # standalone AE — keys are already bare
    with safe_open(ckpt_path, framework="pt", device=device) as f:
        state_dict = {
            k[len(effective_prefix) :]: f.get_tensor(k)
            for k in all_keys
            if k.startswith(effective_prefix)
        }

    copy_state_dict(autoencoder, state_dict)
    return autoencoder.to(device)


# ── fast load (2026-10-28) ────────────────────────────────────────────────────────────────────────
# Today's load took 171 s on an A100 MIG slice (probe 7): 72 s building the model (random init that
# the checkpoint overwrites, plus T5Gemma paged in through mmap), 96 s "copying the state dict"
# — really safetensors' lazy mmap paging the 8.8 GB file in 4 KB at a time. SA3_FAST_LOAD=on (the
# default; off = the old path): pre-read the conditioner's weight files, build without random
# init, read the checkpoint in one sequential pass. Every parameter and buffer comes out bitwise
# as the old path's (probes 7/7b). An fp16 checkpoint (model.fp16.safetensors: the fp32 tensors
# cast once) loads to the same values — the model is cast to fp16 after loading anyway.
_RANDOM_INITS = ("kaiming_uniform_", "kaiming_normal_", "uniform_", "normal_", "trunc_normal_",
                 "xavier_uniform_", "xavier_normal_", "orthogonal_")


class no_random_init:
    """Turn torch.nn.init's random fills into no-ops while a model is built. constant_ / zeros_ /
    ones_ and buffers computed from shapes are left alone: those may not be in the checkpoint."""

    def __enter__(self):
        self._saved = {n: getattr(torch.nn.init, n) for n in _RANDOM_INITS}
        for n in _RANDOM_INITS:
            setattr(torch.nn.init, n, lambda t, *a, **k: t)
        return self

    def __exit__(self, *exc):
        for n, f in self._saved.items():
            setattr(torch.nn.init, n, f)
        return False


def fast_load_mode() -> str:
    mode = os.environ.get("SA3_FAST_LOAD", "on").strip().lower()
    return "off" if mode in ("off", "0", "false", "no") else "on"


def read_checkpoint(ckpt_path: str) -> dict:
    """The checkpoint's tensors, read in ONE sequential pass. `safetensors.load_file` is lazy: it mmaps the
    file and every tensor is paged in 4 KB at a time when it is first copied — 96 s for the 8.8 GB fp32
    checkpoint from node-local disk (~90 MB/s), measured 2026-10-28. One sequential read runs at the
    disk's speed; `safetensors.torch.load` then builds zero-copy views on the buffer."""
    with open(ckpt_path, "rb") as f:
        data = f.read()
    return st_load_bytes(data)


def prefetch_weight_files(model_config) -> int:
    """Read every weight file under a `model_path` the config names (the T5Gemma conditioner) into the
    page cache, sequentially, so HF's mmap loader does not page-fault its way through them."""
    paths = []

    def walk(node):
        if isinstance(node, dict):
            mp = node.get("model_path")
            if isinstance(mp, str) and os.path.isdir(mp):
                for root, _dirs, files in os.walk(mp):
                    for fn in files:
                        if fn.endswith((".safetensors", ".bin", ".pt")):
                            paths.append(os.path.join(root, fn))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(model_config)
    n = 0
    for path in paths:
        with open(path, "rb") as f:
            while f.read(64 << 20):
                pass
        n += os.path.getsize(path)
    return n


def load_diffusion_cond(
    model_config,
    ckpt_path: str,
    device: str = "cuda",
    model_half: bool = False,
    fast: str | None = None,
):
    mode = fast or fast_load_mode()
    if mode == "off":
        model = create_diffusion_cond_from_config(model_config)
        copy_state_dict(model, load_file(ckpt_path))
    else:
        prefetch_weight_files(model_config)
        with no_random_init():
            model = create_diffusion_cond_from_config(model_config)
        copy_state_dict(model, read_checkpoint(ckpt_path))
    model.to(device).eval().requires_grad_(False)
    if model_half:
        model.to(torch.float16)
    return model


def remap_state_dict_keys(state_dict, model_state_dict):
    remapped = {}
    for key, value in state_dict.items():
        if key not in model_state_dict:
            parts = key.split(".")
            for i in range(1, len(parts)):
                candidate = ".".join(parts[:i]) + "." + ".".join(parts[i + 1 :])
                if candidate in model_state_dict:
                    key = candidate
                    break
        remapped[key] = value
    return remapped
