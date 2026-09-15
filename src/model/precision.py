"""
Mixed precision — one opt-in switch (`--amp`) shared by all four trainers.

Off by default, and when off every trainer computes exactly what it did before
this module existed: a disabled GradScaler returns the loss unscaled, makes
`unscale_` a no-op and calls `optimizer.step()` directly. The DAVIS grid was
trained that way and stays that way.

On, the forward pass and loss run under float16 autocast (the T4's tensor
cores) with a GradScaler guarding the backward pass against underflow. Measured
on a T4 with KIBA rows and the grid's batch sizes (`results/speed_test_kiba_t4.md`):
HyperAttentionDTI 2.0x, MolTrans 1.3x, ColdSite-DTI 1.15x, DeepDTA 2.3x, no
non-finite steps. Weights, optimiser state and checkpoints stay float32; only
the arithmetic inside the forward pass changes. Whichever is used is recorded in
every results file, and has to be stated in Methods.

On a CPU the autocast is bfloat16 and the scaler is disabled -- not for real
training, only so the switch can be exercised by the test suite without a GPU.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn


def _cuda(device) -> bool:
    return str(device).startswith("cuda")


def autocast(device, enabled: bool):
    """The context every forward pass and loss runs in."""
    if not enabled:
        return contextlib.nullcontext()
    if _cuda(device):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


def make_scaler(device, enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=bool(enabled) and _cuda(device))


def step(optimizer, scaler, parameters=None, clip: float | None = None) -> None:
    """Clip (on unscaled gradients) and take the optimiser step.

    With the scaler disabled this is `clip_grad_norm_` then `optimizer.step()`,
    exactly the trainers' previous code.
    """
    if clip is not None:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(parameters, max_norm=clip)
    scaler.step(optimizer)
    scaler.update()
