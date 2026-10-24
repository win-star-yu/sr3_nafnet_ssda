"""Minimal EMA wrapper used with the GitHub SR3+NAFNet model.

This file does not change the network architecture.  It keeps a detached
copy of the generator and updates it after each optimizer step.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn


@torch.no_grad()
def make_ema_model(model: nn.Module) -> nn.Module:
    """Create the EMA copy from the current generator weights."""
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema(
    ema_model: nn.Module,
    model: nn.Module,
    decay: float = 0.9999,
) -> None:
    """Update EMA parameters and copy non-parameter buffers."""
    ema_parameters = dict(ema_model.named_parameters())
    for name, parameter in model.named_parameters():
        if name in ema_parameters:
            ema_parameters[name].mul_(decay).add_(
                parameter.detach(), alpha=1.0 - decay
            )

    ema_buffers = dict(ema_model.named_buffers())
    for name, buffer in model.named_buffers():
        if name in ema_buffers:
            ema_buffers[name].copy_(buffer.detach())


def save_ema(ema_model: nn.Module, path: str | Path) -> None:
    """Save the direct state dict used by the checkpoint loader."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ema_model.state_dict(), path)


def load_ema(model: nn.Module, path: str | Path) -> nn.Module:
    """Load an EMA state dict for validation or inference."""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.eval()


# Training-loop usage:
#
# ema_model = make_ema_model(netG)
# ... optimizer.step()
# update_ema(ema_model, netG, decay=0.9999)
# save_ema(ema_model, "checkpoint/I320000_E39_gen_ema.pth")
