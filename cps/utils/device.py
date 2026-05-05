"""Device selection helpers for CUDA, Apple Silicon MPS, and CPU."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    selected: str
    name: str
    cuda_available: bool
    mps_available: bool


def get_device(requested: str = "auto"):
    import torch

    req = str(requested).lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if req == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if req == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        return torch.device("cpu")
    return torch.device(req)


def device_info(requested: str = "auto") -> DeviceInfo:
    import torch

    device = get_device(requested)
    cuda_available = torch.cuda.is_available()
    mps_available = (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    )
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        name = "Apple Silicon MPS"
    else:
        name = "CPU"
    return DeviceInfo(
        requested=requested,
        selected=str(device),
        name=name,
        cuda_available=cuda_available,
        mps_available=mps_available,
    )
