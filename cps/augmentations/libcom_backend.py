"""Loader for the in-repository libcom wrapper."""

from __future__ import annotations

import importlib
import os
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

from cps.paths import project_path


def get_local_image_harmonization_model() -> type[Any]:
    """Return the local libcom ImageHarmonizationModel without importing PyPI libcom."""

    os.environ.setdefault("LIBCOM_MODEL_DIR", str(project_path("data.nosync/libcom_models")))
    _install_local_libcom_namespace()
    try:
        module = importlib.import_module("libcom.image_harmonization.image_harmonization")
    except ImportError as exc:
        missing = f" Missing module: {exc.name}." if getattr(exc, "name", None) else ""
        raise RuntimeError(
            "The in-repository cps/libcom wrapper could not be imported."
            f"{missing} Run `uv sync` so the local libcom runtime dependencies are installed."
        ) from exc
    module.check_gpu_device = _resolve_torch_device
    return module.ImageHarmonizationModel


def _install_local_libcom_namespace() -> None:
    root = project_path("cps/libcom")
    package_dir = root / "libcom"
    if not (package_dir / "image_harmonization").exists():
        raise RuntimeError(f"Local libcom wrapper not found at {package_dir}")

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    existing = sys.modules.get("libcom")
    if existing is not None and _is_local_libcom(existing, package_dir):
        return

    if existing is not None:
        for name in list(sys.modules):
            if name == "libcom" or name.startswith("libcom."):
                del sys.modules[name]

    package = types.ModuleType("libcom")
    package.__file__ = str(package_dir / "__init__.py")
    package.__package__ = "libcom"
    package.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    spec = ModuleSpec("libcom", loader=None, is_package=True)
    spec.submodule_search_locations = [str(package_dir)]
    package.__spec__ = spec
    sys.modules["libcom"] = package


def _is_local_libcom(module: types.ModuleType, package_dir: Path) -> bool:
    local_package_dir = package_dir.resolve()
    paths = [Path(path).resolve() for path in getattr(module, "__path__", [])]
    if local_package_dir in paths:
        return True
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        return Path(module_file).resolve().is_relative_to(local_package_dir)
    except OSError:
        return False


def _resolve_torch_device(device: str | int | Any = "cpu") -> Any:
    import torch

    if isinstance(device, torch.device):
        return device
    if device is None:
        device = "auto"
    if isinstance(device, int):
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device}")
        if device == 0 and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError(f"Local libcom requested CUDA device {device}, but CUDA is unavailable.")

    value = str(device).lower()
    if value == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    if value == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Local libcom requested device='mps', but MPS is unavailable.")
        return torch.device("mps")
    if value == "cpu":
        return torch.device("cpu")
    if value.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Local libcom requested device={device!r}, but CUDA is unavailable."
            )
        return torch.device(value)
    if value.isdigit():
        return _resolve_torch_device(int(value))
    raise RuntimeError(f"Unsupported local libcom device: {device!r}")
