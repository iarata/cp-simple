from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "cps/libcom/libcom/utils/model_download.py"


def load_model_download_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("model_download_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ModelDownloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "huggingface_hub" or name.startswith("modelscope")
        }

    def tearDown(self) -> None:
        for name in list(sys.modules):
            if name == "huggingface_hub" or name.startswith("modelscope"):
                del sys.modules[name]
        sys.modules.update(self._saved_modules)

    def test_huggingface_download_uses_supported_signature_and_preserves_cache(self) -> None:
        model_download = load_model_download_module()
        calls: list[tuple[str, str, str]] = []

        def hf_hub_download(repo_id: str, filename: str, *, cache_dir: str) -> str:
            calls.append((repo_id, filename, cache_dir))
            cache_path = Path(cache_dir) / "hf-cache" / filename
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text("weights")
            return str(cache_path)

        hf_module = types.ModuleType("huggingface_hub")
        hf_module.hf_hub_download = hf_hub_download
        sys.modules["huggingface_hub"] = hf_module
        sys.modules["modelscope"] = None

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            first = Path(model_download.download_file_from_network("PCTNet.pth", tmp))
            second = Path(model_download.download_file_from_network("PCTNet.pth", tmp))

            self.assertEqual(first, second)
            self.assertEqual(first.read_text(), "weights")
            self.assertTrue((Path(tmp) / "hf-cache" / "PCTNet.pth").exists())
            self.assertEqual(calls, [(model_download.hf_repo, "PCTNet.pth", tmp)])

    def test_existing_model_file_short_circuits_download(self) -> None:
        model_download = load_model_download_module()

        def hf_hub_download(repo_id: str, filename: str, *, cache_dir: str) -> str:
            raise AssertionError("network should not be used for an existing model file")

        hf_module = types.ModuleType("huggingface_hub")
        hf_module.hf_hub_download = hf_hub_download
        sys.modules["huggingface_hub"] = hf_module
        sys.modules["modelscope"] = None

        with tempfile.TemporaryDirectory() as tmp:
            expected = Path(tmp) / "IdentityLUT33.txt"
            expected.write_text("lut")

            actual = Path(model_download.download_file_from_network("IdentityLUT33.txt", tmp))

            self.assertEqual(actual, expected)
            self.assertEqual(actual.read_text(), "lut")


if __name__ == "__main__":
    unittest.main()
