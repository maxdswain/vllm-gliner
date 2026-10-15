"""Offline preparation invariants; real checkpoints are covered by smoke_cpu.py."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from forge.model_prep import prepare_model


class FakeTokenizer:
    def __len__(self) -> int:
        """Return the tokenizer's synthetic vocabulary size."""
        return 128

    def save_pretrained(self, output: str | Path) -> None:
        (Path(output) / "tokenizer.json").write_text("{}")


class ModelPrepTests(unittest.TestCase):
    def test_rejects_unsupported_plugin_before_download(self) -> None:
        with pytest.raises(ValueError, match="Unsupported model plugin"):
            prepare_model("fake/legacy", "deberta_gliner", "/tmp/unused")

    def test_switching_models_replaces_weight_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            output = root / "model"
            upstream = root / "upstream"
            upstream.mkdir()
            checkpoint = upstream / "model.safetensors"
            checkpoint.write_bytes(b"fake weights")
            extractor = upstream / "config.json"
            extractor.write_text(json.dumps({"architecture": "boundary"}))
            encoder_dir = upstream / "encoder_config"
            encoder_dir.mkdir()
            encoder = encoder_dir / "config.json"
            encoder.write_text(json.dumps({"model_type": "deberta-v2", "hidden_size": 768}))

            def download(model_id: str, filename: str) -> str:
                return str(upstream / filename)

            with (
                patch("forge.model_prep.list_repo_files", return_value=["model.safetensors"]),
                patch("forge.model_prep.hf_hub_download", side_effect=download),
                patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()),
            ):
                prepare_model("fake/first", "deberta_gliner25", str(output))
                assert json.loads((output / "config.json").read_text())["_source_model"] == "fake/first"
                checkpoint2 = upstream / "other.safetensors"
                checkpoint2.write_bytes(b"different weights")
                (output / "obsolete.safetensors").symlink_to(checkpoint)
                with patch("forge.model_prep.list_repo_files", return_value=["other.safetensors"]):
                    prepare_model("fake/second", "deberta_gliner25", str(output))
                assert not (output / "obsolete.safetensors").exists()
                assert (output / "other.safetensors").resolve() == checkpoint2
                assert json.loads((output / "config.json").read_text())["_source_model"] == "fake/second"

    def test_rejects_wrong_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            source = Path(root)
            (source / "config.json").write_text(json.dumps({"architecture": "boundary"}))
            (source / "encoder_config").mkdir()
            (source / "encoder_config/config.json").write_text(
                json.dumps({"model_type": "deberta-v2"}),
            )
            def download(model_id: str, filename: str) -> str:
                return str(source / filename)

            with (
                patch("forge.model_prep.list_repo_files", return_value=[]),
                patch("forge.model_prep.hf_hub_download", side_effect=download),
                pytest.raises(ValueError, match="not a span GLiNER checkpoint"),
            ):
                prepare_model("fake/boundary", "deberta_gliner2", str(source / "output"))


if __name__ == "__main__":
    unittest.main()
