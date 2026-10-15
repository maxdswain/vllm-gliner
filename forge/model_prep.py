"""Prepare DeBERTa GLiNER2 and GLiNER2.5 checkpoints for vLLM.

The upstream extractor config has a different architecture than vLLM expects;
this writes a local config and symlinks the cached Hugging Face weights. Keep
HF_HOME mounted when serving the result so the symlinks remain valid.
"""

import json
import logging
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

logger = logging.getLogger(__name__)


def _config(
    model_id: str,
    plugin: str,
    encoder: dict[str, object],
    extractor: dict[str, object],
    vocab_size: int,
) -> dict[str, object]:
    boundary = plugin == "deberta_gliner25"
    hidden = encoder.get("hidden_size", 768 if boundary else 1024)
    return {
        "model_type": "gliner25" if boundary else "gliner2",
        "architectures": ["GLiNER25VLLMModel" if boundary else "GLiNER2VLLMModel"],
        "num_hidden_layers": 0,
        "num_attention_heads": 1,
        "hidden_size": hidden,
        "encoder_model_name": extractor.get(
            "model_name", "microsoft/mdeberta-v3-base" if boundary else "microsoft/deberta-v3-large",
        ),
        "vocab_size": vocab_size,
        "encoder_hidden_size": hidden,
        "encoder_num_hidden_layers": encoder.get("num_hidden_layers", 12 if boundary else 24),
        "encoder_num_attention_heads": encoder.get("num_attention_heads", 12 if boundary else 16),
        "encoder_intermediate_size": encoder.get("intermediate_size", 3072 if boundary else 4096),
        "encoder_hidden_act": encoder.get("hidden_act", "gelu"),
        "encoder_hidden_dropout_prob": 0.0,
        "encoder_attention_probs_dropout_prob": 0.0,
        "encoder_max_position_embeddings": encoder.get("max_position_embeddings", 512),
        "encoder_type_vocab_size": encoder.get("type_vocab_size", 0),
        "encoder_layer_norm_eps": encoder.get("layer_norm_eps", 1e-7),
        "encoder_relative_attention": encoder.get("relative_attention", True),
        "encoder_max_relative_positions": encoder.get("max_relative_positions", -1),
        "encoder_position_buckets": encoder.get("position_buckets", 256),
        "encoder_pos_att_type": encoder.get("pos_att_type", ["p2c", "c2p"]),
        "encoder_share_att_key": encoder.get("share_att_key", True),
        "encoder_norm_rel_ebd": encoder.get("norm_rel_ebd", "layer_norm"),
        "encoder_position_biased_input": encoder.get("position_biased_input", False),
        "encoder_pad_token_id": encoder.get("pad_token_id", 0),
        "token_pooling": extractor.get("token_pooling", "first"),
        **(
            {"boundary_head": extractor.get("boundary_head") or {}}
            if boundary
            else {
                "max_width": extractor.get("max_width", 8),
                "counting_layer": extractor.get("counting_layer", "count_lstm"),
            }
        ),
    }


def prepare_model(model_id: str, plugin: str, output_dir: str, *, force: bool = False) -> str:
    """Prepare a supported HF checkpoint, rejecting other encoder families.

    A forced preparation removes the *whole* old directory: otherwise a
    same-named weight symlink can silently point at the previous model.
    """
    if plugin not in {"deberta_gliner2", "deberta_gliner25"}:
        msg = f"Unsupported model plugin: {plugin}"
        raise ValueError(msg)
    from transformers import AutoTokenizer

    files = list_repo_files(model_id)
    extractor = json.loads(Path(hf_hub_download(model_id, "config.json")).read_text())
    encoder = json.loads(Path(hf_hub_download(model_id, "encoder_config/config.json")).read_text())
    boundary = plugin == "deberta_gliner25"
    if boundary != (extractor.get("architecture") == "boundary"):
        msg = f"{model_id} is not a {'boundary' if boundary else 'span'} GLiNER checkpoint"
        raise ValueError(
            msg,
        )
    if encoder.get("model_type") not in {"deberta-v2", "deberta"}:
        msg = f"Only DeBERTa encoders are supported: {encoder.get('model_type')}"
        raise ValueError(msg)

    output = Path(output_dir)
    config_file = output / "config.json"
    if config_file.exists() and not force:
        cached = json.loads(config_file.read_text())
        if cached.get("_source_model") == model_id and cached.get("_source_plugin") == plugin:
            return str(output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer is None:
        msg = f"Could not load tokenizer for {model_id}"
        raise RuntimeError(msg)
    tokenizer.save_pretrained(output)
    config = _config(model_id, plugin, encoder, extractor, len(tokenizer))
    config["_source_model"] = model_id
    config["_source_plugin"] = plugin

    weights = [name for name in files if name.endswith((".safetensors", ".bin", ".pt"))]
    if not weights:
        msg = f"No model weights found for {model_id}"
        raise RuntimeError(msg)
    for name in weights:
        source = hf_hub_download(model_id, name)
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.symlink_to(source)
        except OSError:
            shutil.copy2(source, dest)
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    logger.info("Prepared %s (%s) at %s", model_id, plugin, output)
    return str(output)


def prepare_gliner2_model(
    hf_model_id: str, output_dir: str | None = None, force: bool = False,
) -> str:
    return prepare_model(
        hf_model_id,
        "deberta_gliner2",
        output_dir or f"/tmp/{hf_model_id.replace('/', '--')}-vllm",
        force=force,
    )


def prepare_gliner25_model(
    hf_model_id: str, output_dir: str | None = None, force: bool = False,
) -> str:
    return prepare_model(
        hf_model_id,
        "deberta_gliner25",
        output_dir or f"/tmp/{hf_model_id.replace('/', '--')}-vllm",
        force=force,
    )
