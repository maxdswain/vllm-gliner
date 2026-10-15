"""Registration utilities for vLLM model plugins.

Provides idempotent helpers that handle the boilerplate of registering
custom models and configs with vLLM and HuggingFace.

Usage in a plugin's __init__.py:
    from forge.registration import register_plugin
    register_plugin("my_model_type", MyConfig, "MyModelArch", MyModel)
"""

import logging

from torch.nn import Module
from transformers import PreTrainedConfig

logger = logging.getLogger("forge.registration")


def register_plugin(
    model_type: str,
    config_cls: type[PreTrainedConfig],
    architecture_name: str,
    model_cls: type[Module],
    *,
    aliases: list[str] | None = None,
) -> None:
    """Register a plugin's config and model with HuggingFace + vLLM.

    This is the main entry point — call this in your plugin's __init__.py.

    Parameters
    ----------
    model_type : str
        The model_type string from config.json
        (e.g., "moderncolbert", "gliner_mmbert").
    config_cls : type[PreTrainedConfig]
        HuggingFace config class to register with AutoConfig.
    architecture_name : str
        Architecture name for vLLM ModelRegistry
        (e.g., "ModernBertModel", "GLiNERLinkerModel").
    model_cls : type[Module]
        The model class to register with vLLM.
    aliases : list[str] | None
        Optional list of additional architecture names to register
        the same model class under.

    """
    register_config(model_type, config_cls)
    register_with_vllm(architecture_name, model_cls)
    for alias in aliases or []:
        register_with_vllm(alias, model_cls)


def register_with_vllm(architecture_name: str, model_cls: type[Module]) -> bool:
    """Register a model class with vLLM's ModelRegistry (idempotent).

    Parameters
    ----------
    architecture_name : str
        Architecture name as it appears in model config.
    model_cls : type[Module]
        The model class to register.

    Returns
    -------
    bool
        True if newly registered, False if already registered.

    """
    try:
        from vllm import ModelRegistry

        ModelRegistry.register_model(architecture_name, model_cls)
        logger.debug("Registered '%s' with vLLM", architecture_name)
    except (ValueError, KeyError):
        return False
    except ImportError:
        logger.warning("vLLM not installed, skipping registration")
        return False
    else:
        return True


def register_config(model_type: str, config_cls: type[PreTrainedConfig]) -> bool:
    """Register a config class with HuggingFace's AutoConfig (idempotent).

    Parameters
    ----------
    model_type : str
        The model_type string from config.json.
    config_cls : type[PreTrainedConfig]
        The config class to register.

    Returns
    -------
    bool
        True if newly registered, False if already registered.

    """
    try:
        from transformers import AutoConfig

        try:
            AutoConfig.register(model_type, config_cls, exist_ok=True)
        except TypeError:
            # Older transformers without exist_ok
            try:
                AutoConfig.register(model_type, config_cls)
            except ValueError:
                return False
        logger.debug("Registered config '%s' with AutoConfig", model_type)
    except ImportError:
        logger.warning("transformers not installed, skipping config registration")
        return False
    else:
        return True
