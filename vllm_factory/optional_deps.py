"""Import optional extras with a uniform install hint.

vLLM imports every ``vllm.general_plugins`` ``register()`` at engine start.
A module-level ``import gliner`` / ``import gliner2`` on that path turns a
missing extra into a startup failure for every other plugin. Call ``require``
from ``__init__`` / method bodies, never at module scope on a registration path.
"""

import importlib
from types import ModuleType


def require(module: str, extra: str, *, purpose: str) -> ModuleType:
    r"""Import ``module`` or raise ``ImportError`` naming ``pip install vllm-factory[extra]``.

    Parameters
    ----------
    module : str
        Dotted module name, e.g. ``\"gliner2\"``.
    extra : str
        Extra that provides it, e.g. ``\"gliner2\"``.
    purpose : str
        Short phrase included in the error, e.g. ``\"GLiNER2 pooling\"``.

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    ImportError
        When the extra is not installed.

    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        msg = f"{module} is required for {purpose}; install vllm-factory[{extra}]"
        raise ImportError(
            msg,
        ) from exc
