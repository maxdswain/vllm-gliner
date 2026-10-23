"""FactoryIOProcessor — the ONE file that touches vLLM's IOProcessor ABC.

When vLLM renames/splits/changes the IOProcessor interface, update THIS file.
Both GLiNER IO processors inherit from :class:`FactoryIOProcessor` and
override only the stable ``factory_*`` methods — they never import vLLM
protocol types directly.

Re-exports
----------
This module also re-exports a few vLLM types that plugins legitimately need
(``TokensPrompt``, ``PromptType``).  When vLLM moves them, update the imports
here — plugin code stays untouched.
"""

import contextvars
import threading
import uuid
from collections.abc import Sequence
from typing import Generic, TypeVar

from typing_extensions import override
from vllm.config import VllmConfig
from vllm.entrypoints.pooling.pooling.io_processor import PluginWithIOProcessorPlugins
from vllm.entrypoints.pooling.pooling.protocol import IOProcessorResponse
from vllm.entrypoints.pooling.typing import PoolingServeContext
from vllm.entrypoints.serve.engine.protocol import UsageInfo
from vllm.inputs import PromptType, TokensPrompt
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams, PoolingTask
from vllm.renderers.base import BaseRenderer

# Re-exports for plugins — import from here, not from vllm.*
__all__ = [
    "FactoryIOProcessor",
    "PoolingRequestOutput",
    "PromptType",
    "TokensPrompt",
]


InputT = TypeVar("InputT")


class GLiNERPoolingResponse(IOProcessorResponse[object]):
    """Extend vLLM's plugin response with standard pooling token usage."""

    usage: UsageInfo


_original_post_process_online = PluginWithIOProcessorPlugins.post_process_online


def _post_process_with_usage(self: PluginWithIOProcessorPlugins, ctx: PoolingServeContext) -> None:
    """Add usage only to responses produced by the GLiNER IO processors."""
    if not isinstance(self.io_processor, FactoryIOProcessor):
        _original_post_process_online(self, ctx)
        return
    try:
        output = self.io_processor.post_process(ctx.final_res_batch, request_id=ctx.request_id)
    finally:
        prompt_tokens = self.io_processor._take_prompt_tokens(ctx.request_id)
    ctx.response = GLiNERPoolingResponse(
        request_id=ctx.request_id,
        data=output,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            total_tokens=prompt_tokens,
        ),
    )


# vLLM 0.32 constructs its own IOProcessorResponse after calling post_process;
# its schema has no usage field. Keep the integration here, scoped to our IO
# processors, rather than requiring an additional endpoint plugin.
PluginWithIOProcessorPlugins.post_process_online = _post_process_with_usage


class FactoryIOProcessor(IOProcessor[InputT, object], Generic[InputT]):
    """Shared base class for GLiNER2 and GLiNER2.5 IO processors.

    Plugins inherit this and override the ``factory_*`` methods.  This class
    handles:

    * Delegation from vLLM 0.31's ABC (``parse_data``,
      ``merge_pooling_params``, …) to the shared factory interface.
    * Thread-safe stashing of ``extra_kwargs`` between ``pre_process`` and
      ``merge_pooling_params``.
    * Thread-safe stashing of per-request metadata for ``post_process``.
    * Default ``PoolingParams`` creation (task left as ``None`` so vLLM's
      serving layer resolves it from the model's supported tasks).

    When vLLM renames or reorganises the ABC, update the delegation methods
    below.  The ``factory_*`` signatures stay fixed.
    """

    pooling_task: PoolingTask | None = None

    def __init__(self, vllm_config: VllmConfig, renderer: BaseRenderer) -> None:
        super().__init__(vllm_config, renderer)
        self._extra_lock = threading.Lock()
        self._pending_extra: dict[str, dict[str, object]] = {}
        self._request_meta: dict[str, object] = {}
        self._prompt_tokens: dict[str, int] = {}
        self._request_key: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            f"{self.__class__.__name__}_request_key",
            default=None,
        )

    # ── vLLM ABC delegation (update here when vLLM changes) ──────────────

    @override
    def parse_data(self, data: object) -> InputT:
        """VLLM 0.19+ entry point (replaces deprecated ``parse_request``)."""
        return self.factory_parse(data)

    @override
    def merge_pooling_params(
        self,
        params: PoolingParams | None = None,
    ) -> PoolingParams:
        """Merge parameters for vLLM's pooling task.

        Pass stashed per-request extras through to vLLM's pooling runner.
        """
        request_key = self._request_key.get()
        with self._extra_lock:
            extra = self._pending_extra.pop(request_key, None) if request_key is not None else None

        task = self.pooling_task
        if params is not None:
            if extra is not None:
                params.extra_kwargs = {**(params.extra_kwargs or {}), **extra}
            if task is not None and params.task is None:
                params.task = task
            return params

        return PoolingParams(task=task, extra_kwargs=extra or {})

    def pre_process(
        self,
        prompt: InputT,
        request_id: str | None = None,
        **kwargs: object,
    ) -> PromptType | Sequence[PromptType]:
        request_key = request_id or f"_offline:{uuid.uuid4().hex}"
        self._request_key.set(request_key)
        result = self.factory_pre_process(prompt, request_id)
        prompts = result if isinstance(result, Sequence) else [result]
        token_count = sum(len(item["prompt_token_ids"]) for item in prompts)
        with self._extra_lock:
            self._prompt_tokens[request_key] = token_count
        return result

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs: object,
    ) -> object:
        meta_key = request_id or self._request_key.get() or "_offline"
        with self._extra_lock:
            meta = self._request_meta.pop(meta_key, None)
            if request_id is None:
                self._prompt_tokens.pop(meta_key, None)
        try:
            return self.factory_post_process(model_output, meta)
        finally:
            self._request_key.set(None)

    def _take_prompt_tokens(self, request_id: str) -> int:
        """Consume the length of the encoded prompt(s), including schema tokens."""
        with self._extra_lock:
            return self._prompt_tokens.pop(request_id)

    # ── Stable plugin interface (override these) ─────────────────────────

    def factory_parse(self, data: object) -> InputT:
        """Parse and validate the incoming request data.

        ``data`` is the raw ``request.data`` dict from the HTTP request
        (or the dict passed to ``llm.encode()`` offline).

        Must return a plugin-specific parsed input object (dataclass, etc.).
        """
        raise NotImplementedError

    def factory_pre_process(
        self,
        parsed_input: InputT,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        """Convert the parsed input into a vLLM prompt.

        Typically tokenises text and returns ``TokensPrompt(prompt_token_ids=…)``.
        Use :meth:`_stash` to attach ``extra_kwargs`` and/or per-request
        metadata for later retrieval in ``merge_pooling_params`` and
        ``factory_post_process``.
        """
        raise NotImplementedError

    def factory_post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_meta: object,
    ) -> object:
        """Convert engine output into the final plugin response.

        ``request_meta`` is whatever was stashed via :meth:`_stash` during
        ``factory_pre_process`` (``None`` if nothing was stashed).
        """
        raise NotImplementedError

    # ── Helpers for plugins ───────────────────────────────────────────────

    def _stash(
        self,
        extra_kwargs: dict[str, object] | None = None,
        request_id: str | None = None,
        meta: object = None,
    ) -> None:
        """Thread-safe stash of extra_kwargs and per-request metadata.

        Call this from ``factory_pre_process`` to:
        * Pass ``extra_kwargs`` into ``PoolingParams`` (picked up automatically
          by ``merge_pooling_params``).
        * Store ``meta`` for retrieval in ``factory_post_process``.
        """
        key = request_id or self._request_key.get() or "_offline"
        with self._extra_lock:
            if extra_kwargs is not None:
                self._pending_extra[key] = extra_kwargs
            if meta is not None:
                self._request_meta[key] = meta
