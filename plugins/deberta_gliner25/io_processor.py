"""IO processor for deberta_gliner25 — same HTTP shape as the span plugin."""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from transformers import AutoTokenizer
from typing_extensions import override
from vllm.config import VllmConfig
from vllm.renderers.base import BaseRenderer

from plugins.deberta_gliner25.processor import (
    RUNTIMES,
    boundary_transformer,
    collate_word_cap,
    decode_boundary_output,
    iter_boundary_items,
    preprocess_boundary,
    prompt_schema,
)
from vllm_factory.io.base import (
    FactoryIOProcessor,
    PoolingRequestOutput,
    PromptType,
    TokensPrompt,
)

_ADAPTER_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-:/]{1,128}$")
RuntimeName = Literal["classic", "constrained_classification", "joint_ie"]


@dataclass
class GLiNER25Input:
    """One validated pooling item."""

    text: str
    schema: dict[str, object] = field(default_factory=dict)
    prompt: dict[str, object] = field(default_factory=dict)
    runtime: RuntimeName = "classic"
    threshold: float = 0.5
    include_confidence: bool = False
    include_spans: bool = False
    truncate_overflow_text: bool = False
    adapter: str | None = None


class DeBERTaGLiNER25IOProcessor(FactoryIOProcessor[GLiNER25Input | list[GLiNER25Input]]):
    """Boundary IO processor. HTTP contract matches deberta_gliner2."""

    pooling_task = "embed"

    def __init__(self, vllm_config: VllmConfig, renderer: BaseRenderer) -> None:
        super().__init__(vllm_config, renderer)
        model_id = vllm_config.model_config.model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True,
        )
        token_pooling = getattr(vllm_config.model_config.hf_config, "token_pooling", "first")
        self._transformer = boundary_transformer(
            self._tokenizer, token_pooling=str(token_pooling or "first"),
        )
        raw_max = getattr(vllm_config.model_config, "max_model_len", None)
        self._max_model_len = int(raw_max) if raw_max else None
        self._word_cap = collate_word_cap(self._max_model_len)

    @staticmethod
    def _coerce_bool(value: object, field_name: str) -> bool:
        if isinstance(value, bool):
            return value
        msg = f"'{field_name}' must be a boolean"
        raise ValueError(msg)

    @staticmethod
    def _coerce_adapter(value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            msg = "'adapter' must be a string or null"
            raise ValueError(msg)
        stripped = value.strip()
        if not stripped:
            return None
        if not _ADAPTER_NAME_RE.match(stripped):
            msg = f"'adapter' must match ^[A-Za-z0-9_.\\-:/]{{1,128}}$ — got {value!r}"
            raise ValueError(
                msg,
            )
        return stripped

    @staticmethod
    def _coerce_runtime(value: object) -> RuntimeName:
        if value is None:
            return "classic"
        if value not in RUNTIMES:
            msg = "'runtime' must be classic, constrained_classification, or joint_ie"
            raise ValueError(
                msg,
            )
        return value

    def _parse_one(self, data: dict[str, object]) -> GLiNER25Input:
        """Validate one item and compile its schema for the encoder prompt.

        Parameters
        ----------
        data : dict[str, object]
            One ``/pooling`` item.

        Returns
        -------
        GLiNER25Input
            The parsed item. ``schema`` is the raw dict; ``prompt`` is what
            collate encodes.

        Raises
        ------
        ValueError
            The item or its schema is invalid.

        """
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            msg = "'text' is required"
            raise ValueError(msg)
        threshold = data.get("threshold", 0.5)
        if not isinstance(threshold, (int, float, str)):
            msg = "'threshold' must be a number"
            raise ValueError(msg)
        try:
            threshold = float(threshold)
        except (TypeError, ValueError) as exc:
            msg = "'threshold' must be a number"
            raise ValueError(msg) from exc
        if not 0.0 <= threshold <= 1.0:
            msg = "'threshold' must be between 0 and 1"
            raise ValueError(msg)
        runtime = self._coerce_runtime(data.get("runtime", "classic"))
        raw_schema = data.get("schema")
        labels = data.get("labels")
        if raw_schema is None and labels is not None:
            if runtime != "classic":
                msg = "'labels' is only valid when runtime is classic"
                raise ValueError(msg)
            raw_schema = {"entities": labels}
        elif raw_schema is None:
            msg = "Request must include schema or labels"
            raise ValueError(msg)
        if not isinstance(raw_schema, dict):
            msg = "'schema' must be a mapping"
            raise ValueError(msg)
        prompt = prompt_schema(runtime, raw_schema)
        return GLiNER25Input(
            text=text,
            schema=raw_schema,
            prompt=prompt,
            runtime=runtime,
            threshold=threshold,
            include_confidence=self._coerce_bool(
                data.get("include_confidence", False), "include_confidence",
            ),
            include_spans=self._coerce_bool(
                data.get("include_spans", False), "include_spans",
            ),
            truncate_overflow_text=self._coerce_bool(
                data.get("truncate_overflow_text", False), "truncate_overflow_text",
            ),
            adapter=self._coerce_adapter(data.get("adapter")),
        )

    @override
    def factory_parse(self, data: object) -> GLiNER25Input | list[GLiNER25Input]:
        """Validate a request body holding one item or a batch of them.

        Parameters
        ----------
        data : object
            Raw ``/pooling`` data field: one item, ``text: list``, or a
            list of items.

        Returns
        -------
        GLiNER25Input | list[GLiNER25Input]
            One parsed input, or a list of them for a batch. The shape decides
            whether the rest of the pipeline runs single or batched.

        """
        items = iter_boundary_items(data)
        parsed = [self._parse_one(item) for item in items]
        if len(parsed) == 1:
            return parsed[0]
        return parsed

    def _pre_one(
        self, parsed_input: GLiNER25Input,
    ) -> tuple[TokensPrompt, dict[str, object], dict[str, object]]:
        """Collate one parsed request into a prompt, extras and decode meta.

        Parameters
        ----------
        parsed_input : GLiNER25Input
            One validated request item.

        Returns
        -------
        tuple[TokensPrompt, dict[str, object], dict[str, object]]
            The prompt to schedule, the pooler's compact extras, and the
            metadata ``factory_post_process`` uses for that sequence.

        """
        result = preprocess_boundary(
            self._tokenizer,
            parsed_input.text,
            parsed_input.schema,
            prompt=parsed_input.prompt,
            runtime=parsed_input.runtime,
            threshold=parsed_input.threshold,
            include_confidence=parsed_input.include_confidence,
            include_spans=parsed_input.include_spans,
            truncate_overflow_text=parsed_input.truncate_overflow_text,
            word_cap=self._word_cap,
            max_model_len=self._max_model_len,
            transformer=self._transformer,
        )
        postprocess_meta = {
            "schema_dict": parsed_input.schema,
            "adapter": parsed_input.adapter,
        }
        ids = result["input_ids"]
        extras = result["extra_kwargs"]
        if not isinstance(ids, list) or not isinstance(extras, dict):
            msg = "Invalid GLiNER2.5 preprocessor output"
            raise ValueError(msg)
        prompt = TokensPrompt(prompt_token_ids=[int(token_id) for token_id in ids])
        return prompt, {str(key): value for key, value in extras.items()}, {
            str(key): value for key, value in postprocess_meta.items()
        }

    @override
    def factory_pre_process(
        self,
        parsed_input: GLiNER25Input | list[GLiNER25Input],
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        """Collate every item and stash its extras and decode metadata.

        A batch stashes extras under ``_per_seq`` so each scheduled prompt can
        be given its own ``PoolingParams``; one shared params object would make
        every sequence decode against the first item's schema.

        Parameters
        ----------
        parsed_input : GLiNER25Input | list[GLiNER25Input]
            One parsed input, or a batch of them.
        request_id : str | None
            vLLM request id used to key the stash.

        Returns
        -------
        PromptType | Sequence[PromptType]
            One prompt, or one prompt per batch item in the same order.

        """
        items = parsed_input if isinstance(parsed_input, list) else [parsed_input]
        prompts: list[TokensPrompt] = []
        extras: list[dict[str, object]] = []
        metas: list[dict[str, object]] = []
        for item in items:
            prompt, extra, meta = self._pre_one(item)
            prompts.append(prompt)
            extras.append(extra)
            metas.append(meta)
        if len(prompts) == 1:
            self._stash(extra_kwargs=extras[0], request_id=request_id, meta=metas[0])
            return prompts[0]
        self._stash(
            extra_kwargs={"_per_seq": extras},
            request_id=request_id,
            meta=metas,
        )
        return prompts

    @override
    def factory_post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_meta: object,
    ) -> dict[str, object] | list[dict[str, object]]:
        """Return each sequence's pooler JSON, echoing its adapter.

        Parameters
        ----------
        model_output : Sequence[PoolingRequestOutput]
            Pooler outputs, one per scheduled prompt.
        request_meta : object
            The metadata stashed by ``factory_pre_process``: one
            dict, or a list of them for a batch.

        Returns
        -------
        dict | list[dict]
            One result, or a list of them for a batch, in request order.

        """
        if not model_output or request_meta is None:
            return [] if isinstance(request_meta, list) else {}
        metas = request_meta if isinstance(request_meta, list) else [request_meta]
        results: list[dict[str, object]] = []
        for output, meta in zip(model_output, metas, strict=True):
            raw = output.outputs.data
            if raw is None:
                results.append({})
                continue
            if not isinstance(meta, dict):
                msg = "Invalid GLiNER2.5 response metadata"
                raise ValueError(msg)
            schema = meta.get("schema_dict")
            decoded = decode_boundary_output(raw, schema if isinstance(schema, dict) else {})
            if isinstance(decoded, dict) and meta.get("adapter") is not None:
                decoded.setdefault("adapter", meta["adapter"])
            results.append(decoded if isinstance(decoded, dict) else {})
        if len(results) == 1:
            return results[0]
        return results


def get_processor_cls() -> str:
    """Return the IO processor class path vLLM loads."""
    return "plugins.deberta_gliner25.io_processor.DeBERTaGLiNER25IOProcessor"
