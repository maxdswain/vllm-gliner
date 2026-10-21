"""vLLM 0.32 SystemOne reads for the English GLiNER2.5 Decide checkpoint.

vLLM only creates a native SystemOne handler for built-in read strategies. Its
endpoint-plugin hook runs after that handler is initialized, allowing this
explicitly allowlisted plugin to install a Decide strategy on the existing
route without shadowing the route or changing vLLM's global strategy registry.
"""

from argparse import Namespace
from collections.abc import Mapping, Sequence
from typing import ClassVar

from fastapi import FastAPI
from starlette.datastructures import State
from typing_extensions import override
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.generate.structured_decisions.question_types import (
    Question,
    StructuredDecisionError,
    label_softmax,
)
from vllm.entrypoints.generate.structured_decisions.serving import ServingStructuredDecisions
from vllm.entrypoints.generate.structured_decisions.strategies import (
    DecisionLimits,
    QuestionRead,
    ReadContext,
    ReadStrategy,
)
from vllm.inputs import tokens_input
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.tasks import SupportedTask

from plugins.deberta_gliner2.io_processor import DeBERTaGLiNER2IOProcessor
from plugins.deberta_gliner2.processor import decode_output

DECIDE_MODEL_ID = "fastino/GLiNER2.5-Decide"


class DecideReadStrategy(ReadStrategy):
    """Score every SystemOne question in one GLiNER2 classification pass."""

    task: ClassVar[SupportedTask] = "embed"

    def __init__(self, context: ReadContext) -> None:
        super().__init__(context)
        self._processor = DeBERTaGLiNER2IOProcessor(
            context.engine_client.vllm_config,
            context.online_renderer.renderer,
        )

    @override
    def limits(self) -> DecisionLimits:
        """Keep vLLM's standard question and choice limits."""
        return DecisionLimits(max_questions=64, max_options=26)

    @override
    def num_requests(self, questions: list[Question]) -> int:
        """Batch all questions into one GLiNER2 schema and engine request."""
        del questions
        return 1

    @override
    async def read(
        self,
        questions: list[Question],
        instructions: str | None,
        state: str,
        *,
        request_id: str,
        chat_template_kwargs: dict[str, object] | None,
        lora_request: LoRARequest | None,
        priority: int,
        cache_salt: str | None = None,
        trace_headers: Mapping[str, str] | None = None,
        media: Mapping[str, Sequence[str]] | None = None,
    ) -> list[QuestionRead]:
        """Read closed-set label probabilities directly from the Decide head."""
        if media or chat_template_kwargs or cache_salt is not None:
            msg = "Decide accepts text questions, not media, chat options, or cache salts"
            raise StructuredDecisionError(msg)

        classifications: list[dict[str, object]] = []
        result_keys: list[str] = []
        for question in questions:
            descriptions = {
                option.name: option.description
                for option in question.options
                if option.description is not None
            }
            prompt = "\n".join(part for part in (instructions, question.instructions) if part)
            result_keys.append(f"{question.id}: {prompt}" if prompt else question.id)
            classifications.append(
                {
                    "task": question.id,
                    "labels": [option.name for option in question.options],
                    "label_descriptions": descriptions,
                    "prompt": prompt,
                },
            )

        try:
            parsed = self._processor.factory_parse(
                {"text": state, "schema": {"classifications": classifications}, "threshold": 0},
            )
            prompt, extra, _ = self._processor.prepare_input(parsed, request_id)
        except ValueError as exc:
            raise StructuredDecisionError(str(exc)) from exc
        ids = prompt["prompt_token_ids"]
        result = None
        async for output in self.context.engine_client.encode(
            tokens_input(ids),
            PoolingParams(task="embed", extra_kwargs=extra),
            request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
        ):
            result = output
        if result is None or result.error is not None or result.outputs.data is None:
            msg = "Decide returned no classification data"
            raise RuntimeError(msg)
        decoded = decode_output(result.outputs.data, parsed.schema)
        return [
            QuestionRead(
                probs=label_softmax(decoded[key]["logits"]),
                label_mass=1.0,
                argmax_is_label=True,
                input_tokens=len(ids) if index == 0 else 0,
                output_tokens=0,
            )
            for index, key in enumerate(result_keys)
        ]


class DecideSystemOnePlugin:
    """Opt-in SystemOne handler for the exact English Decide checkpoint."""

    name = "decide_systemone"
    required_tasks: tuple[SupportedTask, ...] | None = ("embed",)

    def attach_router(self, app: FastAPI) -> None:
        """Use vLLM's existing /v1/systemone route and response protocol."""
        del app

    async def init_state(
        self, engine_client: EngineClient | None, state: State, args: Namespace,
    ) -> None:
        """Install the Decide strategy after vLLM initializes its handlers."""
        if engine_client is None:
            return
        config = engine_client.model_config
        if config.architecture != "GLiNER2VLLMModel" or getattr(
            config.hf_config, "_source_model", None,
        ) != DECIDE_MODEL_ID:
            return
        strategy = DecideReadStrategy(
            ReadContext(
                engine_client=engine_client,
                online_renderer=state.online_renderer,
                chat_template=None,
                chat_template_content_format=args.chat_template_content_format,
                default_chat_template_kwargs={},
            ),
        )
        state.serving_structured_decisions = ServingStructuredDecisions(
            state.openai_serving_models,
            strategy,
        )
