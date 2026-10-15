"""Boundary serving processor: GLiNER2 schema parse and prompt collate."""

from plugins.deberta_gliner2.processor import decode_output
from vllm_factory.optional_deps import require

RUNTIMES = ("classic", "constrained_classification", "joint_ie")
_COMPACT_FLAG = "_compact"
# Schema markers, separators and specials share the prompt with the text.
_SCHEMA_TOKEN_HEADROOM = 128
_FIT_ATTEMPTS = 3


def prompt_schema(runtime: str, schema: dict[str, object]) -> dict[str, object]:
    """Parse ``schema`` with the runtime's GLiNER2 parser and return its prompt.

    Parameters
    ----------
    runtime : str
        ``classic``, ``constrained_classification``, or ``joint_ie``.
    schema : dict[str, object]
        Raw schema dict accepted by that runtime's ``from_dict``.

    Returns
    -------
    dict[str, object]
        The dict GLiNER2 collates into the encoder prompt.

    Raises
    ------
    ValueError
        ``runtime`` or ``schema`` is not valid for that parser.

    """
    if runtime not in RUNTIMES:
        msg = "'runtime' must be classic, constrained_classification, or joint_ie"
        raise ValueError(
            msg,
        )
    if not isinstance(schema, dict):
        msg = "'schema' must be an object"
        raise ValueError(msg)
    from pydantic import ValidationError

    try:
        return _compile_prompt(runtime, schema)
    except (ValidationError, TypeError, KeyError) as exc:
        raise ValueError(str(exc)) from exc


def _compile_prompt(runtime: str, schema: dict[str, object]) -> dict[str, object]:
    if runtime == "classic":
        schema_mod = require(
            "gliner2.inference.schema", "gliner2", purpose="GLiNER25 schema parse",
        )
        return schema_mod.Schema.from_dict(schema).build()
    if runtime == "constrained_classification":
        classification = require(
            "gliner2.classification", "gliner2", purpose="GLiNER25 classification parse",
        )
        parsed = classification.ClassificationSchema.from_dict(schema)
        return classification.compile_schema(parsed).build()
    joint_schema = require(
        "gliner2.joint_ie.schema", "gliner2", purpose="GLiNER25 joint schema parse",
    )
    compiler = require(
        "gliner2.joint_ie.compiler", "gliner2", purpose="GLiNER25 joint compile",
    )
    parsed = joint_schema.JointSchema.from_dict(schema)
    return compiler.compile_schema(parsed).build()


def boundary_transformer(tokenizer: object, *, token_pooling: str = "first") -> object:
    """Build a SchemaTransformer bound to ``tokenizer``.

    Parameters
    ----------
    tokenizer : object
        Tokenizer the transformer encodes with.
    token_pooling : str
        Wordpiece aggregation strategy.

    Returns
    -------
    object
        A GLiNER2 SchemaTransformer. Construction is cheap but not free, so
        callers that collate per request hold onto the instance.

    """
    processor_mod = require(
        "gliner2.processor", "gliner2", purpose="GLiNER25 preprocess",
    )
    return processor_mod.SchemaTransformer(
        tokenizer=tokenizer, token_pooling=token_pooling,
    )


def collate_word_cap(max_model_len: int | None) -> int | None:
    """Word budget for truncation, leaving room for the schema tokens.

    Parameters
    ----------
    max_model_len : int | None
        Encoded-token ceiling, or None when unbounded.

    Returns
    -------
    int | None
        Word cap to hand GLiNER2 collate, or None when unbounded. Words are
        not tokens, so this is a starting point that ``preprocess_boundary``
        tightens while the encoded prompt still overflows.

    """
    if not max_model_len:
        return None
    return max(1, int(max_model_len) - _SCHEMA_TOKEN_HEADROOM)


def iter_boundary_items(data: object) -> list[dict[str, object]]:
    """Normalize a /pooling ``data`` payload into per-text dicts.

    Accepts a single item, ``text: list``, or a list of items. Unwraps a
    vLLM request object or ``{"data": ...}`` wrapper when present.
    """
    if hasattr(data, "data"):
        data = data.data
    elif isinstance(data, dict) and "data" in data and not _is_item_dict(data):
        data = data["data"]
    if isinstance(data, list):
        if not data:
            msg = "'data' must be a non-empty list"
            raise ValueError(msg)
        return [_as_item_dict(item) for item in data]
    if isinstance(data, dict):
        text = data.get("text")
        if isinstance(text, list):
            if not text:
                msg = "'text' must be a non-empty list"
                raise ValueError(msg)
            shared = {key: value for key, value in data.items() if key != "text"}
            return [{**shared, "text": item} for item in text]
        return [data]
    msg = "Expected request data dict"
    raise ValueError(msg)


def _is_item_dict(data: dict[str, object]) -> bool:
    return "text" in data or "schema" in data or "labels" in data


def _as_item_dict(item: object) -> dict[str, object]:
    if not isinstance(item, dict):
        msg = "Each batch item must be a dict"
        raise ValueError(msg)
    return item


def compact_boundary_extra(
    text: str,
    schema: dict[str, object],
    *,
    runtime: str = "classic",
    threshold: float = 0.5,
    include_confidence: bool = False,
    include_spans: bool = False,
    max_len: int | None = None,
) -> dict[str, object]:
    """Metadata the pooler needs to call GLiNER2 for one sequence.

    This crosses vLLM V1 ZMQ. ``max_len`` is the word cap the prompt collate
    actually applied, so the pooler's collate lines up with the encoder tokens.

    Parameters
    ----------
    text : str
        Input text to extract from.
    schema : dict[str, object]
        Raw schema the pooler parses with ``from_dict``.
    runtime : str
        Which GLiNER2 batch API decodes this sequence.
    threshold : float
        Score threshold carried through to classic decode.
    include_confidence : bool
        Whether decode should return scores.
    include_spans : bool
        Whether decode should return character spans.
    max_len : int | None
        Word cap applied during collate, or None when unbounded.

    Returns
    -------
    dict[str, object]
        Compact ``extra_kwargs`` for one sequence.

    """
    extra: dict[str, object] = {
        _COMPACT_FLAG: True,
        "text": text,
        "schema": schema,
        "runtime": runtime,
        "threshold": threshold,
        "include_confidence": include_confidence,
        "include_spans": include_spans,
    }
    if max_len is not None:
        extra["max_len"] = int(max_len)
    return extra


def is_compact_extra(extra: dict[str, object]) -> bool:
    """Check whether extras contain reconstructable text/schema instead of tensors."""
    if extra.get(_COMPACT_FLAG):
        return True
    return "text" in extra and "schema" in extra and "text_word_indices" not in extra


def preprocess_boundary(
    tokenizer,
    text: str,
    schema: dict[str, object],
    *,
    prompt: dict[str, object] | None = None,
    runtime: str = "classic",
    threshold: float = 0.5,
    include_confidence: bool = False,
    include_spans: bool = False,
    truncate_overflow_text: bool = False,
    word_cap: int | None = None,
    max_model_len: int | None = None,
    transformer: object = None,
) -> dict[str, object]:
    """Run GLiNER2 boundary collate; stash compact extras for the pooler.

    Collate drops words past its cap without reporting it, so the cap is only
    applied when the caller allowed truncation. The cap counts words while the
    engine limit counts tokens, so an allowed truncation retries with a
    tighter cap until the encoded prompt fits.

    Parameters
    ----------
    tokenizer : object
        Tokenizer used when ``transformer`` is not supplied.
    text : str
        Input text to extract from.
    schema : dict[str, object]
        Raw schema stored for the pooler.
    prompt : dict[str, object] | None
        Collate dict. Defaults to ``schema`` when the raw dict is the prompt.
    runtime : str
        Which GLiNER2 batch API decodes this sequence.
    threshold : float
        Score threshold carried through to classic decode.
    include_confidence : bool
        Whether decode should return scores.
    include_spans : bool
        Whether decode should return character spans.
    truncate_overflow_text : bool
        Whether text past the budget may be dropped.
    word_cap : int | None
        Word budget used when truncating; None leaves text intact.
    max_model_len : int | None
        Encoded-token ceiling the engine can schedule.
    transformer : object
        SchemaTransformer to collate with, built if omitted.

    Returns
    -------
    dict[str, object]
        Dict with ``input_ids``, compact ``extra_kwargs`` and ``schema_dict``.

    Raises
    ------
    ValueError
        The encoded prompt is above ``max_model_len`` and either
            truncation was not requested or it did not converge.

    """
    host = transformer if transformer is not None else boundary_transformer(tokenizer)
    collate_schema = schema if prompt is None else prompt
    cap = word_cap if truncate_overflow_text else None
    for _ in range(_FIT_ATTEMPTS):
        batch = host.collate_fn_inference(
            [(text, collate_schema)],
            architecture="boundary",
            error_policy="raise",
            max_len=cap,
        )
        input_ids = batch.input_ids[0].detach().cpu().tolist()
        if not max_model_len or len(input_ids) <= max_model_len:
            return {
                "input_ids": input_ids,
                "extra_kwargs": compact_boundary_extra(
                    text,
                    schema,
                    runtime=runtime,
                    threshold=threshold,
                    include_confidence=include_confidence,
                    include_spans=include_spans,
                    max_len=cap,
                ),
                "schema_dict": schema,
            }
        if not truncate_overflow_text:
            msg = (
                f"encoded prompt is {len(input_ids)} tokens, above the "
                f"{max_model_len}-token limit; shorten 'text', raise "
                "--max-model-len, or set 'truncate_overflow_text'"
            )
            raise ValueError(
                msg,
            )
        cap = _tighter_word_cap(batch, len(input_ids), max_model_len)
    msg = f"'text' did not fit under {max_model_len} tokens after {_FIT_ATTEMPTS} tries"
    raise ValueError(
        msg,
    )


def _tighter_word_cap(batch: object, tokens: int, budget: int) -> int:
    """Scale the word cap down by the ratio the encoded prompt overflowed by."""
    kept = len(batch.text_tokens[0])
    return max(1, int(kept * budget / tokens) - 1)


def decode_boundary_output(
    raw_output, schema: dict[str, object] | None = None,
) -> dict[str, object]:
    """Unpack the JSON byte tensor produced by the boundary pooler."""
    return decode_output(raw_output, schema or {})
