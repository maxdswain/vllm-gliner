"""Stable pooler protocol — zero vLLM imports.

Defines the contract between business pooling logic (GLiNER, ColBERT, etc.)
and the vLLM adapter layer.  When vLLM changes its Pooler ABC or
PoolingMetadata layout, only ``vllm_adapter.py`` needs updating — everything
that implements ``FactoryPooler`` stays untouched.
"""

import contextlib
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch

NO_ADAPTER = -1


@contextlib.contextmanager
def _no_scope() -> Iterator[None]:
    yield


def null_lora_scope(seq_index: int) -> AbstractContextManager[None]:
    """Return a scope that applies no adapter, for runs without LoRA.

    Parameters
    ----------
    seq_index : int
        Position of the sequence in the scheduled batch.

    Returns
    -------
    AbstractContextManager[None]
        A context manager that does nothing.

    """
    del seq_index
    return _no_scope()


@dataclass
class PoolerContext:
    """Version-independent per-batch metadata that business poolers receive.

    The adapter translates vLLM's ``PoolingMetadata`` into this before
    calling ``FactoryPooler.forward()``.

    ``lora_scope`` is what binds head projections to the right adapter: a
    pooler entering it for sequence ``i`` has every LoRA-wrapped head under it
    apply that sequence's adapter, whatever shape the head projects. Head work
    that mixes sequences must therefore first check that ``lora_slots`` holds
    one value.
    """

    seq_lengths: list[int]
    extra_kwargs: list[dict[str, object]] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    prompt_token_ids: list[torch.Tensor] = field(default_factory=list)
    device: torch.device | None = None
    dtype: torch.dtype | None = None
    lora_slots: tuple[int, ...] = ()
    lora_scope: Callable[[int], AbstractContextManager[None]] = null_lora_scope

    def shares_one_adapter(self) -> bool:
        """Return whether every scheduled sequence carries the same adapter."""
        return len(set(self.lora_slots)) <= 1


def split_hidden_states(
    hidden_states: torch.Tensor,
    seq_lengths: list[int],
) -> list[torch.Tensor]:
    """Split concatenated hidden states into per-sequence tensors.

    Pure torch, no vLLM dependency.  Works on both CPU and GPU.
    """
    parts: list[torch.Tensor] = []
    offset = 0
    for length in seq_lengths:
        parts.append(hidden_states[offset : offset + length])
        offset += length
    return parts


@runtime_checkable
class FactoryPooler(Protocol):
    """Stable interface that business pooling logic implements.

    Implementations must never import from ``vllm.*``.
    """

    def get_tasks(self) -> set[str]:
        """Return the set of pooling task names this pooler supports."""
        ...

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: PoolerContext,
    ) -> list[torch.Tensor | None]:
        """Run business pooling logic.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Concatenated hidden states for all sequences
            in the batch, shape ``(total_tokens, hidden_dim)``.
        ctx : PoolerContext
            Batch metadata (sequence lengths, extra kwargs, etc.).

        Returns
        -------
        list[torch.Tensor | None]
            One tensor per sequence (or ``None`` for unfinished sequences
            when chunked prefill is active).

        """
        ...


class PassthroughPooler:
    """No-op pooler for models that handle everything in ``forward()``.

    Splits concatenated hidden states per sequence and returns them unchanged.
    When wrapped by ``VllmPoolerAdapter``, the adapter delegates to vLLM's
    ``AllPool`` for GPU-efficient splitting with chunked-prefill support,
    so ``forward()`` here serves only as a pure-torch fallback.
    """

    def get_tasks(self) -> set[str]:
        return {"token_embed"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: PoolerContext,
    ) -> list[torch.Tensor | None]:
        return split_hidden_states(hidden_states, ctx.seq_lengths)
