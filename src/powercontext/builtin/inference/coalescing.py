# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Coalesce concurrent embedding calls and cache successful vectors."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from powercontext.builtin.inference.models import EmbeddingResult, InferenceUsage

if TYPE_CHECKING:  # pragma: no cover - typing only
    from powercontext.builtin.inference.models import EmbeddingVector
    from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(slots=True, eq=False)
class _Waiter:
    """One text whose vector every interested caller shares."""

    future: asyncio.Future[EmbeddingVector]
    batch: _Batch | None = None


@dataclass(slots=True, eq=False)
class _Batch:
    """One underlying provider request and the texts it covers."""

    texts: tuple[str, ...]
    waiters: tuple[_Waiter, ...]
    usage: InferenceUsage | None = None


class CoalescingEmbeddingModel:
    """Share in-flight work, bound provider batches and cache successful vectors."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if not isinstance(max_batch_size, int) or isinstance(max_batch_size, bool) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if not isinstance(coalesce_ms, int | float) or isinstance(coalesce_ms, bool) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a non-negative number")
        if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")
        self._model = model
        self.profile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = float(coalesce_ms) / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._waiters: dict[str, _Waiter] = {}
        self._queued: OrderedDict[str, None] = OrderedDict()
        self._flush_handle: asyncio.TimerHandle | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return one vector per input, keeping duplicates in their original slot."""

        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        vectors: dict[str, EmbeddingVector] = {}
        pending: list[tuple[str, _Waiter]] = []
        for text in dict.fromkeys(texts):
            cached = self._cache_lookup(text)
            if cached is not None:
                vectors[text] = cached
                continue
            waiter = self._waiters.get(text)
            if waiter is None:
                waiter = _Waiter(future=asyncio.get_running_loop().create_future())
                self._waiters[text] = waiter
                self._queued[text] = None
                self._schedule_flush()
            pending.append((text, waiter))

        if pending:
            await asyncio.gather(*(asyncio.shield(waiter.future) for _, waiter in pending))
            for text, waiter in pending:
                vectors[text] = waiter.future.result()

        return EmbeddingResult(
            vectors=tuple(vectors[text] for text in texts),
            usage=_sum_usage(pending),
        )

    # ------------------------------------------------------------------ queue

    def _schedule_flush(self) -> None:
        if self._flush_handle is not None:
            return
        loop = asyncio.get_running_loop()
        if self._coalesce_seconds <= 0:
            self._flush_handle = loop.call_soon(self._flush)
        else:
            self._flush_handle = loop.call_later(self._coalesce_seconds, self._flush)

    def _flush(self) -> None:
        self._flush_handle = None
        queued = tuple(self._queued)
        self._queued.clear()
        for offset in range(0, len(queued), self._max_batch_size):
            texts = queued[offset : offset + self._max_batch_size]
            waiters = tuple(self._waiters[text] for text in texts)
            batch = _Batch(texts=texts, waiters=waiters)
            for waiter in waiters:
                waiter.batch = batch
            task = asyncio.get_running_loop().create_task(self._run_batch(batch))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _run_batch(self, batch: _Batch) -> None:
        vectors: tuple[EmbeddingVector, ...] | None = None
        error: BaseException | None = None
        try:
            result = await self._model.embed(batch.texts)
            vectors = tuple(result.vectors)
            if len(vectors) != len(batch.texts):
                raise ValueError("embedding provider returned the wrong number of vectors")
            batch.usage = result.usage
        except asyncio.CancelledError as cancelled:
            error = cancelled
        except BaseException as failure:  # noqa: BLE001 - forwarded to every waiter
            error = failure

        for text, waiter in zip(batch.texts, batch.waiters, strict=True):
            if self._waiters.get(text) is waiter:
                del self._waiters[text]
            if waiter.future.done():
                continue
            if error is not None:
                waiter.future.set_exception(error)
                continue
            vector = vectors[batch.texts.index(text)]
            self._store_cache(text, vector)
            waiter.future.set_result(vector)

    # ------------------------------------------------------------------ cache

    def _cache_lookup(self, text: str) -> EmbeddingVector | None:
        if self._cache_size == 0:
            return None
        vector = self._cache.get(text)
        if vector is not None:
            self._cache.move_to_end(text)
        return vector

    def _store_cache(self, text: str, vector: EmbeddingVector) -> None:
        if self._cache_size == 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def _sum_usage(pending: list[tuple[str, _Waiter]]) -> InferenceUsage:
    batches: list[_Batch] = []
    for _, waiter in pending:
        batch = waiter.batch
        if batch is not None and batch not in batches:
            batches.append(batch)
    requests = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    for batch in batches:
        if batch.usage is None:
            continue
        requests += batch.usage.requests
        if batch.usage.input_tokens is not None:
            input_tokens = (input_tokens or 0) + batch.usage.input_tokens
        if batch.usage.output_tokens is not None:
            output_tokens = (output_tokens or 0) + batch.usage.output_tokens
    return InferenceUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens)


__all__ = ["CoalescingEmbeddingModel"]
