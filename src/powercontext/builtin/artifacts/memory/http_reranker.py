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

"""Score coarse Memory candidates with an offline HTTP cross-encoder service."""

from __future__ import annotations

import math
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision, _validate_bounds
from powercontext.builtin.inference import InferenceUsage

HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"


class HTTPMemoryReranker:
    """Score every coarse candidate in bounded batches and order the whole pool."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = HTTP_RERANK_POLICY_ID,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("reranker endpoint must be a non-empty string")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("reranker batch_size must be a positive integer")
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("reranker policy_id must be a non-empty string")
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Return original ranks ordered by service score, then original rank."""

        _validate_bounds(len(candidates), limit)
        scores: dict[int, float] = {}
        requests = 0
        for offset in range(0, len(candidates), self._batch_size):
            batch = candidates[offset : offset + self._batch_size]
            expected = tuple(range(offset + 1, offset + 1 + len(batch)))
            payload = {
                "query": query,
                "candidates": [
                    {"rank": rank, "text": candidate.text}
                    for rank, candidate in zip(expected, batch, strict=True)
                ],
            }
            response = await self._client.post(self._endpoint, json=payload)
            response.raise_for_status()
            scores.update(_parse_scores(_decode_body(response), expected))
            requests += 1

        ordered = sorted(scores, key=lambda rank: (-scores[rank], rank))
        return MemoryRerankDecision(
            selected_ranks=tuple(ordered[:limit]),
            usage=InferenceUsage(requests=requests),
        )


def _decode_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as error:
        raise ValueError("reranker response must be valid JSON") from error


def _parse_scores(body: Any, expected_ranks: tuple[int, ...]) -> dict[int, float]:
    if not isinstance(body, dict) or set(body) != {"scores"}:
        raise ValueError("reranker response must contain exactly the scores field")
    scores = body["scores"]
    if not isinstance(scores, list):
        raise ValueError("reranker scores must be a list")

    parsed: dict[int, float] = {}
    for item in scores:
        if not isinstance(item, dict) or set(item) != {"rank", "score"}:
            raise ValueError("each reranker score must contain exactly rank and score")
        rank = item["rank"]
        score = item["score"]
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError("reranker score rank must be an integer")
        if not isinstance(score, int | float) or isinstance(score, bool):
            raise ValueError("reranker score must be a number")
        numeric = float(score)
        if not math.isfinite(numeric):
            raise ValueError("reranker score must be finite")
        if rank in parsed:
            raise ValueError("reranker score ranks must be unique")
        parsed[rank] = numeric

    if set(parsed) != set(expected_ranks):
        raise ValueError("reranker scores must cover every candidate rank exactly once")
    return parsed


__all__ = ["HTTP_RERANK_POLICY_ID", "HTTPMemoryReranker"]
