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

"""Propagate Source availability to the Memory and Topic Memory it supports.

A scope registers a dependency DAG: Sources at the roots, Memory entries in the
middle and Topic Memory artifacts on top. Withdrawing a Source deactivates every
descendant whose support policy no longer holds, and restoring it reactivates the
descendants that regain support. Bodies are never rewritten, so historical
citations keep pointing at the exact versions they always named.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.tables import (
    SOURCE_LIFECYCLE_EVENTS_TABLE,
    SOURCE_LIFECYCLE_NODES_TABLE,
)
from powercontext.builtin.runtime.errors import InvalidRuntimeRequestError
from powercontext.builtin.runtime.models import MemoryEntryStateRequest
from powercontext.sources.models import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime

FAMILY_SOURCE = "source"
FAMILY_MEMORY = "memory"
FAMILY_TOPIC = "topic"

POLICY_ALL = "all"
POLICY_ANY = "any"

PURPOSE_GENERATION = "generation"
PURPOSE_AUDIT = "audit"

_PURPOSES = frozenset({PURPOSE_GENERATION, PURPOSE_AUDIT})
_POLICIES = frozenset({POLICY_ALL, POLICY_ANY})
_FAMILIES = frozenset({FAMILY_SOURCE, FAMILY_MEMORY, FAMILY_TOPIC})
_LIFECYCLE_REASON = "source-lifecycle"
_SYNC_ROUNDS = 4


class SourceLifecycleError(ValueError):
    """A lifecycle request cannot be honoured by the registered dependency graph."""


class SourceLifecycleEvent(BaseModel):
    """One Source availability event carrying its lifecycle identity."""

    model_config = ConfigDict(extra="forbid", strict=True)

    node_id: str
    revision: int
    sequence: int
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def _require_text(cls, value: str, info: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @field_validator("revision", "sequence")
    @classmethod
    def _require_positive(cls, value: int, info: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return value


class _Node:
    """One decoded lifecycle node."""

    __slots__ = (
        "applied",
        "available",
        "family",
        "node_id",
        "parents",
        "payload",
        "policy",
        "position",
        "revision",
        "sequence",
    )

    def __init__(
        self,
        *,
        node_id: str,
        family: str,
        payload: str,
        parents: tuple[str, ...],
        policy: str,
        revision: int | None,
        sequence: int | None,
        available: bool,
        applied: bool | None,
        position: int,
    ) -> None:
        self.node_id = node_id
        self.family = family
        self.payload = payload
        self.parents = parents
        self.policy = policy
        self.revision = revision
        self.sequence = sequence
        self.available = available
        self.applied = applied
        self.position = position

    def reference(self) -> SourceRef | MemoryCitation | ArtifactRef:
        if self.family == FAMILY_SOURCE:
            return SourceRef.model_validate(json.loads(self.payload))
        if self.family == FAMILY_MEMORY:
            return MemoryCitation.model_validate(json.loads(self.payload))
        return ArtifactRef.model_validate(json.loads(self.payload))


class LifecycleApplication:
    """Select the per-scope lifecycle service."""

    def __init__(self, runtime: BuiltinRuntime) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str, /) -> SourceLifecycleService:
        return SourceLifecycleService(self._runtime, scope_id)


class SourceLifecycleService:
    """Register one scope's evidence DAG and apply Source availability events."""

    def __init__(self, runtime: BuiltinRuntime, scope_id: str) -> None:
        self._runtime = runtime
        self._scope_id = scope_id

    # ------------------------------------------------------------------ deposit
    async def register_source(self, node_id: str, source: SourceRef, *, revision: int) -> None:
        """Register a Source root bound to one lifecycle revision."""

        if not isinstance(source, SourceRef):
            raise SourceLifecycleError("source must be a SourceRef")
        await self._register(
            node_id,
            family=FAMILY_SOURCE,
            payload=_dump(source),
            parents=(),
            policy=POLICY_ALL,
            revision=_positive_int("revision", revision),
        )

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: Sequence[str],
        policy: str,
    ) -> None:
        """Register a stored Memory entry supported by its parents."""

        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("citation must be a MemoryCitation")
        await self._verify_citation(citation)
        await self._register(
            node_id,
            family=FAMILY_MEMORY,
            payload=_dump(citation),
            parents=parents,
            policy=_policy(policy),
            revision=None,
        )

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        *,
        parents: Sequence[str],
        policy: str,
    ) -> None:
        """Register a Topic Memory artifact supported by its parents."""

        if not isinstance(reference, ArtifactRef):
            raise SourceLifecycleError("reference must be an ArtifactRef")
        await self._verify_artifact(reference)
        await self._register(
            node_id,
            family=FAMILY_TOPIC,
            payload=_dump(reference),
            parents=parents,
            policy=_policy(policy),
            revision=None,
        )

    # -------------------------------------------------------------------- apply
    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        """Apply one Source event and return every available node, sorted."""

        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("event must be a SourceLifecycleEvent")
        async with self._runtime._scoped_operation(self._scope_id), self._runtime._locked(self._scope_id):
            async with self._runtime.database.transaction() as connection:
                nodes = await _load_nodes(connection, self._scope_id)
                source = nodes.get(event.node_id)
                if source is None or source.family != FAMILY_SOURCE:
                    raise SourceLifecycleError(f"unknown Source node {event.node_id!r}")
                if source.revision != event.revision:
                    raise SourceLifecycleError("event revision does not match the registered revision")
                previous = await _load_event(connection, self._scope_id, event.node_id, event.sequence)
                if previous is not None:
                    if previous != (event.revision, event.available):
                        raise SourceLifecycleError("event sequence replayed with different content")
                else:
                    if event.sequence <= (source.sequence or 0):
                        raise SourceLifecycleError("event sequence must increase")
                    await connection.execute(
                        SOURCE_LIFECYCLE_EVENTS_TABLE.insert().values(
                            scope_id=self._scope_id,
                            node_id=event.node_id,
                            sequence=event.sequence,
                            revision=event.revision,
                            available=event.available,
                            reason=event.reason,
                        )
                    )
                    source.sequence = event.sequence
                    source.available = event.available
                    await connection.execute(
                        update(SOURCE_LIFECYCLE_NODES_TABLE)
                        .where(
                            SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == self._scope_id,
                            SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == event.node_id,
                        )
                        .values(sequence=event.sequence, available=event.available)
                    )
                    await self._propagate(connection, nodes)
        await self._sync_memory_states()
        async with self._runtime.database.transaction() as connection:
            nodes = await _load_nodes(connection, self._scope_id)
        return sorted(node for node, item in nodes.items() if item.available)

    # ------------------------------------------------------------------ resolve
    async def resolve(self, node_id: str, *, purpose: str) -> SourceRef | MemoryCitation | ArtifactRef:
        """Return the exact reference a caller may use for ``purpose``."""

        if purpose not in _PURPOSES:
            raise SourceLifecycleError(f"unknown purpose {purpose!r}")
        async with self._runtime._scoped_operation(self._scope_id), self._runtime.database.transaction() as connection:
            nodes = await _load_nodes(connection, self._scope_id)
        node = nodes.get(_identifier(node_id, "node_id"))
        if node is None:
            raise SourceLifecycleError(f"unknown node {node_id!r}")
        reference = node.reference()
        if purpose == PURPOSE_AUDIT:
            return reference
        if not node.available:
            raise SourceLifecycleError(f"node {node_id!r} is not available for generation")
        return reference

    async def inactive_topic_ids(self) -> frozenset[str]:
        """Return the artifact ids of withdrawn Topic Memory artifacts."""

        async with self._runtime._scoped_operation(self._scope_id), self._runtime.database.transaction() as connection:
            nodes = await _load_nodes(connection, self._scope_id)
        inactive: set[str] = set()
        for node in nodes.values():
            if node.family == FAMILY_TOPIC and not node.available:
                inactive.add(json.loads(node.payload)["artifact_id"])
        return frozenset(inactive)

    # ----------------------------------------------------------------- internals
    async def _register(
        self,
        node_id: str,
        *,
        family: str,
        payload: str,
        parents: Sequence[str],
        policy: str,
        revision: int | None,
    ) -> None:
        node = _identifier(node_id, "node_id")
        normalized_parents = _parents(parents, node)
        async with self._runtime._scoped_operation(self._scope_id), self._runtime._locked(self._scope_id):
            async with self._runtime.database.transaction() as connection:
                nodes = await _load_nodes(connection, self._scope_id)
                existing = nodes.get(node)
                if existing is not None:
                    if (
                        existing.family == family
                        and existing.payload == payload
                        and existing.parents == normalized_parents
                        and existing.policy == policy
                        and existing.revision == revision
                    ):
                        return
                    raise SourceLifecycleError(f"node {node!r} is already registered with a different binding")
                if family != FAMILY_SOURCE and not normalized_parents:
                    raise SourceLifecycleError("a derived node needs at least one parent")
                for parent in normalized_parents:
                    if parent not in nodes:
                        raise SourceLifecycleError(f"unknown parent node {parent!r}")
                available = True if family == FAMILY_SOURCE else _derive(normalized_parents, policy, nodes)
                position = max((item.position for item in nodes.values()), default=0) + 1
                await connection.execute(
                    SOURCE_LIFECYCLE_NODES_TABLE.insert().values(
                        scope_id=self._scope_id,
                        node_id=node,
                        family=family,
                        payload=payload,
                        parents=json.dumps(list(normalized_parents)),
                        policy=policy,
                        revision=revision,
                        sequence=None,
                        available=available,
                        applied=None,
                        position=position,
                    )
                )
        await self._sync_memory_states()

    async def _propagate(self, connection: AsyncConnection, nodes: dict[str, _Node]) -> None:
        for node in sorted(nodes.values(), key=lambda item: item.position):
            if node.family == FAMILY_SOURCE:
                continue
            expected = _derive(node.parents, node.policy, nodes)
            if expected == node.available:
                continue
            node.available = expected
            await connection.execute(
                update(SOURCE_LIFECYCLE_NODES_TABLE)
                .where(
                    SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == self._scope_id,
                    SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == node.node_id,
                )
                .values(available=expected)
            )

    async def _sync_memory_states(self) -> None:
        for _ in range(_SYNC_ROUNDS):
            async with self._runtime.database.transaction() as connection:
                nodes = await _load_nodes(connection, self._scope_id)
            pending = sorted(
                (
                    node
                    for node in nodes.values()
                    if node.family == FAMILY_MEMORY and node.applied is not node.available
                ),
                key=lambda item: item.position,
            )
            if not pending:
                return
            application = self._runtime.memory.for_scope(self._scope_id)
            for node in pending:
                citation = node.reference()
                if not isinstance(citation, MemoryCitation):
                    raise SourceLifecycleError("registered Memory node lost its citation")
                try:
                    await application.set_state(
                        MemoryEntryStateRequest(
                            citation=citation,
                            active=node.available,
                            reason=_LIFECYCLE_REASON,
                        )
                    )
                except Exception as error:  # noqa: BLE001 - surfaced as a lifecycle failure
                    raise SourceLifecycleError("memory state propagation failed") from error
                async with self._runtime.database.transaction() as connection:
                    await connection.execute(
                        update(SOURCE_LIFECYCLE_NODES_TABLE)
                        .where(
                            SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == self._scope_id,
                            SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == node.node_id,
                        )
                        .values(applied=node.available)
                    )

    async def _verify_citation(self, citation: MemoryCitation) -> None:
        page = await self._runtime.memory.for_scope(self._scope_id).list(include_inactive=True)
        for record in page.entries:
            if (
                record.entry.entry_id == citation.entry_id
                and record.entry.entry_version_id == citation.entry_version_id
            ):
                return
        raise SourceLifecycleError("citation must name a readable Memory entry version")

    async def _verify_artifact(self, reference: ArtifactRef) -> None:
        try:
            await self._runtime.records.for_scope(self._scope_id).get_artifact_revision(
                reference.family,
                reference.artifact_id,
                reference.revision,
            )
        except Exception as error:  # noqa: BLE001 - any read failure is a binding failure
            raise SourceLifecycleError("reference must name a readable artifact revision") from error


async def _load_nodes(connection: AsyncConnection, scope_id: str) -> dict[str, _Node]:
    result = await connection.execute(
        select(SOURCE_LIFECYCLE_NODES_TABLE).where(SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id)
    )
    nodes: dict[str, _Node] = {}
    for row in result.mappings():
        nodes[row["node_id"]] = _Node(
            node_id=row["node_id"],
            family=row["family"],
            payload=row["payload"],
            parents=tuple(json.loads(row["parents"])),
            policy=row["policy"],
            revision=row["revision"],
            sequence=row["sequence"],
            available=bool(row["available"]),
            applied=None if row["applied"] is None else bool(row["applied"]),
            position=int(row["position"]),
        )
    return nodes


async def _load_event(
    connection: AsyncConnection,
    scope_id: str,
    node_id: str,
    sequence: int,
) -> tuple[int, bool] | None:
    result = await connection.execute(
        select(SOURCE_LIFECYCLE_EVENTS_TABLE.c.revision, SOURCE_LIFECYCLE_EVENTS_TABLE.c.available).where(
            SOURCE_LIFECYCLE_EVENTS_TABLE.c.scope_id == scope_id,
            SOURCE_LIFECYCLE_EVENTS_TABLE.c.node_id == node_id,
            SOURCE_LIFECYCLE_EVENTS_TABLE.c.sequence == sequence,
        )
    )
    row = result.first()
    if row is None:
        return None
    return int(row[0]), bool(row[1])


def _derive(parents: Sequence[str], policy: str, nodes: dict[str, _Node]) -> bool:
    supports = [nodes[parent].available for parent in parents if parent in nodes]
    if not supports:
        return False
    return all(supports) if policy == POLICY_ALL else any(supports)


def _dump(reference: SourceRef | MemoryCitation | ArtifactRef) -> str:
    return json.dumps(reference.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _policy(policy: str) -> str:
    if policy not in _POLICIES:
        raise SourceLifecycleError(f"unknown support policy {policy!r}")
    return policy


def _identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SourceLifecycleError(f"{field} must be a non-blank string without surrounding whitespace")
    return value


def _positive_int(field: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SourceLifecycleError(f"{field} must be a positive integer")
    return value


def _parents(parents: Sequence[str], node: str) -> tuple[str, ...]:
    if isinstance(parents, str):
        raise SourceLifecycleError("parents must be a sequence of node identifiers")
    normalized: list[str] = []
    for parent in parents:
        identifier = _identifier(parent, "parent")
        if identifier == node:
            raise SourceLifecycleError("a node cannot depend on itself")
        if identifier in normalized:
            raise SourceLifecycleError("parents must be distinct")
        normalized.append(identifier)
    return tuple(normalized)