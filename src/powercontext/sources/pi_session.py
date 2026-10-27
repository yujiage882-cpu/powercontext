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

"""Capture the raw visible events of one selected Pi session branch.

A Pi archive is a tree: records carry ``parentId`` links and the selected leaf
plus its ancestors define one conversation. Only the chain matters here, so the
graph is validated first and then walked root-to-leaf; control records keep the
chain connected without producing text evidence, and compaction summaries never
erase the raw events that precede them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import (
    AdapterSourceDefinition,
    Connector,
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    TEXT_EVIDENCE_PROJECTION_KEY,
    TextEvidence,
)
from powercontext.sources.models import Source, SourceMaterialization

PI_EVENT_SOURCE_NAME = "pi-event"
PI_EVENT_SOURCE_VERSION = "1"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
PI_SESSION_CONNECTOR_VERSION = "1"

# Record types that carry a visible body; everything else only links the graph.
_SUMMARY_TYPES = {"compaction": "compaction", "branch_summary": "branch_summary"}
_MESSAGE_ROLES = {"user", "assistant", "toolResult", "bashExecution"}


class PiSessionInput(BaseModel):
    """One Pi v3 archive expressed as exact JSONL bytes."""

    model_config = ConfigDict(frozen=True)

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event of the selected branch, together with its raw record."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, Any] = Field(default_factory=dict)


class PiEventSource(Source):
    """Durable captured event that can be cited as textual evidence."""

    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, Any] = Field(default_factory=dict)

    @property
    def source_id(self) -> str:
        return self.name


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build the delimiter-safe, bounded identity of one Pi event."""

    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session id must be a non-empty string")
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("event id must be a non-empty string")
    source_id = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"pi event source id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return source_id


def _decode_archive(content: bytes) -> list[dict[str, Any]]:
    if not isinstance(content, bytes) or not content:
        raise ValueError("pi archive must not be empty")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("pi archive must be valid utf-8") from error
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"pi archive line is not valid json: {error}") from error
        if not isinstance(record, dict):
            raise ValueError("pi archive records must be json objects")
        records.append(record)
    if not records:
        raise ValueError("pi archive must contain a session header")
    return records


def _visible_parts(message: dict[str, Any]) -> list[str]:
    """Render the visible blocks of one message body, in record order."""

    raw = message.get("content")
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if not isinstance(raw, list):
        return []
    parts: list[str] = []
    for block in raw:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
        elif kind == "toolCall":
            name = block.get("name")
            call_id = block.get("id")
            arguments = block.get("arguments")
            rendered = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            parts.append(f"[tool_call {name} {call_id}] {rendered}")
    return parts


def _render_event_body(record: dict[str, Any]) -> tuple[str, str] | None:
    """Return the (role, body) pair of one record, or None when it has no body."""

    record_type = record.get("type")
    if record_type in _SUMMARY_TYPES:
        summary = record.get("summary")
        if isinstance(summary, str) and summary.strip():
            return _SUMMARY_TYPES[record_type], summary
        return None
    if record_type == "custom_message":
        parts = _visible_parts({"content": record.get("content")})
        return ("custom", "\n".join(parts)) if parts else None
    if record_type != "message":
        return None

    message = record.get("message")
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if role not in _MESSAGE_ROLES:
        return None
    if role == "toolResult":
        parts = _visible_parts(message)
        tool_name = message.get("toolName", "unknown")
        call_id = message.get("toolCallId", "unknown")
        body = f"[tool_result {tool_name} {call_id}] " + "\n".join(parts)
        return (role, body) if parts else None
    if role == "bashExecution":
        command = message.get("command")
        output = message.get("output")
        if not isinstance(command, str):
            return None
        body = "$ " + command + "\n" + (output if isinstance(output, str) else "")
        return (role, body)
    parts = _visible_parts(message)
    return (role, "\n".join(parts)) if parts else None


def parse_pi_session(session: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Return the visible events of the selected branch, ordered root to leaf."""

    records = _decode_archive(session.content)
    header = records[0]
    if header.get("type") != "session" or header.get("version") != 3:
        raise ValueError("pi archive must start with a version 3 session header")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("pi archive session id must be a non-empty string")

    nodes: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records[1:]:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("pi archive records must carry a non-empty id")
        if event_id in nodes:
            raise ValueError(f"duplicate pi archive id {event_id!r}")
        parent_id = record.get("parentId")
        if parent_id is not None and parent_id not in nodes:
            raise ValueError(f"pi archive record {event_id!r} has an unknown parent")
        nodes[event_id] = record
        order.append(event_id)

    if not order:
        raise ValueError("pi archive must contain at least one record")
    leaf_id = session.leaf_id if session.leaf_id is not None else order[-1]
    if not isinstance(leaf_id, str) or not leaf_id.strip():
        raise ValueError("leaf id must be a non-empty string")
    if leaf_id not in nodes:
        raise ValueError(f"pi archive leaf {leaf_id!r} is missing")

    chain: list[dict[str, Any]] = []
    current: str | None = leaf_id
    while current is not None:
        record = nodes[current]
        chain.append(record)
        parent_id = record.get("parentId")
        if parent_id is not None and not isinstance(parent_id, str):
            raise ValueError("pi archive parent ids must be strings")
        current = parent_id
    chain.reverse()

    events: list[PiEventInput] = []
    for record in chain:
        rendered = _render_event_body(record)
        if rendered is None:
            continue
        role, body = rendered
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=record["id"],
                parent_id=record.get("parentId"),
                event_type=str(record.get("type")),
                role=role,
                content=f"{role}: {body}",
                record=record,
            )
        )
    return tuple(events)


class PiEventSourceAdapter:
    """Resolve and read one captured Pi event."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        return PiEventSource(
            name=pi_event_source_id(value.session_id, value.event_id),
            definition_version=PI_EVENT_SOURCE_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose one event as verbatim text evidence with full provenance."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.name,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_ADAPTER = PiEventSourceAdapter()
PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PI_EVENT_SOURCE_ADAPTER,
    version=PI_EVENT_SOURCE_VERSION,
    projections=(PiEventTextEvidenceProjection(),),
)


def _record_digest(record: dict[str, Any]) -> str:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PiSessionConnector(Connector):
    """Import the not-yet-captured events of one selected branch, resumably."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_SESSION_CONNECTOR_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        events = parse_pi_session(PiSessionInput(content=self._path.read_bytes(), leaf_id=self._leaf_id))
        checkpoint = session.checkpoint if isinstance(session.checkpoint, dict) else {}
        imported: dict[str, str] = {}
        raw_imported = checkpoint.get("events")
        if isinstance(raw_imported, dict):
            imported = {
                str(key): str(value) for key, value in raw_imported.items() if isinstance(value, str)
            }

        captured = dict(imported)
        for event in events:
            digest = _record_digest(event.record)
            known = imported.get(event.event_id)
            if known is not None:
                if known != digest:
                    raise ValueError(f"pi archive event {event.event_id!r} changed after capture")
                continue
            await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)
            captured[event.event_id] = digest

        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"events": captured},
        )


__all__ = [
    "PI_EVENT_SOURCE_ADAPTER",
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_EVENT_SOURCE_VERSION",
    "PI_SESSION_CONNECTOR_NAME",
    "PI_SESSION_CONNECTOR_VERSION",
    "PiEventInput",
    "PiEventSource",
    "PiEventSourceAdapter",
    "PiEventTextEvidenceProjection",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
