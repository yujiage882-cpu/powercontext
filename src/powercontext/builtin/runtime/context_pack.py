"""Context pack wire format and prefix-reuse deltas for incremental context transfer.

A context pack is the full body of one ``prepare`` result split into UTF-8 chunks; a delta names
how many leading chunks of a previously sent pack the receiver may keep and carries the remaining
chunks.  The wire format is canonical JSON with Base64 chunk payloads, so a receiver can rebuild the
target body from the base pack and the delta alone, without any sender-side state.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES = 512

_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})


class ContextPackError(ValueError):
    """Raised when a context pack or delta violates the wire contract."""


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode_chunks(chunks: tuple) -> list:
    return [base64.b64encode(chunk).decode("ascii") for chunk in chunks]


def _decode_chunks(raw, label: str) -> tuple:
    if not isinstance(raw, list):
        raise ContextPackError(f"context-pack:{label}-not-list")
    decoded = []
    for value in raw:
        if not isinstance(value, str):
            raise ContextPackError(f"context-pack:{label}-not-text")
        try:
            decoded.append(base64.b64decode(value.encode("ascii"), validate=True))
        except (binascii.Error, UnicodeEncodeError, ValueError) as error:
            raise ContextPackError(f"context-pack:{label}-not-base64") from error
    return tuple(decoded)


def _coerce_chunks(items) -> tuple:
    if isinstance(items, (bytes, bytearray, str)):
        raise ContextPackError("context-pack:items-not-sequence")
    try:
        candidates = tuple(items)
    except TypeError as error:
        raise ContextPackError("context-pack:items-not-sequence") from error
    chunks = []
    for item in candidates:
        if isinstance(item, (bytes, bytearray)):
            chunks.append(bytes(item))
        else:
            raise ContextPackError("context-pack:item-not-bytes")
    return tuple(chunks)


def _digest_of(chunks: tuple) -> str:
    canonical = _canonical({"schema": PACK_SCHEMA, "items": _encode_chunks(chunks)})
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _load_payload(data, schema: str, fields: frozenset) -> dict:
    if isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
    elif isinstance(data, str):
        raw = data.encode("utf-8")
    else:
        raise ContextPackError("context-pack:payload-not-bytes")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ContextPackError("context-pack:payload-not-json") from error
    if not isinstance(payload, dict):
        raise ContextPackError("context-pack:payload-not-object")
    if set(payload) != fields:
        raise ContextPackError("context-pack:payload-fields")
    if payload.get("schema") != schema:
        raise ContextPackError("context-pack:payload-schema")
    return payload


def _validate_digest(value, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ContextPackError(f"context-pack:{label}-invalid")
    return value


def _chunk_text(content: str, chunk_bytes: int) -> tuple:
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        raise ContextPackError("context-pack:chunk-limit")
    if not content:
        return ()
    chunks = []
    current = bytearray()
    for character in content:
        encoded = character.encode("utf-8")
        if len(encoded) > chunk_bytes:
            raise ContextPackError("context-pack:chunk-limit")
        if current and len(current) + len(encoded) > chunk_bytes:
            chunks.append(bytes(current))
            current = bytearray()
        current += encoded
    if current:
        chunks.append(bytes(current))
    return tuple(chunks)


@dataclass(frozen=True, slots=True)
class ContextPack:
    """One complete context body split into UTF-8 chunks."""

    items: tuple
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items) -> "ContextPack":
        chunks = _coerce_chunks(items)
        digest = _digest_of(chunks)
        byte_count = 0
        for _ in range(4):
            payload = {
                "schema": PACK_SCHEMA,
                "items": _encode_chunks(chunks),
                "digest": digest,
                "byte_count": byte_count,
            }
            size = len(_canonical(payload))
            if size == byte_count:
                break
            byte_count = size
        return cls(chunks, digest, byte_count)

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> "ContextPack":
        if not isinstance(content, str):
            raise ContextPackError("context-pack:content-not-text")
        return cls.build(_chunk_text(content, chunk_bytes))

    def to_bytes(self) -> bytes:
        return _canonical(
            {
                "schema": PACK_SCHEMA,
                "items": _encode_chunks(self.items),
                "digest": self.digest,
                "byte_count": self.byte_count,
            }
        )

    @classmethod
    def from_bytes(cls, data) -> "ContextPack":
        payload = _load_payload(data, PACK_SCHEMA, _PACK_FIELDS)
        items = _decode_chunks(payload["items"], "items")
        digest = _validate_digest(payload["digest"], "digest")
        if digest != _digest_of(items):
            raise ContextPackError("context-pack:digest-mismatch")
        byte_count = payload["byte_count"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int):
            raise ContextPackError("context-pack:byte-count-invalid")
        if isinstance(data, (bytes, bytearray)):
            size = len(data)
        else:
            size = len(data.encode("utf-8"))
        if byte_count != size:
            raise ContextPackError("context-pack:byte-count-mismatch")
        return cls(items, digest, byte_count)


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """Prefix reuse over ``base_digest`` plus the chunks a receiver still needs."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> "ContextPackDelta":
        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("context-pack:delta-operand")
        prefix = 0
        for left, right in zip(base.items, target.items):
            if left != right:
                break
            prefix += 1
        return cls(base.digest, target.digest, prefix, tuple(target.items[prefix:]))

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        return _canonical(
            {
                "schema": DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": _encode_chunks(self.appended),
            }
        )

    @classmethod
    def from_bytes(cls, data) -> "ContextPackDelta":
        payload = _load_payload(data, DELTA_SCHEMA, _DELTA_FIELDS)
        prefix_items = payload["prefix_items"]
        if isinstance(prefix_items, bool) or not isinstance(prefix_items, int) or prefix_items < 0:
            raise ContextPackError("context-pack:prefix-invalid")
        appended = _decode_chunks(payload["appended"], "appended")
        return cls(
            _validate_digest(payload["base_digest"], "base-digest"),
            _validate_digest(payload["target_digest"], "target-digest"),
            prefix_items,
            appended,
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild the target pack from ``base`` and ``delta`` after validating both digests."""

    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("context-pack:delta-operand")
    if base.digest != delta.base_digest:
        raise ContextPackError("context-pack:delta-base-mismatch")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("context-pack:delta-prefix-range")
    rebuilt = ContextPack.build(tuple(base.items[: delta.prefix_items]) + tuple(delta.appended))
    if rebuilt.digest != delta.target_digest:
        raise ContextPackError("context-pack:delta-target-mismatch")
    return rebuilt