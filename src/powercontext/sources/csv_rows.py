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

"""Turn one CSV document into row-level Sources with stable identities.

A document keeps its header verbatim, so column names such as ``" id "`` stay
distinguishable from ``"id"``. Every data row becomes its own Source whose id
is derived only from ``(document_id, primary_key_value)``: reordering rows or
editing non-key columns never changes row identity.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Annotated
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, field_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

CSV_ROW_SOURCE_NAME = "csv-row"
CSV_ROW_SOURCE_VERSION = "1"

NonEmptyText = Annotated[str, Field(min_length=1)]


class CSVDocumentInput(BaseModel):
    """One CSV document expressed as an exact header plus data rows."""

    model_config = ConfigDict(frozen=True)

    document_id: NonEmptyText
    primary_key: NonEmptyText
    content: bytes

    @field_validator("document_id")
    @classmethod
    def reject_untrimmed_document_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("document_id must not contain leading or trailing whitespace")
        return value


class CSVRowInput(BaseModel):
    """One row of a CSV document, bound to its document and primary key."""

    model_config = ConfigDict(frozen=True)

    document_id: NonEmptyText
    primary_key: NonEmptyText
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @property
    def primary_key_value(self) -> str:
        index = self.columns.index(self.primary_key)
        return self.values[index]


class CSVRowSource(Source):
    """Durable captured row that can be cited as textual evidence."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @property
    def source_id(self) -> str:
        return self.name


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the delimiter-safe, bounded identity of one CSV row Source."""

    if not isinstance(document_id, str) or not document_id.strip() or document_id != document_id.strip():
        raise ValueError("document_id must be a non-empty trimmed string")
    if not isinstance(primary_key_value, str) or not primary_key_value.strip():
        raise ValueError("primary_key_value must be a non-empty string")
    source_id = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"csv row source id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return source_id


def parse_csv_document(document: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse every data row, ignoring empty records after the header."""

    try:
        text = document.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("csv document must be valid utf-8") from error

    try:
        records = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise ValueError(f"csv document is malformed: {error}") from error

    if not records:
        raise ValueError("csv document must contain a header row")
    header = tuple(records[0])
    if not header:
        raise ValueError("csv document must contain a header row")
    if any(not name.strip() for name in header):
        raise ValueError("csv header names must not be blank")
    if len(set(header)) != len(header):
        raise ValueError("csv header names must be unique")
    if document.primary_key not in header:
        raise ValueError(f"primary key column {document.primary_key!r} is missing from the csv header")

    rows: list[CSVRowInput] = []
    seen: set[str] = set()
    for record in records[1:]:
        if not record:
            continue
        if len(record) != len(header):
            raise ValueError("csv row width must match the header width")
        row = CSVRowInput(
            document_id=document.document_id,
            primary_key=document.primary_key,
            columns=header,
            values=tuple(record),
        )
        key_value = row.primary_key_value
        if not key_value.strip():
            raise ValueError("csv primary key value must not be blank")
        if key_value in seen:
            raise ValueError(f"duplicate csv primary key value {key_value!r}")
        seen.add(key_value)
        rows.append(row)
    return tuple(rows)


class CSVRowSourceAdapter:
    """Resolve and read one row Source of a parsed CSV document."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        return CSVRowSource(
            name=csv_row_source_id(value.document_id, value.primary_key_value),
            definition_version=CSV_ROW_SOURCE_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose one row as canonical JSON text evidence with full provenance."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.name,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_ADAPTER = CSVRowSourceAdapter()
CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSV_ROW_SOURCE_ADAPTER,
    version=CSV_ROW_SOURCE_VERSION,
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSV_ROW_SOURCE_ADAPTER",
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "CSVRowSourceAdapter",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]
