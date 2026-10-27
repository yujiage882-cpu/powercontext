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

"""Capture every physical page of a financial report PDF as page-level evidence.

A report keeps its own identity and revision: the page Source id is derived from
the document id, the SHA-256 of the exact uploaded bytes and the physical page
number, so a revised PDF produces new identities while old citations keep
pointing at the revision they were captured from. Every page, including blank
ones, is returned in physical order; a document that cannot be parsed in full
fails atomically instead of yielding partial pages.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator
from pypdf import PdfReader

from powercontext.builtin.runtime import (
    GetMemoryEntryRequest,
    SearchMemoryRequest,
)
from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import (
    AdapterSourceDefinition,
    TEXT_EVIDENCE_PROJECTION_KEY,
    TextEvidence,
)
from powercontext.sources.models import Source, SourceMaterialization

PDF_PAGE_SOURCE_NAME = "pdf-page"
PDF_PAGE_SOURCE_VERSION = "1"


class PDFDocumentInput(BaseModel):
    """One complete report PDF as its exact uploaded bytes."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def reject_blank_document_id(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("document id must be a non-empty string")
        return value


class PDFPageInput(BaseModel):
    """One physical page of a parsed report."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """Durable captured page that can be cited as textual evidence."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str

    @property
    def source_id(self) -> str:
        return self.name


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build the stable, bounded identity of one physical page."""

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document id must be a non-empty string")
    if not isinstance(document_sha256, str) or len(document_sha256) != 64:
        raise ValueError("document sha256 must be a lowercase hex digest")
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        raise ValueError("page number must be a positive integer")

    prefix = f"pdf:{document_id}:{document_sha256}:{page_number}"
    if len(prefix) <= MAX_SOURCE_ID_LENGTH:
        return prefix
    digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
    source_id = f"pdf:{digest}:{document_sha256}:{page_number}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"pdf page source id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return source_id


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Extract the text of every physical page, in order, or fail atomically."""

    if not isinstance(value.content, bytes) or not value.content:
        raise ValueError("pdf document must not be empty")
    try:
        reader = PdfReader(io.BytesIO(value.content))
    except Exception as error:
        raise ValueError(f"pdf document cannot be parsed: {error}") from error

    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as error:
            raise ValueError(f"pdf document cannot be decrypted: {error}") from error
        if not unlocked:
            raise ValueError("pdf document requires a user password")

    try:
        pages = tuple(reader.pages)
    except Exception as error:
        raise ValueError(f"pdf document pages cannot be read: {error}") from error
    if not pages:
        raise ValueError("pdf document must contain at least one page")

    document_sha256 = hashlib.sha256(value.content).hexdigest()
    parsed: list[PDFPageInput] = []
    for index, page in enumerate(pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as error:
            raise ValueError(f"pdf page {index} cannot be parsed: {error}") from error
        parsed.append(
            PDFPageInput(
                document_id=value.document_id,
                document_sha256=document_sha256,
                page_number=index,
                text=text,
            )
        )
    return tuple(parsed)


class PDFPageSourceAdapter:
    """Resolve and read one captured physical page."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        return PDFPageSource(
            name=pdf_page_source_id(value.document_id, value.document_sha256, value.page_number),
            definition_version=PDF_PAGE_SOURCE_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose one page as ``DOCUMENT_ID\\nPAGE_TEXT`` evidence with provenance."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_ADAPTER = PDFPageSourceAdapter()
PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDF_PAGE_SOURCE_ADAPTER,
    version=PDF_PAGE_SOURCE_VERSION,
    projections=(PDFPageTextEvidenceProjection(),),
)


def _citation_of(hit: Any) -> MemoryCitation:
    return MemoryCitation(
        memory_ref=hit.memory_ref,
        entry_id=hit.entry_id,
        entry_version_id=hit.entry_version_id,
    )


async def search_pdf_pages(
    memory: Any,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Return at most ``limit`` distinct page citations for one natural question."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        return ()

    citations: list[MemoryCitation] = []
    seen: set[tuple[str, str]] = set()
    requested = max(limit * 3, 30)
    while len(citations) < limit:
        page = await memory.search(SearchMemoryRequest(query=question, limit=requested, mode="fts"))
        added = False
        for hit in page.hits:
            identity = (hit.entry_id, hit.entry_version_id)
            if identity in seen:
                continue
            seen.add(identity)
            citations.append(_citation_of(hit))
            added = True
            if len(citations) >= limit:
                break
        if not added or len(page.hits) < requested:
            break
        requested *= 2

    verified: list[MemoryCitation] = []
    verified_pages: set[tuple[str, int] | str] = set()
    for citation in citations:
        entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
        if not entry.sources:
            continue
        pages = {(reference.source_type, reference.source_id) for reference in entry.sources}
        if pages & verified_pages:
            continue
        verified_pages |= pages
        verified.append(citation)
        if len(verified) >= limit:
            break
    return tuple(verified)


__all__ = [
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "PDFPageSourceAdapter",
    "PDFPageTextEvidenceProjection",
    "PDF_PAGE_SOURCE_ADAPTER",
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDF_PAGE_SOURCE_VERSION",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
