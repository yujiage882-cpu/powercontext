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
import re
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


_STOPWORDS = frozenset(
    """a an the of to in on at for and or but as by is are was were be been being with
    from that this these those it its their there which who whom what when where why how
    has have had do does did not no any all more most such than then so if into over under
    between during before after above below up down out off again further once here also may
    can could should would will shall about against among across per""".split()
)
_BOILERPLATE = frozenset(
    """based provide provided information improve improved improving decline declined increase
    increased change changed drove driven useful profile healthy reasonably able give get
    show shows shown report reported reporting continue continued including include includes
    approximately primarily generally significantly substantially compared compare consistent
    accordance respect related relating certain various several many much more less total net
    also however therefore overall""".split()
)
_SYNONYM_GROUPS = """
revenue|sales|net sales|top line|turnover
income|earnings|profit|net income|net earnings|profitability
margin|margins|profit margin|gross margin|operating margin
cash|cash flow|operating cash flow|liquidity|liquid
quick ratio|current ratio|acid test
debt|borrowings|indebtedness|leverage|notes payable|long-term debt
interest|interest expense|interest rate|cost of debt
capex|capital expenditure|capital expenditures|capital spending|capital intensive
expense|expenses|cost|costs|operating expense
dividend|dividends|payout|distribution
buyback|repurchase|repurchases|share repurchase
share|shares|stock|equity|stockholder|shareholder|stockholders|shareholders
eps|earnings per share|diluted|basic
employee|employees|headcount|personnel|workforce|staff
segment|segments|reportable segment|business unit
geograph|geographic|geographies|region|regions|countries|country|worldwide|international|global
legal|litigation|proceedings|lawsuit|lawsuits|claims|contingencies|settlement
risk|risks|uncertainties|exposure
tax|taxes|taxation|effective tax rate|income tax
goodwill|intangible|intangibles|impairment|write-down|writedown
lease|leases|rent|rental|operating lease
supply|suppliers|supplier|supply chain|procurement
acquisition|acquired|acquisitions|merger|mergers|business combination
restructuring|severance|impairment charge
cybersecurity|cyber|security|breach|data protection|privacy
asset|assets|property|properties|facilities|facility|plant|plants
inventory|inventories|stock
receivable|receivables|accounts receivable
payable|payables|accounts payable
backlog|orders|order book
pension|retirement|postretirement|benefit plan
depreciation|amortization|depreciation and amortization
subscriber|subscribers|membership|members|customers
store|stores|retail|outlet|outlets
advertising|marketing|promotion|brand
competition|competitors|competitive|competitor
regulation|regulatory|compliance|law|rules
fiscal|fy|year|annual|fiscal year
quarter|quarters|three months|nine months
growth|grow|grew|growing|increase
decline|decrease|fell|fall|drop|dropped|decreased
"""
_PHRASE_EXPANSIONS = {
    "quick ratio": ("total current assets", "total current liabilities", "current assets", "current liabilities"),
    "current ratio": ("total current assets", "total current liabilities"),
    "inventory turnover": ("cost of sales", "cost of revenue", "inventories", "inventory"),
    "receivable turnover": ("accounts receivable", "net sales"),
    "asset turnover": ("total assets", "net sales"),
    "return on equity": ("net income", "total shareholders equity", "total stockholders equity"),
    "return on assets": ("net income", "total assets"),
    "capital intensive": ("capital expenditures", "property plant and equipment", "capital spending"),
    "capital expenditure": ("property plant and equipment", "purchases of property"),
    "largest liability": ("total liabilities", "accounts payable", "long-term debt"),
    "largest asset": ("total assets", "goodwill", "property plant and equipment"),
    "working capital": ("current assets", "current liabilities"),
    "operating margin": ("operating income", "income from operations"),
    "gross margin": ("gross profit", "cost of sales", "cost of revenue"),
    "net margin": ("net income", "net sales"),
    "liquidity": ("cash and cash equivalents", "current assets", "working capital"),
    "cyclicality": ("cyclical", "economic conditions", "economic downturn"),
    "leverage": ("total debt", "long term debt", "debt to equity"),
    "customer concentration": ("largest customers", "major customers", "top customers"),
    "supplier": ("suppliers", "supply chain", "purchasing"),
    "employee": ("employees", "headcount", "personnel"),
    "geograph": ("geographic", "regions", "countries", "united states", "international"),
    "legal": ("legal proceedings", "litigation", "claims", "contingencies"),
    "goodwill": ("goodwill impairment", "intangible assets"),
    "pension": ("pension plan", "retirement plan", "postretirement"),
    "lease": ("operating lease", "lease obligations", "rent expense"),
    "share repurchase": ("repurchase of common stock", "treasury stock"),
    "dividend": ("dividends declared", "dividends paid"),
    "effective tax rate": ("income tax expense", "pre-tax income"),
    "backlog": ("remaining performance obligations", "order backlog"),
    "same store": ("comparable store sales", "comparable sales"),
    "segment": ("reportable segments", "business segments"),
}

_TOKEN = re.compile(r"[a-z]+|\d+")
_MORPH_SUFFIXES = (("ies", "y"), ("ied", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", ""), ("ly", ""), ("ment", ""))


def _normalize(token: str) -> str:
    if token.isdigit():
        return token
    for suffix, replacement in _MORPH_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)] + replacement
    return token


def _tokens(text: str, *, drop_boilerplate: bool = False) -> tuple:
    out = []
    for token in _TOKEN.findall(text.lower()):
        if token in _STOPWORDS:
            continue
        if drop_boilerplate and token in _BOILERPLATE:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        out.append(_normalize(token))
    return tuple(out)


def _build_synonyms() -> dict:
    table = {}
    for line in _SYNONYM_GROUPS.strip().splitlines():
        items = [item.strip() for item in line.split("|") if item.strip()]
        words = []
        for item in items:
            words.extend(_TOKEN.findall(item.lower()))
        for word in words:
            if word in _STOPWORDS:
                continue
            normalized = _normalize(word)
            others = {_normalize(other) for other in words if other != word}
            others.discard(normalized)
            table.setdefault(normalized, set()).update(others)
    return {key: frozenset(value) for key, value in table.items()}


_SYNONYMS = _build_synonyms()


class _PageIndex(object):
    __slots__ = ("entries", "counters", "lengths", "average_length", "document_frequency", "size")

    def __init__(self, entries, texts) -> None:
        from collections import Counter

        self.entries = entries
        counters = []
        lengths = []
        document_frequency = Counter()
        for text in texts:
            tokens = Counter(_tokens(text))
            counters.append(tokens)
            lengths.append(sum(tokens.values()) or 1)
            for token in tokens:
                document_frequency[token] += 1
        self.counters = tuple(counters)
        self.lengths = tuple(lengths)
        self.size = len(counters)
        self.average_length = (sum(lengths) / self.size) if self.size else 1.0
        self.document_frequency = document_frequency


_INDEX_CACHE = {}


def _page_text(text: str) -> str:
    head, sep, tail = text.partition("\n")
    if sep and 0 < len(head) <= 200 and " " not in head:
        return tail
    return text


async def _page_index(memory):
    page = await memory.list()
    entries = tuple(page.entries)
    if not entries:
        return None
    key = str(getattr(memory, "scope_id", None) or id(memory))
    cached = _INDEX_CACHE.get(key)
    if cached is not None and cached[0] == len(entries):
        return cached[1]
    index = _PageIndex(entries, tuple(_page_text(record.entry.text) for record in entries))
    _INDEX_CACHE[key] = (len(entries), index)
    return index


def _query_terms(question: str, index) -> tuple:
    size = max(index.size, 1)
    primary = list(dict.fromkeys(_tokens(question, drop_boilerplate=True)))
    filtered = [term for term in primary if index.document_frequency.get(term, 0) / size <= 0.7]
    primary = filtered or primary
    extra = []
    lowered = " ".join(_tokens(question))
    for phrase, expansions in _PHRASE_EXPANSIONS.items():
        if phrase in lowered:
            for expansion in expansions:
                for term in _tokens(expansion):
                    if term not in primary and term not in extra:
                        extra.append(term)
    for term in primary:
        for synonym in sorted(_SYNONYMS.get(term, ()))[:10]:
            if synonym not in primary and synonym not in extra:
                extra.append(synonym)
    return primary, extra


def _rank_pages(index, question: str, limit: int) -> list:
    import math

    primary, extra = _query_terms(question, index)
    if not primary:
        return []
    size = max(index.size, 1)
    idf = {}
    for term in set(primary) | set(extra):
        frequency = index.document_frequency.get(term, 0)
        idf[term] = math.log(1 + (size - frequency + 0.5) / (frequency + 0.5))
    scores = []
    for position, counter in enumerate(index.counters):
        length_ratio = index.lengths[position] / index.average_length
        score = 0.0
        for term in primary:
            frequency = counter.get(term, 0)
            if frequency:
                score += (idf[term] ** 1.5) * (frequency * 2.2) / (frequency + 1.2 * (0.25 + 0.75 * length_ratio))
        for term in extra:
            frequency = counter.get(term, 0)
            if frequency:
                score += 0.7 * (idf[term] ** 1.5) * (frequency * 2.2) / (frequency + 1.2 * (0.25 + 0.75 * length_ratio))
        scores.append(score)
    order = sorted(range(len(scores)), key=lambda position: -scores[position])
    ranked = []
    for position in order:
        if scores[position] <= 0 or len(ranked) >= max(limit, 1):
            break
        ranked.append(index.entries[position])
    return ranked


async def search_pdf_pages(
    memory: Memory,
    question: str,
    limit: int = 10,
) -> tuple:
    """Return citations for the pages most likely to support ``question``.

    Candidates come from every indexed page: the scoped workspace API exposes all entries, so
    ranking runs over the full corpus instead of a single fixed full-text query.  The score is
    an IDF-weighted BM25 with finance-domain synonym and phrase expansion, which closes the
    vocabulary gap between a question's wording and a 10-K page's wording.
    """
    text = question.strip() if isinstance(question, str) else ""
    if not text:
        raise ValueError("question must be a non-empty string")
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be an integer")
    if limit < 1:
        raise ValueError("limit must be >= 1")

    index = await _page_index(memory)
    if index is None:
        return ()
    ranked = _rank_pages(index, text, limit)
    citations = []
    seen = set()
    for record in ranked:
        entry = record.entry
        key = (entry.entry_id, entry.entry_version_id)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            MemoryCitation(
                memory_ref=record.memory_ref,
                entry_id=entry.entry_id,
                entry_version_id=entry.entry_version_id,
            )
        )
    return tuple(citations)
