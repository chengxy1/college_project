from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Document:
    doc_id: str
    content: str
    source: str = "通用文档"
    metadata: Dict[str, Any] = field(default_factory=dict)
    base_confidence: float = 0.0
    claims: List[str] = field(default_factory=list)

@dataclass
class ExplanatoryMetadata:
    conflict_tags: List[str] = field(default_factory=list)
    verification_chain: List[str] = field(default_factory=list)
    cf_summary: str = ""

@dataclass
class RankedResult:
    document: Document
    cf_score: float
    metadata: ExplanatoryMetadata
