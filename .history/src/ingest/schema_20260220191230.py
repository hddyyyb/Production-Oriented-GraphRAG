# src/ingest/schema.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import uuid


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Metadata(BaseModel):
    source: str = Field(..., description="Source file path or URL")
    language: Optional[str] = Field(default="unknown")
    author: Optional[str] = None
    tags: Optional[List[str]] = []
    extra: Optional[Dict] = {}


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=generate_uuid)
    doc_id: str
    text: str
    section_title: Optional[str] = None
    position: int = Field(..., description="Order of chunk in document")
    metadata: Optional[Dict] = {}


class Document(BaseModel):
    doc_id: str = Field(default_factory=generate_uuid)
    title: Optional[str] = None
    content: str
    metadata: Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1
    chunks: Optional[List[Chunk]] = []

    def update_timestamp(self):
        self.updated_at = datetime.utcnow()