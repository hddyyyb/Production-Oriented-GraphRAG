# src/ingest/schema.py
'''
Docstring for ingest.schema
    1. doc_id与chunk_id分离
    以后Graph层会用chunk_id建节点。
    2. version字段
    为未来“增量索引”做准备。
    3. metadata嵌套模型
    方便企业扩展。
    4. created_at/updated_at
    支持未来日志追踪。
'''

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