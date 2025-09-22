"""
RAG-specific data structures for forensic analysis

This module provides data structures optimized for the RAG system,
ensuring compatibility with forensic data processing.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class TextChunk:
    """RAG-compatible text chunk with forensic metadata"""
    chunk_id: str
    text: str
    data_type: str
    timestamp: Optional[datetime] = None
    participants: Optional[List[str]] = None
    entities: Optional[Dict[str, Any]] = None
    source_info: Optional[Dict[str, Any]] = None
    
    @property
    def id(self) -> str:
        """Alias for chunk_id for backward compatibility"""
        return self.chunk_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'data_type': self.data_type,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'participants': self.participants or [],
            'entities': self.entities or {},
            'source_info': self.source_info or {}
        }

@dataclass 
class ProcessedDocument:
    """Container for processed forensic documents"""
    file_path: str
    chunks: List[TextChunk]
    metadata: Dict[str, Any]
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'file_path': self.file_path,
            'chunks': [chunk.to_dict() for chunk in self.chunks],
            'metadata': self.metadata,
            'processing_time': self.processing_time
        }