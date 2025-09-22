"""
Database Models for Case-Wise Forensic Analysis System

This module defines the database schema for managing forensic cases,
evidence files, and their associated metadata in Project Sentinel.
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey, Enum, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional, List, Dict, Any
import uuid

Base = declarative_base()

class EvidenceType(PyEnum):
    """Types of evidence that can be stored in a case"""
    CHAT = "chat"
    CALL_LOG = "call_log"
    CONTACT = "contact"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    XML_REPORT = "xml_report"
    JSON_DATA = "json_data"
    CSV_DATA = "csv_data"
    TEXT_REPORT = "text_report"
    OTHER = "other"

class CaseStatus(PyEnum):
    """Status of a forensic case"""
    ACTIVE = "active"
    CLOSED = "closed"
    ARCHIVED = "archived"
    UNDER_REVIEW = "under_review"

class ProcessingStatus(PyEnum):
    """Processing status of evidence"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Case(Base):
    """
    Forensic Case Model
    Represents a forensic investigation case with metadata and associated evidence
    """
    __tablename__ = "cases"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    case_number = Column(String, unique=True, nullable=False, index=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    investigator_name = Column(String, nullable=False)
    investigator_id = Column(String)
    department = Column(String)
    status = Column(Enum(CaseStatus), default=CaseStatus.ACTIVE)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    incident_date = Column(DateTime)
    due_date = Column(DateTime)
    
    # Case metadata
    priority = Column(String, default="medium")  # low, medium, high, critical
    tags = Column(JSON)  # List of tags for categorization
    suspects = Column(JSON)  # List of suspect information
    victims = Column(JSON)  # List of victim information
    case_type = Column(String)  # fraud, cybercrime, drugs, etc.
    jurisdiction = Column(String)
    
    # Processing metadata
    total_evidence_count = Column(Integer, default=0)
    processed_evidence_count = Column(Integer, default=0)
    embedding_collection_name = Column(String)  # ChromaDB collection name for this case
    
    # Relationships
    evidence_files = relationship("Evidence", back_populates="case", cascade="all, delete-orphan")
    case_notes = relationship("CaseNote", back_populates="case", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Case(id='{self.id}', case_number='{self.case_number}', title='{self.title}')>"

class Evidence(Base):
    """
    Evidence Model
    Represents individual pieces of evidence within a forensic case
    """
    __tablename__ = "evidence"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String, ForeignKey("cases.id"), nullable=False)
    
    # File information
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer)  # Size in bytes
    file_hash = Column(String)  # SHA256 hash for integrity
    evidence_type = Column(Enum(EvidenceType), nullable=False)
    
    # Evidence metadata
    title = Column(String)
    description = Column(Text)
    source_device = Column(String)  # Device this evidence came from
    extraction_method = Column(String)  # How the evidence was extracted
    chain_of_custody = Column(JSON)  # Chain of custody information
    
    # Processing information
    processing_status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)
    processing_error = Column(Text)
    processed_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    evidence_date = Column(DateTime)  # When the evidence was originally created
    
    # Embeddings metadata
    has_embeddings = Column(Boolean, default=False)
    embedding_count = Column(Integer, default=0)
    embedding_model = Column(String)  # Model used for embeddings
    
    # Statistics
    entity_count = Column(JSON)  # Count of different entity types found
    chunk_count = Column(Integer, default=0)
    
    # Relationships
    case = relationship("Case", back_populates="evidence_files")
    evidence_chunks = relationship("EvidenceChunk", back_populates="evidence", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Evidence(id='{self.id}', filename='{self.original_filename}', type='{self.evidence_type}')>"

class EvidenceChunk(Base):
    """
    Evidence Chunk Model
    Represents processed chunks of evidence with embeddings
    """
    __tablename__ = "evidence_chunks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    evidence_id = Column(String, ForeignKey("evidence.id"), nullable=False)
    
    # Chunk information
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    chunk_type = Column(String)  # conversation, call_log, contact, etc.
    
    # Metadata
    start_position = Column(Integer)
    end_position = Column(Integer)
    token_count = Column(Integer)
    language = Column(String, default="en")
    
    # Entities and relationships
    entities = Column(JSON)  # Extracted entities from this chunk
    participants = Column(JSON)  # Conversation participants
    timestamps = Column(JSON)  # Relevant timestamps
    
    # Embedding information
    embedding_id = Column(String)  # ID in vector database
    embedding_model = Column(String)
    embedding_created_at = Column(DateTime)
    
    # Relationships
    evidence = relationship("Evidence", back_populates="evidence_chunks")
    
    def __repr__(self):
        return f"<EvidenceChunk(id='{self.id}', evidence_id='{self.evidence_id}', index={self.chunk_index})>"

class CaseNote(Base):
    """
    Case Notes Model
    Represents investigator notes and observations about a case
    """
    __tablename__ = "case_notes"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String, ForeignKey("cases.id"), nullable=False)
    
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    note_type = Column(String, default="general")  # general, observation, lead, conclusion
    
    # Metadata
    author = Column(String, nullable=False)
    is_important = Column(Boolean, default=False)
    tags = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    case = relationship("Case", back_populates="case_notes")
    
    def __repr__(self):
        return f"<CaseNote(id='{self.id}', title='{self.title}', author='{self.author}')>"

class CaseAccess(Base):
    """
    Case Access Model
    Tracks who has access to which cases for security and audit purposes
    """
    __tablename__ = "case_access"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String, ForeignKey("cases.id"), nullable=False)
    
    user_id = Column(String, nullable=False)
    username = Column(String, nullable=False)
    access_level = Column(String, nullable=False)  # read, write, admin
    
    granted_by = Column(String)
    granted_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime)
    
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<CaseAccess(case_id='{self.case_id}', user='{self.username}', level='{self.access_level}')>"

# Database helper functions and session management
class DatabaseManager:
    """
    Database manager for forensic case system
    """
    
    def __init__(self, database_url: str = "sqlite:///data/forensic_cases.db"):
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all tables in the database"""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """Get a database session"""
        return self.SessionLocal()
        
    def drop_tables(self):
        """Drop all tables (use with caution!)"""
        Base.metadata.drop_all(bind=self.engine)
        
    def init_database(self):
        """Initialize the database with tables"""
        self.create_tables()
        print(f"Database initialized at: {self.database_url}")

# Create default database manager instance
db_manager = DatabaseManager()