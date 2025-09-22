"""
Case Management System for Forensic Analysis

This module provides comprehensive case management functionality including:
- Case creation and management
- Evidence file handling
- Case-specific processing
- Database operations
"""

import os
import shutil
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc

from database.models import (
    Case, Evidence, EvidenceChunk, CaseNote,
    EvidenceType, CaseStatus, ProcessingStatus,
    DatabaseManager, db_manager
)

logger = logging.getLogger(__name__)

@dataclass
class CaseCreateRequest:
    """Request model for creating a new case"""
    case_number: str
    title: str
    investigator_name: str
    description: Optional[str] = None
    investigator_id: Optional[str] = None
    department: Optional[str] = None
    incident_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    priority: str = "medium"
    case_type: Optional[str] = None
    jurisdiction: Optional[str] = None
    tags: Optional[List[str]] = None
    suspects: Optional[List[Dict[str, Any]]] = None
    victims: Optional[List[Dict[str, Any]]] = None

@dataclass
class EvidenceUploadRequest:
    """Request model for uploading evidence to a case"""
    case_id: str
    original_filename: str
    evidence_type: EvidenceType
    title: Optional[str] = None
    description: Optional[str] = None
    source_device: Optional[str] = None
    extraction_method: Optional[str] = None
    evidence_date: Optional[datetime] = None

class CaseManager:
    """
    Comprehensive case management system for forensic investigations
    """
    
    def __init__(self, db_manager: DatabaseManager, data_root: str = "data/cases"):
        self.db = db_manager
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
    def create_case(self, request: CaseCreateRequest) -> Case:
        """
        Create a new forensic case
        
        Args:
            request: Case creation request with all necessary information
            
        Returns:
            Created Case object
            
        Raises:
            ValueError: If case_number already exists
        """
        session = self.db.get_session()
        try:
            # Check if case number already exists
            existing_case = session.query(Case).filter(Case.case_number == request.case_number).first()
            if existing_case:
                raise ValueError(f"Case number '{request.case_number}' already exists")
            
            # Create case directory
            case_dir = self.data_root / request.case_number
            case_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for different evidence types
            for evidence_type in EvidenceType:
                (case_dir / evidence_type.value).mkdir(exist_ok=True)
            
            # Generate collection name for vector embeddings
            collection_name = f"case_{request.case_number.replace('-', '_').lower()}"
            
            # Create case object
            case = Case(
                case_number=request.case_number,
                title=request.title,
                description=request.description,
                investigator_name=request.investigator_name,
                investigator_id=request.investigator_id,
                department=request.department,
                incident_date=request.incident_date,
                due_date=request.due_date,
                priority=request.priority,
                case_type=request.case_type,
                jurisdiction=request.jurisdiction,
                tags=request.tags or [],
                suspects=request.suspects or [],
                victims=request.victims or [],
                embedding_collection_name=collection_name
            )
            
            session.add(case)
            session.commit()
            session.refresh(case)
            
            logger.info(f"Created case: {case.case_number} - {case.title}")
            return case
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating case: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_case(self, case_id: str) -> Optional[Case]:
        """Get a case by ID"""
        session = self.db.get_session()
        try:
            return session.query(Case).filter(Case.id == case_id).first()
        finally:
            session.close()
    
    def get_case_by_number(self, case_number: str) -> Optional[Case]:
        """Get a case by case number"""
        session = self.db.get_session()
        try:
            return session.query(Case).filter(Case.case_number == case_number).first()
        finally:
            session.close()
    
    def list_cases(self, status: Optional[CaseStatus] = None, limit: int = 50, offset: int = 0) -> List[Case]:
        """
        List cases with optional filtering
        
        Args:
            status: Filter by case status
            limit: Maximum number of cases to return
            offset: Number of cases to skip
            
        Returns:
            List of cases
        """
        session = self.db.get_session()
        try:
            query = session.query(Case)
            
            if status:
                query = query.filter(Case.status == status)
            
            query = query.order_by(desc(Case.updated_at))
            query = query.offset(offset).limit(limit)
            
            return query.all()
        finally:
            session.close()
    
    def update_case(self, case_id: str, updates: Dict[str, Any]) -> Optional[Case]:
        """Update case information"""
        session = self.db.get_session()
        try:
            case = session.query(Case).filter(Case.id == case_id).first()
            if not case:
                return None
            
            for key, value in updates.items():
                if hasattr(case, key):
                    setattr(case, key, value)
            
            case.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(case)
            
            return case
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating case {case_id}: {str(e)}")
            raise
        finally:
            session.close()
    
    def add_evidence(self, request: EvidenceUploadRequest, file_data: bytes) -> Evidence:
        """
        Add evidence file to a case
        
        Args:
            request: Evidence upload request
            file_data: Raw file data
            
        Returns:
            Created Evidence object
        """
        session = self.db.get_session()
        try:
            # Get case
            case = session.query(Case).filter(Case.id == request.case_id).first()
            if not case:
                raise ValueError(f"Case {request.case_id} not found")
            
            # Calculate file hash
            file_hash = hashlib.sha256(file_data).hexdigest()
            
            # Determine file path
            case_dir = self.data_root / case.case_number
            evidence_dir = case_dir / request.evidence_type.value
            evidence_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = Path(request.original_filename).suffix
            safe_filename = f"{timestamp}_{request.original_filename}"
            file_path = evidence_dir / safe_filename
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            # Create evidence record
            evidence = Evidence()
            evidence.case_id = request.case_id
            evidence.original_filename = request.original_filename
            evidence.file_path = str(file_path)
            evidence.file_size = len(file_data)
            evidence.file_hash = file_hash
            evidence.evidence_type = request.evidence_type.value  # Store enum value as string
            evidence.title = request.title
            evidence.description = request.description
            evidence.source_device = request.source_device
            evidence.extraction_method = request.extraction_method
            evidence.evidence_date = request.evidence_date
            
            session.add(evidence)
            
            # Update case evidence count
            case.total_evidence_count += 1
            case.updated_at = datetime.utcnow()
            
            session.commit()
            session.refresh(evidence)
            
            logger.info(f"Added evidence {evidence.original_filename} to case {case.case_number}")
            return evidence
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding evidence: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_case_evidence(self, case_id: str, evidence_type: Optional[EvidenceType] = None) -> List[Evidence]:
        """Get evidence files for a case"""
        session = self.db.get_session()
        try:
            query = session.query(Evidence).filter(Evidence.case_id == case_id)
            
            if evidence_type:
                query = query.filter(Evidence.evidence_type == evidence_type.value)  # Use enum value
            
            return query.order_by(desc(Evidence.created_at)).all()
        finally:
            session.close()
    
    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        """Get evidence by ID"""
        session = self.db.get_session()
        try:
            return session.query(Evidence).filter(Evidence.id == evidence_id).first()
        finally:
            session.close()
    
    def update_evidence_processing(self, evidence_id: str, status: ProcessingStatus, 
                                 error: Optional[str] = None, progress: Optional[int] = None) -> Optional[Evidence]:
        """Update evidence processing status"""
        session = self.db.get_session()
        try:
            evidence = session.query(Evidence).filter(Evidence.id == evidence_id).first()
            if not evidence:
                return None
            
            evidence.processing_status = status
            if error:
                evidence.processing_error = error
            # Note: progress parameter accepted for compatibility but not stored in DB
            
            if status == ProcessingStatus.COMPLETED:
                evidence.processed_at = datetime.utcnow()
                
                # Update case processed count
                case = session.query(Case).filter(Case.id == evidence.case_id).first()
                if case:
                    case.processed_evidence_count += 1
                    case.updated_at = datetime.utcnow()
            
            session.commit()
            session.refresh(evidence)
            
            return evidence
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating evidence processing: {str(e)}")
            raise
        finally:
            session.close()
    
    def add_case_note(self, case_id: str, title: str, content: str, 
                      author: str, note_type: str = "general", 
                      is_important: bool = False, tags: Optional[List[str]] = None) -> CaseNote:
        """Add a note to a case"""
        session = self.db.get_session()
        try:
            note = CaseNote(
                case_id=case_id,
                title=title,
                content=content,
                author=author,
                note_type=note_type,
                is_important=is_important,
                tags=tags or []
            )
            
            session.add(note)
            session.commit()
            session.refresh(note)
            
            return note
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding case note: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_case_notes(self, case_id: str) -> List[CaseNote]:
        """Get notes for a case"""
        session = self.db.get_session()
        try:
            return session.query(CaseNote).filter(
                CaseNote.case_id == case_id
            ).order_by(desc(CaseNote.created_at)).all()
        finally:
            session.close()
    
    def search_cases(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Case]:
        """
        Search cases by title, description, or case number
        
        Args:
            query: Search query string
            filters: Additional filters (status, case_type, etc.)
            
        Returns:
            List of matching cases
        """
        session = self.db.get_session()
        try:
            search_query = session.query(Case)
            
            # Text search
            if query:
                search_pattern = f"%{query}%"
                search_query = search_query.filter(
                    or_(
                        Case.title.like(search_pattern),
                        Case.description.like(search_pattern),
                        Case.case_number.like(search_pattern)
                    )
                )
            
            # Apply filters
            if filters:
                for key, value in filters.items():
                    if hasattr(Case, key) and value is not None:
                        search_query = search_query.filter(getattr(Case, key) == value)
            
            return search_query.order_by(desc(Case.updated_at)).all()
        finally:
            session.close()
    
    def get_case_statistics(self, case_id: str) -> Dict[str, Any]:
        """Get statistics for a case"""
        session = self.db.get_session()
        try:
            case = session.query(Case).filter(Case.id == case_id).first()
            if not case:
                return {}
            
            # Evidence statistics
            evidence_stats = {}
            for evidence_type in EvidenceType:
                count = session.query(Evidence).filter(
                    and_(Evidence.case_id == case_id, Evidence.evidence_type == evidence_type.value)  # Use enum value
                ).count()
                evidence_stats[evidence_type.value] = count
            
            # Processing statistics
            total_evidence = session.query(Evidence).filter(Evidence.case_id == case_id).count()
            processed_evidence = session.query(Evidence).filter(
                and_(Evidence.case_id == case_id, Evidence.processing_status == ProcessingStatus.COMPLETED.value)  # Use enum value
            ).count()
            
            # Chunk statistics
            total_chunks = session.query(EvidenceChunk).join(Evidence).filter(
                Evidence.case_id == case_id
            ).count()
            
            return {
                "case_info": {
                    "id": case.id,
                    "case_number": case.case_number,
                    "title": case.title,
                    "status": case.status,  # Already a string
                    "created_at": case.created_at,
                    "updated_at": case.updated_at
                },
                "evidence_by_type": evidence_stats,
                "processing": {
                    "total_evidence": total_evidence,
                    "processed_evidence": processed_evidence,
                    "processing_rate": processed_evidence / total_evidence if total_evidence > 0 else 0,
                    "total_chunks": total_chunks
                },
                "embeddings": {
                    "collection_name": case.embedding_collection_name,
                    "has_embeddings": case.processed_evidence_count > 0
                }
            }
        finally:
            session.close()
    
    def delete_case(self, case_id: str, force: bool = False) -> bool:
        """
        Delete a case and all associated data
        
        Args:
            case_id: Case ID to delete
            force: If True, delete even if case has evidence
            
        Returns:
            True if deleted successfully, False otherwise
        """
        session = self.db.get_session()
        try:
            case = session.query(Case).filter(Case.id == case_id).first()
            if not case:
                return False
            
            # Check if case has evidence
            evidence_count = session.query(Evidence).filter(Evidence.case_id == case_id).count()
            if evidence_count > 0 and not force:
                raise ValueError(f"Case has {evidence_count} evidence files. Use force=True to delete.")
            
            # Delete case directory
            case_dir = self.data_root / case.case_number
            if case_dir.exists():
                shutil.rmtree(case_dir)
            
            # Delete from database (cascade will handle related records)
            session.delete(case)
            session.commit()
            
            logger.info(f"Deleted case: {case.case_number}")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting case {case_id}: {str(e)}")
            raise
        finally:
            session.close()

# Default case manager instance
case_manager = CaseManager(db_manager)