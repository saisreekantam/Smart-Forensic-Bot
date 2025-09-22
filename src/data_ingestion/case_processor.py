"""
Case-Aware Data Ingestion Pipeline

This module extends the existing data ingestion system to work with
case-specific contexts, integrating with the case management system
and case-specific vector storage.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Local imports
from .parsers import UFDRDataIngestion
from .preprocessor import DataPreprocessor
from .chunking import ForensicTextChunker
from database.models import Evidence, EvidenceType, ProcessingStatus
from case_management.case_manager import CaseManager
from ai_cores.rag.case_vector_store import CaseVectorStore
from ai_cores.rag.embeddings import ForensicEmbeddingGenerator

logger = logging.getLogger(__name__)

class CaseDataProcessor:
    """
    Case-aware data processing pipeline that integrates with the case management system
    """
    
    def __init__(self, 
                 case_manager: CaseManager,
                 vector_store: CaseVectorStore,
                 embedding_generator: Optional[ForensicEmbeddingGenerator] = None):
        self.case_manager = case_manager
        self.vector_store = vector_store
        
        # Default embedding configuration
        default_config = {
            "primary_model": "all-MiniLM-L6-v2",
            "forensic_model": "all-distilroberta-v1",
            "openai_model": "text-embedding-3-small",
            "openai_api_key": os.getenv("OPENAI_API_KEY")
        }
        
        self.embedding_generator = embedding_generator or ForensicEmbeddingGenerator(default_config)
        
        # Initialize data processing components
        self.ufdr_ingestion = UFDRDataIngestion()
        self.preprocessor = DataPreprocessor()
        self.chunker = ForensicTextChunker()
        
        logger.info("Initialized CaseDataProcessor")
    
    def process_evidence_file(self, 
                            case_id: str, 
                            evidence_id: str, 
                            file_path: str,
                            evidence_type: EvidenceType = None) -> Dict[str, Any]:
        """
        Process an evidence file within a case context
        
        Args:
            case_id: Case identifier
            evidence_id: Evidence identifier from database
            file_path: Path to the evidence file
            evidence_type: Type of evidence (auto-detected if None)
            
        Returns:
            Processing results with statistics and metadata
        """
        try:
            # Get case and evidence information
            case = self.case_manager.get_case(case_id)
            if not case:
                raise ValueError(f"Case {case_id} not found")
            
            evidence = self.case_manager.get_evidence(evidence_id)
            if not evidence:
                raise ValueError(f"Evidence {evidence_id} not found")
            
            # Update processing status
            self.case_manager.update_evidence_processing(
                evidence_id, ProcessingStatus.PROCESSING
            )
            
            logger.info(f"Starting processing of evidence {evidence.original_filename} for case {case.case_number}")
            
            # Step 1: Parse the file based on type
            if evidence_type is None:
                evidence_type = evidence.evidence_type
            
            parsed_document = self._parse_file(file_path, evidence_type)
            
            # Step 2: Preprocess the content
            preprocessed_content = self.preprocessor.process_document(parsed_document)
            
            # Step 3: Create intelligent chunks
            chunks = self.chunker.chunk_document(
                text_content=preprocessed_content.cleaned_content,
                metadata={
                    "case_id": case_id,
                    "evidence_id": evidence_id,
                    "source_file": evidence.original_filename,
                    "evidence_type": evidence_type.value,
                    "processing_timestamp": datetime.now().isoformat()
                }
            )
            
            # Step 4: Generate embeddings for chunks
            embeddings = []
            if chunks:
                embeddings = self._generate_case_embeddings(
                    case_id, evidence_id, chunks, preprocessed_content
                )
            
            # Step 5: Store embeddings in case-specific collection
            embedding_ids = []
            if embeddings:
                embedding_ids = self.vector_store.add_case_embeddings(
                    case_id=case_id,
                    collection_name=case.embedding_collection_name,
                    embeddings=embeddings
                )
            
            # Step 6: Update evidence record with processing results
            evidence_update = {
                "processing_status": ProcessingStatus.COMPLETED,
                "has_embeddings": len(embeddings) > 0,
                "embedding_count": len(embeddings),
                "chunk_count": len(chunks),
                "entity_count": preprocessed_content.extracted_entities,
                "embedding_model": self.embedding_generator.model_name
            }
            
            self.case_manager.update_evidence_processing(evidence_id, ProcessingStatus.COMPLETED)
            
            # Prepare results
            results = {
                "status": "success",
                "case_id": case_id,
                "evidence_id": evidence_id,
                "case_number": case.case_number,
                "evidence_filename": evidence.original_filename,
                "parsed_document": {
                    "document_type": parsed_document.get("document_type", "unknown"),
                    "metadata": parsed_document.get("metadata", {}),
                    "content_summary": {
                        "total_sections": len(parsed_document.get("content", {})),
                        "data_types": list(parsed_document.get("content", {}).keys())
                    }
                },
                "preprocessed_content": {
                    "entity_counts": preprocessed_content.extracted_entities,
                    "conversation_count": len(preprocessed_content.conversations),
                    "cleaned_content_length": len(preprocessed_content.cleaned_content)
                },
                "chunks": {
                    "total_chunks": len(chunks),
                    "chunk_types": [chunk.chunk_type for chunk in chunks],
                    "average_chunk_size": sum(len(chunk.content) for chunk in chunks) / len(chunks) if chunks else 0
                },
                "embeddings": {
                    "total_embeddings": len(embeddings),
                    "embedding_ids": embedding_ids,
                    "collection_name": case.embedding_collection_name,
                    "model_used": self.embedding_generator.model_name
                },
                "processing_time": datetime.now().isoformat(),
                "statistics": {
                    "entities_extracted": sum(preprocessed_content.extracted_entities.values()),
                    "conversations_found": len(preprocessed_content.conversations),
                    "chunks_created": len(chunks),
                    "embeddings_generated": len(embeddings)
                }
            }
            
            logger.info(f"Successfully processed evidence {evidence.original_filename} for case {case.case_number}")
            return results
            
        except Exception as e:
            # Update evidence with error status
            self.case_manager.update_evidence_processing(
                evidence_id, ProcessingStatus.FAILED, str(e)
            )
            
            logger.error(f"Error processing evidence {evidence_id}: {str(e)}")
            return {
                "status": "error",
                "case_id": case_id,
                "evidence_id": evidence_id,
                "error": str(e),
                "processing_time": datetime.now().isoformat()
            }
    
    def process_case_evidence(self, case_id: str, evidence_types: Optional[List[EvidenceType]] = None) -> Dict[str, Any]:
        """
        Process all evidence files for a case
        
        Args:
            case_id: Case identifier
            evidence_types: Optional filter for specific evidence types
            
        Returns:
            Batch processing results
        """
        try:
            case = self.case_manager.get_case(case_id)
            if not case:
                raise ValueError(f"Case {case_id} not found")
            
            # Get evidence files for the case
            evidence_files = self.case_manager.get_case_evidence(case_id)
            
            if evidence_types:
                evidence_files = [e for e in evidence_files if e.evidence_type in evidence_types]
            
            # Filter for unprocessed evidence
            unprocessed_evidence = [
                e for e in evidence_files 
                if e.processing_status in [ProcessingStatus.PENDING, ProcessingStatus.FAILED]
            ]
            
            logger.info(f"Processing {len(unprocessed_evidence)} evidence files for case {case.case_number}")
            
            results = {
                "case_id": case_id,
                "case_number": case.case_number,
                "total_evidence": len(evidence_files),
                "processed_evidence": [],
                "failed_evidence": [],
                "statistics": {
                    "successful_processing": 0,
                    "failed_processing": 0,
                    "total_chunks": 0,
                    "total_embeddings": 0
                }
            }
            
            # Process each evidence file
            for evidence in unprocessed_evidence:
                try:
                    result = self.process_evidence_file(
                        case_id=case_id,
                        evidence_id=evidence.id,
                        file_path=evidence.file_path,
                        evidence_type=evidence.evidence_type
                    )
                    
                    if result["status"] == "success":
                        results["processed_evidence"].append(result)
                        results["statistics"]["successful_processing"] += 1
                        results["statistics"]["total_chunks"] += result["chunks"]["total_chunks"]
                        results["statistics"]["total_embeddings"] += result["embeddings"]["total_embeddings"]
                    else:
                        results["failed_evidence"].append(result)
                        results["statistics"]["failed_processing"] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing evidence {evidence.id}: {str(e)}")
                    results["failed_evidence"].append({
                        "evidence_id": evidence.id,
                        "filename": evidence.original_filename,
                        "error": str(e)
                    })
                    results["statistics"]["failed_processing"] += 1
            
            logger.info(f"Batch processing completed for case {case.case_number}: "
                       f"{results['statistics']['successful_processing']} successful, "
                       f"{results['statistics']['failed_processing']} failed")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing for case {case_id}: {str(e)}")
            return {
                "status": "error",
                "case_id": case_id,
                "error": str(e)
            }
    
    def _parse_file(self, file_path: str, evidence_type: EvidenceType) -> Dict[str, Any]:
        """Parse file based on evidence type"""
        file_path_obj = Path(file_path)
        
        if evidence_type == EvidenceType.XML_REPORT:
            return self.ufdr_ingestion.parse_xml_file(str(file_path_obj))
        elif evidence_type == EvidenceType.CSV_DATA:
            return self.ufdr_ingestion.parse_csv_file(str(file_path_obj))
        elif evidence_type == EvidenceType.JSON_DATA:
            return self.ufdr_ingestion.parse_json_file(str(file_path_obj))
        elif evidence_type == EvidenceType.TEXT_REPORT:
            return self.ufdr_ingestion.parse_txt_file(str(file_path_obj))
        else:
            # For other file types, create basic structure
            return {
                "document_type": evidence_type.value,
                "metadata": {
                    "filename": file_path_obj.name,
                    "file_size": file_path_obj.stat().st_size if file_path_obj.exists() else 0
                },
                "content": {
                    "raw_content": "File type not yet supported for automatic processing"
                }
            }
    
    def _generate_case_embeddings(self, case_id: str, evidence_id: str, 
                                chunks: List[Any], preprocessed_content: Any) -> List[Any]:
        """Generate embeddings for chunks with case context"""
        embeddings = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Add case-specific metadata to chunk
                chunk.metadata.update({
                    "case_id": case_id,
                    "evidence_id": evidence_id,
                    "chunk_index": i
                })
                
                # Generate embedding
                embedding = self.embedding_generator.generate_embedding(
                    text=chunk.content,
                    metadata=chunk.metadata
                )
                
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Error generating embedding for chunk {i}: {str(e)}")
                continue
        
        return embeddings
    
    def get_case_processing_status(self, case_id: str) -> Dict[str, Any]:
        """Get processing status for all evidence in a case"""
        try:
            case = self.case_manager.get_case(case_id)
            if not case:
                return {"error": f"Case {case_id} not found"}
            
            evidence_files = self.case_manager.get_case_evidence(case_id)
            
            status_summary = {
                "case_id": case_id,
                "case_number": case.case_number,
                "total_evidence": len(evidence_files),
                "by_status": {},
                "by_type": {},
                "processing_progress": 0,
                "last_updated": case.updated_at.isoformat() if case.updated_at else None
            }
            
            # Count by status
            for status in ProcessingStatus:
                count = len([e for e in evidence_files if e.processing_status == status])
                status_summary["by_status"][status.value] = count
            
            # Count by type
            for evidence_type in EvidenceType:
                count = len([e for e in evidence_files if e.evidence_type == evidence_type])
                if count > 0:
                    status_summary["by_type"][evidence_type.value] = count
            
            # Calculate progress
            completed = status_summary["by_status"].get("completed", 0)
            if status_summary["total_evidence"] > 0:
                status_summary["processing_progress"] = (completed / status_summary["total_evidence"]) * 100
            
            return status_summary
            
        except Exception as e:
            logger.error(f"Error getting processing status for case {case_id}: {str(e)}")
            return {"error": str(e)}

# Default instance for easy importing
case_data_processor = None  # Will be initialized when needed with proper dependencies