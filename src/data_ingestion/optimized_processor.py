"""
Optimized Asynchronous Case Data Processor

This module provides high-performance data processing with:
- Asynchronous processing for better performance
- Batch embedding generation to reduce API calls
- Smart model selection for different complexity levels
- Real-time progress tracking
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import os

try:
    import torch
except ImportError:
    torch = None

from src.database.models import Evidence, EvidenceType, ProcessingStatus
from src.case_management.case_manager import CaseManager
from src.ai_cores.rag.case_vector_store import CaseVectorStore
from src.ai_cores.rag.embeddings import ForensicEmbeddingGenerator
from .parsers import UFDRDataIngestion
from .preprocessor import DataPreprocessor
from .chunking import ForensicTextChunker

logger = logging.getLogger(__name__)

class OptimizedCaseProcessor:
    """
    High-performance case data processor with async operations
    """
    
    def __init__(self, 
                 case_manager: CaseManager,
                 vector_store: CaseVectorStore,
                 max_workers: int = 4):
        self.case_manager = case_manager
        self.vector_store = vector_store
        self.max_workers = max_workers
        
        # Initialize components
        self.embedding_config = {
            "primary_model": "all-MiniLM-L6-v2",
            "batch_size": 32,  # Process embeddings in batches
            "max_chunk_size": 512,  # Smaller chunks for faster processing
            "use_gpu": torch.cuda.is_available() if torch is not None else False
        }
        
        self.embedding_generator = ForensicEmbeddingGenerator(self.embedding_config)
        self.ufdr_ingestion = UFDRDataIngestion()
        self.preprocessor = DataPreprocessor()
        self.chunker = ForensicTextChunker()
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_evidence_fast(self, case_id: str, evidence_id: str, 
                                  file_path: str, evidence_type: Optional[EvidenceType] = None) -> Dict[str, Any]:
        """
        Fast asynchronous evidence processing with real-time updates
        """
        start_time = time.time()
        
        try:
            # Get case and evidence info
            case = self.case_manager.get_case(case_id)
            evidence = self.case_manager.get_evidence(evidence_id)
            
            if not case or not evidence:
                raise ValueError(f"Case {case_id} or evidence {evidence_id} not found")
            
            # Update processing status
            await self._update_processing_status(evidence_id, ProcessingStatus.PROCESSING, 0)
            
            logger.info(f"🚀 Fast processing started for {evidence.original_filename}")
            
            # Step 1: Parse file (async)
            parsed_document = await self._parse_file_async(file_path, evidence_type or evidence.evidence_type)
            await self._update_processing_status(evidence_id, ProcessingStatus.PROCESSING, 20)
            
            # Step 2: Preprocess content (async)
            preprocessed_content = await self._preprocess_async(parsed_document)
            await self._update_processing_status(evidence_id, ProcessingStatus.PROCESSING, 40)
            
            # Step 3: Create optimized chunks
            chunks = await self._create_chunks_async(preprocessed_content, {
                "case_id": case_id,
                "evidence_id": evidence_id,
                "source_file": evidence.original_filename,
                "evidence_type": (evidence_type or evidence.evidence_type).value
            })
            await self._update_processing_status(evidence_id, ProcessingStatus.PROCESSING, 60)
            
            # Step 4: Batch generate embeddings
            if chunks:
                embeddings = await self._generate_embeddings_batch(case_id, evidence_id, chunks)
                await self._update_processing_status(evidence_id, ProcessingStatus.PROCESSING, 80)
                
                # Step 5: Store embeddings in vector store
                embedding_ids = await self._store_embeddings_async(case_id, case.embedding_collection_name, embeddings)
                await self._update_processing_status(evidence_id, ProcessingStatus.PROCESSING, 90)
            else:
                embeddings = []
                embedding_ids = []
            
            # Update final status
            await self._update_processing_status(evidence_id, ProcessingStatus.COMPLETED, 100)
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Fast processing completed for {evidence.original_filename} in {processing_time:.2f}s")
            
            return {
                "status": "completed",
                "case_id": case_id,
                "evidence_id": evidence_id,
                "processing_time_seconds": processing_time,
                "chunks_created": len(chunks),
                "embeddings_generated": len(embeddings),
                "embedding_ids": embedding_ids
            }
            
        except Exception as e:
            await self._update_processing_status(evidence_id, ProcessingStatus.FAILED, 0, str(e))
            logger.error(f"❌ Fast processing failed for evidence {evidence_id}: {str(e)}")
            
            return {
                "status": "error",
                "case_id": case_id,
                "evidence_id": evidence_id,
                "error": str(e),
                "processing_time_seconds": time.time() - start_time
            }
    
    async def _parse_file_async(self, file_path: str, evidence_type: EvidenceType):
        """Async file parsing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.ufdr_ingestion.process_file,
            file_path
        )
    
    async def _preprocess_async(self, document):
        """Async preprocessing"""
        loop = asyncio.get_event_loop()
        
        # Convert UFDRDocument to dictionary format expected by preprocessor
        if hasattr(document, 'content'):
            raw_content = document.content
        else:
            raw_content = document
            
        return await loop.run_in_executor(
            self.executor,
            self.preprocessor.preprocess_document,
            raw_content
        )
    
    async def _create_chunks_async(self, content, metadata):
        """Async chunking with optimized parameters"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._create_optimized_chunks,
            content,
            metadata
        )
    
    def _create_optimized_chunks(self, content, metadata):
        """Create smaller, more focused chunks for faster processing"""
        return self.chunker.chunk_document(
            preprocessed_document=content,
            source_file=metadata.get("source_file", "unknown")
        )
    
    async def _generate_embeddings_batch(self, case_id: str, evidence_id: str, chunks: List[Any]) -> List[Any]:
        """Generate embeddings in batches for better performance"""
        embeddings = []
        batch_size = self.embedding_config["batch_size"]
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            # Process batch
            batch_embeddings = await self._process_embedding_batch(case_id, evidence_id, batch_chunks, i)
            embeddings.extend(batch_embeddings)
            
            # Small delay to prevent overwhelming the embedding model
            await asyncio.sleep(0.1)
        
        return embeddings
    
    async def _process_embedding_batch(self, case_id: str, evidence_id: str, 
                                     batch_chunks: List[Any], start_index: int) -> List[Any]:
        """Process a batch of chunks for embedding generation"""
        loop = asyncio.get_event_loop()
        
        def generate_batch():
            embeddings = []
            texts = []
            metadata_list = []
            
            # Collect all texts and metadata first
            for i, chunk in enumerate(batch_chunks):
                try:
                    # Handle both TextChunk objects and dictionaries
                    if hasattr(chunk, 'content') and hasattr(chunk, 'metadata'):
                        # TextChunk object
                        content = chunk.content
                        chunk_metadata = chunk.metadata.copy()
                    elif isinstance(chunk, dict):
                        # Dictionary format
                        content = chunk.get('content', str(chunk))
                        chunk_metadata = chunk.get('metadata', {})
                    else:
                        # Fallback: convert to string
                        content = str(chunk)
                        chunk_metadata = {}
                    
                    chunk_metadata.update({
                        "case_id": case_id,
                        "evidence_id": evidence_id,
                        "chunk_index": start_index + i
                    })
                    
                    texts.append(content)
                    metadata_list.append(chunk_metadata)
                    
                except Exception as e:
                    logger.warning(f"Failed to prepare chunk {start_index + i}: {e}")
                    continue
            
            # Generate embeddings for all texts at once
            if texts:
                embeddings = self.embedding_generator.generate_embeddings(
                    texts=texts,
                    metadata_list=metadata_list
                )
            
            return embeddings
        
        return await loop.run_in_executor(self.executor, generate_batch)
    
    async def _store_embeddings_async(self, case_id: str, collection_name: str, embeddings: List[Any]) -> List[str]:
        """Store embeddings asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.vector_store.add_case_embeddings,
            case_id,
            collection_name,
            embeddings
        )
    
    async def _update_processing_status(self, evidence_id: str, status: ProcessingStatus, 
                                      progress: int, error_message: str = None):
        """Update processing status with progress tracking"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self.case_manager.update_evidence_processing,
            evidence_id,
            status,
            error_message,
            progress
        )
    
    async def process_multiple_evidence_concurrent(self, case_id: str, 
                                                 evidence_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple evidence files concurrently"""
        logger.info(f"🚀 Starting concurrent processing of {len(evidence_items)} evidence files")
        
        # Create tasks for all evidence items
        tasks = [
            self.process_evidence_fast(
                case_id=case_id,
                evidence_id=item["evidence_id"],
                file_path=item["file_path"],
                evidence_type=item.get("evidence_type")
            )
            for item in evidence_items
        ]
        
        # Process all concurrently with progress tracking
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Evidence {evidence_items[i]['evidence_id']} failed: {result}")
                processed_results.append({
                    "status": "error",
                    "evidence_id": evidence_items[i]["evidence_id"],
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)