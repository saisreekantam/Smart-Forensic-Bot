"""
Main RAG Pipeline Orchestrator for Forensic Analysis

This module provides the complete RAG system that orchestrates all components:
embeddings, vector storage, retrieval, and response generation for forensic data analysis.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local imports
from .embeddings import ForensicEmbeddingGenerator, ForensicEmbedding, EmbeddingMetadata
from .vector_store import create_vector_store, BaseVectorStore, SearchFilter
from .retrieval import AdvancedRetriever, RetrievalQuery, RetrievalContext, RankedResult
from .generation import AdvancedResponseGenerator, GenerationContext, GeneratedResponse

# Import data processing components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_ingestion.chunking import ForensicTextChunker
from .data_structures import TextChunk, ProcessedDocument

logger = logging.getLogger(__name__)

@dataclass
class RAGQuery:
    """Complete query object for the RAG system"""
    text: str
    query_id: Optional[str] = None
    query_type: str = "general"  # 'general', 'entity', 'temporal', 'relationship', 'factual'
    filters: Optional[SearchFilter] = None
    context: Optional[RetrievalContext] = None
    max_results: int = 10
    include_reasoning: bool = True
    response_style: str = "professional"
    provider_preference: Optional[str] = None

@dataclass
class RAGResponse:
    """Complete response from the RAG system"""
    query_id: str
    response: GeneratedResponse
    retrieved_results: List[RankedResult]
    processing_time: float
    metadata: Dict[str, Any]
    timestamp: datetime

class ForensicRAGSystem:
    """
    Complete RAG system for forensic data analysis
    
    Features:
    - Multi-format data ingestion and processing
    - Advanced embedding generation with forensic optimization
    - Hybrid vector storage with multiple backends
    - Context-aware retrieval with sophisticated ranking
    - Multi-LLM response generation with forensic prompts
    - Comprehensive evaluation and monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initialized = False
        
        # Core components
        self.embedding_generator = None
        self.vector_store = None
        self.retriever = None
        self.response_generator = None
        self.chunker = None
        
        # State management
        self.conversation_histories: Dict[str, RetrievalContext] = {}
        self.query_cache: Dict[str, RAGResponse] = {}
        self.performance_metrics: Dict[str, Any] = {}
        
        # Initialize components
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all RAG system components"""
        try:
            logger.info("Initializing Forensic RAG System...")
            
            # 1. Initialize embedding generator
            embedding_config = self.config.get('embedding_config', {})
            self.embedding_generator = ForensicEmbeddingGenerator(embedding_config)
            logger.info("✓ Embedding generator initialized")
            
            # 2. Initialize vector store
            vector_config = self.config.get('vector_config', {})
            # Add embedding dimension from the primary model
            model_info = self.embedding_generator.get_model_info()
            primary_model = list(model_info.keys())[0]
            vector_config['embedding_dimension'] = model_info[primary_model]['dimension']
            
            self.vector_store = create_vector_store(vector_config)
            logger.info("✓ Vector store initialized")
            
            # 3. Initialize retriever
            retrieval_config = self.config.get('retrieval_config', {})
            self.retriever = AdvancedRetriever(
                self.vector_store,
                self.embedding_generator,
                retrieval_config
            )
            logger.info("✓ Advanced retriever initialized")
            
            # 4. Initialize response generator
            generation_config = self.config.get('generation_config', {})
            self.response_generator = AdvancedResponseGenerator(generation_config)
            logger.info("✓ Response generator initialized")
            
            # 5. Initialize chunker for data processing
            chunking_config = self.config.get('chunking_config', {})
            self.chunker = ForensicTextChunker(chunking_config)
            logger.info("✓ Text chunker initialized")
            
            self.initialized = True
            logger.info("🎉 Forensic RAG System fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def ingest_processed_data(self, processed_data_path: str) -> Dict[str, Any]:
        """
        Ingest processed forensic data into the RAG system
        
        Args:
            processed_data_path: Path to processed JSON data
        
        Returns:
            Ingestion statistics
        """
        if not self.initialized:
            raise RuntimeError("RAG system not initialized")
        
        start_time = datetime.now()
        logger.info(f"Starting data ingestion from: {processed_data_path}")
        
        try:
            # Load processed data
            with open(processed_data_path, 'r') as f:
                processed_data = json.load(f)
            
            ingestion_stats = {
                "total_files": 0,
                "total_chunks": 0,
                "total_embeddings": 0,
                "processing_errors": [],
                "data_type_distribution": {},
                "start_time": start_time.isoformat()
            }
            
            # Process each file
            if "processed_files" in processed_data:
                for file_data in processed_data["processed_files"]:
                    try:
                        file_stats = self._process_file_data(file_data)
                        ingestion_stats["total_files"] += 1
                        ingestion_stats["total_chunks"] += file_stats["chunks"]
                        ingestion_stats["total_embeddings"] += file_stats["embeddings"]
                        
                        # Update data type distribution
                        for data_type, count in file_stats["data_types"].items():
                            if data_type not in ingestion_stats["data_type_distribution"]:
                                ingestion_stats["data_type_distribution"][data_type] = 0
                            ingestion_stats["data_type_distribution"][data_type] += count
                        
                    except Exception as e:
                        error_msg = f"Error processing file {file_data.get('file_path', 'unknown')}: {str(e)}"
                        ingestion_stats["processing_errors"].append(error_msg)
                        logger.error(error_msg)
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            ingestion_stats["processing_time"] = processing_time
            ingestion_stats["end_time"] = end_time.isoformat()
            
            logger.info(f"✓ Data ingestion completed in {processing_time:.2f}s")
            logger.info(f"  - Files processed: {ingestion_stats['total_files']}")
            logger.info(f"  - Chunks created: {ingestion_stats['total_chunks']}")
            logger.info(f"  - Embeddings generated: {ingestion_stats['total_embeddings']}")
            logger.info(f"  - Errors: {len(ingestion_stats['processing_errors'])}")
            
            return ingestion_stats
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise
    
    def _process_file_data(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual file data into chunks and embeddings"""
        
        file_stats = {
            "chunks": 0,
            "embeddings": 0,
            "data_types": {}
        }
        
        if file_data.get("status") != "success":
            logger.warning(f"Skipping failed file: {file_data.get('file_path')}")
            return file_stats
        
        ufdr_document = file_data.get("ufdr_document", {})
        file_path = ufdr_document.get("file_path", "unknown")
        case_id = ufdr_document.get("case_id")
        
        # Process different content types
        content = ufdr_document.get("content", {})
        
        # Process communications
        if "communications" in content:
            stats = self._process_communications(content["communications"], file_path, case_id)
            self._merge_stats(file_stats, stats)
        
        # Process contacts
        if "contacts" in content:
            stats = self._process_contacts(content["contacts"], file_path, case_id)
            self._merge_stats(file_stats, stats)
        
        # Process call logs
        if "call_logs" in content:
            stats = self._process_call_logs(content["call_logs"], file_path, case_id)
            self._merge_stats(file_stats, stats)
        
        # Process chunked content
        if "chunked_content" in ufdr_document:
            stats = self._process_chunked_content(ufdr_document["chunked_content"], file_path, case_id)
            self._merge_stats(file_stats, stats)
        
        return file_stats
    
    def _process_communications(self, communications: List[Dict], file_path: str, case_id: str) -> Dict[str, Any]:
        """Process communication data"""
        stats = {"chunks": 0, "embeddings": 0, "data_types": {}}
        
        for comm in communications:
            # Create conversation chunks
            conversation_text = self._format_conversation(comm)
            
            if conversation_text:
                # Create chunk
                chunk = TextChunk(
                    chunk_id=f"comm_{comm.get('id', 'unknown')}",
                    text=conversation_text,
                    data_type="conversation",
                    timestamp=self._parse_timestamp(comm.get('timestamp')),
                    participants=comm.get('participants', []),
                    entities=comm.get('entities', {}),
                    source_info={"file_path": file_path, "case_id": case_id}
                )
                
                # Generate and store embeddings
                embedding_metadata = EmbeddingMetadata(
                    chunk_id=chunk.chunk_id,
                    data_type=chunk.data_type,
                    timestamp=chunk.timestamp,
                    participants=chunk.participants,
                    entities=chunk.entities,
                    source_file=file_path,
                    case_id=case_id
                )
                
                forensic_embeddings = self.embedding_generator.generate_embeddings(
                    [chunk.text],
                    [embedding_metadata],
                    embedding_type="semantic"
                )
                
                # Store in vector database
                self.vector_store.add_embeddings(forensic_embeddings)
                
                stats["chunks"] += 1
                stats["embeddings"] += len(forensic_embeddings)
                stats["data_types"]["conversation"] = stats["data_types"].get("conversation", 0) + 1
        
        return stats
    
    def _process_contacts(self, contacts: List[Dict], file_path: str, case_id: str) -> Dict[str, Any]:
        """Process contact data"""
        stats = {"chunks": 0, "embeddings": 0, "data_types": {}}
        
        for contact in contacts:
            # Format contact information
            contact_text = self._format_contact(contact)
            
            if contact_text:
                chunk = TextChunk(
                    chunk_id=f"contact_{contact.get('id', 'unknown')}",
                    text=contact_text,
                    data_type="contact",
                    entities=self._extract_contact_entities(contact),
                    source_info={"file_path": file_path, "case_id": case_id}
                )
                
                embedding_metadata = EmbeddingMetadata(
                    chunk_id=chunk.chunk_id,
                    data_type=chunk.data_type,
                    entities=chunk.entities,
                    source_file=file_path,
                    case_id=case_id
                )
                
                forensic_embeddings = self.embedding_generator.generate_embeddings(
                    [chunk.text],
                    [embedding_metadata],
                    embedding_type="entity"
                )
                
                self.vector_store.add_embeddings(forensic_embeddings)
                
                stats["chunks"] += 1
                stats["embeddings"] += len(forensic_embeddings)
                stats["data_types"]["contact"] = stats["data_types"].get("contact", 0) + 1
        
        return stats
    
    def _process_call_logs(self, call_logs: List[Dict], file_path: str, case_id: str) -> Dict[str, Any]:
        """Process call log data"""
        stats = {"chunks": 0, "embeddings": 0, "data_types": {}}
        
        for call in call_logs:
            # Format call information
            call_text = self._format_call_log(call)
            
            if call_text:
                chunk = TextChunk(
                    chunk_id=f"call_{call.get('id', 'unknown')}",
                    text=call_text,
                    data_type="call_log",
                    timestamp=self._parse_timestamp(call.get('timestamp')),
                    entities=self._extract_call_entities(call),
                    source_info={"file_path": file_path, "case_id": case_id}
                )
                
                embedding_metadata = EmbeddingMetadata(
                    chunk_id=chunk.chunk_id,
                    data_type=chunk.data_type,
                    timestamp=chunk.timestamp,
                    entities=chunk.entities,
                    source_file=file_path,
                    case_id=case_id
                )
                
                forensic_embeddings = self.embedding_generator.generate_embeddings(
                    [chunk.text],
                    [embedding_metadata],
                    embedding_type="temporal"
                )
                
                self.vector_store.add_embeddings(forensic_embeddings)
                
                stats["chunks"] += 1
                stats["embeddings"] += len(forensic_embeddings)
                stats["data_types"]["call_log"] = stats["data_types"].get("call_log", 0) + 1
        
        return stats
    
    def _process_chunked_content(self, chunked_content: List[Dict], file_path: str, case_id: str) -> Dict[str, Any]:
        """Process pre-chunked content"""
        stats = {"chunks": 0, "embeddings": 0, "data_types": {}}
        
        for chunk_data in chunked_content:
            chunk_text = chunk_data.get("text", "")
            if not chunk_text:
                continue
            
            # Create embedding metadata
            embedding_metadata = EmbeddingMetadata(
                chunk_id=chunk_data.get("chunk_id", f"chunk_{len(stats)}"),
                data_type=chunk_data.get("data_type", "document"),
                timestamp=self._parse_timestamp(chunk_data.get("timestamp")),
                participants=chunk_data.get("participants"),
                entities=chunk_data.get("entities"),
                source_file=file_path,
                case_id=case_id
            )
            
            # Generate embeddings
            forensic_embeddings = self.embedding_generator.generate_embeddings(
                [chunk_text],
                [embedding_metadata],
                embedding_type="semantic"
            )
            
            # Store in vector database
            self.vector_store.add_embeddings(forensic_embeddings)
            
            stats["chunks"] += 1
            stats["embeddings"] += len(forensic_embeddings)
            data_type = chunk_data.get("data_type", "document")
            stats["data_types"][data_type] = stats["data_types"].get(data_type, 0) + 1
        
        return stats
    
    def query(self, rag_query: RAGQuery) -> RAGResponse:
        """
        Main query interface for the RAG system
        
        Args:
            rag_query: Complete query object
        
        Returns:
            RAG response with generated answer and metadata
        """
        if not self.initialized:
            raise RuntimeError("RAG system not initialized")
        
        start_time = datetime.now()
        query_id = rag_query.query_id or f"query_{int(start_time.timestamp())}"
        
        logger.info(f"Processing RAG query: {rag_query.text[:100]}...")
        
        try:
            # Step 1: Create retrieval query
            retrieval_query = RetrievalQuery(
                text=rag_query.text,
                query_type=rag_query.query_type,
                filters=rag_query.filters,
                max_results=rag_query.max_results
            )
            
            # Step 2: Retrieve relevant documents
            retrieved_results = self.retriever.retrieve(
                retrieval_query,
                context=rag_query.context
            )
            
            logger.info(f"Retrieved {len(retrieved_results)} relevant documents")
            
            # Step 3: Generate response
            generation_context = GenerationContext(
                query=rag_query.text,
                retrieved_results=retrieved_results,
                conversation_history=getattr(rag_query.context, 'conversation_history', None) if rag_query.context else None,
                response_style=rag_query.response_style,
                include_sources=True,
                include_confidence=True
            )
            
            generated_response = self.response_generator.generate_response(
                rag_query.text,
                generation_context,
                provider_preference=rag_query.provider_preference
            )
            
            # Step 4: Create complete response
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            response = RAGResponse(
                query_id=query_id,
                response=generated_response,
                retrieved_results=retrieved_results,
                processing_time=processing_time,
                metadata={
                    "query_type": rag_query.query_type,
                    "num_retrieved": len(retrieved_results),
                    "avg_similarity": sum(r.result.similarity_score for r in retrieved_results) / max(len(retrieved_results), 1),
                    "response_length": len(generated_response.response_text),
                    "provider_used": generated_response.model_used
                },
                timestamp=end_time
            )
            
            logger.info(f"✓ Query processed in {processing_time:.2f}s (confidence: {generated_response.confidence_score:.2f})")
            
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            # Return error response
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            error_response = GeneratedResponse(
                response_text=f"Error processing query: {str(e)}",
                confidence_score=0.0,
                sources_used=[],
                model_used="error",
                warnings=[str(e)]
            )
            
            return RAGResponse(
                query_id=query_id,
                response=error_response,
                retrieved_results=[],
                processing_time=processing_time,
                metadata={"error": str(e)},
                timestamp=end_time
            )
    
    def batch_query(self, queries: List[RAGQuery]) -> List[RAGResponse]:
        """Process multiple queries in batch"""
        logger.info(f"Processing batch of {len(queries)} queries")
        
        responses = []
        for query in queries:
            try:
                response = self.query(query)
                responses.append(response)
            except Exception as e:
                logger.error(f"Batch query failed for: {query.text[:50]}... - {e}")
                # Add error response
                error_response = RAGResponse(
                    query_id=query.query_id or f"error_{len(responses)}",
                    response=GeneratedResponse(
                        response_text=f"Batch processing error: {str(e)}",
                        confidence_score=0.0,
                        sources_used=[],
                        model_used="error"
                    ),
                    retrieved_results=[],
                    processing_time=0.0,
                    metadata={"error": str(e)},
                    timestamp=datetime.now()
                )
                responses.append(error_response)
        
        return responses
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            "system_status": "initialized" if self.initialized else "not_initialized",
            "timestamp": datetime.now().isoformat()
        }
        
        if self.initialized:
            # Vector store statistics
            stats["vector_store"] = self.vector_store.get_statistics()
            
            # Embedding model information
            stats["embedding_models"] = self.embedding_generator.get_model_info()
            
            # Available LLM providers
            stats["llm_providers"] = self.response_generator.get_available_providers()
            
            # Conversation contexts
            stats["active_conversations"] = len(self.conversation_histories)
            
            # Cache statistics
            stats["query_cache_size"] = len(self.query_cache)
        
        return stats
    
    def clear_cache(self):
        """Clear query cache and conversation histories"""
        self.query_cache.clear()
        self.conversation_histories.clear()
        logger.info("Cleared system cache")
    
    def export_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Export conversation history"""
        if conversation_id in self.conversation_histories:
            context = self.conversation_histories[conversation_id]
            return {
                "conversation_id": conversation_id,
                "history": context.conversation_history,
                "previous_results": [asdict(r) for r in context.previous_results],
                "domain_focus": context.domain_focus
            }
        return None
    
    # Helper methods
    def _merge_stats(self, main_stats: Dict, new_stats: Dict):
        """Merge statistics dictionaries"""
        main_stats["chunks"] += new_stats["chunks"]
        main_stats["embeddings"] += new_stats["embeddings"]
        
        for data_type, count in new_stats["data_types"].items():
            if data_type not in main_stats["data_types"]:
                main_stats["data_types"][data_type] = 0
            main_stats["data_types"][data_type] += count
    
    def _format_conversation(self, comm: Dict) -> str:
        """Format communication data for embedding"""
        if "messages" in comm:
            messages = []
            for msg in comm["messages"]:
                sender = msg.get("sender", "Unknown")
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")
                messages.append(f"[{timestamp}] {sender}: {content}")
            return "\n".join(messages)
        return str(comm)
    
    def _format_contact(self, contact: Dict) -> str:
        """Format contact data for embedding"""
        parts = []
        if contact.get("name"):
            parts.append(f"Name: {contact['name']}")
        if contact.get("phone_number"):
            parts.append(f"Phone: {contact['phone_number']}")
        if contact.get("email"):
            parts.append(f"Email: {contact['email']}")
        if contact.get("notes"):
            parts.append(f"Notes: {contact['notes']}")
        return " | ".join(parts)
    
    def _format_call_log(self, call: Dict) -> str:
        """Format call log data for embedding"""
        parts = []
        if call.get("timestamp"):
            parts.append(f"Time: {call['timestamp']}")
        if call.get("direction"):
            parts.append(f"Direction: {call['direction']}")
        if call.get("number"):
            parts.append(f"Number: {call['number']}")
        if call.get("duration"):
            parts.append(f"Duration: {call['duration']}s")
        if call.get("status"):
            parts.append(f"Status: {call['status']}")
        if call.get("location"):
            parts.append(f"Location: {call['location']}")
        return " | ".join(parts)
    
    def _extract_contact_entities(self, contact: Dict) -> Dict[str, List[str]]:
        """Extract entities from contact data"""
        entities = {}
        
        if contact.get("phone_number"):
            entities["phone"] = [contact["phone_number"]]
        
        if contact.get("email"):
            entities["email"] = [contact["email"]]
        
        if contact.get("name"):
            entities["person"] = [contact["name"]]
        
        return entities
    
    def _extract_call_entities(self, call: Dict) -> Dict[str, List[str]]:
        """Extract entities from call data"""
        entities = {}
        
        if call.get("number"):
            entities["phone"] = [call["number"]]
        
        if call.get("location"):
            entities["coordinates"] = [call["location"]]
        
        return entities
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse timestamp string to datetime object"""
        if not timestamp_str:
            return None
        
        try:
            # Try ISO format first
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except Exception:
            try:
                # Try other common formats
                return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                logger.warning(f"Could not parse timestamp: {timestamp_str}")
                return None

# Factory function
def create_forensic_rag_system(config: Dict[str, Any]) -> ForensicRAGSystem:
    """Create and initialize a complete forensic RAG system"""
    return ForensicRAGSystem(config)