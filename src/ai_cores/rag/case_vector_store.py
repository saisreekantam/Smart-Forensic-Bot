"""
Case-Specific Vector Store Manager for Forensic Analysis

This module extends the existing vector store to support case-specific
collections, ensuring that each forensic case has isolated embeddings
and can be searched independently.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json

# ChromaDB imports
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from .chroma_manager import ChromaDBManager
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

# Local imports
from .vector_store import BaseVectorStore, SearchResult, SearchFilter
from .embeddings import ForensicEmbedding, EmbeddingMetadata

logger = logging.getLogger(__name__)

class CaseVectorStore:
    """
    Case-specific vector store manager that handles multiple ChromaDB collections,
    one for each forensic case
    """
    
    def __init__(self, persist_directory: str = "data/vector_db"):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Use shared ChromaDB client manager to prevent conflicts
        self.client = ChromaDBManager.get_client(
            persist_directory=str(self.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Track active collections for each case
        self._case_collections: Dict[str, Any] = {}
        
        logger.info(f"Initialized CaseVectorStore at {self.persist_directory}")
    
    def get_case_collection(self, case_id: str, collection_name: str) -> Any:
        """
        Get or create a ChromaDB collection for a specific case
        
        Args:
            case_id: Unique case identifier
            collection_name: Name for the collection (usually case_<case_number>)
            
        Returns:
            ChromaDB collection for the case
        """
        if case_id in self._case_collections:
            return self._case_collections[case_id]
        
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=collection_name)
            logger.info(f"Retrieved existing collection for case {case_id}: {collection_name}")
        except Exception:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "case_id": case_id,
                    "created_at": datetime.now().isoformat(),
                    "description": f"Vector embeddings for forensic case {case_id}"
                }
            )
            logger.info(f"Created new collection for case {case_id}: {collection_name}")
        
        self._case_collections[case_id] = collection
        return collection
    
    def add_case_embeddings(
        self, 
        case_id: str, 
        collection_name: str, 
        embeddings: List[ForensicEmbedding]
    ) -> List[str]:
        """
        Add embeddings to a specific case's collection
        
        Args:
            case_id: Case identifier
            collection_name: Collection name for the case
            embeddings: List of ForensicEmbedding objects to add
            
        Returns:
            List of embedding IDs that were added
        """
        if not embeddings:
            return []
        
        collection = self.get_case_collection(case_id, collection_name)
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings_array = []
        
        for embedding in embeddings:
            ids.append(embedding.id)
            documents.append(embedding.text)
            
            # Convert metadata to dict and add case context
            metadata_dict = {
                "case_id": case_id,
                "chunk_id": embedding.metadata.chunk_id,
                "source_file": embedding.metadata.source_file,
                "data_type": embedding.metadata.data_type,
                "timestamp": embedding.metadata.timestamp.isoformat() if embedding.metadata.timestamp else None,
                "participants": json.dumps(embedding.metadata.participants or []),
                "entities": json.dumps(embedding.metadata.entities or {}),
                "confidence_score": embedding.metadata.confidence_score,
                "language": embedding.metadata.language,
                "processing_version": embedding.metadata.processing_version,
                "evidence_id": getattr(embedding.metadata, 'evidence_id', None),
                "chunk_index": getattr(embedding.metadata, 'chunk_index', None)
            }
            
            metadatas.append(metadata_dict)
            embeddings_array.append(embedding.vector.tolist())
        
        try:
            collection.add(
                embeddings=embeddings_array,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(embeddings)} embeddings to case {case_id} collection {collection_name}")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding embeddings to case {case_id}: {str(e)}")
            raise
    
    def search_case(
        self,
        case_id: str,
        collection_name: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[SearchFilter] = None
    ) -> List[SearchResult]:
        """
        Search within a specific case's embeddings
        
        Args:
            case_id: Case to search within
            collection_name: Collection name for the case
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Additional search filters
            
        Returns:
            List of search results from the case
        """
        try:
            collection = self.get_case_collection(case_id, collection_name)
            
            # Build where clause for ChromaDB
            where_clause = {"case_id": case_id}
            
            if filters:
                if filters.data_types:
                    where_clause["data_type"] = {"$in": filters.data_types}
                
                if filters.source_files:
                    where_clause["source_file"] = {"$in": filters.source_files}
                
                if filters.confidence_threshold > 0:
                    where_clause["confidence_score"] = {"$gte": filters.confidence_threshold}
                
                if filters.exclude_ids:
                    where_clause["$not"] = {"$in": filters.exclude_ids}
            
            # Perform search
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_clause if len(where_clause) > 1 else None
            )
            
            # Convert to SearchResult objects
            search_results = []
            
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0.0
                    
                    # Convert back to EmbeddingMetadata
                    embedding_metadata = EmbeddingMetadata(
                        chunk_id=metadata.get('chunk_id', ''),
                        source_file=metadata.get('source_file', ''),
                        data_type=metadata.get('data_type', ''),
                        timestamp=datetime.fromisoformat(metadata['timestamp']) if metadata.get('timestamp') else None,
                        participants=json.loads(metadata.get('participants', '[]')),
                        entities=json.loads(metadata.get('entities', '{}')),
                        confidence_score=metadata.get('confidence_score', 0.0),
                        language=metadata.get('language', 'en'),
                        processing_version=metadata.get('processing_version', '1.0')
                    )
                    
                    # Add case-specific metadata
                    embedding_metadata.evidence_id = metadata.get('evidence_id')
                    embedding_metadata.chunk_index = metadata.get('chunk_index')
                    
                    search_result = SearchResult(
                        id=doc_id,
                        text=results['documents'][0][i] if results['documents'] else '',
                        metadata=embedding_metadata,
                        similarity_score=1.0 - distance,  # Convert distance to similarity
                        rank=i + 1
                    )
                    
                    search_results.append(search_result)
            
            logger.info(f"Found {len(search_results)} results for case {case_id}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching case {case_id}: {str(e)}")
            return []
    
    def get_case_statistics(self, case_id: str, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics for a case's vector store
        
        Args:
            case_id: Case identifier
            collection_name: Collection name
            
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.get_case_collection(case_id, collection_name)
            
            # Get collection count
            count = collection.count()
            
            # Get a sample of metadata to analyze data types
            sample_results = collection.peek(limit=min(100, count))
            
            data_types = set()
            source_files = set()
            languages = set()
            
            if sample_results.get('metadatas'):
                for metadata in sample_results['metadatas']:
                    if metadata.get('data_type'):
                        data_types.add(metadata['data_type'])
                    if metadata.get('source_file'):
                        source_files.add(metadata['source_file'])
                    if metadata.get('language'):
                        languages.add(metadata['language'])
            
            return {
                "case_id": case_id,
                "collection_name": collection_name,
                "total_embeddings": count,
                "data_types": list(data_types),
                "source_files": list(source_files),
                "languages": list(languages),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics for case {case_id}: {str(e)}")
            return {
                "case_id": case_id,
                "collection_name": collection_name,
                "error": str(e)
            }
    
    def delete_case_embeddings(self, case_id: str, collection_name: str, embedding_ids: List[str]) -> bool:
        """
        Delete specific embeddings from a case collection
        
        Args:
            case_id: Case identifier
            collection_name: Collection name
            embedding_ids: List of embedding IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.get_case_collection(case_id, collection_name)
            collection.delete(ids=embedding_ids)
            
            logger.info(f"Deleted {len(embedding_ids)} embeddings from case {case_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embeddings from case {case_id}: {str(e)}")
            return False
    
    def delete_case_collection(self, case_id: str, collection_name: str) -> bool:
        """
        Delete entire collection for a case
        
        Args:
            case_id: Case identifier
            collection_name: Collection name to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(name=collection_name)
            
            # Remove from cache
            if case_id in self._case_collections:
                del self._case_collections[case_id]
            
            logger.info(f"Deleted collection {collection_name} for case {case_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection for case {case_id}: {str(e)}")
            return False
    
    def list_case_collections(self) -> List[Dict[str, Any]]:
        """
        List all case collections in the vector store
        
        Returns:
            List of collection information
        """
        try:
            collections = self.client.list_collections()
            
            collection_info = []
            for collection in collections:
                metadata = collection.metadata or {}
                collection_info.append({
                    "name": collection.name,
                    "case_id": metadata.get("case_id", "unknown"),
                    "created_at": metadata.get("created_at", "unknown"),
                    "description": metadata.get("description", ""),
                    "count": collection.count()
                })
            
            return collection_info
            
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return []
    
    def backup_case_collection(self, case_id: str, collection_name: str, backup_path: str) -> bool:
        """
        Backup a case collection to file
        
        Args:
            case_id: Case identifier
            collection_name: Collection name
            backup_path: Path to save backup
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.get_case_collection(case_id, collection_name)
            
            # Get all data from collection
            all_data = collection.get()
            
            backup_data = {
                "case_id": case_id,
                "collection_name": collection_name,
                "backup_timestamp": datetime.now().isoformat(),
                "data": all_data
            }
            
            # Save to file
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            logger.info(f"Backed up case {case_id} collection to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error backing up case {case_id}: {str(e)}")
            return False

# Default case vector store instance
case_vector_store = CaseVectorStore()