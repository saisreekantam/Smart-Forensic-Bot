"""
Advanced Vector Database System for Forensic RAG

This module provides a sophisticated vector storage and retrieval system optimized for forensic data,
supporting multiple backends (ChromaDB, FAISS) with advanced indexing and filtering capabilities.
"""

import numpy as np
import logging
import json
import pickle
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import uuid
from abc import ABC, abstractmethod

# Vector database backends
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.utils import embedding_functions
    from .chroma_manager import ChromaDBManager
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

# Local imports
from .embeddings import ForensicEmbedding, EmbeddingMetadata

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Container for search results with forensic context"""
    id: str
    text: str
    metadata: EmbeddingMetadata
    similarity_score: float
    embedding: Optional[np.ndarray] = None
    rank: int = 0

@dataclass
class SearchFilter:
    """Advanced filtering options for forensic searches"""
    data_types: Optional[List[str]] = None
    date_range: Optional[Tuple[datetime, datetime]] = None
    participants: Optional[List[str]] = None
    entities: Optional[Dict[str, List[str]]] = None
    sensitivity_levels: Optional[List[str]] = None
    case_ids: Optional[List[str]] = None
    source_files: Optional[List[str]] = None
    confidence_threshold: float = 0.0
    exclude_ids: Optional[List[str]] = None

class BaseVectorStore(ABC):
    """Abstract base class for vector storage backends"""
    
    @abstractmethod
    def add_embeddings(self, embeddings: List[ForensicEmbedding]) -> List[str]:
        """Add embeddings to the vector store"""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[SearchFilter] = None
    ) -> List[SearchResult]:
        """Search for similar embeddings"""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """Delete embeddings by IDs"""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        pass

class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-based vector storage with advanced forensic features"""
    
    def __init__(self, config: Dict[str, Any]):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        self.config = config
        self.persist_directory = Path(config.get('persist_directory', './data/vector_db'))
        self.collection_name = config.get('collection_name', 'ufdr_embeddings')
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Use shared ChromaDB client manager to prevent conflicts
        self.client = ChromaDBManager.get_client(
            persist_directory=str(self.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing ChromaDB collection: {self.collection_name}")
        except Exception:
            # Create new collection with custom metadata schema
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Forensic UFDR embeddings for RAG system",
                    "created_at": datetime.now().isoformat(),
                    "schema_version": "1.0"
                }
            )
            logger.info(f"Created new ChromaDB collection: {self.collection_name}")
    
    def add_embeddings(self, embeddings: List[ForensicEmbedding]) -> List[str]:
        """Add forensic embeddings to ChromaDB"""
        if not embeddings:
            return []
        
        ids = []
        documents = []
        embedding_vectors = []
        metadatas = []
        
        for fe in embeddings:
            # Generate unique ID if not present
            doc_id = fe.metadata.chunk_id or str(uuid.uuid4())
            ids.append(doc_id)
            
            # Prepare document text
            documents.append(fe.text)
            
            # Prepare embedding vector
            embedding_vectors.append(fe.embedding.tolist())
            
            # Prepare metadata for ChromaDB (must be JSON serializable)
            metadata = self._prepare_metadata_for_chroma(fe)
            metadatas.append(metadata)
        
        try:
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embedding_vectors,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(embeddings)} embeddings to ChromaDB")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add embeddings to ChromaDB: {e}")
            raise
    
    def _prepare_metadata_for_chroma(self, forensic_embedding: ForensicEmbedding) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB storage (JSON serializable)"""
        metadata = {
            "data_type": forensic_embedding.metadata.data_type,
            "model_name": forensic_embedding.model_name,
            "embedding_type": forensic_embedding.embedding_type,
            "confidence_score": forensic_embedding.confidence_score,
            "sensitivity_level": forensic_embedding.metadata.sensitivity_level,
            "created_at": datetime.now().isoformat()
        }
        
        # Add optional fields if present
        if forensic_embedding.metadata.timestamp:
            metadata["timestamp"] = forensic_embedding.metadata.timestamp.isoformat()
        
        if forensic_embedding.metadata.participants:
            metadata["participants"] = ",".join(forensic_embedding.metadata.participants)
        
        if forensic_embedding.metadata.entities:
            # Flatten entities for ChromaDB
            for entity_type, entity_list in forensic_embedding.metadata.entities.items():
                metadata[f"entities_{entity_type}"] = ",".join(entity_list)
        
        if forensic_embedding.metadata.case_id:
            metadata["case_id"] = forensic_embedding.metadata.case_id
        
        if forensic_embedding.metadata.source_file:
            metadata["source_file"] = forensic_embedding.metadata.source_file
        
        return metadata
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[SearchFilter] = None
    ) -> List[SearchResult]:
        """Search ChromaDB with advanced filtering"""
        
        # Prepare where clause for filtering
        where_clause = self._build_chroma_where_clause(filters) if filters else None
        
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, 100),  # ChromaDB limit
                where=where_clause,
                include=["documents", "metadatas", "distances", "embeddings"]
            )
            
            # Convert results to SearchResult objects
            search_results = []
            
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    # Extract data
                    text = results["documents"][0][i]
                    metadata_dict = results["metadatas"][0][i]
                    distance = results["distances"][0][i]
                    embedding = np.array(results["embeddings"][0][i]) if results["embeddings"] else None
                    
                    # Convert distance to similarity (ChromaDB returns L2 distance)
                    similarity = 1 / (1 + distance)
                    
                    # Reconstruct metadata
                    metadata = self._reconstruct_metadata_from_chroma(metadata_dict)
                    
                    search_result = SearchResult(
                        id=doc_id,
                        text=text,
                        metadata=metadata,
                        similarity_score=similarity,
                        embedding=embedding,
                        rank=i + 1
                    )
                    
                    search_results.append(search_result)
            
            # Apply additional client-side filtering if needed
            if filters:
                search_results = self._apply_additional_filters(search_results, filters)
            
            logger.info(f"ChromaDB search returned {len(search_results)} results")
            return search_results[:top_k]
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []
    
    def _build_chroma_where_clause(self, filters: SearchFilter) -> Dict[str, Any]:
        """Build ChromaDB where clause from search filters"""
        where = {}
        
        if filters.data_types:
            if len(filters.data_types) == 1:
                where["data_type"] = {"$eq": filters.data_types[0]}
            else:
                where["data_type"] = {"$in": filters.data_types}
        
        if filters.case_ids:
            if len(filters.case_ids) == 1:
                where["case_id"] = {"$eq": filters.case_ids[0]}
            else:
                where["case_id"] = {"$in": filters.case_ids}
        
        if filters.sensitivity_levels:
            if len(filters.sensitivity_levels) == 1:
                where["sensitivity_level"] = {"$eq": filters.sensitivity_levels[0]}
            else:
                where["sensitivity_level"] = {"$in": filters.sensitivity_levels}
        
        if filters.confidence_threshold > 0:
            where["confidence_score"] = {"$gte": filters.confidence_threshold}
        
        # Date range filtering (if timestamps are present)
        if filters.date_range:
            start_date, end_date = filters.date_range
            where["timestamp"] = {
                "$gte": start_date.isoformat(),
                "$lte": end_date.isoformat()
            }
        
        return where if where else None
    
    def _reconstruct_metadata_from_chroma(self, metadata_dict: Dict[str, Any]) -> EmbeddingMetadata:
        """Reconstruct EmbeddingMetadata from ChromaDB metadata"""
        
        # Parse timestamp
        timestamp = None
        if "timestamp" in metadata_dict:
            try:
                timestamp = datetime.fromisoformat(metadata_dict["timestamp"])
            except Exception:
                pass
        
        # Parse participants
        participants = None
        if "participants" in metadata_dict and metadata_dict["participants"]:
            participants = metadata_dict["participants"].split(",")
        
        # Parse entities
        entities = {}
        for key, value in metadata_dict.items():
            if key.startswith("entities_") and value:
                entity_type = key.replace("entities_", "")
                entities[entity_type] = value.split(",")
        
        return EmbeddingMetadata(
            chunk_id=metadata_dict.get("chunk_id", ""),
            data_type=metadata_dict.get("data_type", "unknown"),
            timestamp=timestamp,
            participants=participants,
            entities=entities if entities else None,
            sensitivity_level=metadata_dict.get("sensitivity_level", "standard"),
            source_file=metadata_dict.get("source_file"),
            case_id=metadata_dict.get("case_id")
        )
    
    def _apply_additional_filters(
        self,
        results: List[SearchResult],
        filters: SearchFilter
    ) -> List[SearchResult]:
        """Apply additional client-side filters"""
        filtered_results = []
        
        for result in results:
            # Filter by participants
            if filters.participants:
                if not result.metadata.participants:
                    continue
                if not any(p in result.metadata.participants for p in filters.participants):
                    continue
            
            # Filter by entities
            if filters.entities:
                if not result.metadata.entities:
                    continue
                match_found = False
                for entity_type, entity_values in filters.entities.items():
                    if entity_type in result.metadata.entities:
                        if any(ev in result.metadata.entities[entity_type] for ev in entity_values):
                            match_found = True
                            break
                if not match_found:
                    continue
            
            # Filter by excluded IDs
            if filters.exclude_ids and result.id in filters.exclude_ids:
                continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def delete(self, ids: List[str]) -> bool:
        """Delete embeddings by IDs"""
        try:
            # Filter to only include IDs that exist
            existing_results = self.collection.get(ids=ids, include=[])
            existing_ids = existing_results["ids"] if existing_results["ids"] else []
            
            if existing_ids:
                self.collection.delete(ids=existing_ids)
                logger.info(f"Deleted {len(existing_ids)} embeddings from ChromaDB")
                return True
            else:
                logger.warning("No matching IDs found for deletion")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ChromaDB collection statistics"""
        try:
            # Get basic collection info
            collection_count = self.collection.count()
            
            # Get sample of metadata for analysis
            sample_results = self.collection.get(
                limit=min(100, collection_count),
                include=["metadatas"]
            )
            
            stats = {
                "total_embeddings": collection_count,
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory)
            }
            
            if sample_results["metadatas"]:
                # Analyze metadata
                data_types = {}
                models = {}
                sensitivity_levels = {}
                
                for metadata in sample_results["metadatas"]:
                    # Count data types
                    data_type = metadata.get("data_type", "unknown")
                    data_types[data_type] = data_types.get(data_type, 0) + 1
                    
                    # Count models
                    model = metadata.get("model_name", "unknown")
                    models[model] = models.get(model, 0) + 1
                    
                    # Count sensitivity levels
                    sensitivity = metadata.get("sensitivity_level", "standard")
                    sensitivity_levels[sensitivity] = sensitivity_levels.get(sensitivity, 0) + 1
                
                stats.update({
                    "data_type_distribution": data_types,
                    "model_distribution": models,
                    "sensitivity_distribution": sensitivity_levels
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get ChromaDB statistics: {e}")
            return {"error": str(e)}

class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector storage for high-performance similarity search"""
    
    def __init__(self, config: Dict[str, Any]):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")
        
        self.config = config
        self.dimension = config.get('dimension', 384)
        self.index_type = config.get('index_type', 'flat')  # 'flat', 'ivf', 'hnsw'
        self.persist_directory = Path(config.get('persist_directory', './data/vector_db'))
        self.index_file = self.persist_directory / 'faiss.index'
        self.metadata_file = self.persist_directory / 'metadata.pkl'
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index
        self.index = self._create_index()
        self.id_to_metadata: Dict[str, Dict[str, Any]] = {}
        self.next_id = 0
        
        # Load existing index if available
        self._load_index()
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration"""
        if self.index_type == 'flat':
            # Simple flat index for exact search
            index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
        elif self.index_type == 'ivf':
            # IVF index for faster approximate search
            nlist = self.config.get('nlist', 100)  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        elif self.index_type == 'hnsw':
            # HNSW index for very fast approximate search
            m = self.config.get('hnsw_m', 16)  # Number of bi-directional links
            index = faiss.IndexHNSWFlat(self.dimension, m)
        else:
            logger.warning(f"Unknown index type {self.index_type}, using flat")
            index = faiss.IndexFlatIP(self.dimension)
        
        logger.info(f"Created FAISS index: {type(index).__name__}")
        return index
    
    def _load_index(self):
        """Load existing FAISS index and metadata"""
        try:
            if self.index_file.exists() and self.metadata_file.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(self.index_file))
                
                # Load metadata
                with open(self.metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.id_to_metadata = data.get('id_to_metadata', {})
                    self.next_id = data.get('next_id', 0)
                
                logger.info(f"Loaded FAISS index with {self.index.ntotal} embeddings")
            else:
                logger.info("No existing FAISS index found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            # Reset if loading fails
            self.index = self._create_index()
            self.id_to_metadata = {}
            self.next_id = 0
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))
            
            # Save metadata
            data = {
                'id_to_metadata': self.id_to_metadata,
                'next_id': self.next_id
            }
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.debug("Saved FAISS index and metadata")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def add_embeddings(self, embeddings: List[ForensicEmbedding]) -> List[str]:
        """Add embeddings to FAISS index"""
        if not embeddings:
            return []
        
        # Prepare embedding matrix
        embedding_matrix = np.array([fe.embedding for fe in embeddings]).astype(np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embedding_matrix)
        
        # Generate IDs and store metadata
        ids = []
        for i, fe in enumerate(embeddings):
            doc_id = fe.metadata.chunk_id or f"doc_{self.next_id}"
            ids.append(doc_id)
            
            # Store metadata
            metadata = {
                'internal_id': self.next_id,
                'text': fe.text,
                'metadata': asdict(fe.metadata),
                'model_name': fe.model_name,
                'embedding_type': fe.embedding_type,
                'confidence_score': fe.confidence_score,
                'created_at': datetime.now().isoformat()
            }
            self.id_to_metadata[doc_id] = metadata
            self.next_id += 1
        
        # Add to FAISS index
        self.index.add(embedding_matrix)
        
        # Train index if needed (for IVF)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if self.index.ntotal >= self.config.get('min_train_size', 1000):
                logger.info("Training FAISS index...")
                self.index.train(embedding_matrix)
        
        # Save to disk
        self._save_index()
        
        logger.info(f"Added {len(embeddings)} embeddings to FAISS index")
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[SearchFilter] = None
    ) -> List[SearchResult]:
        """Search FAISS index with post-filtering"""
        
        if self.index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search with larger k to account for filtering
        search_k = min(top_k * 5, self.index.ntotal)  # Search more to allow for filtering
        
        try:
            # Perform FAISS search
            similarities, indices = self.index.search(query_embedding, search_k)
            
            # Convert results to SearchResult objects
            search_results = []
            
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                # Find the document ID for this internal index
                doc_id = None
                metadata_dict = None
                
                for doc_id_key, meta in self.id_to_metadata.items():
                    if meta['internal_id'] == idx:
                        doc_id = doc_id_key
                        metadata_dict = meta
                        break
                
                if not doc_id or not metadata_dict:
                    continue
                
                # Reconstruct metadata
                metadata = EmbeddingMetadata(**metadata_dict['metadata'])
                
                search_result = SearchResult(
                    id=doc_id,
                    text=metadata_dict['text'],
                    metadata=metadata,
                    similarity_score=float(similarity),
                    rank=i + 1
                )
                
                # Apply filters
                if filters and not self._passes_filters(search_result, filters):
                    continue
                
                search_results.append(search_result)
                
                # Stop when we have enough results
                if len(search_results) >= top_k:
                    break
            
            logger.info(f"FAISS search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    def _passes_filters(self, result: SearchResult, filters: SearchFilter) -> bool:
        """Check if a result passes the given filters"""
        
        # Data type filter
        if filters.data_types and result.metadata.data_type not in filters.data_types:
            return False
        
        # Date range filter
        if filters.date_range and result.metadata.timestamp:
            start_date, end_date = filters.date_range
            if not (start_date <= result.metadata.timestamp <= end_date):
                return False
        
        # Participants filter
        if filters.participants:
            if not result.metadata.participants:
                return False
            if not any(p in result.metadata.participants for p in filters.participants):
                return False
        
        # Entities filter
        if filters.entities:
            if not result.metadata.entities:
                return False
            match_found = False
            for entity_type, entity_values in filters.entities.items():
                if entity_type in result.metadata.entities:
                    if any(ev in result.metadata.entities[entity_type] for ev in entity_values):
                        match_found = True
                        break
            if not match_found:
                return False
        
        # Sensitivity level filter
        if filters.sensitivity_levels and result.metadata.sensitivity_level not in filters.sensitivity_levels:
            return False
        
        # Case ID filter
        if filters.case_ids and result.metadata.case_id not in filters.case_ids:
            return False
        
        # Source file filter
        if filters.source_files and result.metadata.source_file not in filters.source_files:
            return False
        
        # Confidence threshold
        # Note: For FAISS, confidence is in metadata, need to check if available
        
        # Exclude IDs filter
        if filters.exclude_ids and result.id in filters.exclude_ids:
            return False
        
        return True
    
    def delete(self, ids: List[str]) -> bool:
        """Delete embeddings by IDs (FAISS doesn't support efficient deletion)"""
        logger.warning("FAISS doesn't support efficient deletion. Consider rebuilding index.")
        
        # Remove from metadata
        deleted_count = 0
        for doc_id in ids:
            if doc_id in self.id_to_metadata:
                del self.id_to_metadata[doc_id]
                deleted_count += 1
        
        if deleted_count > 0:
            self._save_index()
            logger.info(f"Removed {deleted_count} embeddings from metadata")
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get FAISS index statistics"""
        stats = {
            "total_embeddings": self.index.ntotal,
            "index_type": type(self.index).__name__,
            "dimension": self.dimension,
            "is_trained": getattr(self.index, 'is_trained', True),
            "persist_directory": str(self.persist_directory)
        }
        
        # Analyze metadata
        if self.id_to_metadata:
            data_types = {}
            models = {}
            
            for metadata in self.id_to_metadata.values():
                # Count data types
                data_type = metadata.get('metadata', {}).get('data_type', 'unknown')
                data_types[data_type] = data_types.get(data_type, 0) + 1
                
                # Count models
                model = metadata.get('model_name', 'unknown')
                models[model] = models.get(model, 0) + 1
            
            stats.update({
                "data_type_distribution": data_types,
                "model_distribution": models
            })
        
        return stats

class HybridVectorStore(BaseVectorStore):
    """Hybrid vector store combining ChromaDB and FAISS for optimal performance"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.primary_backend = config.get('primary_backend', 'chroma')
        
        # Initialize backends
        self.stores: Dict[str, BaseVectorStore] = {}
        
        # Initialize ChromaDB if available
        if CHROMADB_AVAILABLE and config.get('use_chroma', True):
            try:
                chroma_config = config.get('chroma_config', {})
                self.stores['chroma'] = ChromaVectorStore(chroma_config)
                logger.info("Initialized ChromaDB backend")
            except Exception as e:
                logger.warning(f"Failed to initialize ChromaDB: {e}")
        
        # Initialize FAISS if available
        if FAISS_AVAILABLE and config.get('use_faiss', False):
            try:
                faiss_config = config.get('faiss_config', {})
                faiss_config['dimension'] = config.get('embedding_dimension', 384)
                self.stores['faiss'] = FAISSVectorStore(faiss_config)
                logger.info("Initialized FAISS backend")
            except Exception as e:
                logger.warning(f"Failed to initialize FAISS: {e}")
        
        if not self.stores:
            raise RuntimeError("No vector store backends available")
        
        # Set primary store
        if self.primary_backend in self.stores:
            self.primary_store = self.stores[self.primary_backend]
        else:
            self.primary_store = list(self.stores.values())[0]
            self.primary_backend = list(self.stores.keys())[0]
        
        logger.info(f"Using {self.primary_backend} as primary vector store")
    
    def add_embeddings(self, embeddings: List[ForensicEmbedding]) -> List[str]:
        """Add embeddings to all available backends"""
        results = {}
        
        for backend_name, store in self.stores.items():
            try:
                ids = store.add_embeddings(embeddings)
                results[backend_name] = ids
                logger.info(f"Added {len(ids)} embeddings to {backend_name}")
            except Exception as e:
                logger.error(f"Failed to add embeddings to {backend_name}: {e}")
        
        # Return IDs from primary store
        return results.get(self.primary_backend, [])
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[SearchFilter] = None
    ) -> List[SearchResult]:
        """Search using the primary backend with fallback"""
        
        # Try primary backend first
        try:
            results = self.primary_store.search(query_embedding, top_k, filters)
            if results:
                return results
        except Exception as e:
            logger.error(f"Primary backend {self.primary_backend} search failed: {e}")
        
        # Try fallback backends
        for backend_name, store in self.stores.items():
            if backend_name == self.primary_backend:
                continue
            
            try:
                results = store.search(query_embedding, top_k, filters)
                if results:
                    logger.info(f"Used fallback backend {backend_name} for search")
                    return results
            except Exception as e:
                logger.error(f"Fallback backend {backend_name} search failed: {e}")
        
        return []
    
    def delete(self, ids: List[str]) -> bool:
        """Delete from all backends"""
        success = False
        
        for backend_name, store in self.stores.items():
            try:
                if store.delete(ids):
                    success = True
                    logger.info(f"Deleted from {backend_name}")
            except Exception as e:
                logger.error(f"Failed to delete from {backend_name}: {e}")
        
        return success
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from all backends"""
        stats = {
            "primary_backend": self.primary_backend,
            "available_backends": list(self.stores.keys()),
            "backend_stats": {}
        }
        
        for backend_name, store in self.stores.items():
            try:
                stats["backend_stats"][backend_name] = store.get_statistics()
            except Exception as e:
                stats["backend_stats"][backend_name] = {"error": str(e)}
        
        return stats

def create_vector_store(config: Dict[str, Any]) -> BaseVectorStore:
    """Factory function to create appropriate vector store"""
    
    store_type = config.get('store_type', 'auto')
    
    if store_type == 'auto':
        # Auto-select based on availability
        if CHROMADB_AVAILABLE:
            return ChromaVectorStore(config)
        elif FAISS_AVAILABLE:
            faiss_config = config.copy()
            faiss_config['dimension'] = config.get('embedding_dimension', 384)
            return FAISSVectorStore(faiss_config)
        else:
            raise RuntimeError("No vector store backends available")
    
    elif store_type == 'chroma':
        return ChromaVectorStore(config)
    
    elif store_type == 'faiss':
        faiss_config = config.copy()
        faiss_config['dimension'] = config.get('embedding_dimension', 384)
        return FAISSVectorStore(faiss_config)
    
    elif store_type == 'hybrid':
        return HybridVectorStore(config)
    
    else:
        raise ValueError(f"Unknown store type: {store_type}")