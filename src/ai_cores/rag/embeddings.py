"""
Advanced Multi-Model Embedding System for Forensic Data Analysis

This module provides sophisticated embedding generation optimized for different types of forensic data,
including conversations, entities, temporal patterns, and relationships.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import re
from datetime import datetime
import json

# Core embedding models
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import openai
import google.generativeai as genai

# Specialized models for forensic analysis
try:
    from InstructorEmbedding import INSTRUCTOR
except ImportError:
    INSTRUCTOR = None

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingMetadata:
    """Metadata for embeddings with forensic context"""
    chunk_id: str
    data_type: str  # 'conversation', 'contact', 'call_log', 'entity', 'document'
    timestamp: Optional[datetime] = None
    participants: Optional[List[str]] = None
    entities: Optional[Dict[str, List[str]]] = None
    sensitivity_level: str = "standard"  # 'low', 'standard', 'high', 'critical'
    source_file: Optional[str] = None
    case_id: Optional[str] = None

@dataclass
class ForensicEmbedding:
    """Container for forensic embeddings with rich metadata"""
    embedding: np.ndarray
    text: str
    metadata: EmbeddingMetadata
    model_name: str
    embedding_type: str  # 'semantic', 'entity', 'temporal', 'relationship'
    confidence_score: float = 1.0

class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Generate embeddings for input texts"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass

class SentenceTransformerModel(BaseEmbeddingModel):
    """Wrapper for SentenceTransformers models"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "auto"):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Loaded SentenceTransformer model: {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model {model_name}: {e}")
            # Fallback to a smaller model
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
            self.model_name = "all-MiniLM-L6-v2"
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Generate embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=kwargs.get('batch_size', 32),
            show_progress_bar=kwargs.get('show_progress', False),
            convert_to_numpy=True,
            normalize_embeddings=kwargs.get('normalize', True)
        )
        return embeddings
    
    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

class InstructorModel(BaseEmbeddingModel):
    """Advanced instruction-based embedding model for forensic queries"""
    
    def __init__(self, model_name: str = "hkunlp/instructor-xl", device: str = "auto"):
        if INSTRUCTOR is None:
            raise ImportError("InstructorEmbedding not available. Install with: pip install InstructorEmbedding")
        
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.model = INSTRUCTOR(model_name, device=self.device)
            logger.info(f"Loaded Instructor model: {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Instructor model: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], instructions: Optional[List[str]] = None, **kwargs) -> np.ndarray:
        """Generate instruction-based embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Default forensic instructions if none provided
        if instructions is None:
            instructions = ["Represent the forensic document for retrieval:"] * len(texts)
        elif len(instructions) == 1 and len(texts) > 1:
            instructions = instructions * len(texts)
        
        # Combine instructions with texts
        instruction_text_pairs = list(zip(instructions, texts))
        
        embeddings = self.model.encode(
            instruction_text_pairs,
            batch_size=kwargs.get('batch_size', 16),
            show_progress_bar=kwargs.get('show_progress', False)
        )
        return embeddings
    
    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """OpenAI embedding model wrapper"""
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model_name = model_name
        if api_key:
            openai.api_key = api_key
        
        # Model dimensions mapping
        self.dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Generate OpenAI embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            response = openai.embeddings.create(
                model=self.model_name,
                input=texts
            )
            embeddings = np.array([data.embedding for data in response.data])
            return embeddings
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    def get_dimension(self) -> int:
        return self.dimension_map.get(self.model_name, 1536)

class ForensicEmbeddingGenerator:
    """
    Advanced embedding generator specialized for forensic data analysis.
    
    Features:
    - Multi-model support with automatic fallbacks
    - Forensic-specific preprocessing and entity awareness
    - Context-aware embedding generation for different data types
    - Temporal and relationship embeddings
    - Confidence scoring and quality assessment
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, BaseEmbeddingModel] = {}
        self.entity_patterns = self._load_forensic_patterns()
        
        self._initialize_models()
    
    def _load_forensic_patterns(self) -> Dict[str, str]:
        """Load forensic entity patterns for enhanced processing"""
        return {
            "phone": r"(\+?\d{1,4}[\s-]?)?\(?\d{1,4}\)?[\s-]?\d{1,4}[\s-]?\d{1,9}",
            "crypto_btc": r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b",
            "crypto_eth": r"\b0x[a-fA-F0-9]{40}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "ip": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
            "imei": r"\b\d{15}\b"
        }
    
    def _initialize_models(self):
        """Initialize embedding models with fallbacks"""
        
        # Primary semantic model
        try:
            primary_model = self.config.get('primary_model', 'all-mpnet-base-v2')
            self.models['semantic'] = SentenceTransformerModel(primary_model)
            logger.info(f"Initialized primary semantic model: {primary_model}")
        except Exception as e:
            logger.warning(f"Failed to load primary model, using fallback: {e}")
            self.models['semantic'] = SentenceTransformerModel('all-MiniLM-L6-v2')
        
        # Forensic-specialized model
        try:
            forensic_model = self.config.get('forensic_model', 'sentence-transformers/all-distilroberta-v1')
            self.models['forensic'] = SentenceTransformerModel(forensic_model)
        except Exception as e:
            logger.warning(f"Failed to load forensic model: {e}")
            self.models['forensic'] = self.models['semantic']  # Use semantic as fallback
        
        # Entity-focused model - use same dimension as primary
        try:
            entity_model = self.config.get('entity_model', 'all-MiniLM-L6-v2')  # Changed to same as primary
            self.models['entity'] = SentenceTransformerModel(entity_model)
        except Exception as e:
            logger.warning(f"Failed to load entity model: {e}")
            self.models['entity'] = self.models['semantic']
        
        # Advanced instruction model (optional)
        if self.config.get('use_instructor', False):
            try:
                self.models['instructor'] = InstructorModel()
                logger.info("Initialized Instructor model for advanced queries")
            except Exception as e:
                logger.warning(f"Instructor model not available: {e}")
        
        # OpenAI model (optional)
        if self.config.get('openai_api_key'):
            try:
                self.models['openai'] = OpenAIEmbeddingModel(
                    model_name=self.config.get('openai_model', 'text-embedding-3-small'),
                    api_key=self.config['openai_api_key']
                )
                logger.info("Initialized OpenAI embedding model")
            except Exception as e:
                logger.warning(f"OpenAI model not available: {e}")
    
    def _preprocess_forensic_text(self, text: str, data_type: str) -> str:
        """Forensic-specific text preprocessing"""
        if not text:
            return ""
        
        # Preserve entity structure while cleaning
        processed_text = text.strip()
        
        # Add context markers for different data types
        if data_type == "conversation":
            processed_text = f"[CONVERSATION] {processed_text}"
        elif data_type == "contact":
            processed_text = f"[CONTACT] {processed_text}"
        elif data_type == "call_log":
            processed_text = f"[CALL] {processed_text}"
        elif data_type == "entity":
            processed_text = f"[ENTITY] {processed_text}"
        
        return processed_text
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract forensic entities from text"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))
        
        return entities
    
    def _calculate_confidence(self, text: str, embedding: np.ndarray, model_name: str) -> float:
        """Calculate confidence score for embedding quality"""
        base_confidence = 1.0
        
        # Penalize very short texts
        if len(text.split()) < 3:
            base_confidence *= 0.8
        
        # Reward presence of forensic entities
        entities = self._extract_entities(text)
        if entities:
            base_confidence *= 1.1
        
        # Check embedding norm (should be around 1 for normalized embeddings)
        norm = np.linalg.norm(embedding)
        if 0.8 <= norm <= 1.2:
            base_confidence *= 1.0
        else:
            base_confidence *= 0.9
        
        return min(base_confidence, 1.0)
    
    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        metadata_list: Optional[List[EmbeddingMetadata]] = None,
        embedding_type: str = "semantic",
        model_preference: Optional[str] = None
    ) -> List[ForensicEmbedding]:
        """
        Generate forensic embeddings with rich metadata
        
        Args:
            texts: Input texts to embed
            metadata_list: Optional metadata for each text
            embedding_type: Type of embedding ('semantic', 'entity', 'temporal', 'relationship')
            model_preference: Preferred model name
        
        Returns:
            List of ForensicEmbedding objects
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Default metadata if none provided
        if metadata_list is None:
            metadata_list = [
                EmbeddingMetadata(
                    chunk_id=f"chunk_{i}",
                    data_type="unknown"
                ) for i in range(len(texts))
            ]
        
        # Select appropriate model
        model_name = model_preference or self._select_best_model(embedding_type)
        model = self.models.get(model_name, self.models['semantic'])
        
        # Preprocess texts based on their data types
        processed_texts = [
            self._preprocess_forensic_text(text, metadata.data_type)
            for text, metadata in zip(texts, metadata_list)
        ]
        
        # Generate embeddings
        try:
            embeddings = model.encode(processed_texts, batch_size=32, normalize=True)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
        except Exception as e:
            logger.error(f"Embedding generation failed with {model_name}: {e}")
            # Fallback to semantic model
            model = self.models['semantic']
            model_name = 'semantic'
            embeddings = model.encode(processed_texts, batch_size=32, normalize=True)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
        
        # Create ForensicEmbedding objects
        forensic_embeddings = []
        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadata_list)):
            # Extract entities and update metadata
            entities = self._extract_entities(text)
            if entities and metadata.entities is None:
                metadata.entities = entities
            
            # Calculate confidence
            confidence = self._calculate_confidence(text, embedding, model_name)
            
            forensic_embedding = ForensicEmbedding(
                embedding=embedding,
                text=text,
                metadata=metadata,
                model_name=model_name,
                embedding_type=embedding_type,
                confidence_score=confidence
            )
            
            forensic_embeddings.append(forensic_embedding)
        
        logger.info(f"Generated {len(forensic_embeddings)} embeddings using {model_name}")
        return forensic_embeddings
    
    def _select_best_model(self, embedding_type: str) -> str:
        """Select the best model for the given embedding type"""
        model_mapping = {
            "semantic": "semantic",
            "entity": "entity",
            "temporal": "forensic",
            "relationship": "forensic",
            "conversation": "forensic"
        }
        
        preferred = model_mapping.get(embedding_type, "semantic")
        
        # Check if instructor model is available for complex queries
        if embedding_type in ["relationship", "temporal"] and "instructor" in self.models:
            return "instructor"
        
        # Check if preferred model is available
        if preferred in self.models:
            return preferred
        
        # Fallback to semantic
        return "semantic"
    
    def generate_query_embedding(
        self,
        query: str,
        query_type: str = "general",
        instruction: Optional[str] = None
    ) -> ForensicEmbedding:
        """
        Generate embedding for user queries with forensic context
        
        Args:
            query: User query text
            query_type: Type of query ('general', 'entity', 'temporal', 'relationship')
            instruction: Optional instruction for advanced models
        
        Returns:
            ForensicEmbedding for the query
        """
        # Create metadata for query
        metadata = EmbeddingMetadata(
            chunk_id="query",
            data_type="query",
            timestamp=datetime.now()
        )
        
        # Use instructor model if available and instruction provided
        if instruction and "instructor" in self.models:
            try:
                model = self.models["instructor"]
                embedding = model.encode([query], instructions=[instruction])[0]
                model_name = "instructor"
            except Exception as e:
                logger.warning(f"Instructor model failed, using fallback: {e}")
                model_name = self._select_best_model(query_type)
                model = self.models[model_name]
                embedding = model.encode([query])[0]
        else:
            # Use best model for query type
            model_name = self._select_best_model(query_type)
            model = self.models[model_name]
            
            # Add query context
            context_query = f"[QUERY] {query}"
            embedding = model.encode([context_query])[0]
        
        confidence = self._calculate_confidence(query, embedding, model_name)
        
        return ForensicEmbedding(
            embedding=embedding,
            text=query,
            metadata=metadata,
            model_name=model_name,
            embedding_type=query_type,
            confidence_score=confidence
        )
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about loaded models"""
        info = {}
        for name, model in self.models.items():
            info[name] = {
                "type": type(model).__name__,
                "dimension": model.get_dimension(),
                "model_name": getattr(model, 'model_name', 'unknown')
            }
        return info

# Utility functions for embedding analysis
def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings"""
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )

def batch_cosine_similarity(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray
) -> np.ndarray:
    """Calculate cosine similarity between query and corpus embeddings"""
    # Normalize embeddings
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    corpus_norm = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    
    # Calculate similarities
    similarities = np.dot(corpus_norm, query_norm)
    return similarities

def analyze_embedding_quality(embeddings: List[ForensicEmbedding]) -> Dict[str, Any]:
    """Analyze the quality of generated embeddings"""
    if not embeddings:
        return {"error": "No embeddings provided"}
    
    # Extract numerical embeddings
    embedding_matrix = np.array([fe.embedding for fe in embeddings])
    
    # Calculate statistics
    mean_confidence = np.mean([fe.confidence_score for fe in embeddings])
    embedding_norms = np.linalg.norm(embedding_matrix, axis=1)
    
    # Analyze diversity (average pairwise distance)
    n_embeddings = len(embeddings)
    if n_embeddings > 1:
        pairwise_similarities = []
        for i in range(n_embeddings):
            for j in range(i + 1, n_embeddings):
                sim = cosine_similarity(embedding_matrix[i], embedding_matrix[j])
                pairwise_similarities.append(sim)
        
        avg_similarity = np.mean(pairwise_similarities)
        diversity_score = 1 - avg_similarity  # Higher diversity = lower average similarity
    else:
        diversity_score = 1.0
    
    return {
        "total_embeddings": n_embeddings,
        "mean_confidence": float(mean_confidence),
        "embedding_dimension": embedding_matrix.shape[1],
        "mean_norm": float(np.mean(embedding_norms)),
        "std_norm": float(np.std(embedding_norms)),
        "diversity_score": float(diversity_score),
        "model_distribution": _get_model_distribution(embeddings)
    }

def _get_model_distribution(embeddings: List[ForensicEmbedding]) -> Dict[str, int]:
    """Get distribution of models used for embeddings"""
    distribution = {}
    for embedding in embeddings:
        model = embedding.model_name
        distribution[model] = distribution.get(model, 0) + 1
    return distribution