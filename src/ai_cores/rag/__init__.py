"""
Forensic RAG System - AI-Powered UFDR Analysis

This package provides a complete Retrieval-Augmented Generation (RAG) system
optimized for forensic data analysis and Universal Forensic Data Record (UFDR) processing.

Components:
- embeddings: Advanced embedding generation with forensic optimization
- vector_store: Hybrid vector storage with ChromaDB and FAISS support
- case_vector_store: Case-specific vector storage for isolated case management
- retrieval: Advanced retrieval with forensic context awareness
- generation: Multi-LLM response generation with source attribution
- rag_system: Complete RAG pipeline orchestration
- evaluation: Comprehensive evaluation metrics and testing
"""

from .embeddings import (
    ForensicEmbeddingGenerator,
    ForensicEmbedding,
    EmbeddingMetadata,
    cosine_similarity,
    batch_cosine_similarity,
    analyze_embedding_quality
)

# Data structures
from .data_structures import TextChunk, ProcessedDocument

from .vector_store import (
    create_vector_store,
    BaseVectorStore,
    ChromaVectorStore,
    FAISSVectorStore,
    HybridVectorStore,
    SearchResult,
    SearchFilter
)

# Case-specific vector storage
from .case_vector_store import CaseVectorStore, case_vector_store

from .retrieval import (
    AdvancedRetriever,
    RetrievalQuery,
    RetrievalContext,
    RankedResult,
    ForensicQueryProcessor
)

from .generation import (
    AdvancedResponseGenerator,
    GenerationContext,
    GeneratedResponse,
    ForensicPromptEngine,
    OpenAIProvider,
    GoogleProvider,
    HuggingFaceProvider
)

from .rag_system import (
    ForensicRAGSystem,
    RAGQuery,
    RAGResponse,
    create_forensic_rag_system
)

from .evaluation import (
    RAGSystemEvaluator,
    RetrievalEvaluator,
    GenerationEvaluator,
    RetrievalGroundTruth,
    GenerationGroundTruth,
    RetrievalMetrics,
    GenerationMetrics,
    EndToEndMetrics,
    create_sample_ground_truth
)

__version__ = "1.0.0"
__author__ = "Forensic AI Team"
__description__ = "Advanced RAG system for forensic data analysis"

# Default configuration template
DEFAULT_CONFIG = {
    "embedding_config": {
        "primary_model": "all-mpnet-base-v2",
        "forensic_model": "sentence-transformers/all-distilroberta-v1",
        "entity_model": "sentence-transformers/paraphrase-albert-small-v2",
        "use_instructor": False,
        "openai_api_key": None,
        "openai_model": "text-embedding-3-small"
    },
    "vector_config": {
        "store_type": "auto",  # 'auto', 'chroma', 'faiss', 'hybrid'
        "persist_directory": "./data/vector_db",
        "collection_name": "ufdr_embeddings",
        "use_chroma": True,
        "use_faiss": False,
        "chroma_config": {
            "persist_directory": "./data/chroma_db",
            "collection_name": "ufdr_embeddings"
        },
        "faiss_config": {
            "index_type": "flat",  # 'flat', 'ivf', 'hnsw'
            "persist_directory": "./data/faiss_db"
        }
    },
    "retrieval_config": {
        "ranking_weights": {
            "semantic_similarity": 0.4,
            "keyword_match": 0.2,
            "entity_match": 0.2,
            "temporal_relevance": 0.1,
            "confidence_score": 0.1
        },
        "tfidf_max_features": 5000,
        "max_similar_results": 3,
        "spacy_model": "en_core_web_sm"
    },
    "generation_config": {
        "primary_provider": "openai",
        "openai_config": {
            "api_key": None,
            "model": "gpt-3.5-turbo",
            "max_tokens": 1000,
            "temperature": 0.3
        },
        "google_config": {
            "api_key": None,
            "model": "gemini-pro",
            "max_tokens": 1000,
            "temperature": 0.3
        },
        "huggingface_config": {
            "model": "microsoft/DialoGPT-small",
            "device": "auto",
            "max_length": 1000,
            "temperature": 0.3
        }
    },
    "chunking_config": {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "max_chunk_size": 1000,
        "min_chunk_size": 100
    }
}

def get_default_config():
    """Get the default configuration for the RAG system"""
    return DEFAULT_CONFIG.copy()

__all__ = [
    # Core system
    "ForensicRAGSystem",
    "create_forensic_rag_system",
    "RAGQuery",
    "RAGResponse",
    
    # Embeddings
    "ForensicEmbeddingGenerator",
    "ForensicEmbedding",
    "EmbeddingMetadata",
    "cosine_similarity",
    "batch_cosine_similarity",
    "analyze_embedding_quality",
    
    # Vector store
    "create_vector_store",
    "BaseVectorStore",
    "ChromaVectorStore",
    "FAISSVectorStore",
    "HybridVectorStore",
    "SearchResult",
    "SearchFilter",
    
    # Retrieval
    "AdvancedRetriever",
    "RetrievalQuery",
    "RetrievalContext",
    "RankedResult",
    "ForensicQueryProcessor",
    
    # Generation
    "AdvancedResponseGenerator",
    "GenerationContext",
    "GeneratedResponse",
    "ForensicPromptEngine",
    "OpenAIProvider",
    "GoogleProvider",
    "HuggingFaceProvider",
    
    # Evaluation
    "RAGSystemEvaluator",
    "RetrievalEvaluator",
    "GenerationEvaluator",
    "RetrievalGroundTruth",
    "GenerationGroundTruth",
    "RetrievalMetrics",
    "GenerationMetrics",
    "EndToEndMetrics",
    "create_sample_ground_truth",
    
    # Configuration
    "DEFAULT_CONFIG",
    "get_default_config"
]