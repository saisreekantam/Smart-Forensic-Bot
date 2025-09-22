"""
Vector store implementation for forensic knowledge retrieval
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime

from ..core.config import config

class ForensicVectorStore:
    """Vector store for forensic knowledge and case data"""
    
    def __init__(self, embedding_model: str = None):
        self.embedding_model = SentenceTransformer(
            embedding_model or config.embedding_model
        )
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadata = []
        
        # Load existing index if available
        self.index_path = config.vectordb_path / "forensic_index.faiss"
        self.docs_path = config.vectordb_path / "documents.pkl"
        self.metadata_path = config.vectordb_path / "metadata.json"
        
        self.load_index()
    
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """
        Add documents to the vector store
        
        Args:
            documents: List of text documents to add
            metadata: Optional metadata for each document
        """
        if metadata is None:
            metadata = [{"timestamp": datetime.now().isoformat()} for _ in documents]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents)
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        # Save the updated index
        self.save_index()
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of tuples (document, score, metadata)
        """
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS index
        scores, indices = self.index.search(
            np.array(query_embedding).astype('float32'), k
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((
                    self.documents[idx],
                    float(score),
                    self.metadata[idx] if idx < len(self.metadata) else {}
                ))
        
        return results
    
    def add_forensic_case(self, case_data: Dict[str, Any]):
        """
        Add forensic case data to the knowledge base
        
        Args:
            case_data: Dictionary containing case information
        """
        # Extract text content from case data
        text_content = []
        metadata = []
        
        # Case summary
        if 'summary' in case_data:
            text_content.append(f"Case Summary: {case_data['summary']}")
            metadata.append({
                'type': 'case_summary',
                'case_id': case_data.get('case_id', 'unknown'),
                'timestamp': datetime.now().isoformat()
            })
        
        # Evidence descriptions
        if 'evidence' in case_data:
            for i, evidence in enumerate(case_data['evidence']):
                if isinstance(evidence, dict) and 'description' in evidence:
                    text_content.append(f"Evidence {i+1}: {evidence['description']}")
                    metadata.append({
                        'type': 'evidence',
                        'case_id': case_data.get('case_id', 'unknown'),
                        'evidence_id': evidence.get('id', f'evidence_{i+1}'),
                        'timestamp': datetime.now().isoformat()
                    })
        
        # Analysis findings
        if 'findings' in case_data:
            for i, finding in enumerate(case_data['findings']):
                if isinstance(finding, str):
                    text_content.append(f"Finding: {finding}")
                    metadata.append({
                        'type': 'finding',
                        'case_id': case_data.get('case_id', 'unknown'),
                        'finding_id': f'finding_{i+1}',
                        'timestamp': datetime.now().isoformat()
                    })
        
        if text_content:
            self.add_documents(text_content, metadata)
    
    def add_forensic_knowledge(self, knowledge_base: List[Dict[str, Any]]):
        """
        Add general forensic knowledge to the vector store
        
        Args:
            knowledge_base: List of knowledge items
        """
        documents = []
        metadata = []
        
        for item in knowledge_base:
            if 'content' in item:
                documents.append(item['content'])
                metadata.append({
                    'type': 'knowledge',
                    'category': item.get('category', 'general'),
                    'source': item.get('source', 'unknown'),
                    'timestamp': datetime.now().isoformat()
                })
        
        if documents:
            self.add_documents(documents, metadata)
    
    def save_index(self):
        """Save the FAISS index and associated data"""
        config.vectordb_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))
        
        # Save documents
        with open(self.docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def load_index(self):
        """Load the FAISS index and associated data"""
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                
                if self.docs_path.exists():
                    with open(self.docs_path, 'rb') as f:
                        self.documents = pickle.load(f)
                
                if self.metadata_path.exists():
                    with open(self.metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                        
            except Exception as e:
                print(f"Error loading index: {e}")
                # Reset to empty index
                self.index = faiss.IndexFlatL2(self.dimension)
                self.documents = []
                self.metadata = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal,
            'dimension': self.dimension,
            'types': self._get_type_distribution()
        }
    
    def _get_type_distribution(self) -> Dict[str, int]:
        """Get distribution of document types"""
        type_counts = {}
        for meta in self.metadata:
            doc_type = meta.get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        return type_counts