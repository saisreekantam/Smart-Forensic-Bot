"""
Advanced Retrieval System for Forensic RAG

This module provides sophisticated retrieval capabilities with multi-stage ranking,
relevance scoring, and forensic-specific context preservation.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from collections import defaultdict
import math

# NLP and similarity libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Local imports
from .embeddings import ForensicEmbedding, ForensicEmbeddingGenerator
from .vector_store import BaseVectorStore, SearchResult, SearchFilter

logger = logging.getLogger(__name__)

@dataclass
class RetrievalQuery:
    """Enhanced query object for forensic retrieval"""
    text: str
    query_type: str = "general"  # 'general', 'entity', 'temporal', 'relationship', 'factual'
    filters: Optional[SearchFilter] = None
    context: Optional[str] = None
    intent: Optional[str] = None
    entities_of_interest: Optional[List[str]] = None
    time_focus: Optional[datetime] = None
    max_results: int = 10
    min_similarity: float = 0.1
    boost_factors: Optional[Dict[str, float]] = None

@dataclass
class RetrievalContext:
    """Context for multi-turn conversations and related queries"""
    conversation_history: List[str]
    previous_results: List[SearchResult]
    user_feedback: Optional[Dict[str, Any]] = None
    domain_focus: Optional[str] = None

@dataclass
class RankedResult:
    """Enhanced search result with detailed ranking information"""
    result: SearchResult
    ranking_score: float
    ranking_factors: Dict[str, float]
    relevance_explanation: str
    contextual_importance: float = 1.0

class ForensicQueryProcessor:
    """Advanced query processing for forensic contexts"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.entity_patterns = self._load_entity_patterns()
        
        # Initialize NLP components if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                model_name = config.get('spacy_model', 'en_core_web_sm')
                self.nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
    
    def _load_entity_patterns(self) -> Dict[str, str]:
        """Load forensic entity patterns"""
        return {
            "phone": r"(\+?\d{1,4}[\s-]?)?\(?\d{1,4}\)?[\s-]?\d{1,4}[\s-]?\d{1,9}",
            "crypto_btc": r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b",
            "crypto_eth": r"\b0x[a-fA-F0-9]{40}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "ip": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
            "imei": r"\b\d{15}\b",
            "coordinates": r"\b\d{1,3}\.\d{1,6}°?\s*[NS],?\s*\d{1,3}\.\d{1,6}°?\s*[EW]\b"
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to extract intent, entities, and context"""
        analysis = {
            "entities": {},
            "intent": "general",
            "query_type": "general",
            "temporal_references": [],
            "urgency_level": "normal",
            "specificity_score": 0.0
        }
        
        query_lower = query.lower()
        
        # Extract entities using regex patterns
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                analysis["entities"][entity_type] = matches
        
        # Extract entities using spaCy if available
        if self.nlp:
            try:
                doc = self.nlp(query)
                spacy_entities = {}
                for ent in doc.ents:
                    entity_type = ent.label_.lower()
                    if entity_type not in spacy_entities:
                        spacy_entities[entity_type] = []
                    spacy_entities[entity_type].append(ent.text)
                
                # Merge with regex entities
                for entity_type, entities in spacy_entities.items():
                    if entity_type in analysis["entities"]:
                        analysis["entities"][entity_type].extend(entities)
                    else:
                        analysis["entities"][entity_type] = entities
                
                # Analyze temporal references
                temporal_entities = [ent for ent in doc.ents if ent.label_ in ["DATE", "TIME"]]
                analysis["temporal_references"] = [ent.text for ent in temporal_entities]
                
            except Exception as e:
                logger.warning(f"spaCy analysis failed: {e}")
        
        # Determine query intent and type
        analysis.update(self._classify_query_intent(query_lower))
        
        # Calculate specificity score
        analysis["specificity_score"] = self._calculate_specificity(query, analysis)
        
        return analysis
    
    def _classify_query_intent(self, query_lower: str) -> Dict[str, str]:
        """Classify query intent and type"""
        
        # Intent keywords
        intents = {
            "search": ["find", "search", "look for", "locate", "show me"],
            "analysis": ["analyze", "examine", "investigate", "review", "assess"],
            "comparison": ["compare", "contrast", "difference", "similar", "match"],
            "timeline": ["when", "timeline", "chronology", "sequence", "order"],
            "relationship": ["connection", "relationship", "link", "associate", "related"],
            "summary": ["summarize", "overview", "brief", "summary", "outline"]
        }
        
        # Query type keywords
        query_types = {
            "entity": ["who", "what", "where", "phone", "email", "address", "crypto"],
            "temporal": ["when", "time", "date", "timeline", "before", "after", "during"],
            "relationship": ["connection", "relationship", "between", "with", "related"],
            "factual": ["how", "why", "what happened", "details", "information"]
        }
        
        detected_intent = "general"
        detected_type = "general"
        
        # Check intents
        for intent, keywords in intents.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intent = intent
                break
        
        # Check query types
        for qtype, keywords in query_types.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_type = qtype
                break
        
        return {"intent": detected_intent, "query_type": detected_type}
    
    def _calculate_specificity(self, query: str, analysis: Dict[str, Any]) -> float:
        """Calculate how specific the query is (0.0 to 1.0)"""
        score = 0.0
        
        # Base score from query length
        word_count = len(query.split())
        if word_count > 5:
            score += 0.2
        if word_count > 10:
            score += 0.2
        
        # Bonus for entities
        entity_count = sum(len(entities) for entities in analysis["entities"].values())
        score += min(entity_count * 0.15, 0.4)
        
        # Bonus for temporal references
        if analysis["temporal_references"]:
            score += 0.2
        
        # Bonus for specific forensic terms
        forensic_terms = [
            "case", "investigation", "evidence", "suspect", "communication",
            "transaction", "device", "location", "contact", "message"
        ]
        forensic_count = sum(1 for term in forensic_terms if term in query.lower())
        score += min(forensic_count * 0.1, 0.2)
        
        return min(score, 1.0)

class AdvancedRetriever:
    """
    Advanced retrieval system for forensic data with multi-stage ranking
    
    Features:
    - Multi-stage retrieval (embedding + keyword + entity matching)
    - Advanced ranking with multiple factors
    - Context-aware retrieval for conversations
    - Temporal and entity-based boosting
    - Relevance feedback incorporation
    """
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_generator: ForensicEmbeddingGenerator,
        config: Dict[str, Any]
    ):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.config = config
        self.query_processor = ForensicQueryProcessor(config)
        
        # Initialize TF-IDF for keyword matching if available
        self.tfidf_vectorizer = None
        self.tfidf_corpus = None
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=config.get('tfidf_max_features', 5000),
                stop_words='english',
                ngram_range=(1, 2)
            )
        
        # Ranking weights
        self.ranking_weights = config.get('ranking_weights', {
            'semantic_similarity': 0.4,
            'keyword_match': 0.2,
            'entity_match': 0.2,
            'temporal_relevance': 0.1,
            'confidence_score': 0.1
        })
        
        logger.info("Initialized AdvancedRetriever")
    
    def retrieve(
        self,
        query: RetrievalQuery,
        context: Optional[RetrievalContext] = None
    ) -> List[RankedResult]:
        """
        Main retrieval method with multi-stage ranking
        
        Args:
            query: Enhanced query object
            context: Optional context from previous interactions
        
        Returns:
            List of ranked results with detailed scoring
        """
        
        # Step 1: Analyze query
        query_analysis = self.query_processor.analyze_query(query.text)
        logger.info(f"Query analysis: {query_analysis['intent']}/{query_analysis['query_type']}")
        
        # Step 2: Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(
            query.text,
            query_type=query.query_type,
            instruction=self._get_instruction_for_query_type(query.query_type)
        )
        
        # Step 3: Initial vector search
        initial_results = self._vector_search(query_embedding, query, query_analysis)
        
        if not initial_results:
            logger.warning("No initial results found")
            return []
        
        # Step 4: Multi-stage ranking
        ranked_results = self._multi_stage_ranking(
            initial_results,
            query,
            query_analysis,
            context
        )
        
        # Step 5: Apply final filters and limits
        final_results = self._apply_final_filters(ranked_results, query)
        
        logger.info(f"Retrieved {len(final_results)} results for query: {query.text[:50]}...")
        return final_results
    
    def _get_instruction_for_query_type(self, query_type: str) -> str:
        """Get instruction prompt for different query types"""
        instructions = {
            "entity": "Represent the forensic document for entity-based retrieval:",
            "temporal": "Represent the forensic document for temporal analysis:",
            "relationship": "Represent the forensic document for relationship analysis:",
            "factual": "Represent the forensic document for factual information retrieval:",
            "general": "Represent the forensic document for retrieval:"
        }
        return instructions.get(query_type, instructions["general"])
    
    def _vector_search(
        self,
        query_embedding: ForensicEmbedding,
        query: RetrievalQuery,
        query_analysis: Dict[str, Any]
    ) -> List[SearchResult]:
        """Perform initial vector-based search"""
        
        # Calculate search multiplier based on query specificity
        specificity = query_analysis.get("specificity_score", 0.5)
        search_multiplier = 2 if specificity < 0.3 else 1.5 if specificity < 0.6 else 1.2
        
        search_k = min(int(query.max_results * search_multiplier), 100)
        
        # Perform vector search
        results = self.vector_store.search(
            query_embedding.embedding,
            top_k=search_k,
            filters=query.filters
        )
        
        return results
    
    def _multi_stage_ranking(
        self,
        initial_results: List[SearchResult],
        query: RetrievalQuery,
        query_analysis: Dict[str, Any],
        context: Optional[RetrievalContext]
    ) -> List[RankedResult]:
        """Apply multi-stage ranking to initial results"""
        
        ranked_results = []
        
        for result in initial_results:
            # Calculate individual ranking factors
            factors = self._calculate_ranking_factors(
                result, query, query_analysis, context
            )
            
            # Calculate overall ranking score
            ranking_score = self._calculate_overall_score(factors)
            
            # Generate relevance explanation
            explanation = self._generate_relevance_explanation(factors, result)
            
            # Calculate contextual importance
            contextual_importance = self._calculate_contextual_importance(
                result, query, context
            )
            
            ranked_result = RankedResult(
                result=result,
                ranking_score=ranking_score,
                ranking_factors=factors,
                relevance_explanation=explanation,
                contextual_importance=contextual_importance
            )
            
            ranked_results.append(ranked_result)
        
        # Sort by ranking score
        ranked_results.sort(key=lambda x: x.ranking_score, reverse=True)
        
        return ranked_results
    
    def _calculate_ranking_factors(
        self,
        result: SearchResult,
        query: RetrievalQuery,
        query_analysis: Dict[str, Any],
        context: Optional[RetrievalContext]
    ) -> Dict[str, float]:
        """Calculate individual ranking factors"""
        
        factors = {}
        
        # 1. Semantic similarity (from vector search)
        factors['semantic_similarity'] = result.similarity_score
        
        # 2. Keyword matching
        factors['keyword_match'] = self._calculate_keyword_match(
            query.text, result.text
        )
        
        # 3. Entity matching
        factors['entity_match'] = self._calculate_entity_match(
            query_analysis.get("entities", {}),
            result.metadata.entities or {}
        )
        
        # 4. Temporal relevance
        factors['temporal_relevance'] = self._calculate_temporal_relevance(
            query_analysis.get("temporal_references", []),
            query.time_focus,
            result.metadata.timestamp
        )
        
        # 5. Confidence score
        factors['confidence_score'] = getattr(result, 'confidence_score', 1.0)
        
        # 6. Data type relevance
        factors['data_type_relevance'] = self._calculate_data_type_relevance(
            query.query_type, result.metadata.data_type
        )
        
        # 7. Context relevance (if context provided)
        if context:
            factors['context_relevance'] = self._calculate_context_relevance(
                result, context
            )
        else:
            factors['context_relevance'] = 0.5  # Neutral
        
        # 8. Apply boost factors if specified
        if query.boost_factors:
            for factor_name, boost in query.boost_factors.items():
                if factor_name in factors:
                    factors[factor_name] *= boost
        
        return factors
    
    def _calculate_keyword_match(self, query_text: str, result_text: str) -> float:
        """Calculate keyword matching score"""
        if not SKLEARN_AVAILABLE:
            # Simple fallback
            query_words = set(query_text.lower().split())
            result_words = set(result_text.lower().split())
            intersection = query_words.intersection(result_words)
            return len(intersection) / max(len(query_words), 1)
        
        try:
            # Use TF-IDF for more sophisticated matching
            texts = [query_text, result_text]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            similarity = sklearn_cosine_similarity(
                tfidf_matrix[0:1], tfidf_matrix[1:2]
            )[0][0]
            return float(similarity)
        except Exception:
            # Fallback to simple matching
            query_words = set(query_text.lower().split())
            result_words = set(result_text.lower().split())
            intersection = query_words.intersection(result_words)
            return len(intersection) / max(len(query_words), 1)
    
    def _calculate_entity_match(
        self,
        query_entities: Dict[str, List[str]],
        result_entities: Dict[str, List[str]]
    ) -> float:
        """Calculate entity matching score"""
        if not query_entities or not result_entities:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        # Entity type weights
        entity_weights = {
            'phone': 1.0,
            'crypto_btc': 1.0,
            'crypto_eth': 1.0,
            'email': 0.8,
            'ip': 0.6,
            'person': 0.9,
            'org': 0.7
        }
        
        for entity_type, query_values in query_entities.items():
            if entity_type in result_entities:
                weight = entity_weights.get(entity_type, 0.5)
                
                # Calculate overlap
                query_set = set(query_values)
                result_set = set(result_entities[entity_type])
                overlap = len(query_set.intersection(result_set))
                
                if overlap > 0:
                    # Jaccard similarity
                    jaccard = overlap / len(query_set.union(result_set))
                    total_score += jaccard * weight
                
                total_weight += weight
        
        return total_score / max(total_weight, 1.0)
    
    def _calculate_temporal_relevance(
        self,
        query_temporal_refs: List[str],
        query_time_focus: Optional[datetime],
        result_timestamp: Optional[datetime]
    ) -> float:
        """Calculate temporal relevance score"""
        if not result_timestamp:
            return 0.5  # Neutral for missing timestamps
        
        score = 0.5  # Base score
        
        # If query has specific time focus
        if query_time_focus:
            time_diff = abs((result_timestamp - query_time_focus).total_seconds())
            
            # Score based on time proximity (within 24 hours = 1.0, decay over time)
            if time_diff <= 24 * 3600:  # Within 24 hours
                score = 1.0
            elif time_diff <= 7 * 24 * 3600:  # Within 1 week
                score = 0.8
            elif time_diff <= 30 * 24 * 3600:  # Within 1 month
                score = 0.6
            else:
                score = 0.3
        
        # Boost for recent data (within last 30 days of extraction)
        if result_timestamp:
            # Ensure both timestamps are timezone-naive for comparison
            now = datetime.now()
            if hasattr(result_timestamp, 'tzinfo') and result_timestamp.tzinfo is not None:
                # Convert timezone-aware to naive using local timezone
                result_timestamp = result_timestamp.replace(tzinfo=None)
            
            days_old = (now - result_timestamp).days
            if days_old <= 30:
                score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_data_type_relevance(
        self,
        query_type: str,
        result_data_type: str
    ) -> float:
        """Calculate relevance based on data types"""
        
        # Data type affinity matrix
        affinity_matrix = {
            "entity": {
                "contact": 1.0,
                "entity": 1.0,
                "conversation": 0.7,
                "call_log": 0.8,
                "document": 0.5
            },
            "temporal": {
                "call_log": 1.0,
                "conversation": 1.0,
                "document": 0.8,
                "contact": 0.3,
                "entity": 0.2
            },
            "relationship": {
                "conversation": 1.0,
                "call_log": 0.9,
                "contact": 0.8,
                "document": 0.6,
                "entity": 0.4
            },
            "factual": {
                "document": 1.0,
                "conversation": 0.8,
                "contact": 0.6,
                "call_log": 0.5,
                "entity": 0.3
            },
            "general": {
                "conversation": 0.9,
                "document": 0.9,
                "call_log": 0.8,
                "contact": 0.7,
                "entity": 0.6
            }
        }
        
        return affinity_matrix.get(query_type, {}).get(result_data_type, 0.5)
    
    def _calculate_context_relevance(
        self,
        result: SearchResult,
        context: RetrievalContext
    ) -> float:
        """Calculate relevance based on conversation context"""
        score = 0.5  # Base score
        
        # Check if result participants match context participants
        if (result.metadata.participants and 
            hasattr(context, 'participants') and context.participants):
            
            result_participants = set(result.metadata.participants)
            context_participants = set(context.participants)
            overlap = len(result_participants.intersection(context_participants))
            
            if overlap > 0:
                score += 0.3
        
        # Check if similar results were previously relevant
        if context.previous_results:
            for prev_result in context.previous_results[-5:]:  # Check last 5
                if (result.metadata.data_type == prev_result.metadata.data_type or
                    result.metadata.case_id == prev_result.metadata.case_id):
                    score += 0.2
                    break
        
        return min(score, 1.0)
    
    def _calculate_overall_score(self, factors: Dict[str, float]) -> float:
        """Calculate weighted overall ranking score"""
        score = 0.0
        
        for factor_name, factor_value in factors.items():
            weight = self.ranking_weights.get(factor_name, 0.0)
            score += factor_value * weight
        
        return score
    
    def _generate_relevance_explanation(
        self,
        factors: Dict[str, float],
        result: SearchResult
    ) -> str:
        """Generate human-readable explanation for relevance"""
        
        explanations = []
        
        # Top factors
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
        
        for factor_name, factor_value in sorted_factors[:3]:
            if factor_value > 0.7:
                if factor_name == 'semantic_similarity':
                    explanations.append("high semantic similarity")
                elif factor_name == 'keyword_match':
                    explanations.append("strong keyword match")
                elif factor_name == 'entity_match':
                    explanations.append("matching entities")
                elif factor_name == 'temporal_relevance':
                    explanations.append("temporal relevance")
                elif factor_name == 'data_type_relevance':
                    explanations.append(f"relevant data type ({result.metadata.data_type})")
        
        if not explanations:
            explanations.append("general relevance")
        
        return f"Relevant due to: {', '.join(explanations)}"
    
    def _calculate_contextual_importance(
        self,
        result: SearchResult,
        query: RetrievalQuery,
        context: Optional[RetrievalContext]
    ) -> float:
        """Calculate contextual importance for result diversification"""
        
        importance = 1.0
        
        # Boost based on data type variety
        if hasattr(query, '_seen_data_types'):
            if result.metadata.data_type not in query._seen_data_types:
                importance += 0.2
                query._seen_data_types.add(result.metadata.data_type)
        else:
            query._seen_data_types = {result.metadata.data_type}
        
        # Boost based on sensitivity level
        sensitivity_boost = {
            'critical': 1.3,
            'high': 1.2,
            'standard': 1.0,
            'low': 0.9
        }
        importance *= sensitivity_boost.get(result.metadata.sensitivity_level, 1.0)
        
        return importance
    
    def _apply_final_filters(
        self,
        ranked_results: List[RankedResult],
        query: RetrievalQuery
    ) -> List[RankedResult]:
        """Apply final filtering and limits"""
        
        # Filter by minimum similarity
        filtered_results = [
            r for r in ranked_results 
            if r.result.similarity_score >= query.min_similarity
        ]
        
        # Apply diversity filtering to avoid too many similar results
        diverse_results = self._apply_diversity_filtering(
            filtered_results,
            max_similar=self.config.get('max_similar_results', 3)
        )
        
        # Limit to max results
        final_results = diverse_results[:query.max_results]
        
        return final_results
    
    def _apply_diversity_filtering(
        self,
        results: List[RankedResult],
        max_similar: int = 3
    ) -> List[RankedResult]:
        """Apply diversity filtering to avoid redundant results"""
        
        if len(results) <= max_similar:
            return results
        
        diverse_results = []
        seen_signatures = []
        
        for result in results:
            # Create signature for similarity checking
            signature = (
                result.result.metadata.data_type,
                result.result.metadata.case_id,
                result.result.metadata.source_file
            )
            
            # Count similar signatures
            similar_count = sum(1 for sig in seen_signatures if sig == signature)
            
            if similar_count < max_similar:
                diverse_results.append(result)
                seen_signatures.append(signature)
        
        return diverse_results

# Utility functions for retrieval analysis
def analyze_retrieval_performance(
    queries: List[str],
    retriever: AdvancedRetriever,
    ground_truth: Optional[List[List[str]]] = None
) -> Dict[str, Any]:
    """Analyze retrieval performance across multiple queries"""
    
    results = {
        "total_queries": len(queries),
        "avg_results_per_query": 0.0,
        "avg_ranking_score": 0.0,
        "coverage_by_data_type": defaultdict(int),
        "query_type_distribution": defaultdict(int)
    }
    
    total_results = 0
    total_ranking_score = 0.0
    
    for i, query_text in enumerate(queries):
        query = RetrievalQuery(text=query_text)
        retrieved_results = retriever.retrieve(query)
        
        total_results += len(retrieved_results)
        
        for ranked_result in retrieved_results:
            total_ranking_score += ranked_result.ranking_score
            results["coverage_by_data_type"][ranked_result.result.metadata.data_type] += 1
        
        # Analyze query
        query_analysis = retriever.query_processor.analyze_query(query_text)
        results["query_type_distribution"][query_analysis["query_type"]] += 1
    
    if total_results > 0:
        results["avg_results_per_query"] = total_results / len(queries)
        results["avg_ranking_score"] = total_ranking_score / total_results
    
    return dict(results)