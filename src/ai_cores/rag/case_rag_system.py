"""
Case-Aware RAG System for Forensic Investigation

This module provides a specialized RAG system that works within case contexts,
integrating with the case management system and providing intelligent responses
for forensic investigation queries.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

# Local imports
from .rag_system import ForensicRAGSystem
from .case_vector_store import CaseVectorStore
from .embeddings import ForensicEmbeddingGenerator
from src.database.models import Case, Evidence, EvidenceType
from src.case_management.case_manager import CaseManager

logger = logging.getLogger(__name__)

class CaseRAGSystem:
    """
    Case-aware RAG system for forensic investigation
    
    This system provides intelligent question-answering capabilities
    specifically for forensic cases, with case-isolated vector searches
    and forensic-optimized response generation.
    """
    
    def __init__(self, 
                 case_manager: CaseManager,
                 vector_store: CaseVectorStore,
                 embedding_generator: Optional[ForensicEmbeddingGenerator] = None):
        self.case_manager = case_manager
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator or ForensicEmbeddingGenerator()
        
        # Initialize LLM clients for different models
        self.llm_clients = self._initialize_llm_clients()
        
        logger.info("Initialized CaseRAGSystem")
    
    def _initialize_llm_clients(self) -> Dict[str, Any]:
        """Initialize various LLM clients"""
        clients = {}
        
        try:
            # Google Gemini 2.5 Pro (primary)
            import google.generativeai as genai
            import os
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                clients["gemini"] = genai.GenerativeModel('gemini-2.0-flash-exp')
                logger.info("Initialized Gemini client")
        except Exception as e:
            logger.warning(f"Could not initialize Gemini: {str(e)}")
        
        try:
            # OpenAI GPT (backup)
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                clients["openai"] = openai.OpenAI(api_key=api_key)
                logger.info("Initialized OpenAI client")
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI: {str(e)}")
        
        return clients
    
    async def query_case(self, 
                        case_id: str, 
                        query: str, 
                        context: Optional[Dict[str, Any]] = None,
                        top_k: int = 10) -> Dict[str, Any]:
        """
        Query a specific case using RAG
        
        Args:
            case_id: Case identifier
            query: User's question
            context: Additional context (conversation history, filters, etc.)
            top_k: Number of top results to retrieve
            
        Returns:
            Comprehensive response with answer, sources, and metadata
        """
        try:
            # Get case information
            case = self.case_manager.get_case(case_id)
            if not case:
                return {
                    "status": "error",
                    "message": f"Case {case_id} not found"
                }
            
            # Check if case has processed evidence
            if case.processed_evidence_count == 0:
                return {
                    "status": "no_data",
                    "message": "No processed evidence available for this case",
                    "suggestion": "Please upload and process evidence files first"
                }
            
            # Step 1: Generate query embedding
            query_embedding = self._generate_query_embedding(query, context)
            
            # Step 2: Search case-specific vector store
            search_results = self.vector_store.search_case(
                case_id=case_id,
                collection_name=case.embedding_collection_name,
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            # Step 3: Analyze and rank results
            ranked_results = self._rank_results_for_context(search_results, query, context)
            
            # Step 4: Generate response using LLM
            response = await self._generate_forensic_response(
                case=case,
                query=query,
                search_results=ranked_results,
                context=context
            )
            
            # Step 5: Add case-specific metadata
            response.update({
                "case_id": case_id,
                "case_number": case.case_number,
                "case_title": case.title,
                "search_metadata": {
                    "total_results": len(search_results),
                    "used_results": len(ranked_results),
                    "collection_name": case.embedding_collection_name,
                    "query_timestamp": datetime.now().isoformat()
                }
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error querying case {case_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}"
            }
    
    def _generate_query_embedding(self, query: str, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Generate embedding for the user query"""
        
        # Enhance query with context if available
        enhanced_query = query
        if context:
            if context.get("evidence_types"):
                enhanced_query += f" [Evidence types: {', '.join(context['evidence_types'])}]"
            if context.get("time_range"):
                enhanced_query += f" [Time range: {context['time_range']}]"
            if context.get("participants"):
                enhanced_query += f" [Participants: {', '.join(context['participants'])}]"
        
        # Generate embedding using the forensic embedding generator
        embedding = self.embedding_generator.generate_embedding(
            text=enhanced_query,
            metadata={"query_type": "case_investigation", "context": context or {}}
        )
        
        return embedding.vector
    
    def _rank_results_for_context(self, 
                                search_results: List[Any], 
                                query: str, 
                                context: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Re-rank search results based on forensic investigation context
        """
        if not search_results:
            return []
        
        # Apply forensic-specific ranking factors
        for result in search_results:
            score = result.similarity_score
            
            # Boost results based on evidence type relevance
            if self._is_evidence_type_relevant(result, query):
                score *= 1.2
            
            # Boost results with specific forensic entities
            if self._contains_forensic_entities(result, query):
                score *= 1.15
            
            # Boost recent or time-relevant results
            if self._is_temporally_relevant(result, context):
                score *= 1.1
            
            # Boost results with multiple participants (for communication evidence)
            if self._has_multiple_participants(result):
                score *= 1.05
            
            result.forensic_score = score
        
        # Sort by forensic score and return top results
        ranked_results = sorted(search_results, key=lambda x: x.forensic_score, reverse=True)
        return ranked_results[:8]  # Limit to top 8 for response generation
    
    def _is_evidence_type_relevant(self, result: Any, query: str) -> bool:
        """Check if evidence type matches query intent"""
        query_lower = query.lower()
        evidence_type = result.metadata.data_type.lower()
        
        # Define relevance mappings
        relevance_map = {
            "chat": ["message", "chat", "conversation", "talk", "text", "whatsapp", "telegram"],
            "call_log": ["call", "phone", "dial", "ring", "duration", "contact"],
            "contact": ["contact", "address", "phone book", "friend", "connection"],
            "document": ["document", "report", "file", "text", "note"]
        }
        
        relevant_terms = relevance_map.get(evidence_type, [])
        return any(term in query_lower for term in relevant_terms)
    
    def _contains_forensic_entities(self, result: Any, query: str) -> bool:
        """Check if result contains entities mentioned in query"""
        query_lower = query.lower()
        entities = result.metadata.entities or {}
        
        # Check if any entity values appear in the query
        for entity_type, entity_list in entities.items():
            if isinstance(entity_list, list):
                for entity in entity_list:
                    if entity.lower() in query_lower:
                        return True
        
        return False
    
    def _is_temporally_relevant(self, result: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check temporal relevance"""
        if not context or not context.get("time_range"):
            return False
        
        # Implement temporal relevance logic
        return True  # Simplified for now
    
    def _has_multiple_participants(self, result: Any) -> bool:
        """Check if result involves multiple participants"""
        participants = result.metadata.participants or []
        return len(participants) > 1
    
    async def _generate_forensic_response(self, 
                                        case: Case, 
                                        query: str, 
                                        search_results: List[Any],
                                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate forensic investigation response using LLM"""
        
        # Try Gemini first, then fallback to OpenAI, then rule-based
        if "gemini" in self.llm_clients:
            return await self._generate_gemini_response(case, query, search_results, context)
        elif "openai" in self.llm_clients:
            return await self._generate_openai_response(case, query, search_results, context)
        else:
            return self._generate_rule_based_response(case, query, search_results, context)
    
    async def _generate_gemini_response(self, 
                                      case: Case, 
                                      query: str, 
                                      search_results: List[Any],
                                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate response using Google Gemini"""
        try:
            prompt = self._build_forensic_prompt(case, query, search_results, context)
            
            # Generate response with Gemini
            response = self.llm_clients["gemini"].generate_content(prompt)
            
            return {
                "status": "success",
                "response": response.text,
                "model_used": "gemini-2.0-flash-exp",
                "confidence": 0.9,
                "sources": self._format_sources(search_results),
                "reasoning": "AI analysis using advanced language model"
            }
            
        except Exception as e:
            logger.error(f"Error with Gemini response: {str(e)}")
            return await self._generate_openai_response(case, query, search_results, context)
    
    async def _generate_openai_response(self, 
                                      case: Case, 
                                      query: str, 
                                      search_results: List[Any],
                                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate response using OpenAI GPT"""
        try:
            prompt = self._build_forensic_prompt(case, query, search_results, context)
            
            response = self.llm_clients["openai"].chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert forensic investigator AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return {
                "status": "success",
                "response": response.choices[0].message.content,
                "model_used": "gpt-4",
                "confidence": 0.85,
                "sources": self._format_sources(search_results),
                "reasoning": "AI analysis using GPT-4"
            }
            
        except Exception as e:
            logger.error(f"Error with OpenAI response: {str(e)}")
            return self._generate_rule_based_response(case, query, search_results, context)
    
    def _generate_rule_based_response(self, 
                                    case: Case, 
                                    query: str, 
                                    search_results: List[Any],
                                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate rule-based response as fallback"""
        
        if not search_results:
            return {
                "status": "no_results",
                "response": f"I couldn't find specific information about '{query}' in case {case.case_number}. Please try rephrasing your question or check if relevant evidence has been processed.",
                "model_used": "rule_based",
                "confidence": 0.6,
                "sources": [],
                "reasoning": "No relevant evidence found"
            }
        
        # Create a summary response
        response_parts = [
            f"Based on the evidence in case {case.case_number}, I found {len(search_results)} relevant items:"
        ]
        
        for i, result in enumerate(search_results[:3]):
            response_parts.append(
                f"\n{i+1}. From {result.metadata.source_file} ({result.metadata.data_type}): "
                f"{result.text[:150]}..."
            )
        
        if len(search_results) > 3:
            response_parts.append(f"\n...and {len(search_results) - 3} more results.")
        
        response_parts.append(
            f"\n\nNote: This is a basic analysis. For detailed insights, please ensure advanced AI models are configured."
        )
        
        return {
            "status": "success",
            "response": "".join(response_parts),
            "model_used": "rule_based",
            "confidence": 0.6,
            "sources": self._format_sources(search_results),
            "reasoning": "Rule-based analysis of search results"
        }
    
    def _build_forensic_prompt(self, 
                             case: Case, 
                             query: str, 
                             search_results: List[Any],
                             context: Optional[Dict[str, Any]] = None) -> str:
        """Build comprehensive forensic investigation prompt"""
        
        prompt = f"""
You are an expert forensic investigator AI assistant analyzing evidence for criminal investigations.

CASE INFORMATION:
- Case Number: {case.case_number}
- Case Title: {case.title}
- Case Type: {case.case_type or 'General Investigation'}
- Lead Investigator: {case.investigator_name}
- Department: {case.department or 'Unknown'}
- Case Status: {case.status}
- Priority Level: {case.priority}

INVESTIGATION QUERY: {query}

RELEVANT EVIDENCE FOUND:
"""
        
        for i, result in enumerate(search_results):
            prompt += f"""
Evidence Item {i+1}:
- Source: {result.metadata.source_file}
- Type: {result.metadata.data_type}
- Relevance Score: {result.similarity_score:.3f}
- Content: {result.text[:300]}
- Metadata: {result.metadata.entities or {}}
- Timestamp: {result.metadata.timestamp or 'Unknown'}
- Participants: {result.metadata.participants or []}

"""
        
        if context and context.get("conversation_history"):
            prompt += f"\nCONVERSATION HISTORY:\n"
            for msg in context["conversation_history"][-3:]:
                prompt += f"- {msg.get('role', 'user')}: {msg.get('content', '')}\n"
        
        prompt += f"""

FORENSIC ANALYSIS INSTRUCTIONS:
1. Analyze the evidence in the context of criminal investigation
2. Identify patterns, connections, and suspicious activities
3. Highlight key findings that answer the investigator's query
4. Note any gaps in evidence or areas needing further investigation
5. Use professional forensic terminology
6. Cite specific evidence sources
7. If applicable, suggest next investigative steps
8. Maintain objectivity and distinguish between facts and inferences

RESPONSE REQUIREMENTS:
- Provide a clear, factual analysis
- Structure your response logically
- Include specific evidence references
- Highlight any red flags or suspicious patterns
- Suggest additional investigation areas if relevant
- Keep the response focused on the query while providing necessary context

Please provide your forensic analysis of this evidence in response to the query.
"""
        
        return prompt
    
    def _format_sources(self, search_results: List[Any]) -> List[Dict[str, Any]]:
        """Format search results as source references"""
        sources = []
        
        for i, result in enumerate(search_results):
            sources.append({
                "id": result.id,
                "source_file": result.metadata.source_file,
                "evidence_type": result.metadata.data_type,
                "content_preview": result.text[:200] + "..." if len(result.text) > 200 else result.text,
                "similarity_score": round(result.similarity_score, 3),
                "timestamp": result.metadata.timestamp.isoformat() if result.metadata.timestamp else None,
                "participants": result.metadata.participants,
                "entities": result.metadata.entities,
                "rank": i + 1
            })
        
        return sources
    
    def get_case_summary(self, case_id: str) -> Dict[str, Any]:
        """Generate a comprehensive summary of the case"""
        try:
            case = self.case_manager.get_case(case_id)
            if not case:
                return {"error": "Case not found"}
            
            case_stats = self.case_manager.get_case_statistics(case_id)
            vector_stats = self.vector_store.get_case_statistics(case_id, case.embedding_collection_name)
            
            return {
                "case_overview": {
                    "case_number": case.case_number,
                    "title": case.title,
                    "status": case.status,  # Already a string
                    "investigator": case.investigator_name,
                    "created": case.created_at.isoformat(),
                    "last_updated": case.updated_at.isoformat()
                },
                "evidence_summary": case_stats.get("evidence_by_type", {}),
                "processing_status": case_stats.get("processing", {}),
                "ai_readiness": {
                    "has_embeddings": case.processed_evidence_count > 0,
                    "total_embeddings": vector_stats.get("total_embeddings", 0),
                    "searchable_content": vector_stats.get("total_embeddings", 0) > 0
                },
                "investigation_suggestions": self._generate_investigation_suggestions(case_stats)
            }
            
        except Exception as e:
            logger.error(f"Error generating case summary: {str(e)}")
            return {"error": str(e)}
    
    def _generate_investigation_suggestions(self, case_stats: Dict[str, Any]) -> List[str]:
        """Generate investigation suggestions based on available evidence"""
        suggestions = []
        evidence_types = case_stats.get("evidence_by_type", {})
        
        if evidence_types.get("chat", 0) > 0:
            suggestions.extend([
                "Analyze communication patterns and frequency",
                "Identify key participants and their relationships",
                "Look for suspicious language or coded communications",
                "Check for deleted or encrypted messages"
            ])
        
        if evidence_types.get("call_log", 0) > 0:
            suggestions.extend([
                "Examine call patterns and timing",
                "Identify frequent contacts and unknown numbers",
                "Analyze call duration patterns",
                "Check for calls during suspicious time periods"
            ])
        
        if evidence_types.get("contact", 0) > 0:
            suggestions.extend([
                "Map contact relationships and networks",
                "Identify contacts with multiple aliases",
                "Check for international or suspicious contacts"
            ])
        
        # General suggestions
        suggestions.extend([
            "Cross-reference evidence across different sources",
            "Look for timeline correlations between different evidence types",
            "Identify gaps in evidence that need investigation"
        ])
        
        return suggestions[:8]  # Limit to 8 suggestions

# Default instance for easy importing
case_rag_system = None  # Will be initialized when dependencies are available