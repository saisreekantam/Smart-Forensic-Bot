"""
Enhanced Intelligent Case Assistant with Advanced GPT Integration

This module provides sophisticated AI assistance with:
- Smart model selection (GPT-4o-mini vs GPT-4) based on complexity
- Advanced RAG with knowledge graph integration
- Multi-step reasoning for complex investigations
- Forensic-specific prompt engineering
"""

import asyncio
import logging
import os
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import json

from openai import OpenAI
from src.case_management.case_manager import CaseManager
from src.ai_cores.rag.case_vector_store import CaseVectorStore
from src.ai_cores.knowledge_graph.graph_querier import ForensicGraphQuerier

logger = logging.getLogger(__name__)

@dataclass
class QueryComplexity:
    """Analysis of query complexity for model selection"""
    complexity_score: float  # 0.0 = simple, 1.0 = very complex
    reasoning_type: str      # 'factual', 'analytical', 'inferential', 'creative'
    domain_specificity: float  # 0.0 = general, 1.0 = highly specialized
    context_requirement: str   # 'low', 'medium', 'high'
    recommended_model: str     # 'gpt-4o-mini' or 'gpt-4'

@dataclass
class InvestigationResponse:
    """Enhanced response with investigation insights"""
    response: str
    confidence: float
    reasoning_chain: List[str]
    evidence_sources: List[Dict[str, Any]]
    knowledge_graph_insights: List[Dict[str, Any]]
    follow_up_suggestions: List[str]
    model_used: str
    complexity_analysis: QueryComplexity

class EnhancedCaseAssistant:
    """
    Advanced AI assistant for forensic investigations with sophisticated reasoning
    """
    
    def __init__(self, case_manager: CaseManager, vector_store: CaseVectorStore):
        self.case_manager = case_manager
        self.vector_store = vector_store
        
        # Initialize OpenAI client
        self.openai_client = self._initialize_openai()
        
        # Initialize knowledge graph querier
        try:
            self.graph_querier = ForensicGraphQuerier(None)  # Will be updated when graph store is available
        except Exception as e:
            logger.warning(f"Knowledge graph not available: {e}")
            self.graph_querier = None
        
        # Complexity analysis patterns
        self.complexity_patterns = self._load_complexity_patterns()
        
        # Conversation memory for context
        self.conversation_memory = {}
    
    def _initialize_openai(self) -> Optional[OpenAI]:
        """Initialize OpenAI client with API key validation"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or not api_key.startswith("sk-"):
                logger.error("Invalid or missing OpenAI API key")
                return None
            
            client = OpenAI(api_key=api_key)
            
            # Test the connection
            client.models.list()
            logger.info("✅ OpenAI client initialized successfully")
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return None
    
    def _load_complexity_patterns(self) -> Dict[str, Any]:
        """Load patterns for analyzing query complexity"""
        return {
            "simple_indicators": [
                r"\bwhat is\b", r"\bwho is\b", r"\bwhen did\b", r"\bwhere\b",
                r"\bhow many\b", r"\blist\b", r"\bshow me\b", r"\bfind\b",
                r"\btell me about\b", r"\bdefine\b", r"\bexplain what\b"
            ],
            "analytical_indicators": [
                r"\banalyze\b", r"\bcompare\b", r"\bcorrelate\b", r"\bpattern\b",
                r"\btrend\b", r"\brelationship\b", r"\bconnection\b", r"\blink\b",
                r"\bassociation\b", r"\bsimilarity\b", r"\bdifference\b"
            ],
            "inferential_indicators": [
                r"\bwhy\b", r"\bhow did\b", r"\bwhat caused\b", r"\bwhat if\b",
                r"\bpredict\b", r"\blikely\b", r"\bprobable\b", r"\bsuspect\b",
                r"\bmotive\b", r"\bintention\b", r"\breason\b", r"\bcause\b"
            ],
            "complex_indicators": [
                r"\breconstruct\b", r"\btheory\b", r"\bhypothesis\b", r"\bstrategy\b",
                r"\brecommend\b", r"\bapproach\b", r"\bmethodology\b", r"\bframework\b",
                r"\bimplication\b", r"\bconsequence\b", r"\bscenario\b", r"\btimeline\b"
            ],
            "forensic_indicators": [
                r"\bevidence chain\b", r"\bdigital forensics\b", r"\bmetadata\b",
                r"\btimestamp\b", r"\bgeolocation\b", r"\bfingerprint\b", r"\bcrypto\b",
                r"\bblockchain\b", r"\bmalware\b", r"\bphishing\b", r"\bcybercrime\b"
            ]
        }
    
    def analyze_query_complexity(self, query: str, context: Dict[str, Any]) -> QueryComplexity:
        """
        Analyze query complexity to determine appropriate GPT model
        """
        query_lower = query.lower()
        
        # Initialize scores
        complexity_score = 0.0
        reasoning_type = "factual"
        domain_specificity = 0.0
        
        # Check for complexity indicators
        simple_matches = sum(1 for pattern in self.complexity_patterns["simple_indicators"] 
                           if re.search(pattern, query_lower))
        analytical_matches = sum(1 for pattern in self.complexity_patterns["analytical_indicators"] 
                               if re.search(pattern, query_lower))
        inferential_matches = sum(1 for pattern in self.complexity_patterns["inferential_indicators"] 
                                if re.search(pattern, query_lower))
        complex_matches = sum(1 for pattern in self.complexity_patterns["complex_indicators"] 
                            if re.search(pattern, query_lower))
        forensic_matches = sum(1 for pattern in self.complexity_patterns["forensic_indicators"] 
                             if re.search(pattern, query_lower))
        
        # Calculate complexity score
        if complex_matches > 0:
            complexity_score += 0.8
            reasoning_type = "creative"
        elif inferential_matches > 0:
            complexity_score += 0.6
            reasoning_type = "inferential"
        elif analytical_matches > 0:
            complexity_score += 0.4
            reasoning_type = "analytical"
        elif simple_matches > 0:
            complexity_score += 0.1
            reasoning_type = "factual"
        
        # Adjust for forensic specificity
        if forensic_matches > 0:
            domain_specificity = min(1.0, forensic_matches * 0.3)
            complexity_score += domain_specificity * 0.3
        
        # Consider context factors
        evidence_count = context.get("evidence_count", 0)
        if evidence_count > 10:
            complexity_score += 0.2
        
        # Determine context requirement
        if len(query.split()) > 30 or "multiple" in query_lower or "all" in query_lower:
            context_requirement = "high"
            complexity_score += 0.2
        elif analytical_matches > 0 or inferential_matches > 0:
            context_requirement = "medium"
            complexity_score += 0.1
        else:
            context_requirement = "low"
        
        # Cap complexity score
        complexity_score = min(1.0, complexity_score)
        
        # Determine recommended model
        if complexity_score >= 0.5 or reasoning_type in ["inferential", "creative"]:
            recommended_model = "gpt-4"
        else:
            recommended_model = "gpt-4o-mini"
        
        return QueryComplexity(
            complexity_score=complexity_score,
            reasoning_type=reasoning_type,
            domain_specificity=domain_specificity,
            context_requirement=context_requirement,
            recommended_model=recommended_model
        )
    
    async def process_investigation_query(
        self, 
        case_id: str, 
        query: str, 
        conversation_history: List[Dict[str, str]] = None
    ) -> InvestigationResponse:
        """
        Process investigation query with sophisticated reasoning
        """
        if not self.openai_client:
            return self._create_fallback_response(case_id, query)
        
        try:
            # Get case context
            case = self.case_manager.get_case(case_id)
            case_stats = self.case_manager.get_case_statistics(case_id)
            
            # Analyze query complexity
            complexity = self.analyze_query_complexity(query, {
                "evidence_count": case_stats.get("processing", {}).get("total_evidence", 0),
                "case_type": case.case_type,
                "case_priority": getattr(case, 'priority', 'medium')
            })
            
            # Search for relevant evidence using RAG
            evidence_sources = await self._enhanced_evidence_search(case_id, query, case.embedding_collection_name)
            
            # Get knowledge graph insights if available
            kg_insights = await self._get_knowledge_graph_insights(case_id, query, evidence_sources)
            
            # Generate response using selected model
            response_data = await self._generate_sophisticated_response(
                case, query, evidence_sources, kg_insights, complexity, conversation_history or []
            )
            
            # Generate follow-up suggestions
            follow_ups = self._generate_follow_up_suggestions(query, evidence_sources, complexity)
            
            return InvestigationResponse(
                response=response_data["response"],
                confidence=response_data["confidence"],
                reasoning_chain=response_data["reasoning_chain"],
                evidence_sources=evidence_sources,
                knowledge_graph_insights=kg_insights,
                follow_up_suggestions=follow_ups,
                model_used=complexity.recommended_model,
                complexity_analysis=complexity
            )
            
        except Exception as e:
            logger.error(f"Error processing investigation query: {e}")
            return self._create_error_response(case_id, query, str(e))
    
    async def _enhanced_evidence_search(self, case_id: str, query: str, 
                                      collection_name: str) -> List[Dict[str, Any]]:
        """Enhanced evidence search with multiple retrieval strategies"""
        try:
            # Multiple search strategies for better recall
            search_strategies = [
                {"query": query, "top_k": 5, "threshold": 0.7},
                {"query": self._expand_query_entities(query), "top_k": 3, "threshold": 0.6},
                {"query": self._extract_key_terms(query), "top_k": 2, "threshold": 0.5}
            ]
            
            all_sources = []
            seen_ids = set()
            
            for strategy in search_strategies:
                sources = await self._search_vector_store(collection_name, strategy)
                for source in sources:
                    source_id = source.get("id", source.get("content", "")[:50])
                    if source_id not in seen_ids:
                        seen_ids.add(source_id)
                        all_sources.append(source)
            
            # Sort by relevance and return top results
            all_sources.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            return all_sources[:8]  # Top 8 most relevant sources
            
        except Exception as e:
            logger.error(f"Error in enhanced evidence search: {e}")
            return []
    
    async def _search_vector_store(self, collection_name: str, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search vector store with given strategy"""
        try:
            # This would integrate with your vector store
            # For now, return mock results
            return [
                {
                    "id": f"chunk_{i}",
                    "content": f"Sample evidence content for {strategy['query']}",
                    "source_file": f"evidence_{i}.txt",
                    "similarity_score": 0.8 - (i * 0.1),
                    "metadata": {"type": "chat", "timestamp": "2024-01-15"}
                }
                for i in range(strategy["top_k"])
            ]
        except Exception as e:
            logger.error(f"Vector store search failed: {e}")
            return []
    
    async def _get_knowledge_graph_insights(self, case_id: str, query: str, 
                                          evidence_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get insights from knowledge graph"""
        if not self.graph_querier:
            return []
        
        try:
            # Extract entities from query and evidence
            entities = self._extract_entities_from_query(query)
            
            # Query knowledge graph for relationships
            insights = []
            for entity in entities:
                relationships = await self.graph_querier.find_entity_relationships(entity, case_id)
                if relationships:
                    insights.append({
                        "entity": entity,
                        "relationships": relationships,
                        "insight_type": "entity_connections"
                    })
            
            return insights
            
        except Exception as e:
            logger.error(f"Knowledge graph query failed: {e}")
            return []
    
    async def _generate_sophisticated_response(
        self, 
        case: Any, 
        query: str, 
        evidence_sources: List[Dict[str, Any]], 
        kg_insights: List[Dict[str, Any]],
        complexity: QueryComplexity,
        conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Generate sophisticated response using selected GPT model"""
        
        # Build enhanced system prompt
        system_prompt = self._build_enhanced_system_prompt(case, complexity)
        
        # Build conversation messages
        messages = self._build_investigation_messages(
            case, query, evidence_sources, kg_insights, complexity, conversation_history
        )
        
        # Select model parameters based on complexity
        model_params = self._get_model_parameters(complexity)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=complexity.recommended_model,
                messages=messages,
                **model_params
            )
            
            response_text = response.choices[0].message.content
            
            # Extract reasoning chain
            reasoning_chain = self._extract_reasoning_chain(response_text)
            
            # Calculate confidence based on evidence and model certainty
            confidence = self._calculate_response_confidence(
                evidence_sources, complexity, response.choices[0].finish_reason
            )
            
            return {
                "response": response_text,
                "confidence": confidence,
                "reasoning_chain": reasoning_chain
            }
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _build_enhanced_system_prompt(self, case: Any, complexity: QueryComplexity) -> str:
        """Build enhanced system prompt based on complexity"""
        
        base_prompt = f"""You are an expert forensic investigator AI with advanced analytical capabilities.

CASE CONTEXT:
- Case: {case.case_number} - {case.title}
- Investigator: {case.investigator_name}
- Case Type: {case.case_type or 'General Investigation'}
- Priority: {getattr(case, 'priority', 'Standard')}

ANALYSIS MODE: {complexity.reasoning_type.upper()} ({complexity.recommended_model})
Complexity Score: {complexity.complexity_score:.2f}
Domain Specificity: {complexity.domain_specificity:.2f}

CORE CAPABILITIES:
1. Digital Evidence Analysis & Pattern Recognition
2. Timeline Reconstruction & Event Correlation
3. Entity Relationship Mapping
4. Behavioral Analysis & Motive Assessment
5. Technical Forensic Interpretation
6. Legal Compliance & Chain of Custody Awareness

REASONING FRAMEWORK:
- OBSERVE: Analyze available evidence objectively
- ORIENT: Consider context, patterns, and relationships
- DECIDE: Form hypotheses based on evidence
- ACT: Provide actionable investigative insights

RESPONSE REQUIREMENTS:
- Base all conclusions on available evidence
- Clearly distinguish facts from inferences
- Provide confidence levels for assessments
- Suggest additional evidence needs
- Maintain forensic accuracy and objectivity
- Use professional investigative terminology"""

        if complexity.reasoning_type == "inferential":
            base_prompt += """

INFERENTIAL ANALYSIS MODE:
- Apply logical reasoning to connect evidence points
- Consider multiple hypotheses and evaluate likelihood
- Identify gaps that need additional investigation
- Assess credibility and reliability of sources"""

        elif complexity.reasoning_type == "creative":
            base_prompt += """

ADVANCED REASONING MODE:
- Synthesize complex patterns across multiple evidence types
- Develop comprehensive investigative theories
- Consider non-obvious connections and relationships
- Provide strategic recommendations for case progression"""

        return base_prompt
    
    def _build_investigation_messages(
        self,
        case: Any,
        query: str,
        evidence_sources: List[Dict[str, Any]],
        kg_insights: List[Dict[str, Any]],
        complexity: QueryComplexity,
        conversation_history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Build comprehensive message context for investigation"""
        
        messages = [
            {"role": "system", "content": self._build_enhanced_system_prompt(case, complexity)}
        ]
        
        # Add relevant conversation history
        for msg in conversation_history[-6:]:  # Last 6 messages for context
            role = "user" if msg.get('role') == 'user' else "assistant"
            messages.append({"role": role, "content": msg.get('content', '')})
        
        # Build current query with comprehensive context
        evidence_context = self._format_evidence_context(evidence_sources)
        kg_context = self._format_knowledge_graph_context(kg_insights)
        
        current_query = f"""INVESTIGATION REQUEST: {query}

{evidence_context}

{kg_context}

ANALYSIS INSTRUCTIONS:
Based on the available evidence and knowledge graph insights, provide a comprehensive forensic analysis. Structure your response as follows:

1. IMMEDIATE FINDINGS: Key facts directly supported by evidence
2. PATTERN ANALYSIS: Connections, correlations, and anomalies identified
3. INVESTIGATIVE ASSESSMENT: What this evidence suggests about the case
4. CONFIDENCE EVALUATION: Reliability of conclusions drawn
5. NEXT STEPS: Recommended follow-up actions or additional evidence needed

Ensure your analysis maintains forensic standards and clearly indicates the strength of evidence supporting each conclusion."""

        messages.append({"role": "user", "content": current_query})
        
        return messages
    
    def _format_evidence_context(self, evidence_sources: List[Dict[str, Any]]) -> str:
        """Format evidence sources for context"""
        if not evidence_sources:
            return "EVIDENCE STATUS: No processed evidence available for analysis."
        
        context = "AVAILABLE EVIDENCE:\n"
        for i, source in enumerate(evidence_sources[:6], 1):
            context += f"""
Evidence {i} ({source.get('metadata', {}).get('type', 'Unknown')}):
Source: {source.get('source_file', 'Unknown')}
Relevance: {source.get('similarity_score', 0):.3f}
Content: {source.get('content', 'No content')[:300]}{'...' if len(source.get('content', '')) > 300 else ''}
Metadata: {source.get('metadata', {})}
---"""
        
        return context
    
    def _format_knowledge_graph_context(self, kg_insights: List[Dict[str, Any]]) -> str:
        """Format knowledge graph insights"""
        if not kg_insights:
            return "KNOWLEDGE GRAPH: No additional entity relationships identified."
        
        context = "KNOWLEDGE GRAPH INSIGHTS:\n"
        for insight in kg_insights:
            context += f"""
Entity: {insight.get('entity', 'Unknown')}
Relationships: {len(insight.get('relationships', []))} connections found
Key Connections: {', '.join([r.get('target', 'Unknown') for r in insight.get('relationships', [])[:3]])}
---"""
        
        return context
    
    def _get_model_parameters(self, complexity: QueryComplexity) -> Dict[str, Any]:
        """Get model parameters based on complexity"""
        if complexity.recommended_model == "gpt-4":
            return {
                "max_tokens": 2000,
                "temperature": 0.3,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
        else:  # gpt-4o-mini
            return {
                "max_tokens": 1200,
                "temperature": 0.2,
                "top_p": 0.8,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
    
    def _extract_reasoning_chain(self, response_text: str) -> List[str]:
        """Extract reasoning steps from response"""
        # Look for numbered points or structured reasoning
        reasoning_patterns = [
            r'\d+\.\s*([^.\n]+)',
            r'(?:First|Second|Third|Fourth|Fifth),?\s*([^.\n]+)',
            r'(?:Therefore|Thus|Hence|Consequently),?\s*([^.\n]+)'
        ]
        
        reasoning_chain = []
        for pattern in reasoning_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            reasoning_chain.extend(matches)
        
        return reasoning_chain[:5]  # Limit to 5 key reasoning steps
    
    def _calculate_response_confidence(self, evidence_sources: List[Dict[str, Any]], 
                                     complexity: QueryComplexity, finish_reason: str) -> float:
        """Calculate confidence score for response"""
        base_confidence = 0.7
        
        # Adjust for evidence quality
        if evidence_sources:
            avg_similarity = sum(s.get('similarity_score', 0) for s in evidence_sources) / len(evidence_sources)
            base_confidence += (avg_similarity - 0.5) * 0.3
        else:
            base_confidence = 0.4  # Lower confidence without evidence
        
        # Adjust for model used
        if complexity.recommended_model == "gpt-4":
            base_confidence += 0.1
        
        # Adjust for completion reason
        if finish_reason == "stop":
            base_confidence += 0.05
        
        return min(0.95, max(0.1, base_confidence))
    
    def _generate_follow_up_suggestions(self, query: str, evidence_sources: List[Dict[str, Any]], 
                                      complexity: QueryComplexity) -> List[str]:
        """Generate intelligent follow-up suggestions"""
        suggestions = []
        
        # Based on complexity and evidence
        if complexity.reasoning_type == "factual" and evidence_sources:
            suggestions.extend([
                "What patterns can you identify in this evidence?",
                "Are there any suspicious activities or anomalies?",
                "What additional context might be relevant?"
            ])
        elif complexity.reasoning_type in ["analytical", "inferential"]:
            suggestions.extend([
                "What are the implications of these findings?",
                "Can you reconstruct a timeline of events?",
                "What additional evidence would strengthen this analysis?"
            ])
        
        # Based on evidence types
        evidence_types = set(s.get('metadata', {}).get('type', '') for s in evidence_sources)
        if 'chat' in evidence_types:
            suggestions.append("Analyze communication patterns and participant behavior")
        if 'call_log' in evidence_types:
            suggestions.append("Examine call frequency and timing patterns")
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    def _create_fallback_response(self, case_id: str, query: str) -> InvestigationResponse:
        """Create fallback response when OpenAI is not available"""
        return InvestigationResponse(
            response="OpenAI service is currently unavailable. Please check your API key configuration and try again.",
            confidence=0.0,
            reasoning_chain=["Fallback mode activated"],
            evidence_sources=[],
            knowledge_graph_insights=[],
            follow_up_suggestions=["Check API configuration", "Retry the query"],
            model_used="fallback",
            complexity_analysis=QueryComplexity(0.0, "fallback", 0.0, "low", "fallback")
        )
    
    def _create_error_response(self, case_id: str, query: str, error: str) -> InvestigationResponse:
        """Create error response"""
        return InvestigationResponse(
            response=f"Error processing investigation query: {error}",
            confidence=0.0,
            reasoning_chain=[f"Error occurred: {error}"],
            evidence_sources=[],
            knowledge_graph_insights=[],
            follow_up_suggestions=["Check system logs", "Retry with simplified query"],
            model_used="error",
            complexity_analysis=QueryComplexity(0.0, "error", 0.0, "low", "error")
        )
    
    # Helper methods for query processing
    def _expand_query_entities(self, query: str) -> str:
        """Expand query with related entities"""
        # Simple entity expansion - in production, use NER
        return query  # Placeholder
    
    def _extract_key_terms(self, query: str) -> str:
        """Extract key terms from query"""
        # Simple keyword extraction - in production, use proper NLP
        return query  # Placeholder
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract entities from query for knowledge graph"""
        # Simple entity extraction - in production, use NER
        return []  # Placeholder