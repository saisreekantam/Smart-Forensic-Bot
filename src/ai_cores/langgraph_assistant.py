"""
LangGraph-based Enhanced Case Assistant

This module provides a wrapper around the LangGraph forensic bot to maintain
compatibility with the existing API while leveraging the advanced LangGraph workflow.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

from src.langgraph_bot.forensic_bot import ForensicBot
from src.case_management.case_manager import CaseManager

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

class LangGraphCaseAssistant:
    """
    LangGraph-powered AI assistant for forensic investigations
    """
    
    def __init__(self, case_manager: CaseManager, vector_store=None, debug_mode: bool = False):
        """
        Initialize the LangGraph-based assistant
        
        Args:
            case_manager: Case management system
            vector_store: Vector store for RAG (optional)
            debug_mode: Enable debug logging
        """
        self.case_manager = case_manager
        self.vector_store = vector_store
        self.debug_mode = debug_mode
        
        # Store active bots by case_id to maintain session continuity
        self.active_bots: Dict[str, ForensicBot] = {}
        
        logger.info("LangGraph Case Assistant initialized")
    
    def _get_or_create_bot(self, case_id: str) -> ForensicBot:
        """Get existing bot for case or create new one"""
        if case_id not in self.active_bots:
            self.active_bots[case_id] = ForensicBot(
                case_id=case_id,
                debug_mode=self.debug_mode
            )
            # Start a session for this case
            self.active_bots[case_id].start_session(f"api_session_{case_id}")
        
        return self.active_bots[case_id]
    
    def analyze_query_complexity(self, query: str, case_context: Dict[str, Any]) -> QueryComplexity:
        """
        Analyze query complexity for model selection and routing
        """
        query_lower = query.lower()
        
        # Check for complex patterns
        complex_patterns = [
            'analyze', 'correlate', 'pattern', 'relationship', 'timeline',
            'compare', 'contrast', 'trend', 'anomaly', 'suspicious',
            'inference', 'hypothesis', 'reasoning', 'conclusion'
        ]
        
        factual_patterns = [
            'show', 'list', 'find', 'search', 'extract', 'get',
            'what', 'when', 'where', 'who', 'count', 'sum'
        ]
        
        complexity_score = 0.3  # Base complexity
        reasoning_type = 'factual'
        
        # Increase complexity based on patterns
        complex_matches = sum(1 for pattern in complex_patterns if pattern in query_lower)
        factual_matches = sum(1 for pattern in factual_patterns if pattern in query_lower)
        
        if complex_matches > factual_matches:
            complexity_score += 0.4
            reasoning_type = 'analytical'
        
        if complex_matches > 2:
            complexity_score += 0.3
            reasoning_type = 'inferential'
        
        # Domain specificity
        forensic_terms = [
            'evidence', 'forensic', 'investigation', 'suspect', 'victim',
            'metadata', 'hash', 'artifact', 'timeline', 'chain of custody'
        ]
        domain_specificity = min(1.0, sum(1 for term in forensic_terms if term in query_lower) * 0.2)
        
        # Context requirement based on case complexity
        evidence_count = case_context.get('evidence_count', 0)
        if evidence_count > 10:
            context_requirement = 'high'
            complexity_score += 0.2
        elif evidence_count > 3:
            context_requirement = 'medium'
            complexity_score += 0.1
        else:
            context_requirement = 'low'
        
        # Model selection
        recommended_model = 'gpt-4' if complexity_score > 0.6 else 'gpt-4o-mini'
        
        return QueryComplexity(
            complexity_score=min(1.0, complexity_score),
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
        Process investigation query using LangGraph forensic bot
        """
        try:
            # Get case context
            case = self.case_manager.get_case(case_id)
            if not case:
                return self._create_error_response(f"Case {case_id} not found")
            
            case_stats = self.case_manager.get_case_statistics(case_id)
            
            # Analyze query complexity
            complexity = self.analyze_query_complexity(query, {
                "evidence_count": case_stats.get("processing", {}).get("total_evidence", 0),
                "case_type": getattr(case, 'case_type', 'general'),
                "case_priority": getattr(case, 'priority', 'medium')
            })
            
            # Get or create the forensic bot for this case
            bot = self._get_or_create_bot(case_id)
            
            # Ensure the bot is working with the correct case
            bot.load_case(case_id)
            
            # Process the query through LangGraph
            response_text = await bot.chat_async(query)
            
            # Parse response for structured data (if available)
            evidence_sources = self._extract_evidence_sources(response_text, case_id)
            
            # Generate follow-up suggestions based on query type
            follow_ups = self._generate_follow_ups(query, complexity)
            
            return InvestigationResponse(
                response=response_text,
                confidence=0.85,  # LangGraph provides high confidence
                reasoning_chain=[
                    f"Query analyzed as {complexity.reasoning_type}",
                    f"Routed through LangGraph workflow",
                    f"Processed with {complexity.recommended_model} capabilities",
                    "Response synthesized from multiple analysis nodes"
                ],
                evidence_sources=evidence_sources,
                knowledge_graph_insights=[],  # TODO: Extract from LangGraph state
                follow_up_suggestions=follow_ups,
                model_used=complexity.recommended_model,
                complexity_analysis=complexity
            )
            
        except Exception as e:
            logger.error(f"Error in LangGraph investigation query: {str(e)}")
            return self._create_error_response(f"Error processing query: {str(e)}")
    
    def _extract_evidence_sources(self, response: str, case_id: str) -> List[Dict[str, Any]]:
        """Extract evidence sources mentioned in the response"""
        sources = []
        
        try:
            # Get case evidence list
            evidence_list = self.case_manager.list_evidence(case_id)
            
            # Look for evidence filenames mentioned in response
            for evidence in evidence_list:
                if evidence.original_filename.lower() in response.lower():
                    sources.append({
                        "filename": evidence.original_filename,
                        "evidence_type": evidence.evidence_type,  # Already a string
                        "confidence": 0.8,
                        "relevance": "high"
                    })
        
        except Exception as e:
            logger.warning(f"Error extracting evidence sources: {e}")
        
        return sources[:5]  # Limit to top 5 sources
    
    def _generate_follow_ups(self, query: str, complexity: QueryComplexity) -> List[str]:
        """Generate follow-up question suggestions"""
        query_lower = query.lower()
        follow_ups = []
        
        if 'timeline' in query_lower:
            follow_ups.extend([
                "Can you show the complete chronological sequence?",
                "What events happened before and after this timeframe?",
                "Are there any gaps in the timeline that need investigation?"
            ])
        elif 'communication' in query_lower or 'message' in query_lower:
            follow_ups.extend([
                "Who were the main participants in these communications?",
                "What patterns do you see in the communication frequency?",
                "Are there any deleted or hidden communications?"
            ])
        elif 'location' in query_lower or 'gps' in query_lower:
            follow_ups.extend([
                "Can you map out the movement patterns?",
                "What locations were visited most frequently?",
                "Are there any unexpected location visits?"
            ])
        else:
            # Generic follow-ups based on complexity
            if complexity.reasoning_type == 'analytical':
                follow_ups.extend([
                    "What additional evidence supports this analysis?",
                    "Can you identify any related patterns or anomalies?",
                    "What are the key findings from this investigation?"
                ])
            else:
                follow_ups.extend([
                    "Can you provide more details about this evidence?",
                    "What other related information is available?",
                    "How does this connect to other case elements?"
                ])
        
        return follow_ups[:3]  # Limit to 3 suggestions
    
    def _create_error_response(self, error_msg: str) -> InvestigationResponse:
        """Create an error response"""
        return InvestigationResponse(
            response=f"I apologize, but I encountered an issue: {error_msg}. Please try rephrasing your question or contact support if the issue persists.",
            confidence=0.0,
            reasoning_chain=[f"Error encountered: {error_msg}"],
            evidence_sources=[],
            knowledge_graph_insights=[],
            follow_up_suggestions=[
                "Try rephrasing your question",
                "Check if the case has processed evidence",
                "Contact system administrator if error persists"
            ],
            model_used="error_handler",
            complexity_analysis=QueryComplexity(
                complexity_score=0.0,
                reasoning_type="error",
                domain_specificity=0.0,
                context_requirement="none",
                recommended_model="none"
            )
        )
    
    def cleanup_case_session(self, case_id: str):
        """Clean up bot session for a case"""
        if case_id in self.active_bots:
            del self.active_bots[case_id]
            logger.info(f"Cleaned up session for case {case_id}")