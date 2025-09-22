"""
RAG Analyzer Node

This module implements the RAG (Retrieval-Augmented Generation) analysis node
that connects with the existing ForensicRAGSystem for intelligent question answering.
"""

import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

# Add the src directory to the path to import RAG system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ..state import ForensicBotState, add_workflow_step
from src.ai_cores.rag.rag_system import ForensicRAGSystem, RAGQuery, create_forensic_rag_system
from src.ai_cores.rag.vector_store import SearchFilter

class RAGAnalyzer:
    """Forensic RAG analysis coordinator"""
    
    def __init__(self):
        self.rag_system: Optional[ForensicRAGSystem] = None
        self.initialized = False
    
    def initialize_rag_system(self, case_id: Optional[str] = None) -> bool:
        """
        Initialize the RAG system for forensic analysis
        
        Args:
            case_id: Optional case ID for case-specific analysis
            
        Returns:
            bool: True if initialization successful
        """
        try:
            # Use default configuration for now
            config = {
                "embedding_config": {
                    "primary_model": "all-mpnet-base-v2",
                    "use_instructor": False
                },
                "vector_config": {
                    "store_type": "auto",
                    "persist_directory": "./data/vector_db",
                    "collection_name": f"case_{case_id}" if case_id else "general_forensic"
                },
                "retrieval_config": {
                    "top_k": 5,
                    "score_threshold": 0.3,
                    "use_reranking": True
                },
                "generation_config": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.1,
                    "max_tokens": 1000
                }
            }
            
            self.rag_system = create_forensic_rag_system(config)
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"RAG system initialization failed: {e}")
            return False
    
    def analyze_query(
        self, 
        query: str, 
        case_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a query using the RAG system
        
        Args:
            query: User's question or query
            case_id: Optional case ID for context
            context: Additional context for the query
            
        Returns:
            Dict containing the analysis results
        """
        if not self.initialized:
            if not self.initialize_rag_system(case_id):
                return {
                    "success": False,
                    "error": "Failed to initialize RAG system",
                    "response": "I'm currently unable to process your query. Please try again later."
                }
        
        try:
            # Create RAG query
            rag_query = RAGQuery(
                text=query,
                query_id=f"rag_query_{case_id or 'default'}",
                query_type="general",
                filters=SearchFilter(case_ids=[case_id]) if case_id else None,
                max_results=5
            )
            
            # Process the query
            rag_response = self.rag_system.query(rag_query)
            
            return {
                "success": True,
                "response": rag_response.response.response_text,
                "sources": [result.metadata for result in rag_response.retrieved_results],
                "confidence": rag_response.response.confidence_score,
                "retrieved_documents_count": len(rag_response.retrieved_results),
                "response_metadata": rag_response.metadata
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"I encountered an error while processing your query: {str(e)}"
            }

def rag_analyzer(state: ForensicBotState) -> ForensicBotState:
    """
    Analyze user queries using the RAG system
    
    This node uses the existing ForensicRAGSystem to answer questions
    about evidence and cases using retrieval-augmented generation.
    
    Args:
        state: Current forensic bot state
        
    Returns:
        ForensicBotState: Updated state with RAG analysis results
    """
    start_time = datetime.now()
    
    try:
        # Get the user's query from the last message
        if not state["messages"]:
            raise ValueError("No messages found in state")
        
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            query = last_message.content
        else:
            # Try to get query from routing context
            query = state.get("routing_context", {}).get("original_message", "")
            if not query:
                raise ValueError("No user query found")
        
        # Initialize RAG analyzer
        if "rag_analyzer_instance" not in state["tool_results"]:
            state["tool_results"]["rag_analyzer_instance"] = RAGAnalyzer()
        
        rag_analyzer_instance = state["tool_results"]["rag_analyzer_instance"]
        
        # Prepare context for the query
        context = {
            "investigation_phase": state["investigation_phase"],
            "current_focus": state.get("current_focus"),
            "session_id": state["session_id"],
            "entity_memory": list(state["entity_memory"].keys()),
            "active_evidence_count": len(state["active_evidence"]),
            "key_findings": state.get("key_findings", [])
        }
        
        # Add relevant cached information
        if state["semantic_search_cache"]:
            context["cached_searches"] = list(state["semantic_search_cache"].keys())
        
        # Analyze the query
        analysis_result = rag_analyzer_instance.analyze_query(
            query=query,
            case_id=state["current_case_id"],
            context=context
        )
        
        # Update state with results
        state["analysis_results"]["last_rag_query"] = {
            "query": query,
            "result": analysis_result,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the results
        cache_key = f"rag_{hash(query)}"
        state["semantic_search_cache"][cache_key] = analysis_result
        
        # Update RAG context
        state["rag_context"].update({
            "last_query": query,
            "last_response": analysis_result.get("response", ""),
            "last_confidence": analysis_result.get("confidence", 0.0),
            "sources_count": analysis_result.get("retrieved_documents_count", 0)
        })
        
        # Add AI response to messages
        response_text = analysis_result.get("response", "I couldn't process your query.")
        
        # Enhance response with confidence and source information
        if analysis_result.get("success", False):
            confidence = analysis_result.get("confidence", 0.0)
            sources_count = analysis_result.get("retrieved_documents_count", 0)
            
            if sources_count > 0:
                response_text += f"\n\n*Based on {sources_count} evidence sources"
                if confidence > 0:
                    response_text += f" (Confidence: {confidence:.1%})"
                response_text += "*"
        
        ai_message = AIMessage(content=response_text)
        state["messages"].append(ai_message)
        
        # Update recommendations if applicable
        if analysis_result.get("success", False) and sources_count > 0:
            recommendation = f"Analyzed query using RAG system with {sources_count} relevant evidence sources"
            if recommendation not in state["recommendations"]:
                state["recommendations"].append(recommendation)
        
        # Add to tools used
        if "rag_analyzer" not in state["tools_used"]:
            state["tools_used"].append("rag_analyzer")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        add_workflow_step(
            state,
            node_name="rag_analyzer",
            action="analyze_query",
            input_data={
                "query": query,
                "case_id": state["current_case_id"],
                "context_keys": list(context.keys())
            },
            output_data={
                "success": analysis_result.get("success", False),
                "confidence": analysis_result.get("confidence", 0.0),
                "sources_count": analysis_result.get("retrieved_documents_count", 0),
                "response_length": len(response_text)
            },
            execution_time=execution_time,
            success=analysis_result.get("success", False)
        )
        
    except Exception as e:
        # Handle errors gracefully
        error_msg = f"RAG analysis error: {str(e)}"
        state["last_tool_error"] = error_msg
        
        # Provide fallback response
        fallback_response = (
            "I encountered an issue while analyzing your query. "
            "This might be because the evidence hasn't been processed yet, "
            "or there's a configuration issue. Please try rephrasing your question "
            "or ensure evidence has been loaded into the system."
        )
        
        ai_message = AIMessage(content=fallback_response)
        state["messages"].append(ai_message)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        add_workflow_step(
            state,
            node_name="rag_analyzer",
            action="analyze_query",
            input_data={"error": str(e)},
            output_data={"fallback_response": True},
            execution_time=execution_time,
            success=False
        )
    
    return state

def enhance_query_with_context(
    query: str, 
    state: ForensicBotState
) -> str:
    """
    Enhance the user query with relevant context from the investigation
    
    Args:
        query: Original user query
        state: Current forensic bot state
        
    Returns:
        str: Enhanced query with context
    """
    context_parts = []
    
    # Add case context
    if state["current_case_id"]:
        context_parts.append(f"Case ID: {state['current_case_id']}")
    
    # Add current focus
    if state.get("current_focus"):
        context_parts.append(f"Current focus: {state['current_focus']}")
    
    # Add key entities
    if state["entity_memory"]:
        key_entities = list(state["entity_memory"].keys())[:3]  # Top 3 entities
        context_parts.append(f"Key entities: {', '.join(key_entities)}")
    
    # Add recent findings
    if state.get("key_findings"):
        recent_findings = state["key_findings"][-2:]  # Last 2 findings
        context_parts.append(f"Recent findings: {'; '.join(recent_findings)}")
    
    if context_parts:
        enhanced_query = f"{query}\n\nContext: {' | '.join(context_parts)}"
        return enhanced_query
    
    return query