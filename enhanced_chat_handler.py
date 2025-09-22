"""
Enhanced Chat Handler with Knowledge Graph and Case Memory Integration
Integrates with case memory system, knowledge graphs, and intelligent reporting
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from openai import AsyncOpenAI
import os
import uuid
from dotenv import load_dotenv
from simple_search_system import SimpleSearchSystem

# Import our new enhanced systems
from src.ai_cores.case_memory import case_memory, BotInteraction
from src.ai_cores.enhanced_knowledge_graph import enhanced_kg_db
from src.ai_cores.intelligent_report_generator import get_report_generator

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class EnhancedChatHandler:
    """
    Enhanced chat handler with knowledge graph and case memory integration
    """
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = AsyncOpenAI(api_key=self.openai_api_key)
        self.search_system = SimpleSearchSystem()
        self.case_memory = case_memory
        self.knowledge_graph = enhanced_kg_db
        self.report_generator = get_report_generator()
    
    async def process_query(self, case_id: str, query: str, session_id: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Enhanced query processing with knowledge graph and case memory integration
        
        Args:
            case_id: The case ID to search in
            query: User's question
            session_id: Chat session ID for tracking
            conversation_history: Previous conversation messages
            
        Returns:
            Response with answer, sources, and enhanced insights
        """
        start_time = datetime.now()
        
        try:
            # First, determine if this is a forensic query or general conversation
            is_forensic_query = await self._is_forensic_query(query)
            query_type = "forensic" if is_forensic_query else "general"
            
            if not is_forensic_query:
                # Handle as general conversation
                response = await self._handle_general_conversation(query, conversation_history)
                confidence_score = 0.8
                evidence_sources = []
                entities_found = []
                relationships_discovered = []
            else:
                # Enhanced forensic query processing
                response_data = await self._handle_enhanced_forensic_query(case_id, query, conversation_history)
                response = response_data["response"]
                confidence_score = response_data["confidence_score"]
                evidence_sources = response_data["evidence_sources"]
                entities_found = response_data["entities_found"]
                relationships_discovered = response_data["relationships_discovered"]
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create and store bot interaction in case memory
            interaction = BotInteraction(
                id=str(uuid.uuid4()),
                case_id=case_id,
                session_id=session_id,
                timestamp=datetime.now(),
                user_query=query,
                bot_response=response,
                entities_found=entities_found,
                relationships_discovered=relationships_discovered,
                evidence_sources=evidence_sources,
                query_type=query_type,
                confidence_score=confidence_score,
                processing_time=processing_time,
                metadata={
                    "model_used": "gpt-4",
                    "search_performed": is_forensic_query,
                    "conversation_length": len(conversation_history) if conversation_history else 0
                }
            )
            
            # Store interaction in case memory
            self.case_memory.store_interaction(interaction)
            
            # Process interaction for knowledge graph updates
            if is_forensic_query:
                kg_result = self.knowledge_graph.process_interaction_for_knowledge(interaction)
                logger.info(f"Knowledge graph updated: {kg_result.get('entities_processed', 0)} entities, "
                           f"{kg_result.get('relationships_processed', 0)} relationships")
            
            return {
                "response": response,
                "case_id": case_id,
                "session_id": session_id,
                "query_type": query_type,
                "confidence_score": confidence_score,
                "evidence_sources": evidence_sources,
                "processing_time": processing_time,
                "entities_found": entities_found,
                "relationships_discovered": relationships_discovered,
                "interaction_id": interaction.id,
                "conversation_history": conversation_history
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": f"I encountered an error processing your query: {str(e)}",
                "case_id": case_id,
                "session_id": session_id,
                "query_type": "error",
                "confidence_score": 0.0,
                "evidence_sources": [],
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "error": str(e)
            }

    async def _handle_enhanced_forensic_query(self, case_id: str, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Enhanced forensic query handling with knowledge graph integration
        """
        try:
            # Get existing knowledge graph for context
            kg_data = self.knowledge_graph.get_case_knowledge_graph(case_id)
            
            # Perform search as before
            search_results = self.search_system.search_case_data(case_id, query)
            
            # Extract entities and relationships from query
            query_entities = []
            query_relationships = []
            
            # Check if query mentions known entities
            if kg_data and kg_data.get("entities"):
                for entity in kg_data["entities"]:
                    entity_value = entity.get("value", "").lower()
                    if entity_value and entity_value in query.lower():
                        query_entities.append({
                            "id": entity["id"],
                            "type": entity["type"],
                            "value": entity["value"],
                            "confidence": entity["confidence"],
                            "context": f"mentioned_in_query: {query}"
                        })
            
            # Enhanced prompt with knowledge graph context
            enhanced_prompt = self._build_enhanced_forensic_prompt(
                query, search_results, kg_data, conversation_history
            )
            
            # Get response from OpenAI
            response = await self._get_openai_response(enhanced_prompt, conversation_history)
            
            return {
                "response": response,
                "confidence_score": self._calculate_enhanced_confidence(search_results, kg_data, query),
                "evidence_sources": [result.get("source", "unknown") for result in search_results],
                "entities_found": query_entities,
                "relationships_discovered": query_relationships
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced forensic query handling: {e}")
            # Fallback to basic processing
            return await self._handle_basic_forensic_query(case_id, query, conversation_history)

    def _build_enhanced_forensic_prompt(self, query: str, search_results: List[Dict], kg_data: Dict[str, Any], conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Build enhanced forensic prompt with knowledge graph context
        """
        prompt = """You are an expert forensic digital investigator with access to case evidence and a comprehensive knowledge graph. 
Your role is to analyze evidence and provide detailed, accurate insights based on the available data.

"""
        
        # Add knowledge graph context
        if kg_data and kg_data.get("summary"):
            summary = kg_data["summary"]
            prompt += f"""
CASE KNOWLEDGE GRAPH SUMMARY:
- Total Entities: {summary.get('total_entities', 0)}
- Total Relationships: {summary.get('total_relationships', 0)}
- Knowledge Density: {summary.get('knowledge_density', 0.0):.3f}

KEY ENTITIES IN CASE:
"""
            for entity in summary.get("key_entities", [])[:5]:
                prompt += f"- {entity.get('value', 'N/A')} ({entity.get('type', 'unknown')}): "
                prompt += f"Importance {entity.get('importance', 0.0):.2f}, {entity.get('mentions', 0)} mentions\n"
            
            prompt += "\nIMPORTANT RELATIONSHIPS:\n"
            for rel in summary.get("important_relationships", [])[:3]:
                prompt += f"- {rel.get('entity1', 'Entity1')} → {rel.get('entity2', 'Entity2')} "
                prompt += f"({rel.get('type', 'related')}): Strength {rel.get('strength', 0.0):.2f}\n"
        
        # Add search results
        if search_results:
            prompt += f"\n\nEVIDENCE SEARCH RESULTS for query '{query}':\n"
            for i, result in enumerate(search_results[:5], 1):
                prompt += f"\n{i}. Source: {result.get('source', 'Unknown')}\n"
                prompt += f"   Relevance: {result.get('relevance', 0.0):.2f}\n"
                prompt += f"   Content: {result.get('content', 'No content')[:300]}...\n"
        
        # Add conversation context
        if conversation_history:
            prompt += "\n\nCONVERSATION HISTORY:\n"
            for msg in conversation_history[-3:]:  # Last 3 messages for context
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:150]
                prompt += f"{role.upper()}: {content}...\n"
        
        prompt += f"""

USER QUERY: {query}

INSTRUCTIONS:
1. Analyze the evidence and knowledge graph to answer the user's question
2. Reference specific evidence sources when making claims
3. Highlight any new entities or relationships discovered
4. If the query relates to existing entities in the knowledge graph, provide additional context
5. Be specific about confidence levels and limitations
6. Format your response clearly with evidence citations
7. If you identify important new connections or insights, highlight them

Provide a comprehensive, evidence-based response:
"""
        
        return prompt

    def _calculate_enhanced_confidence(self, search_results: List[Dict], kg_data: Dict[str, Any], query: str) -> float:
        """
        Calculate enhanced confidence score considering knowledge graph data
        """
        base_confidence = 0.5
        
        # Search results quality
        if search_results:
            avg_relevance = sum(result.get("relevance", 0.0) for result in search_results) / len(search_results)
            search_boost = min(0.3, avg_relevance * 0.3)
            base_confidence += search_boost
        
        # Knowledge graph context boost
        if kg_data and kg_data.get("summary"):
            total_entities = kg_data["summary"].get("total_entities", 0)
            total_relationships = kg_data["summary"].get("total_relationships", 0)
            
            # More entities and relationships = higher confidence
            kg_boost = min(0.2, (total_entities + total_relationships) * 0.01)
            base_confidence += kg_boost
        
        # Query specificity boost
        query_words = len(query.split())
        if query_words >= 5:
            base_confidence += 0.1
        
        return min(1.0, max(0.0, base_confidence))

    async def _handle_basic_forensic_query(self, case_id: str, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Fallback to basic forensic query handling
        """
        search_results = self.search_system.search_case_data(case_id, query)
        
        prompt = f"""You are a forensic digital investigator. Analyze the following evidence to answer the user's query.

EVIDENCE:
"""
        
        for i, result in enumerate(search_results[:5], 1):
            prompt += f"\n{i}. Source: {result.get('source', 'Unknown')}\n"
            prompt += f"   Content: {result.get('content', 'No content')[:200]}...\n"
        
        prompt += f"\n\nUSER QUERY: {query}\n\nProvide a detailed analysis based on the evidence:"
        
        response = await self._get_openai_response(prompt, conversation_history)
        
        return {
            "response": response,
            "confidence_score": 0.7,
            "evidence_sources": [{"source": result.get("source", "unknown"), "content": result.get("content", "")[:200]} for result in search_results],
            "entities_found": [],
            "relationships_discovered": []
        }

    async def _handle_general_conversation(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Handle general conversation queries that aren't forensic-related
        """
        # Check for common greetings and responses
        greetings = {
            'hi': "Hello! I'm your forensic analysis assistant. I can help you investigate cases, search through evidence, and analyze digital communications. What case would you like to explore today?",
            'hello': "Hello! I'm here to help with your forensic investigation. I can search through evidence, analyze communications, and provide insights about your cases. How can I assist you?",
            'hey': "Hey there! I'm your digital forensics assistant. I can help analyze evidence, search through case data, and provide investigative insights. What would you like to investigate?",
            'thanks': "You're welcome! I'm here whenever you need help with forensic analysis or case investigation.",
            'thank you': "You're very welcome! Feel free to ask me anything about your forensic cases or evidence analysis.",
            'good': "Great! I'm here to help with your forensic investigations. What case or evidence would you like to analyze?",
            'ok': "Perfect! I'm ready to help with your forensic analysis. What case would you like to work on?",
            'okay': "Sounds good! I can help you investigate cases, search evidence, or analyze digital communications. What do you need?"
        }
        
        query_lower = query.lower().strip()
        
        # Check for exact matches
        if query_lower in greetings:
            return greetings[query_lower]
        
        # Check for partial matches
        for greeting, response in greetings.items():
            if greeting in query_lower:
                return response
        
        # For other general queries, provide a helpful response
        messages = [
            {
                "role": "system",
                "content": """You are a helpful forensic investigation assistant. The user is making a general comment or asking a non-forensic question. 
Respond helpfully but always guide them back to how you can help with forensic analysis, case investigation, or evidence analysis.
Keep responses concise and professional."""
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating general conversation response: {e}")
            return "I'm here to help with your forensic investigations. What case or evidence would you like to analyze?"

    async def _is_forensic_query(self, query: str) -> bool:
        """
        Determine if a query is forensic/investigation related or general conversation
        """
        try:
            # Quick keyword-based classification first
            forensic_keywords = [
                'evidence', 'call', 'message', 'contact', 'phone', 'sms', 'communication',
                'timeline', 'suspect', 'investigation', 'analyze', 'search', 'find',
                'who called', 'when', 'where', 'forensic', 'data', 'record', 'log',
                'what happened', 'show me', 'list', 'summary', 'report', 'activity'
            ]
            
            query_lower = query.lower()
            
            # If it contains obvious forensic keywords, likely forensic
            if any(keyword in query_lower for keyword in forensic_keywords):
                return True
            
            # If it's very short and casual, likely not forensic
            casual_patterns = [
                'hi', 'hello', 'hey', 'thanks', 'thank you', 'ok', 'okay', 'yes', 'no',
                'good', 'great', 'cool', 'nice', 'wow', 'amazing'
            ]
            
            if query_lower.strip() in casual_patterns:
                return False
            
            # Use AI classification for ambiguous cases
            messages = [
                {
                    "role": "system",
                    "content": """You are a query classifier. Determine if a user query is:
1. FORENSIC: Related to digital forensics, investigation, evidence analysis, case data, or asking questions about evidence
2. GENERAL: General conversation, greetings, thanks, or unrelated to forensic investigation

Respond with only "FORENSIC" or "GENERAL" - nothing else."""
                },
                {
                    "role": "user",
                    "content": f"Classify this query: '{query}'"
                }
            ]
            
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
                max_tokens=10
            )
            
            classification = response.choices[0].message.content.strip().upper()
            return classification == "FORENSIC"
            
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            # Default to forensic for safety
            return True

    async def _get_openai_response(self, prompt: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Get response from OpenAI with conversation context
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": prompt
                }
            ]
            
            # Add conversation history if provided
            if conversation_history:
                for msg in conversation_history[-5:]:  # Last 5 messages for context
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error getting OpenAI response: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."

    # Report generation methods
    async def generate_case_report(self, case_id: str, report_type: str = "detailed_analysis") -> Dict[str, Any]:
        """
        Generate a comprehensive case report using the intelligent report generator
        """
        try:
            report = self.report_generator.generate_report(case_id, report_type)
            
            return {
                "success": True,
                "report": {
                    "report_id": report.report_id,
                    "title": report.title,
                    "generated_at": report.generated_at.isoformat(),
                    "report_type": report.report_type,
                    "confidence_score": report.confidence_score,
                    "sections": [
                        {
                            "title": section.title,
                            "content": section.content,
                            "confidence": section.confidence
                        }
                        for section in report.sections
                    ],
                    "insights": report.insights,
                    "recommendations": report.recommendations,
                    "statistics": report.statistics
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating case report: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_case_insights(self, case_id: str) -> Dict[str, Any]:
        """
        Get case insights and investigation summary
        """
        try:
            insights = self.case_memory.get_case_insights(case_id)
            stats = self.case_memory.get_case_memory_stats(case_id)
            summary = self.case_memory.get_investigation_summary(case_id)
            
            return {
                "success": True,
                "case_id": case_id,
                "insights": [
                    {
                        "title": insight.title,
                        "description": insight.description,
                        "type": insight.insight_type,
                        "priority": insight.priority,
                        "confidence": insight.confidence,
                        "discovered_at": insight.discovered_at.isoformat()
                    }
                    for insight in insights
                ],
                "statistics": {
                    "total_interactions": stats.total_interactions,
                    "unique_entities": stats.unique_entities_mentioned,
                    "investigation_focus": stats.investigation_focus_areas[:5],
                    "query_patterns": stats.common_query_patterns[:5]
                },
                "investigation_health": summary.get("investigation_health", {})
            }
            
        except Exception as e:
            logger.error(f"Error getting case insights: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_knowledge_graph_summary(self, case_id: str) -> Dict[str, Any]:
        """
        Get knowledge graph summary for a case
        """
        try:
            kg_data = self.knowledge_graph.get_case_knowledge_graph(case_id)
            
            return {
                "success": True,
                "case_id": case_id,
                "knowledge_graph": {
                    "entities": len(kg_data.get("entities", [])),
                    "relationships": len(kg_data.get("relationships", [])),
                    "summary": kg_data.get("summary", {}),
                    "key_entities": kg_data.get("summary", {}).get("key_entities", [])[:10],
                    "important_relationships": kg_data.get("summary", {}).get("important_relationships", [])[:10]
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting knowledge graph summary: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Global instance for use in other modules
enhanced_chat_handler = EnhancedChatHandler()

# For backward compatibility with existing code
SimpleChatHandler = EnhancedChatHandler
simple_chat_handler = enhanced_chat_handler