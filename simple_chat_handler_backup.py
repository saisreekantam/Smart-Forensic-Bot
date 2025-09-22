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
from src.ai_cores.intelligent_report_generator import report_generator

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
        self.report_generator = report_generator
    
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
            search_results = self.search_system.search_evidence(case_id, query)
            
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
        search_results = self.search_system.search_evidence(case_id, query)
        
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
            "evidence_sources": [result.get("source", "unknown") for result in search_results],
            "entities_found": [],
            "relationships_discovered": []
        }
                return await self._handle_general_conversation(query, conversation_history)
            
            # Handle as forensic investigation query
            # Get case summary
            case_summary = self.search_system.get_case_summary(case_id)
            
            if case_summary.get("total_records", 0) == 0:
                return {
                    "response": "⚠️ This case doesn't have any processed evidence yet. Please upload and process evidence files first to enable AI analysis.",
                    "sources": [],
                    "confidence": 1.0,
                    "case_context": {"case_id": case_id, "evidence_count": 0},
                    "timestamp": datetime.now().isoformat()
                }
            
            # Search for relevant evidence
            # Use enhanced search for message content if query suggests it
            if self._is_message_query(query):
                search_results_dict = self.search_system.search_message_content(case_id, query, max_results=10)
                search_results = search_results_dict.get("results", []) if isinstance(search_results_dict, dict) else []
            else:
                search_results = self.search_system.search_case_data(case_id, query, max_results=10)
                # Ensure it's always a list
                if not isinstance(search_results, list):
                    search_results = []
            
            # Build context for OpenAI
            context = self._build_context(case_summary, search_results, query)
            
            # Generate response using OpenAI
            response_text = await self._generate_forensic_response(context, query, conversation_history)
            
            # Format sources
            sources = self._format_sources(search_results)
            
            return {
                "response": response_text,
                "sources": sources,
                "confidence": 0.9 if search_results else 0.3,
                "case_context": {
                    "case_id": case_id,
                    "evidence_count": case_summary.get("total_records", 0),
                    "search_results_count": len(search_results)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while analyzing the evidence. Please try rephrasing your question or contact support if the issue persists."
    
    def _is_message_query(self, query: str) -> bool:
        """Check if the query is specifically about message content"""
        message_indicators = [
            'message', 'text', 'sms', 'chat', 'conversation', 'said', 'wrote',
            'told', 'sent', 'received', 'replied', 'content', 'communication'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in message_indicators)
    
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
            logger.error(f"Error classifying query: {str(e)}")
            # Default to forensic if classification fails
            return True
    
    async def _handle_general_conversation(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Handle general conversation that's not forensic-related
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful and friendly forensic investigation AI assistant. 
When users engage in general conversation (greetings, thanks, casual chat), respond naturally and warmly while staying in character as a forensic AI.

Keep responses:
- Brief and friendly
- Professional but approachable  
- Helpful in guiding them toward investigation tasks when appropriate
- No more than 2-3 sentences

You can mention your forensic capabilities casually if it fits the conversation naturally."""
                }
            ]
            
            # Add conversation history if provided
            if conversation_history:
                for msg in conversation_history[-3:]:  # Last 3 messages for context
                    messages.append({
                        "role": "user" if msg.get("role") == "user" else "assistant",
                        "content": msg.get("content", "")
                    })
            
            messages.append({
                "role": "user", 
                "content": query
            })
            
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            
            return {
                "response": response.choices[0].message.content.strip(),
                "sources": [],
                "confidence": 1.0,
                "case_context": {"type": "general_conversation"},
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error handling general conversation: {str(e)}")
            return {
                "response": "Hello! I'm here to help with your forensic investigation. Feel free to ask me about evidence, case data, or anything investigation-related.",
                "sources": [],
                "confidence": 1.0,
                "case_context": {"type": "general_conversation", "error": str(e)},
                "timestamp": datetime.now().isoformat()
            }
    
    def _build_context(self, case_summary: Dict[str, Any], search_results: List[Dict[str, Any]], query: str) -> str:
        """Build context string for OpenAI"""
        
        # Safety checks
        if not isinstance(case_summary, dict):
            case_summary = {}
        if not isinstance(search_results, list):
            search_results = []
        
        context = f"""You are a forensic investigation AI assistant analyzing Case ID. 

CASE SUMMARY:
- Total Evidence Files: {case_summary.get('total_files', 0)}
- Total Records: {case_summary.get('total_records', 0)}
- File Types: {', '.join(case_summary.get('file_types', []))}

DATA SOURCES:
"""
        
        for source in case_summary.get('data_sources', []):
            if isinstance(source, dict):
                context += f"- {source.get('file', 'unknown')} ({source.get('type', 'unknown')}): {source.get('records', 0)} records\n"
        
        if search_results:
            context += f"\nRELEVANT EVIDENCE (Top {len(search_results)} matches for query: '{query}'):\n\n"
            
            for i, result in enumerate(search_results, 1):
                if not isinstance(result, dict):
                    continue
                    
                data = result.get('data', {})
                source_file = result.get('source_file', 'unknown')
                
                context += f"[Evidence {i}] From: {source_file}\n"
                
                # Format data based on type
                if result.get('record_type') == 'call_log':
                    context += f"  Call: {data.get('direction', '')} call to/from {data.get('phone_number', '')} ({data.get('contact_name', 'Unknown')})\n"
                    context += f"  Time: {data.get('timestamp', '')}, Duration: {data.get('duration', '')} seconds\n"
                    context += f"  Status: {data.get('status', '')}\n"
                elif result.get('record_type') == 'message' or 'message' in source_file.lower():
                    context += f"  Message: From {data.get('sender', '')} to {data.get('recipient', '')}\n"
                    context += f"  Time: {data.get('timestamp', '')}\n"
                    context += f"  Platform: {data.get('platform', '')}\n"
                    context += f"  Content: {data.get('message', '')[:200]}{'...' if len(str(data.get('message', ''))) > 200 else ''}\n"
                else:
                    # Generic data display
                    context += f"  Content: {result.get('searchable_text', '')[:200]}{'...' if len(result.get('searchable_text', '')) > 200 else ''}\n"
                
                context += f"  Relevance: {result.get('relevance_score', 0):.2f}\n\n"
        else:
            context += f"\nNo specific evidence found matching the query: '{query}'\n"
            context += "However, you can still provide general assistance about the case based on the available data sources.\n"
        
        return context
    
    async def _generate_forensic_response(self, context: str, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate response using OpenAI"""
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert forensic investigation AI assistant. You help investigators analyze digital evidence and answer questions about cases.

INSTRUCTIONS:
1. Always base your answers on the provided evidence data
2. Be specific and cite the evidence sources when possible
3. If the evidence shows suspicious activity, highlight it clearly
4. Provide actionable insights for the investigation
5. Use forensic terminology appropriately
6. If you don't have enough evidence to answer something, say so clearly
7. Format your response in a clear, professional manner
8. Include relevant phone numbers, timestamps, and other key details when available

Your responses should be thorough but concise, and always focused on helping the investigation."""
            }
        ]
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                messages.append({
                    "role": "user" if msg.get("role") == "user" else "assistant",
                    "content": msg.get("content", "")
                })
        
        # Add current context and query
        messages.append({
            "role": "user",
            "content": f"""CASE EVIDENCE AND CONTEXT:
{context}

INVESTIGATOR QUESTION: {query}

Please analyze the evidence and provide a detailed response to help with the investigation."""
        })
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return f"I'm having trouble generating a response right now. However, based on the evidence, I found {len([r for r in context.split('[Evidence') if 'Evidence' in r]) - 1} relevant pieces of evidence. Please try rephrasing your question or contact technical support."
    
    def _format_sources(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format search results as sources"""
        sources = []
        
        # Safety check
        if not search_results or not isinstance(search_results, list):
            return sources
        
        for result in search_results[:5]:  # Top 5 sources
            if not isinstance(result, dict):
                continue
                
            source = {
                "source_file": result.get("source_file", "unknown"),
                "record_type": result.get("record_type", "unknown"),
                "relevance_score": result.get("relevance_score", 0),
                "snippet": result.get("searchable_text", "")[:200],
                "data_summary": self._summarize_data(result.get("data", {}))
            }
            sources.append(source)
        
        return sources
    
    def _summarize_data(self, data: Dict[str, Any]) -> str:
        """Create a brief summary of the data"""
        if not data:
            return "No data available"
        
        # Check for common fields and create summary
        if "phone_number" in data and "duration" in data:
            return f"Call to/from {data.get('phone_number', '')} for {data.get('duration', '')} seconds"
        elif "sender" in data and "message" in data:
            return f"Message from {data.get('sender', '')} to {data.get('recipient', '')}"
        elif "contact_name" in data:
            return f"Contact: {data.get('contact_name', '')} - {data.get('phone_number', '')}"
        else:
            # Generic summary
            key_fields = [k for k in data.keys() if k in ["name", "phone", "number", "time", "date", "content"]]
            if key_fields:
                return f"Contains: {', '.join(key_fields)}"
            else:
                return f"Data record with {len(data)} fields"

# Global instance
simple_chat_handler = SimpleChatHandler()

async def process_case_query(case_id: str, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Global function to process a case query
    """
    return await simple_chat_handler.process_query(case_id, query, conversation_history)

if __name__ == "__main__":
    import asyncio
    
    async def test_chat():
        handler = SimpleChatHandler()
        
        # Test case ID
        case_id = "c0c91912-9ba2-4ecc-9af6-22e3296d562c"
        
        # Test queries
        queries = [
            "Who is Alex Rivera?",
            "Show me all calls from +1-555-0987",
            "What suspicious messages are in this case?",
            "Give me a summary of this case"
        ]
        
        for query in queries:
            print(f"\n=== Query: {query} ===")
            result = await handler.process_query(case_id, query)
            print(f"Response: {result['response']}")
            print(f"Sources: {len(result['sources'])}")
            print(f"Confidence: {result['confidence']}")
    
    # Run test
    asyncio.run(test_chat())