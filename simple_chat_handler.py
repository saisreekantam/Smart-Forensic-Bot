"""
Simple Chat Handler using our Search System
A replacement for the complex LangGraph system that actually works with our processed data
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from simple_search_system import SimpleSearchSystem

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class SimpleChatHandler:
    """
    Simple chat handler that uses our search system and OpenAI directly
    """
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = AsyncOpenAI(api_key=self.openai_api_key)
        self.search_system = SimpleSearchSystem()
    
    async def process_query(self, case_id: str, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a user query by first determining if it's forensic-related or general conversation
        
        Args:
            case_id: The case ID to search in
            query: User's question
            conversation_history: Previous conversation messages
            
        Returns:
            Response with answer and sources
        """
        try:
            # First, determine if this is a forensic query or general conversation
            is_forensic_query = await self._is_forensic_query(query)
            
            if not is_forensic_query:
                # Handle as general conversation
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