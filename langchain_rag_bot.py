"""
Super Intelligent Forensic Investigation Bot
Integrates LangChain with existing forensic analysis systems
Uses GPT-5/GPT-5-mini based on query complexity
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import asyncio
import json
from enum import Enum

# LangChain imports
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool, StructuredTool
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.agents.agent import AgentExecutor
from pydantic import BaseModel, Field

# Import existing forensic systems (using exact paths from your project)
from src.case_management.case_manager import CaseManager
from src.ai_cores.rag.case_rag_system import CaseRAGSystem
from src.ai_cores.rag.rag_system import RAGQuery, RAGResponse
from src.ai_cores.knowledge_graph.graph_builder import ForensicGraphBuilder
from src.ai_cores.knowledge_graph.graph_querier import ForensicGraphQuerier
from src.ai_cores.knowledge_graph.graph_store import create_graph_store
from src.database.models import Case, Evidence
from src.data_ingestion.evidence_handlers import evidence_handler_factory

logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    """Query complexity levels for model selection"""
    SIMPLE = "simple"      # GPT-5-mini - basic facts, single evidence queries
    MODERATE = "moderate"  # GPT-5-mini - cross-evidence correlation, timeline
    COMPLEX = "complex"    # GPT-5 - multi-step reasoning, analysis
    EXPERT = "expert"      # GPT-5 - complex investigation strategy, legal analysis

class ForensicIntelligentBot:
    """
    Super intelligent forensic investigation bot that integrates all existing systems
    """
    
    def __init__(self, openai_api_key: str, case_manager: CaseManager):
        """Initialize the intelligent bot with existing systems"""
        self.openai_api_key = openai_api_key
        self.case_manager = case_manager
        
        # Initialize AI models
        self.gpt5_mini = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.1,
            openai_api_key=openai_api_key,
            max_tokens=2000
        )
        
        self.gpt5 = ChatOpenAI(
            model="gpt-4o", 
            temperature=0.2,
            openai_api_key=openai_api_key,
            max_tokens=4000
        )
        
        # Initialize existing systems
        self.rag_systems: Dict[str, CaseRAGSystem] = {}
        self.graph_builders: Dict[str, ForensicGraphBuilder] = {}
        self.graph_queriers: Dict[str, ForensicGraphQuerier] = {}
        
        # Enhanced conversation memory and context
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history", 
            k=20,  # Increased memory window
            return_messages=True,
            output_key="output"
        )
        
        # Context storage for smart responses
        self.conversation_context = {
            "last_search_results": [],
            "mentioned_entities": set(),
            "current_investigation_focus": None,
            "previous_queries": []
        }
        
        # Initialize LangChain tools
        self.tools = self._create_forensic_tools()
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        logger.info("✅ Forensic Intelligent Bot initialized successfully")
    
    def _needs_forensic_tools(self, query: str) -> bool:
        """Determine if the query requires forensic search tools or can be answered conversationally"""
        query_lower = query.lower()
        
        # Forensic keywords that indicate need for evidence search
        forensic_keywords = [
            'evidence', 'case', 'phone', 'call', 'message', 'chat', 'communication',
            'crypto', 'bitcoin', 'wallet', 'transaction', 'suspect', 'victim',
            'forensic', 'investigation', 'analyze', 'search', 'find', 'who',
            'when', 'where', 'what', 'show me', 'tell me about', 'timeline',
            'pattern', 'relationship', 'contact', 'email', 'address', 'location'
        ]
        
        # Casual conversation patterns
        casual_patterns = [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'thanks', 'thank you', 'goodbye', 'bye', 'see you',
            'what can you do', 'help', 'who are you', 'what are you'
        ]
        
        # Check for casual conversation first
        if any(pattern in query_lower for pattern in casual_patterns):
            return False
            
        # Check for forensic keywords
        if any(keyword in query_lower for keyword in forensic_keywords):
            return True
            
        # Default to conversation mode for unclear queries
        return False
    
    async def _handle_casual_conversation(self, query: str, model) -> str:
        """Handle casual conversation with memory and context"""
        
        # Get conversation history
        chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
        history_text = ""
        
        if chat_history:
            # Format recent conversation history
            recent_history = chat_history[-6:]  # Last 3 exchanges
            for msg in recent_history:
                if hasattr(msg, 'content'):
                    role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
                    history_text += f"{role}: {msg.content}\n"
        
        # Build context about current investigation state
        context_info = ""
        if self.conversation_context["mentioned_entities"]:
            entities = list(self.conversation_context["mentioned_entities"])[:5]
            context_info += f"Previously mentioned: {', '.join(entities)}\n"
        
        if self.conversation_context["current_investigation_focus"]:
            context_info += f"Current focus: {self.conversation_context['current_investigation_focus']}\n"
        
        system_message = f"""You are an intelligent forensic investigation assistant with memory of our conversation.

CONVERSATION HISTORY:
{history_text}

INVESTIGATION CONTEXT:
{context_info}

You should:
1. Remember our previous conversation and refer to it naturally
2. If this is a follow-up question about previous forensic analysis, use that context intelligently
3. For greetings and casual conversation, respond warmly but professionally
4. If they ask about previous results without needing new searches, answer from memory
5. Only suggest searching for new evidence if they ask something you haven't analyzed before

Respond naturally and conversationally while maintaining your forensic expertise."""

        messages = [
            HumanMessage(content=system_message),
            HumanMessage(content=f"User: {query}")
        ]
        
        response = await model.agenerate([messages])
        result = response.generations[0][0].text.strip()
        
        # Save interaction to memory
        self.memory.save_context({"input": query}, {"output": result})
        
        return result
    
    def _build_conversation_context(self, current_query: str) -> str:
        """Build rich context from conversation history and stored data"""
        context_parts = []
        
        # Recent conversation history
        chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
        if chat_history:
            context_parts.append("RECENT CONVERSATION:")
            for msg in chat_history[-6:]:  # Last 3 exchanges
                if hasattr(msg, 'content'):
                    role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
                    content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                    context_parts.append(f"{role}: {content}")
        
        # Previously mentioned entities
        if self.conversation_context["mentioned_entities"]:
            entities = list(self.conversation_context["mentioned_entities"])[:10]
            context_parts.append(f"\nPREVIOUSLY DISCUSSED ENTITIES: {', '.join(entities)}")
        
        # Recent queries for pattern recognition
        if self.conversation_context["previous_queries"]:
            recent_queries = self.conversation_context["previous_queries"][-3:]
            context_parts.append(f"\nRECENT QUERIES: {' | '.join(recent_queries)}")
        
        # Current investigation focus
        if self.conversation_context["current_investigation_focus"]:
            context_parts.append(f"\nCURRENT FOCUS: {self.conversation_context['current_investigation_focus']}")
        
        return "\n".join(context_parts) if context_parts else "No previous conversation context."
    
    def _extract_and_store_entities(self, query: str, response: str) -> None:
        """Extract and store important entities from query and response"""
        import re
        
        text_to_analyze = f"{query} {response}"
        
        # Extract phone numbers
        phone_pattern = r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
        phones = re.findall(phone_pattern, text_to_analyze)
        for phone in phones:
            clean_phone = re.sub(r'[^\d+]', '', phone)
            if len(clean_phone) >= 10:
                self.conversation_context["mentioned_entities"].add(clean_phone)
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text_to_analyze)
        for email in emails:
            self.conversation_context["mentioned_entities"].add(email)
        
        # Extract crypto addresses (simplified pattern)
        crypto_pattern = r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b'
        crypto_addresses = re.findall(crypto_pattern, text_to_analyze)
        for addr in crypto_addresses:
            self.conversation_context["mentioned_entities"].add(addr)
        
        # Extract names (basic pattern for capitalized words)
        name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        names = re.findall(name_pattern, text_to_analyze)
        for name in names:
            if name not in ["Building", "Case", "User Query", "Action Input"]:  # Filter out common false positives
                self.conversation_context["mentioned_entities"].add(name)
        
        # Update investigation focus based on query content
        if any(word in query.lower() for word in ['phone', 'call', 'contact']):
            self.conversation_context["current_investigation_focus"] = "Communication Analysis"
        elif any(word in query.lower() for word in ['crypto', 'bitcoin', 'wallet', 'payment']):
            self.conversation_context["current_investigation_focus"] = "Financial Investigation"
        elif any(word in query.lower() for word in ['location', 'building', 'address', 'meet']):
            self.conversation_context["current_investigation_focus"] = "Location Intelligence"
    
    def _generate_smart_follow_ups(self, query: str, response: str) -> List[str]:
        """Generate intelligent follow-up suggestions based on context and response"""
        suggestions = []
        
        # Analyze what was found in the response
        response_lower = response.lower()
        
        # If phone numbers were mentioned
        if any(phone in self.conversation_context["mentioned_entities"] for phone in ['+1-555-0987', '+1-555-0123', '+44-7911-123456']):
            suggestions.extend([
                "Who else was in contact with these phone numbers?",
                "What's the timeline of these communications?",
                "Are there any suspicious patterns in the call logs?"
            ])
        
        # If crypto was mentioned
        if any(word in response_lower for word in ['crypto', 'bitcoin', 'wallet', 'btc']):
            suggestions.extend([
                "What other cryptocurrency transactions occurred?",
                "Can you trace the wallet addresses?",
                "What are the amounts involved in these transactions?"
            ])
        
        # If locations were mentioned
        if any(word in response_lower for word in ['building', 'location', 'address', 'meet']):
            suggestions.extend([
                "What other locations are mentioned in the evidence?",
                "Are there any GPS coordinates in the data?",
                "What's the significance of these meeting places?"
            ])
        
        # Based on investigation focus
        if self.conversation_context["current_investigation_focus"] == "Communication Analysis":
            suggestions.extend([
                "What apps were used for communication?",
                "Show me the message timestamps",
                "Who initiated the conversations?"
            ])
        
        # Remove duplicates and limit
        suggestions = list(dict.fromkeys(suggestions))[:4]
        return suggestions
    
    def _create_forensic_tools(self) -> List[Tool]:
        """Create LangChain tools that interface with existing forensic systems"""
        
        # Define input schema for search tool
        class SearchInput(BaseModel):
            query: str = Field(description="The search query to look for in forensic evidence files")
        
        def search_evidence(query: str) -> str:
            """Search all available evidence using direct file access"""
            try:
                # Search in sample data files
                evidence_found = []
                sample_dir = Path("data/sample")
                
                if sample_dir.exists():
                    for file_path in sample_dir.glob("*"):
                        if file_path.suffix in ['.csv', '.json', '.xml', '.txt']:
                            try:
                                content = file_path.read_text(encoding='utf-8')
                                # Simple search in content
                                if any(term.lower() in content.lower() for term in query.split()):
                                    evidence_found.append({
                                        'file': file_path.name,
                                        'content_preview': content[:500] + "..." if len(content) > 500 else content
                                    })
                            except Exception as e:
                                continue
                
                if evidence_found:
                    result = f"Found {len(evidence_found)} relevant evidence files:\n"
                    for evidence in evidence_found[:3]:  # Limit to first 3 results
                        result += f"\n📄 {evidence['file']}:\n{evidence['content_preview']}\n"
                    return result
                else:
                    return f"No evidence found matching query: {query}"
                    
            except Exception as e:
                return f"Error searching evidence: {str(e)}"
        
        def query_knowledge_graph(case_id: str, query_type: str, entity_value: str = "") -> str:
            """Query knowledge graph for entity relationships and patterns"""
            try:
                # Get or create graph systems for case
                if case_id not in self.graph_queriers:
                    case = self.case_manager.get_case(case_id)
                    if not case:
                        return f"Error: Case {case_id} not found"
                    
                    # Initialize graph systems using your exact methods
                    graph_store = create_graph_store("memory")
                    self.graph_builders[case_id] = ForensicGraphBuilder("memory")
                    self.graph_queriers[case_id] = ForensicGraphQuerier(graph_store)
                
                querier = self.graph_queriers[case_id]
                
                # Execute different query types using your exact methods
                if query_type == "communications":
                    results = querier.find_communications(entity_value)
                    if not results:
                        return f"No communications found for {entity_value}"
                    
                    comms = []
                    for comm in results[:5]:
                        participant = comm['participant']
                        direction = comm['direction']
                        comms.append(f"- {direction.title()}: {participant.value} ({participant.type})")
                    
                    return f"Communications for {entity_value}:\n" + "\n".join(comms)
                
                elif query_type == "ownership":
                    results = querier.find_ownership_network(entity_value)
                    if "error" in results:
                        return results["error"]
                    
                    owned_items = [f"- {item['item'].value} ({item['item'].type})" 
                                  for item in results.get('owned_items', [])]
                    
                    return f"Assets owned by {entity_value}:\n" + "\n".join(owned_items)
                
                elif query_type == "crypto":
                    results = querier.find_crypto_network()
                    crypto_info = []
                    for crypto in results[:5]:
                        crypto_info.append(f"- {crypto['entity'].value}")
                    
                    return "Cryptocurrency addresses found:\n" + "\n".join(crypto_info)
                
                elif query_type == "insights":
                    insights = querier.get_graph_insights()
                    return f"""Graph Analysis:
- Total Entities: {insights['graph_size']['entities']}
- Total Relationships: {insights['graph_size']['relationships']}
- Crypto Addresses: {insights['crypto_addresses']}
- Communication Patterns: {insights['communication_patterns']}"""
                
                else:
                    return f"Unknown query type: {query_type}"
                    
            except Exception as e:
                logger.error(f"Error querying knowledge graph: {e}")
                return f"Error querying knowledge graph: {str(e)}"
        
        def get_case_info(case_id: str) -> str:
            """Get comprehensive case information"""
            try:
                # Use your exact method
                case = self.case_manager.get_case(case_id)
                if not case:
                    return f"Case {case_id} not found"
                
                # Get evidence list using your exact method
                evidence_list = self.case_manager.get_case_evidence(case_id)
                
                evidence_summary = []
                for evidence in evidence_list[:10]:  # Limit to first 10
                    evidence_summary.append(f"- {evidence.file_name} ({evidence.evidence_type})")
                
                return f"""Case Information:
Case Number: {case.case_number}
Title: {case.title}
Investigator: {case.investigator_name}
Status: {case.status}
Priority: {case.priority}
Created: {case.created_at}

Evidence ({len(evidence_list)} total):
{chr(10).join(evidence_summary)}"""
                
            except Exception as e:
                logger.error(f"Error getting case info: {e}")
                return f"Error getting case information: {str(e)}"
        
        def analyze_timeline(case_id: str, start_date: str = "", end_date: str = "") -> str:
            """Analyze case timeline and events"""
            try:
                # Search for temporal events using RAG
                if start_date and end_date:
                    query = f"events between {start_date} and {end_date}"
                else:
                    query = "timeline chronological events sequence"
                
                # Use RAG system for timeline analysis
                if case_id not in self.rag_systems:
                    case = self.case_manager.get_case(case_id)
                    if not case:
                        return f"Error: Case {case_id} not found"
                    self.rag_systems[case_id] = CaseRAGSystem(case)
                
                # Create temporal RAG query using your exact parameters
                rag_query = RAGQuery(
                    text=query,
                    query_type="temporal",
                    max_results=15,
                    include_reasoning=True
                )
                
                response = self.rag_systems[case_id].query(rag_query)
                
                return f"""Timeline Analysis:
{response.response.content}

Key temporal patterns identified from evidence sources."""
                
            except Exception as e:
                logger.error(f"Error analyzing timeline: {e}")
                return f"Error analyzing timeline: {str(e)}"
        
        def investigate_entity(case_id: str, entity_name: str) -> str:
            """Deep investigation of a specific entity (person, phone, address, etc.)"""
            try:
                # Combine RAG search and knowledge graph analysis
                results = []
                
                # 1. RAG search for entity
                if case_id in self.rag_systems:
                    rag_query = RAGQuery(
                        text=f"information about {entity_name}",
                        query_type="entity",
                        max_results=8
                    )
                    rag_response = self.rag_systems[case_id].query(rag_query)
                    results.append(f"Evidence Analysis:\n{rag_response.response.content}")
                
                # 2. Knowledge graph analysis
                if case_id in self.graph_queriers:
                    querier = self.graph_queriers[case_id]
                    
                    # Try different query types
                    comms = querier.find_communications(entity_name)
                    if comms:
                        comm_list = [f"- {c['participant'].value}" for c in comms[:3]]
                        results.append(f"Communications:\n" + "\n".join(comm_list))
                
                return "\n\n".join(results) if results else f"No information found for {entity_name}"
                
            except Exception as e:
                logger.error(f"Error investigating entity: {e}")
                return f"Error investigating entity: {str(e)}"
        
        # Create LangChain tools
        tools = [
            StructuredTool.from_function(
                name="search_evidence",
                description="Search through all available forensic evidence files for specific information, patterns, or entities. Input should be the search query as a string.",
                func=search_evidence,
                args_schema=SearchInput
            ),
            Tool(
                name="query_knowledge_graph", 
                description="Query knowledge graph for entity relationships. Format: case_id|||query_type|||entity_value. Query types: communications, ownership, crypto, insights",
                func=lambda params: query_knowledge_graph(*params.split("|||")) if params.count("|||") >= 2 else query_knowledge_graph(params, "insights", "")
            ),
            Tool(
                name="get_case_information",
                description="Get basic case information including evidence list, case details, and status.",
                func=get_case_info
            ),
            Tool(
                name="analyze_timeline",
                description="Analyze case timeline and temporal relationships. Format: case_id|||start_date|||end_date (dates optional)",
                func=lambda params: analyze_timeline(*params.split("|||")) if "|||" in params else analyze_timeline(params, "", "")
            ),
            Tool(
                name="investigate_entity",
                description="Deep investigation of specific entity (person, phone, email, address). Format: case_id|||entity_name",
                func=lambda params: investigate_entity(*params.split("|||", 1)) if "|||" in params else investigate_entity(params, "")
            )
        ]
        
        return tools
    
    def _analyze_query_complexity(self, query: str, case_id: str = "") -> QueryComplexity:
        """Analyze query complexity to select appropriate GPT model"""
        
        # Simple patterns - use GPT-5-mini
        simple_patterns = [
            "what is", "who is", "when did", "where is",
            "show me", "list", "find", "get information"
        ]
        
        # Complex patterns - use GPT-5  
        complex_patterns = [
            "analyze", "investigate", "correlate", "compare",
            "pattern", "suspicious", "timeline", "relationship",
            "strategy", "recommend", "assess", "evaluate"
        ]
        
        # Expert patterns - use GPT-5
        expert_patterns = [
            "legal", "court", "prosecution", "defense",
            "expert opinion", "forensic analysis", "chain of custody"
        ]
        
        query_lower = query.lower()
        
        # Check for expert level
        if any(pattern in query_lower for pattern in expert_patterns):
            return QueryComplexity.EXPERT
        
        # Check for complex level  
        if any(pattern in query_lower for pattern in complex_patterns):
            return QueryComplexity.COMPLEX
            
        # Check for simple level
        if any(pattern in query_lower for pattern in simple_patterns):
            return QueryComplexity.SIMPLE
            
        # Default to moderate
        return QueryComplexity.MODERATE
    
    def _get_forensic_system_prompt(self, complexity: QueryComplexity) -> str:
        """Get system prompt based on query complexity"""
        
        base_prompt = """You are a highly skilled forensic investigator AI assistant specializing in digital evidence analysis. 
        
Your expertise includes:
- Digital forensics and evidence analysis
- Criminal investigation techniques
- Communication pattern analysis
- Financial crime investigation
- Timeline reconstruction
- Network analysis
- Legal procedures and chain of custody

Guidelines:
- Always maintain forensic integrity in your analysis
- Cite specific evidence sources when making claims
- Be objective and factual in your responses
- Consider multiple investigative angles
- Suggest next steps when appropriate
- Flag potential areas for further investigation
"""
        
        if complexity == QueryComplexity.SIMPLE:
            return base_prompt + """
            
Focus on providing clear, direct answers to basic questions. Keep responses concise and factual."""
            
        elif complexity == QueryComplexity.MODERATE:
            return base_prompt + """
            
Provide thorough analysis connecting multiple pieces of evidence. Include reasoning for your conclusions."""
            
        elif complexity == QueryComplexity.COMPLEX:
            return base_prompt + """
            
Conduct deep analytical investigation. Consider complex relationships, patterns, and multi-step reasoning. 
Provide strategic investigation recommendations."""
            
        elif complexity == QueryComplexity.EXPERT:
            return base_prompt + """
            
Provide expert-level forensic analysis suitable for legal proceedings. Consider all forensic best practices,
potential challenges, and provide comprehensive investigation strategy with legal considerations."""
            
        return base_prompt
    
    def _initialize_agents(self) -> Dict[str, AgentExecutor]:
        """Initialize LangChain agents for different complexity levels"""
        
        agents = {}
        
        # Simple/Moderate agent with GPT-5-mini
        agents['simple'] = initialize_agent(
            tools=self.tools,
            llm=self.gpt5_mini,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            max_iterations=3,
            early_stopping_method="generate"
        )
        
        # Complex/Expert agent with GPT-5
        agents['complex'] = initialize_agent(
            tools=self.tools,
            llm=self.gpt5,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            max_iterations=5,
            early_stopping_method="generate"
        )
        
        return agents
    
    async def investigate(self, query: str, case_id: str = None, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main investigation method - processes queries with full intelligence
        """
        try:
            start_time = datetime.now()
            
            # First, determine if this requires forensic tools or is just casual conversation
            needs_forensic_search = self._needs_forensic_tools(query)
            
            # Analyze query complexity
            complexity = self._analyze_query_complexity(query, case_id)
            
            # Select appropriate model
            if complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]:
                model = self.gpt5_mini
                model_used = "gpt-4o-mini"
            else:
                model = self.gpt5
                model_used = "gpt-4o"
            
            if not needs_forensic_search:
                # Handle as normal conversation without tools
                response = await self._handle_casual_conversation(query, model)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    "status": "success",
                    "response": response,
                    "metadata": {
                        "query_complexity": complexity.value,
                        "model_used": model_used,
                        "processing_time": processing_time,
                        "case_id": case_id,
                        "timestamp": datetime.now().isoformat(),
                        "tools_used": [],
                        "conversation_mode": True
                    },
                    "suggestions": [],
                    "confidence": 1.0
                }
            else:
                # Use forensic tools for investigation
                agent = self.agents['simple'] if complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE] else self.agents['complex']
                
                # Build rich context from conversation history
                context_info = self._build_conversation_context(query)
                
                # Enhanced query with conversation context
                available_data = "Available forensic evidence files: messages_case001.csv, call_logs_case001.csv, structured_data_case001.json, sample_ufdr_case001.xml, and other sample data."
                full_query = f"""
CONVERSATION CONTEXT:
{context_info}

FORENSIC DATA AVAILABLE:
{available_data}

CURRENT INVESTIGATION QUERY: {query}

Instructions: Use the available forensic tools to search for evidence. Maintain awareness of previous conversation context and provide intelligent responses that build on our discussion.
"""
                
                # Get system prompt
                system_prompt = self._get_forensic_system_prompt(complexity)
                
                # Execute investigation using LangChain agent
                logger.info(f"🔍 Investigating query with {model_used} (complexity: {complexity.value})")
                
                response = await agent.arun(full_query)
                
                # Extract and store entities from this interaction
                self._extract_and_store_entities(query, response)
                
                # Save interaction to memory
                self.memory.save_context({"input": query}, {"output": response})
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Prepare comprehensive response
                investigation_result = {
                    "status": "success",
                    "response": response,
                    "metadata": {
                        "query_complexity": complexity.value,
                        "model_used": model_used,
                        "processing_time": processing_time,
                        "case_id": case_id,
                        "timestamp": datetime.now().isoformat(),
                        "tools_used": [tool.name for tool in self.tools],
                        "conversation_mode": False,
                        "context_entities": list(self.conversation_context["mentioned_entities"]),
                        "investigation_focus": self.conversation_context["current_investigation_focus"]
                    },
                    "suggestions": self._generate_smart_follow_ups(query, response),
                    "confidence": self._assess_response_confidence(response, complexity)
                }
                
                logger.info(f"✅ Investigation completed in {processing_time:.2f}s")
                return investigation_result
            
        except Exception as e:
            logger.error(f"❌ Investigation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "response": f"I apologize, but I encountered an error while investigating: {str(e)}",
                "confidence": 0.0,
                "metadata": {
                    "query_complexity": complexity.value if 'complexity' in locals() else "unknown",
                    "model_used": "none",
                    "processing_time": 0,
                    "case_id": case_id
                }
            }
    
    def _generate_follow_up_suggestions(self, query: str, response: str, case_id: str) -> List[str]:
        """Generate intelligent follow-up questions based on the investigation"""
        
        suggestions = []
        
        # Entity-based suggestions
        if "phone" in query.lower() or "call" in query.lower():
            suggestions.extend([
                "Who else was in contact with this phone number?",
                "What was the timeline of these communications?",
                "Are there any suspicious call patterns?"
            ])
        
        # Timeline-based suggestions
        if "when" in query.lower() or "time" in query.lower():
            suggestions.extend([
                "What other events occurred around this timeframe?",
                "Are there any corroborating timestamps in other evidence?"
            ])
        
        # General investigation suggestions
        if not suggestions:
            suggestions = [
                f"What other evidence is related to this finding?",
                f"Show me the knowledge graph connections for entities mentioned",
                f"Analyze the timeline around these events"
            ]
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _assess_response_confidence(self, response: str, complexity: QueryComplexity) -> float:
        """Assess confidence level of the response"""
        
        confidence = 0.7  # Base confidence
        
        # Adjust based on complexity
        if complexity == QueryComplexity.SIMPLE:
            confidence = 0.9
        elif complexity == QueryComplexity.MODERATE:
            confidence = 0.8
        elif complexity == QueryComplexity.COMPLEX:
            confidence = 0.7
        elif complexity == QueryComplexity.EXPERT:
            confidence = 0.6
        
        # Adjust based on response indicators
        if "Error" in response or "not found" in response:
            confidence *= 0.5
        elif "evidence shows" in response.lower() or "based on" in response.lower():
            confidence *= 1.1
        
        return min(confidence, 1.0)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "active_rag_systems": len(self.rag_systems),
            "active_graph_builders": len(self.graph_builders),
            "active_graph_queriers": len(self.graph_queriers),
            "memory_conversations": len(self.memory.chat_memory.messages),
            "available_tools": [tool.name for tool in self.tools],
            "models_configured": ["gpt-5", "gpt-5-mini"],
            "status": "operational"
        }

# Factory function to create the bot
def create_forensic_bot(openai_api_key: str, database_url: str = None) -> ForensicIntelligentBot:
    """
    Factory function to create and configure the forensic bot
    
    Args:
        openai_api_key: OpenAI API key for GPT models
        database_url: Database URL for case management (optional)
    
    Returns:
        Configured ForensicIntelligentBot instance
    """
    
    # Initialize database manager and case manager with your existing system
    from src.database.models import DatabaseManager, db_manager
    # Use the global db_manager instance instead of creating new one
    case_manager = CaseManager(db_manager)
    
    # Create and return the bot
    bot = ForensicIntelligentBot(openai_api_key, case_manager)
    
    logger.info("🤖 Forensic Intelligent Bot created and ready for investigations!")
    
    return bot

# Interactive test interface
if __name__ == "__main__":
    async def interactive_bot():
        print("🔍 FORENSIC INVESTIGATION BOT - Interactive Test Mode")
        print("=" * 60)
        
        try:
            # Initialize bot
            print("🤖 Initializing bot...")
            bot = create_forensic_bot(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                database_url=os.getenv("DATABASE_URL", "sqlite:///data/forensic_cases.db")
            )
            print("✅ Bot initialized successfully!")
            print()
            
            # Interactive loop
            print("📋 Available data in sample files:")
            print("   - messages_case001.csv (Chat communications)")
            print("   - call_logs_case001.csv (Call records)")
            print("   - Various other forensic evidence files")
            print()
            print("💡 Example queries:")
            print("   - 'What phone numbers appear in the evidence?'")
            print("   - 'Show me cryptocurrency wallet addresses'")
            print("   - 'Who are the key people mentioned?'")
            print("   - 'Find suspicious communication patterns'")
            print("   - 'What evidence do we have about building 7?'")
            print("   - 'Show me all WhatsApp messages'")
            print()
            
            while True:
                print("-" * 60)
                query = input("Enter your investigation query (or 'quit' to exit): ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                print(f"\n🔍 Investigating: '{query}'")
                print("⏳ Processing...")
                
                start_time = datetime.now()
                result = await bot.investigate(query=query)
                
                print("\n" + "="*60)
                print("📊 INVESTIGATION RESULT")
                print("="*60)
                print(f"🤖 Response: {result['response']}")
                print(f"📈 Confidence: {result.get('confidence', 'N/A')}")
                print(f"🧠 Model Used: {result['metadata']['model_used']}")
                print(f"⏱️  Processing Time: {result['metadata']['processing_time']:.2f}s")
                print(f"🔧 Query Complexity: {result['metadata']['query_complexity']}")
                
                if 'suggestions' in result:
                    print(f"\n💡 Follow-up suggestions:")
                    for i, suggestion in enumerate(result['suggestions'][:3], 1):
                        print(f"   {i}. {suggestion}")
                
                print()
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Run interactive mode
    asyncio.run(interactive_bot())