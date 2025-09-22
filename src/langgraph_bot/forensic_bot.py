"""
Main Forensic Bot Application

This module implements the main LangGraph-based forensic investigation bot
that orchestrates all analysis capabilities into an intelligent assistant.
"""

import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, Any, Optional, List

# Import state management
from .state import ForensicBotState, create_initial_state, update_state_activity

# Import node functions
from .nodes.conversation_router import conversation_router, route_conversation, conversation_handler
from .nodes.rag_analyzer import rag_analyzer
from .nodes.evidence_processor import evidence_processor
from .nodes.knowledge_graph_reasoner import knowledge_graph_reasoner
from .nodes.pattern_detector import pattern_detector
from .nodes.synthesis_engine import synthesis_engine
from .nodes.report_generator import report_generator

class ForensicBot:
    """
    Main Forensic Investigation Bot using LangGraph
    
    This bot provides intelligent assistance for forensic investigations,
    combining evidence analysis, pattern detection, knowledge graph reasoning,
    and comprehensive reporting capabilities.
    """
    
    def __init__(
        self, 
        case_id: Optional[str] = None,
        debug_mode: bool = False,
        memory_path: Optional[str] = None
    ):
        """
        Initialize the Forensic Bot
        
        Args:
            case_id: Optional case ID to work with
            debug_mode: Enable debug mode for detailed logging
            memory_path: Path for persistent memory storage
        """
        self.case_id = case_id
        self.debug_mode = debug_mode
        self.memory_path = memory_path or "./data/bot_memory.db"
        
        # Initialize the LangGraph workflow
        self.workflow = self._create_workflow()
        self.memory = None  # TODO: Re-enable when SqliteSaver import is fixed
        self.app = self.workflow.compile()  # Compile without checkpointer for now
        
        # Session management
        self.session_id = None
        self.current_thread = None
        
        print("🔍 Forensic Investigation Bot initialized successfully!")
        if case_id:
            print(f"📁 Working with case: {case_id}")
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for the forensic bot"""
        
        # Create the state graph
        workflow = StateGraph(ForensicBotState)
        
        # Add all the nodes
        workflow.add_node("conversation_router", conversation_router)
        workflow.add_node("conversation_handler", conversation_handler)
        workflow.add_node("rag_analyzer", rag_analyzer)
        workflow.add_node("evidence_processor", evidence_processor)
        workflow.add_node("knowledge_graph_reasoner", knowledge_graph_reasoner)
        workflow.add_node("pattern_detector", pattern_detector)
        workflow.add_node("synthesis_engine", synthesis_engine)
        workflow.add_node("report_generator", report_generator)
        
        # Set the entry point
        workflow.set_entry_point("conversation_router")
        
        # Add conditional edges from the conversation router
        workflow.add_conditional_edges(
            "conversation_router",
            route_conversation,
            {
                "conversation_handler": "conversation_handler",
                "rag_analyzer": "rag_analyzer",
                "evidence_processor": "evidence_processor",
                "knowledge_graph_reasoner": "knowledge_graph_reasoner",
                "pattern_detector": "pattern_detector", 
                "synthesis_engine": "synthesis_engine",
                "report_generator": "report_generator"
            }
        )
        
        # All nodes lead to END for now (can be extended for multi-step workflows)
        workflow.add_edge("conversation_handler", END)
        workflow.add_edge("rag_analyzer", END)
        workflow.add_edge("evidence_processor", END)
        workflow.add_edge("knowledge_graph_reasoner", END)
        workflow.add_edge("pattern_detector", END)
        workflow.add_edge("synthesis_engine", END)
        workflow.add_edge("report_generator", END)
        
        return workflow
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new conversation session
        
        Args:
            session_id: Optional session ID, will generate one if not provided
            
        Returns:
            str: Session ID
        """
        import uuid
        
        if session_id is None:
            session_id = f"forensic_session_{uuid.uuid4()}"
        
        self.session_id = session_id
        self.current_thread = {"configurable": {"thread_id": session_id}}
        
        # Initialize the session state
        initial_state = create_initial_state(
            case_id=self.case_id,
            session_id=session_id,
            debug_mode=self.debug_mode
        )
        
        if self.debug_mode:
            print(f"🚀 Started new session: {session_id}")
        
        return session_id
    
    def chat(self, message: str) -> str:
        """
        Send a message to the forensic bot and get a response
        
        Args:
            message: User's message
            
        Returns:
            str: Bot's response
        """
        if not self.session_id:
            self.start_session()
        
        try:
            # Create human message
            human_message = HumanMessage(content=message)
            
            # Create initial state with the message for now (no persistent memory)
            state = create_initial_state(
                case_id=self.case_id,
                session_id=self.session_id,
                debug_mode=self.debug_mode
            )
            state["messages"] = [human_message]
            
            # Process the message through the workflow
            result = self.app.invoke(state)
            
            # Extract the bot's response
            if result and "messages" in result and result["messages"]:
                # Get the last AI message
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage):
                        return msg.content
            
            # Fallback response
            return "I received your message but couldn't generate a proper response. Please try again."
            
        except Exception as e:
            error_message = f"I encountered an error while processing your message: {str(e)}"
            if self.debug_mode:
                print(f"❌ Chat error: {e}")
                import traceback
                traceback.print_exc()
            return error_message
    
    async def chat_async(self, message: str) -> str:
        """
        Async version of chat method
        
        Args:
            message: User's message
            
        Returns:
            str: Bot's response
        """
        return self.chat(message)  # For now, just call the sync version
    
    def load_case(self, case_id: str) -> bool:
        """
        Load a specific case for analysis
        
        Args:
            case_id: Case identifier
            
        Returns:
            bool: True if case loaded successfully
        """
        try:
            self.case_id = case_id
            
            # Update current session state if active
            if self.session_id and self.current_thread:
                try:
                    current_state = self.app.get_state(self.current_thread)
                    if current_state.values:
                        state = current_state.values
                        state["current_case_id"] = case_id
                        
                        # Add a system message about case change
                        case_change_msg = AIMessage(
                            content=f"Switched to case {case_id}. I'm ready to help with this investigation."
                        )
                        state["messages"].append(case_change_msg)
                        
                        # Update the state
                        self.app.update_state(self.current_thread, state)
                except Exception as e:
                    if self.debug_mode:
                        print(f"⚠️ Error updating state with new case: {e}")
            
            if self.debug_mode:
                print(f"📁 Loaded case: {case_id}")
            
            return True
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Error loading case {case_id}: {e}")
            return False
    
    def get_session_state(self) -> Optional[Dict[str, Any]]:
        """
        Get the current session state
        
        Returns:
            Optional[Dict[str, Any]]: Current state or None if no session
        """
        # TODO: Implement when persistent memory is available
        if self.debug_mode:
            print("⚠️ Session state tracking not available without persistent memory")
        return None
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history for the current session
        
        Returns:
            List[Dict[str, Any]]: Conversation history
        """
        state = self.get_session_state()
        if not state or "messages" not in state:
            return []
        
        history = []
        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                history.append({
                    "type": "human",
                    "content": message.content,
                    "timestamp": datetime.now().isoformat()  # Simplified
                })
            elif isinstance(message, AIMessage):
                history.append({
                    "type": "ai",
                    "content": message.content,
                    "timestamp": datetime.now().isoformat()  # Simplified
                })
        
        return history
    
    def get_investigation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current investigation state
        
        Returns:
            Dict[str, Any]: Investigation summary
        """
        state = self.get_session_state()
        if not state:
            return {"error": "No active session"}
        
        return {
            "case_id": state.get("current_case_id"),
            "session_id": state.get("session_id"),
            "investigation_phase": state.get("investigation_phase"),
            "entities_count": len(state.get("entity_memory", {})),
            "events_count": len(state.get("timeline_memory", [])),
            "patterns_count": len(state.get("pattern_memory", {})),
            "evidence_count": len(state.get("active_evidence", [])),
            "key_findings": state.get("key_findings", []),
            "recommendations": state.get("recommendations", []),
            "tools_used": state.get("tools_used", []),
            "confidence_scores": state.get("confidence_scores", {}),
            "last_activity": state.get("last_activity")
        }
    
    def export_session_data(self, format: str = "json") -> str:
        """
        Export session data for backup or analysis
        
        Args:
            format: Export format ('json' or 'summary')
            
        Returns:
            str: Exported data
        """
        state = self.get_session_state()
        if not state:
            return "No active session to export"
        
        if format == "json":
            import json
            # Convert non-serializable objects to strings
            export_data = {}
            for key, value in state.items():
                try:
                    json.dumps(value)  # Test if serializable
                    export_data[key] = value
                except:
                    export_data[key] = str(value)
            
            return json.dumps(export_data, indent=2, default=str)
        
        elif format == "summary":
            summary = self.get_investigation_summary()
            
            export_text = f"""
Forensic Investigation Session Export
=====================================

Case ID: {summary.get('case_id', 'Unknown')}
Session ID: {summary.get('session_id', 'Unknown')}
Investigation Phase: {summary.get('investigation_phase', 'Unknown')}
Last Activity: {summary.get('last_activity', 'Unknown')}

Analysis Summary:
- Entities Identified: {summary.get('entities_count', 0)}
- Events Processed: {summary.get('events_count', 0)}
- Patterns Detected: {summary.get('patterns_count', 0)}
- Evidence Items: {summary.get('evidence_count', 0)}

Tools Used: {', '.join(summary.get('tools_used', []))}

Key Findings:
"""
            
            for i, finding in enumerate(summary.get("key_findings", []), 1):
                export_text += f"{i}. {finding}\n"
            
            export_text += "\nRecommendations:\n"
            for i, rec in enumerate(summary.get("recommendations", []), 1):
                export_text += f"{i}. {rec}\n"
            
            return export_text.strip()
        
        else:
            return f"Unsupported export format: {format}"
    
    def reset_session(self):
        """Reset the current session"""
        if self.session_id:
            self.start_session()  # This will create a new session
            if self.debug_mode:
                print("🔄 Session reset")
    
    def shutdown(self):
        """Shutdown the bot and cleanup resources"""
        if self.debug_mode:
            print("🛑 Forensic Bot shutting down...")
        
        # Could add cleanup logic here if needed
        self.session_id = None
        self.current_thread = None

def create_forensic_bot(
    case_id: Optional[str] = None,
    debug_mode: bool = False,
    memory_path: Optional[str] = None
) -> ForensicBot:
    """
    Factory function to create a ForensicBot instance
    
    Args:
        case_id: Optional case ID to work with
        debug_mode: Enable debug mode
        memory_path: Path for persistent memory storage
        
    Returns:
        ForensicBot: Configured bot instance
    """
    return ForensicBot(
        case_id=case_id,
        debug_mode=debug_mode,
        memory_path=memory_path
    )

# Example usage and testing
if __name__ == "__main__":
    # Create a forensic bot for testing
    bot = create_forensic_bot(debug_mode=True)
    
    # Start a session
    session_id = bot.start_session()
    print(f"Started session: {session_id}")
    
    # Test conversation
    test_messages = [
        "Hello, I need help with a forensic investigation",
        "Can you analyze the evidence in case DEMO-2024-001?",
        "What patterns have you detected?",
        "Generate a summary report"
    ]
    
    print("\n🤖 Testing Forensic Bot:")
    print("=" * 50)
    
    for message in test_messages:
        print(f"\n👤 User: {message}")
        response = bot.chat(message)
        print(f"🤖 Bot: {response}")
    
    # Show investigation summary
    print("\n📊 Investigation Summary:")
    print("=" * 50)
    summary = bot.get_investigation_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Shutdown
    bot.shutdown()