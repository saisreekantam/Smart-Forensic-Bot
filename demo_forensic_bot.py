"""
Standalone LangGraph Forensic Bot Demo

This demo shows a working LangGraph forensic bot with basic functionality
without complex dependencies.
"""

import sys
import os
from typing import Dict, Any, Optional, TypedDict, Annotated, Sequence, List
from datetime import datetime
import uuid

# Core LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Basic state structure for demo
class SimpleBotState(TypedDict):
    """Simplified state for demo bot"""
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    current_case_id: Optional[str]
    session_id: Optional[str]
    investigation_phase: str
    key_findings: List[str]
    conversation_context: Dict[str, Any]
    last_activity: str

def create_demo_state(case_id: Optional[str] = None, session_id: Optional[str] = None) -> SimpleBotState:
    """Create initial state for demo"""
    return {
        "messages": [],
        "current_case_id": case_id,
        "session_id": session_id or f"demo_{uuid.uuid4()}",
        "investigation_phase": "conversation",
        "key_findings": [],
        "conversation_context": {},
        "last_activity": datetime.now().isoformat()
    }

# Demo node functions
def conversation_node(state: SimpleBotState) -> SimpleBotState:
    """Handle general conversation"""
    last_message = state["messages"][-1] if state["messages"] else None
    
    if isinstance(last_message, HumanMessage):
        user_input = last_message.content.lower()
        
        # Simple response logic
        if "hello" in user_input or "hi" in user_input:
            response = "Hello! I'm the LangGraph Forensic Investigation Bot. I can help you analyze evidence, detect patterns, and generate reports. How can I assist you today?"
        
        elif "evidence" in user_input:
            response = "I can help you process and analyze evidence files. I support various types including chat logs, call records, documents, images, and more. Please specify the case ID and evidence type you'd like to analyze."
        
        elif "pattern" in user_input:
            response = "I can detect various patterns in forensic data including communication patterns, temporal patterns, behavioral anomalies, financial patterns, and geographical patterns. Would you like me to analyze patterns for a specific case?"
        
        elif "report" in user_input:
            response = "I can generate comprehensive forensic reports including executive summaries, detailed analysis, evidence inventory, timeline reports, entity analysis, and recommendations. Which type of report would you like?"
        
        elif "case" in user_input:
            response = "I can work with forensic cases. I can load case data, analyze evidence, track investigation progress, and generate insights. Please provide a case ID to get started."
        
        elif "knowledge graph" in user_input:
            response = "I can create and analyze knowledge graphs showing relationships between entities, events, and evidence. This helps identify connections and patterns that might not be obvious from individual pieces of evidence."
        
        elif "rag" in user_input or "search" in user_input or "query" in user_input:
            response = "I have RAG (Retrieval Augmented Generation) capabilities that allow me to search through case evidence and provide intelligent answers based on the available data. What would you like to search for?"
        
        else:
            response = "I understand you're asking about forensic investigation. I can help with evidence processing, pattern detection, knowledge graph analysis, RAG queries, and report generation. Could you be more specific about what you need?"
        
        # Add findings to state
        state["key_findings"].append(f"User inquiry: {user_input}")
        state["conversation_context"]["last_intent"] = "general_conversation"
    
    else:
        response = "I'm ready to help with your forensic investigation. Please tell me what you need assistance with."
    
    # Create AI response
    ai_message = AIMessage(content=response)
    state["messages"].append(ai_message)
    state["last_activity"] = datetime.now().isoformat()
    
    return state

def analysis_node(state: SimpleBotState) -> SimpleBotState:
    """Handle analysis requests"""
    last_message = state["messages"][-1] if state["messages"] else None
    
    if isinstance(last_message, HumanMessage):
        response = """I'm analyzing the available evidence. Here's what I found:

🔍 Evidence Analysis Results:
- Processed chat logs, call records, and document files
- Identified key entities: persons, phone numbers, locations
- Detected communication patterns and timeline events
- Found potential anomalies requiring investigation

📊 Pattern Detection:
- Communication frequency spikes detected
- Unusual activity patterns identified
- Potential relationship networks mapped

🧠 Knowledge Graph Insights:
- Entity relationships established
- Event correlations identified
- Investigation leads generated

Would you like me to generate a detailed report or focus on a specific aspect?"""
    
    else:
        response = "I'm ready to perform analysis. Please provide details about what you'd like me to analyze."
    
    # Add analysis findings
    state["key_findings"].extend([
        "Evidence analysis completed",
        "Patterns detected in communication data",
        "Knowledge graph relationships identified"
    ])
    
    state["investigation_phase"] = "analysis"
    state["conversation_context"]["last_intent"] = "analysis"
    
    ai_message = AIMessage(content=response)
    state["messages"].append(ai_message)
    state["last_activity"] = datetime.now().isoformat()
    
    return state

def router_node(state: SimpleBotState) -> SimpleBotState:
    """Route conversations to appropriate handlers"""
    # This is just a pass-through for demo - routing happens in conditional edges
    return state

def route_conversation(state: SimpleBotState) -> str:
    """Determine which node to route to"""
    last_message = state["messages"][-1] if state["messages"] else None
    
    if isinstance(last_message, HumanMessage):
        user_input = last_message.content.lower()
        
        # Simple routing logic
        analysis_keywords = ["analyze", "pattern", "evidence", "detect", "process", "report", "graph", "rag"]
        
        if any(keyword in user_input for keyword in analysis_keywords):
            return "analysis_node"
    
    return "conversation_node"

class DemoForensicBot:
    """Demo version of the Forensic Bot"""
    
    def __init__(self, memory_path: str = "./demo_memory.db"):
        self.memory_path = memory_path
        self.workflow = self._create_workflow()
        self.memory = SqliteSaver.from_conn_string(self.memory_path)
        self.app = self.workflow.compile(checkpointer=self.memory)
        self.current_thread = None
        
    def _create_workflow(self) -> StateGraph:
        """Create the demo workflow"""
        workflow = StateGraph(SimpleBotState)
        
        # Add nodes
        workflow.add_node("router", router_node)
        workflow.add_node("conversation_node", conversation_node)
        workflow.add_node("analysis_node", analysis_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional routing
        workflow.add_conditional_edges(
            "router",
            route_conversation,
            {
                "conversation_node": "conversation_node",
                "analysis_node": "analysis_node"
            }
        )
        
        # All nodes end
        workflow.add_edge("conversation_node", END)
        workflow.add_edge("analysis_node", END)
        
        return workflow
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new session"""
        if session_id is None:
            session_id = f"demo_session_{uuid.uuid4()}"
        
        self.current_thread = {"configurable": {"thread_id": session_id}}
        return session_id
    
    def chat(self, message: str) -> str:
        """Chat with the bot"""
        if not self.current_thread:
            self.start_session()
        
        try:
            # Get or create state
            try:
                current_state = self.app.get_state(self.current_thread)
                if current_state.values:
                    state = current_state.values
                else:
                    state = create_demo_state()
            except:
                state = create_demo_state()
            
            # Add user message
            user_message = HumanMessage(content=message)
            state["messages"].append(user_message)
            
            # Process through workflow
            result = self.app.invoke(state, config=self.current_thread)
            
            # Extract response
            if result and "messages" in result:
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage):
                        return msg.content
            
            return "I'm here to help with forensic investigations. What would you like to know?"
            
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try again."
    
    def get_summary(self) -> Dict[str, Any]:
        """Get session summary"""
        if not self.current_thread:
            return {"error": "No active session"}
        
        try:
            current_state = self.app.get_state(self.current_thread)
            if current_state.values:
                state = current_state.values
                return {
                    "session_id": state.get("session_id"),
                    "investigation_phase": state.get("investigation_phase"),
                    "key_findings": state.get("key_findings", []),
                    "message_count": len(state.get("messages", [])),
                    "last_activity": state.get("last_activity")
                }
        except:
            pass
        
        return {"error": "Could not get session summary"}
    
    def cleanup(self):
        """Cleanup resources"""
        if os.path.exists(self.memory_path):
            os.remove(self.memory_path)

def demo():
    """Run the demo"""
    print("🔍 LangGraph Forensic Bot Demo")
    print("=" * 50)
    
    # Create bot
    bot = DemoForensicBot()
    session_id = bot.start_session()
    print(f"Started session: {session_id}")
    
    # Demo conversation
    test_messages = [
        "Hello, I need help with a forensic investigation",
        "Can you analyze evidence for case DEMO-2024-001?",
        "What patterns can you detect in communication data?",
        "Generate a summary report of your findings"
    ]
    
    print("\n🤖 Demo Conversation:")
    print("-" * 30)
    
    for message in test_messages:
        print(f"\n👤 User: {message}")
        response = bot.chat(message)
        print(f"🤖 Bot: {response[:200]}...")
    
    # Show summary
    print(f"\n📊 Session Summary:")
    summary = bot.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    bot.cleanup()
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    demo()