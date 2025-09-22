"""
Simplified test for LangGraph Forensic Bot

This test script creates a minimal version of the bot to check core functionality
without complex dependencies.
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

def test_minimal_bot():
    """Test minimal bot functionality with basic nodes only"""
    print("🧪 Testing Minimal LangGraph Forensic Bot")
    print("=" * 50)
    
    try:
        # Test basic imports first
        from langgraph.graph import StateGraph, END
        from langchain_core.messages import HumanMessage, AIMessage
        print("✅ LangGraph core imports successful")
        
        # Test state import
        from src.langgraph_bot.state import ForensicBotState, create_initial_state
        print("✅ State management imports successful")
        
        # Test conversation router (minimal dependencies)
        from src.langgraph_bot.nodes.conversation_router import conversation_router, conversation_handler
        
        # Create a simple route function that only routes to available nodes
        def simple_route_conversation(state):
            """Simple routing for minimal test"""
            return "conversation_handler"
        
        print("✅ Conversation router imports successful")
        
        # Create a minimal workflow without persistent memory
        workflow = StateGraph(ForensicBotState)
        workflow.add_node("conversation_router", conversation_router)
        workflow.add_node("conversation_handler", conversation_handler)
        
        workflow.set_entry_point("conversation_router")
        
        workflow.add_conditional_edges(
            "conversation_router",
            simple_route_conversation,
            {
                "conversation_handler": "conversation_handler"
            }
        )
        
        workflow.add_edge("conversation_handler", END)
        
        # Compile the workflow without checkpointer for now
        app = workflow.compile()
        print("✅ Minimal workflow compiled successfully")
        
        # Test initial state creation
        initial_state = create_initial_state(
            case_id="TEST-001",
            session_id="test_session",
            debug_mode=True
        )
        print("✅ Initial state created successfully")
        
        # Test a simple conversation
        test_message = HumanMessage(content="hi")
        initial_state["messages"] = [test_message]
        
        result = app.invoke(initial_state)
        
        print("✅ Workflow execution successful")
        
        # Check for response
        if result and "messages" in result:
            for msg in result["messages"]:
                if isinstance(msg, AIMessage):
                    print(f"🤖 Bot responded: {msg.content[:100]}...")
                    break
        
        print("✅ Minimal LangGraph Forensic Bot test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Minimal bot test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_minimal_bot()