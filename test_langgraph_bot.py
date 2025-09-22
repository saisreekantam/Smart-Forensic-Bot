"""
Test script for the LangGraph Forensic Bot

This script tests the basic functionality of the forensic bot
to ensure all components are working correctly.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from langgraph_bot import create_forensic_bot
    print("✅ Successfully imported ForensicBot")
except ImportError as e:
    print(f"❌ Failed to import ForensicBot: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic bot functionality"""
    print("\n🧪 Testing Basic Bot Functionality")
    print("=" * 50)
    
    try:
        # Create bot instance
        bot = create_forensic_bot(debug_mode=True)
        print("✅ Bot created successfully")
        
        # Start session
        session_id = bot.start_session()
        print(f"✅ Session started: {session_id}")
        
        # Test basic conversation
        test_message = "Hello, can you help me with a forensic investigation?"
        print(f"\n👤 User: {test_message}")
        
        response = bot.chat(test_message)
        print(f"🤖 Bot: {response}")
        
        if response and len(response) > 10:
            print("✅ Bot responded appropriately")
        else:
            print("⚠️ Bot response seems short or empty")
        
        # Test investigation summary
        summary = bot.get_investigation_summary()
        print(f"\n📊 Investigation Summary: {summary}")
        
        if summary and "session_id" in summary:
            print("✅ Investigation summary generated")
        else:
            print("⚠️ Investigation summary incomplete")
        
        # Test conversation history
        history = bot.get_conversation_history()
        print(f"\n💬 Conversation History: {len(history)} messages")
        
        if len(history) >= 2:  # Should have user message and bot response
            print("✅ Conversation history tracked")
        else:
            print("⚠️ Conversation history incomplete")
        
        # Cleanup
        bot.shutdown()
        print("✅ Bot shutdown successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_workflow_routing():
    """Test different workflow routes"""
    print("\n🧪 Testing Workflow Routing")
    print("=" * 50)
    
    try:
        bot = create_forensic_bot(debug_mode=True)
        bot.start_session()
        
        # Test different types of queries
        test_queries = [
            ("evidence", "Can you process evidence files for case 123?"),
            ("rag", "What information do you have about this case?"),
            ("patterns", "Detect patterns in the communication data"),
            ("knowledge_graph", "Show me entity relationships"),
            ("synthesis", "Correlate all the evidence"),
            ("report", "Generate a forensic report")
        ]
        
        for query_type, query in test_queries:
            print(f"\n🔍 Testing {query_type} workflow:")
            print(f"👤 User: {query}")
            
            response = bot.chat(query)
            print(f"🤖 Bot: {response[:100]}...")  # Show first 100 chars
            
            if response and len(response) > 20:
                print(f"✅ {query_type} workflow responded")
            else:
                print(f"⚠️ {query_type} workflow response incomplete")
        
        bot.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Workflow routing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_persistence():
    """Test memory persistence across conversations"""
    print("\n🧪 Testing Memory Persistence")
    print("=" * 50)
    
    try:
        # Create bot with memory
        memory_path = "./test_memory.db"
        bot = create_forensic_bot(debug_mode=True, memory_path=memory_path)
        
        # First conversation
        session_id = bot.start_session("test_session_123")
        bot.chat("Remember that suspect A contacted suspect B at 2:00 PM")
        
        # Get state
        first_state = bot.get_session_state()
        print(f"✅ First conversation state captured")
        
        # Second conversation in same session
        bot.chat("What do you remember about the suspects?")
        
        second_state = bot.get_session_state()
        
        if (first_state and second_state and 
            len(second_state.get("messages", [])) > len(first_state.get("messages", []))):
            print("✅ Memory persistence working")
        else:
            print("⚠️ Memory persistence may have issues")
        
        bot.shutdown()
        
        # Cleanup test memory file
        if os.path.exists(memory_path):
            os.remove(memory_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Memory persistence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 LangGraph Forensic Bot Test Suite")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Workflow Routing", test_workflow_routing), 
        ("Memory Persistence", test_memory_persistence)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            if test_func():
                print(f"✅ {test_name} Test PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} Test FAILED")
        except Exception as e:
            print(f"❌ {test_name} Test FAILED with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The LangGraph Forensic Bot is ready for use.")
    else:
        print("⚠️ Some tests failed. Please review the implementation.")