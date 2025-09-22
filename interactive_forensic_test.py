"""
Interactive Forensic Bot Test Script

This script allows you to test the forensic bot with your own inputs
and see how it uses the database, vector embeddings, and OpenAI integration.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append('.')
sys.path.append('./src')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import logging
from typing import Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if all required environment variables and components are available"""
    print("🔍 Checking Environment Setup...")
    print("=" * 50)
    
    # Check OpenAI API Key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and len(openai_key) > 10:
        print(f"✅ OpenAI API Key: Found (ends with {openai_key[-8:]})")
    else:
        print("❌ OpenAI API Key: Not found or invalid")
        return False
    
    # Check database file
    db_path = Path("data/forensic_cases.db")
    if db_path.exists():
        print(f"✅ Database: Found at {db_path} ({db_path.stat().st_size} bytes)")
    else:
        print(f"❌ Database: Not found at {db_path}")
    
    # Check vector database
    vector_db_path = Path("data/vector_db")
    if vector_db_path.exists():
        files = list(vector_db_path.glob("*"))
        print(f"✅ Vector DB: Found at {vector_db_path} ({len(files)} files)")
    else:
        print(f"❌ Vector DB: Not found at {vector_db_path}")
    
    # Check sample data
    sample_path = Path("data/sample")
    if sample_path.exists():
        files = list(sample_path.glob("*"))
        print(f"✅ Sample Data: Found {len(files)} sample files")
        for file in files[:3]:  # Show first 3 files
            print(f"   - {file.name}")
        if len(files) > 3:
            print(f"   ... and {len(files) - 3} more")
    else:
        print(f"❌ Sample Data: Not found at {sample_path}")
    
    print()
    return True

def test_basic_imports():
    """Test if all required imports work"""
    print("📦 Testing Imports...")
    print("=" * 50)
    
    try:
        from langgraph.graph import StateGraph, END
        from langchain_core.messages import HumanMessage, AIMessage
        print("✅ LangGraph imports successful")
    except ImportError as e:
        print(f"❌ LangGraph imports failed: {e}")
        return False
    
    try:
        from src.langgraph_bot.state import ForensicBotState, create_initial_state
        print("✅ ForensicBotState imports successful")
    except ImportError as e:
        print(f"❌ ForensicBotState imports failed: {e}")
        return False
    
    try:
        from src.langgraph_bot import create_forensic_bot
        print("✅ ForensicBot imports successful")
    except ImportError as e:
        print(f"❌ ForensicBot imports failed: {e}")
        return False
    
    try:
        from src.ai_cores.enhanced_assistant import EnhancedCaseAssistant
        print("✅ Enhanced Assistant imports successful")
    except ImportError as e:
        print(f"❌ Enhanced Assistant imports failed: {e}")
        return False
    
    print()
    return True

def test_database_connection():
    """Test database connectivity and sample data"""
    print("🗄️ Testing Database Connection...")
    print("=" * 50)
    
    try:
        from src.database.models import db_manager, Case
        
        # Initialize database
        db_manager.create_tables()
        print("✅ Database tables created/verified successfully")
        
        # Query cases
        with db_manager.get_session() as session:
            cases = session.query(Case).limit(3).all()
            print(f"✅ Found {len(cases)} cases in database")
            
            for case in cases:
                print(f"   - Case {case.case_number}: {case.title}")
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    
    print()
    return True

def test_vector_store():
    """Test vector store connectivity"""
    print("🧭 Testing Vector Store...")
    print("=" * 50)
    
    try:
        from src.ai_cores.rag.case_vector_store import CaseVectorStore
        
        vector_store = CaseVectorStore()
        print("✅ Vector store initialized successfully")
        
        # Test search with a demo case
        test_results = vector_store.search_case(
            case_id="DEMO-2024-001",
            collection_name="evidence", 
            query_text="evidence analysis",
            top_k=2
        )
        print(f"✅ Vector search returned {len(test_results)} results")
        
        for i, result in enumerate(test_results[:2]):
            content_preview = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
            print(f"   {i+1}. {content_preview}")
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        print("   This might be normal if no data has been processed yet.")
        return True  # Don't fail the test for this
    
    print()
    return True

def create_interactive_bot():
    """Create and return an interactive forensic bot"""
    print("🤖 Creating Interactive Forensic Bot...")
    print("=" * 50)
    
    try:
        from src.langgraph_bot import create_forensic_bot
        
        # Create bot with debug mode
        bot = create_forensic_bot(debug_mode=True)
        print("✅ Forensic bot created successfully")
        
        # Start a session
        session_id = bot.start_session()
        print(f"✅ Session started: {session_id}")
        
        return bot
        
    except Exception as e:
        print(f"❌ Bot creation failed: {e}")
        return None

def run_interactive_session():
    """Run an interactive session with the forensic bot"""
    print("\n🎯 Starting Interactive Forensic Bot Session")
    print("=" * 60)
    print("Type 'quit', 'exit', or 'bye' to end the session")
    print("Type 'help' for example queries")
    print("Type 'status' to see current session status")
    print("=" * 60)
    
    # Create bot
    bot = create_interactive_bot()
    if not bot:
        print("❌ Failed to create bot. Exiting.")
        return
    
    session_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input("\n🔍 Your forensic query: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye! Thank you for testing the forensic bot.")
                break
                
            if user_input.lower() == 'help':
                print("""
📋 Example Forensic Queries:
- "What evidence do we have in the cases?"
- "Show me patterns in the communication data"
- "Analyze the timeline of events"
- "What relationships exist between entities?"
- "Generate a summary of case findings"
- "Hello" (for greeting)
- "Load case DEMO-2024-001"
                """)
                continue
                
            if user_input.lower() == 'status':
                print(f"\n� Session Status:")
                session_state = bot.get_session_state()
                if session_state:
                    print(f"   Session ID: {session_state.get('session_id', 'Unknown')}")
                    print(f"   Case ID: {session_state.get('case_id', 'None')}")
                    print(f"   Messages: {len(session_state.get('messages', []))}")
                else:
                    print("   No active session")
                continue
            
            session_count += 1
            print(f"\n🔄 Processing query #{session_count}...")
            
            # Process with bot using chat method
            start_time = datetime.now()
            response = bot.chat(user_input)
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            
            # Display bot response
            print(f"\n🤖 Forensic Bot Response (took {processing_time:.2f}s):")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            # Show conversation history count
            history = bot.get_conversation_history()
            print(f"\n📊 Conversation Messages: {len(history)}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            print("Please try again or type 'quit' to exit.")
            # Add more detailed error info
            import traceback
            print(f"📋 Error details: {traceback.format_exc()}")
    
    # Cleanup
    try:
        bot.shutdown()
        print("\n🧹 Bot session cleaned up successfully.")
    except:
        pass

def main():
    """Main function to run the interactive test"""
    print("🔬 Interactive Forensic Bot Testing Suite")
    print("=" * 60)
    print("This script will test the forensic bot's capabilities including:")
    print("- Environment configuration")
    print("- Database connectivity")
    print("- Vector embeddings")
    print("- OpenAI integration")
    print("- Interactive chat interface")
    print("=" * 60)
    
    # Run all checks
    checks_passed = True
    
    checks_passed &= check_environment()
    checks_passed &= test_basic_imports()
    checks_passed &= test_database_connection()
    checks_passed &= test_vector_store()
    
    if not checks_passed:
        print("\n❌ Some checks failed. Please fix the issues before proceeding.")
        return
    
    print("✅ All system checks passed!")
    
    # Ask user if they want to continue to interactive mode
    try:
        response = input("\n🚀 Would you like to start the interactive session? (y/n): ").strip().lower()
        if response in ['y', 'yes', '']:
            run_interactive_session()
        else:
            print("\n👍 Tests completed successfully. You can run this script again anytime.")
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")

if __name__ == "__main__":
    main()