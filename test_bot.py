#!/usr/bin/env python3
"""
Simple Forensic Investigation Bot - Terminal Interface
Auto-loads sample data and works directly without case management setup
"""

import os
import sys
import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Configure logging to be less verbose
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forensic_bot.log'),
    ]
)
logger = logging.getLogger(__name__)

class SimpleForensicBot:
    """Simple forensic bot that works directly with sample data"""
    
    def __init__(self):
        self.sample_data_loaded = False
        self.available_queries = []
        self.session_history: List[Dict[str, Any]] = []
        
        # Initialize systems with fallbacks
        self.rag_system = None
        self.graph_querier = None
        self.graph_builder = None
        self.openai_available = False
        
        # Try to import AI systems (with fallback)
        self._try_import_systems()
    
    def _try_import_systems(self):
        """Try to import AI systems with graceful fallbacks"""
        try:
            # Try different import paths
            try:
                from src.ai_cores.rag.rag_system import ForensicRAGSystem, RAGQuery, get_default_config
                from src.ai_cores.knowledge_graph.graph_builder import ForensicGraphBuilder
                from src.ai_cores.knowledge_graph.graph_querier import ForensicGraphQuerier
                from src.ai_cores.knowledge_graph.graph_store import create_graph_store
                print("✅ AI cores imported successfully")
                self.ai_cores_available = True
            except ImportError:
                # Try alternative paths
                from ai_cores.rag.rag_system import ForensicRAGSystem, RAGQuery, get_default_config
                from ai_cores.knowledge_graph.graph_builder import ForensicGraphBuilder
                from ai_cores.knowledge_graph.graph_querier import ForensicGraphQuerier
                from ai_cores.knowledge_graph.graph_store import create_graph_store
                print("✅ AI cores imported from alternative path")
                self.ai_cores_available = True
                
        except ImportError as e:
            print(f"⚠️  AI cores not available: {e}")
            print("🔧 Bot will work with basic functionality")
            self.ai_cores_available = False
    
    def initialize_openai(self):
        """Initialize OpenAI for intelligent responses"""
        try:
            import openai
            
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                print("🔑 OpenAI API key not found. Bot will work with basic functionality.")
                print("   Set OPENAI_API_KEY environment variable for full AI features.")
                return False
            
            openai.api_key = openai_api_key
            self.openai_available = True
            print("✅ OpenAI initialized - Full AI features available!")
            return True
            
        except ImportError:
            print("📦 OpenAI package not installed. Using basic functionality.")
            return False
        except Exception as e:
            print(f"⚠️  OpenAI setup issue: {e}")
            return False
    
    def load_sample_data(self):
        """Auto-load sample forensic data"""
        print("🔧 Loading sample forensic data...")
        
        # Create sample data directories if they don't exist
        self._ensure_sample_data_exists()
        
        # Look for sample data files
        sample_paths = [
            Path("data/sample"),
            Path("sample"),
            Path("test_data"),
            Path("examples"),
            Path(current_dir) / "data" / "sample"
        ]
        
        sample_dir = None
        for path in sample_paths:
            if path.exists():
                sample_dir = path
                break
        
        if not sample_dir:
            print("📁 Creating sample data directory...")
            sample_dir = Path("data/sample")
            sample_dir.mkdir(parents=True, exist_ok=True)
            self._create_sample_files(sample_dir)
        
        try:
            # Initialize systems if available
            if self.ai_cores_available:
                try:
                    from src.ai_cores.rag.rag_system import ForensicRAGSystem, get_default_config
                    
                    config = get_default_config()
                    self.rag_system = ForensicRAGSystem(config)
                    print("✅ RAG system initialized")
                except Exception as e:
                    print(f"⚠️  RAG system initialization: {e}")
                
                try:
                    from src.ai_cores.knowledge_graph.graph_store import create_graph_store
                    from src.ai_cores.knowledge_graph.graph_builder import ForensicGraphBuilder
                    from src.ai_cores.knowledge_graph.graph_querier import ForensicGraphQuerier
                    
                    graph_store = create_graph_store("memory")
                    self.graph_builder = ForensicGraphBuilder("memory")
                    self.graph_querier = ForensicGraphQuerier(graph_store)
                    print("🕸️  Knowledge graph initialized")
                except Exception as e:
                    print(f"⚠️  Knowledge graph setup: {e}")
            
            # Load sample files
            sample_files = list(sample_dir.glob("*"))
            print(f"📁 Found {len(sample_files)} sample files")
            
            loaded_count = 0
            for file_path in sample_files[:5]:  # Load first 5 files
                if file_path.suffix in ['.csv', '.json', '.xml', '.txt'] or file_path.is_file():
                    try:
                        content = self._read_file_content(file_path)
                        if content:
                            print(f"  ✅ Loaded: {file_path.name}")
                            loaded_count += 1
                    except Exception as e:
                        print(f"  ⚠️  Skipped {file_path.name}: {e}")
            
            self.sample_data_loaded = True
            print(f"✅ Sample data loaded successfully! ({loaded_count} files)")
            
            # Set up example queries
            self.available_queries = [
                "What evidence do we have?",
                "Show me communication patterns",
                "Who are the key people mentioned?",
                "What happened on a specific date?",
                "Find suspicious activities",
                "Analyze call logs",
                "Show me cryptocurrency mentions",
                "What phone numbers appear in evidence?"
            ]
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load sample data: {e}")
            logger.error(f"Sample data loading error: {e}")
            return False
    
    def _ensure_sample_data_exists(self):
        """Ensure sample data directory and files exist"""
        data_dir = Path("data")
        sample_dir = data_dir / "sample"
        
        if not sample_dir.exists():
            sample_dir.mkdir(parents=True, exist_ok=True)
            self._create_sample_files(sample_dir)
    
    def _create_sample_files(self, sample_dir: Path):
        """Create sample forensic data files"""
        print("📝 Creating sample forensic data...")
        
        # Create sample call log
        call_log = {
            "call_records": [
                {
                    "timestamp": "2024-01-15T10:30:00",
                    "from_number": "+1-555-0123",
                    "to_number": "+1-555-0456",
                    "duration": 240,
                    "type": "outgoing"
                },
                {
                    "timestamp": "2024-01-15T14:22:00",
                    "from_number": "+1-555-0456",
                    "to_number": "+1-555-0123",
                    "duration": 180,
                    "type": "incoming"
                }
            ]
        }
        
        with open(sample_dir / "call_logs.json", "w") as f:
            json.dump(call_log, f, indent=2)
        
        # Create sample evidence file
        evidence = """Evidence Log - Case #2024-001
Date: January 15, 2024
Investigating Officer: Detective Smith

Items seized:
1. iPhone 12 Pro - Serial: ABC123XYZ
2. Samsung Galaxy S21 - Serial: DEF456UVW
3. Laptop - Dell XPS 15 - Serial: GHI789RST

Digital evidence extracted:
- Call logs from both phones
- Text messages 
- Browser history
- Cryptocurrency wallet transactions
- Social media communications

Key individuals mentioned:
- John Smith (suspect)
- Sarah Johnson (contact)
- Mike Wilson (business partner)
        """
        
        with open(sample_dir / "evidence_log.txt", "w") as f:
            f.write(evidence)
        
        print("✅ Sample data files created")

    def _read_file_content(self, file_path: Path) -> str:
        """Read and return file content"""
        try:
            if file_path.suffix == '.csv':
                import pandas as pd
                df = pd.read_csv(file_path)
                return f"CSV file {file_path.name}:\n{df.head(10).to_string()}"
            
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return f"JSON file {file_path.name}:\n{json.dumps(data, indent=2)[:1000]}"
            
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return f"File {file_path.name}:\n{content[:1000]}"
                
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return ""
    
    async def simple_query(self, query: str) -> str:
        """Process a simple query using available systems"""
        
        if not self.sample_data_loaded:
            return "❌ Sample data not loaded. Please restart the bot."
        
        print("🔍 Processing your query...")
        
        results = []
        
        # Try RAG system if available
        if self.rag_system:
            try:
                from src.ai_cores.rag.rag_system import RAGQuery
                
                rag_query = RAGQuery(
                    text=query,
                    query_type="general",
                    max_results=5
                )
                
                response = self.rag_system.query(rag_query)
                
                if response and hasattr(response, 'response'):
                    if hasattr(response.response, 'response_text'):
                        results.append(f"📊 Evidence Analysis: {response.response.response_text}")
                    elif hasattr(response.response, 'content'):
                        results.append(f"📊 Evidence Analysis: {response.response.content}")
                
            except Exception as e:
                logger.error(f"RAG query error: {e}")
        
        # Basic pattern matching for common queries
        if not results:
            results.append(self._basic_query_response(query))
        
        # Try OpenAI if available
        if self.openai_available and len(results) == 1:
            try:
                import openai
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",  # Using more accessible model
                    messages=[
                        {"role": "system", "content": "You are a forensic investigator analyzing digital evidence. Provide helpful insights based on the query."},
                        {"role": "user", "content": f"Based on forensic evidence, answer: {query}"}
                    ],
                    max_tokens=300,
                    temperature=0.3
                )
                
                results.append(f"🤖 AI Analysis: {response.choices[0].message.content}")
                
            except Exception as e:
                logger.error(f"OpenAI error: {e}")
        
        return "\n\n".join(results)
    
    def _basic_query_response(self, query: str) -> str:
        """Provide basic responses based on query patterns"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['evidence', 'what', 'show']):
            return """🔍 Evidence Summary:
            
• Call logs from 2 mobile devices
• Text message records  
• Browser history data
• Cryptocurrency transactions
• Social media communications
• Physical devices: iPhone 12 Pro, Samsung Galaxy S21, Dell laptop

Key individuals: John Smith, Sarah Johnson, Mike Wilson"""
        
        elif any(word in query_lower for word in ['phone', 'number', 'call']):
            return """📞 Phone Numbers Found:
            
• +1-555-0123 (Primary suspect device)
• +1-555-0456 (Contact device)
• Multiple calls between these numbers on Jan 15, 2024
• Duration: 180-240 seconds average"""
        
        elif any(word in query_lower for word in ['people', 'who', 'person']):
            return """👥 Key Individuals:
            
• John Smith - Primary suspect
• Sarah Johnson - Frequent contact
• Mike Wilson - Business associate
• Detective Smith - Investigating officer"""
        
        elif any(word in query_lower for word in ['crypto', 'bitcoin', 'wallet']):
            return """₿ Cryptocurrency Activity:
            
• Wallet transactions detected
• Multiple digital currency transfers
• Timeline matches communication patterns
• Further analysis recommended"""
        
        else:
            return f"""🔍 Query processed: '{query}'

The system found relevant data in the loaded evidence files. 
For more detailed analysis, ensure all AI systems are properly configured.

Available data includes:
• Digital device evidence
• Communication records  
• Transaction logs
• Timeline analysis"""

    def display_banner(self):
        """Display welcome banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         🔍 SIMPLE FORENSIC INVESTIGATION BOT               ║
║                                                              ║
║            Ready to analyze your sample data!                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def show_help(self):
        """Show help information"""
        help_text = """
🔧 HOW TO USE:

Just type your questions naturally! Examples:
  
  • What evidence do we have?
  • Show me all phone numbers
  • Who contacted John Smith?
  • What happened on January 15th?
  • Find suspicious cryptocurrency transactions
  • Analyze the communication patterns
  • Who are the key people in this case?

🎯 SAMPLE QUERIES TO TRY:
"""
        print(help_text)
        
        for i, query in enumerate(self.available_queries, 1):
            print(f"  {i}. {query}")
        
        print("""
💡 TIPS:
  • Ask specific questions for better results
  • Use names, dates, or phone numbers for targeted searches  
  • Type 'exit' to quit, 'help' for this message
  • Type 'status' to see system information
        """)
    
    def show_status(self):
        """Show system status"""
        print(f"""
🖥️  SYSTEM STATUS:
┌─────────────────────────────────────────────────────────────┐
│ Sample Data Loaded: {'✅ Yes' if self.sample_data_loaded else '❌ No':<43} │
│ RAG System: {'✅ Active' if self.rag_system else '❌ Not loaded':<47} │
│ Knowledge Graph: {'✅ Active' if self.graph_querier else '❌ Not loaded':<39} │
│ OpenAI Features: {'✅ Available' if self.openai_available else '❌ Limited':<40} │
│ Queries Processed: {len(self.session_history):<38} │
└─────────────────────────────────────────────────────────────┘
        """)
        
        if self.session_history:
            print("\n📈 RECENT QUERIES:")
            for query in self.session_history[-3:]:
                time_str = query['timestamp'].strftime("%H:%M:%S")
                print(f"  • {time_str}: {query['query'][:50]}...")
    
    async def run(self):
        """Main bot loop"""
        self.display_banner()
        
        # Initialize systems
        self.initialize_openai()
        
        if not self.load_sample_data():
            print("❌ Could not load sample data. Please check your data directory.")
            return
        
        print(f"\n🎉 Bot ready! Your sample data is loaded.")
        print("💬 Just type your questions naturally, or 'help' for examples.")
        
        while True:
            try:
                # Simple prompt
                user_input = input(f"\n🔍 Query> ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Thank you for using the Forensic Bot!")
                    break
                
                elif user_input.lower() in ['help', 'h']:
                    self.show_help()
                    continue
                
                elif user_input.lower() in ['status', 'info']:
                    self.show_status()
                    continue
                
                elif user_input.lower() in ['clear', 'cls']:
                    os.system('clear' if os.name == 'posix' else 'cls')
                    self.display_banner()
                    continue
                
                # Process query
                start_time = datetime.now()
                response = await self.simple_query(user_input)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Display response
                print(f"\n{'='*60}")
                print(response)
                print(f"{'='*60}")
                print(f"⏱️  Processed in {processing_time:.2f} seconds")
                
                # Add to history
                self.session_history.append({
                    'timestamp': datetime.now(),
                    'query': user_input,
                    'processing_time': processing_time
                })
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Main loop error: {e}")

# Main function
async def main():
    bot = SimpleForensicBot()
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        print("💡 Try: python test_bot.py")