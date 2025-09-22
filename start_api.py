#!/usr/bin/env python3
"""
Simple Project Sentinel Startup Script
Starts the FastAPI backend with OpenAI chatbot integration
"""

import os
import sys
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def check_openai_key():
    """Check if OpenAI API key is configured"""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or not openai_key.startswith("sk-"):
        print("⚠️  Warning: OpenAI API key not configured properly!")
        print("   The chatbot will use fallback responses without the API key.")
        print("   To enable AI features, set OPENAI_API_KEY in the .env file.")
        print("   Example: OPENAI_API_KEY=sk-your-actual-api-key-here")
        return False
    else:
        print("✅ OpenAI API key configured")
        return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/cases",
        "data/vector_db",
        "data/processed", 
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")

def start_api_server():
    """Start the FastAPI server"""
    try:
        print("🚀 Starting Project Sentinel API server...")
        print("📊 Backend API: http://localhost:8000")
        print("📋 API Docs: http://localhost:8000/docs")
        print("🛑 Press Ctrl+C to stop")
        
        # Import the FastAPI app
        from src.api.case_api import app
        
        # Start the server
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🔍 Project Sentinel - Forensic Investigation Platform")
    print("=" * 60)
    
    # Setup
    create_directories()
    check_openai_key()
    
    print("\n🎯 Starting API server...")
    start_api_server()