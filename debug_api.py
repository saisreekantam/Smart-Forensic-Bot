#!/usr/bin/env python3
"""Debug API startup issues"""

import sys
import os
sys.path.append('src')

print("🔍 Testing API module imports...")

try:
    print("1. Testing database import...")
    from src.database.models import Case, Evidence, EvidenceType, ProcessingStatus
    print("✅ Database models imported successfully")
    
    print("2. Testing case manager import...")
    from src.case_management.case_manager import CaseManager
    print("✅ Case manager imported successfully")
    
    print("3. Testing LangGraph assistant import...")
    from src.ai_cores.langgraph_assistant import LangGraphCaseAssistant
    print("✅ LangGraph assistant imported successfully")
    
    print("4. Testing API module import...")
    from src.api import case_api
    print("✅ API module imported successfully")
    
    print("5. Testing database connection...")
    import sqlite3
    conn = sqlite3.connect('data/forensic_cases.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM cases")
    case_count = cursor.fetchone()[0]
    print(f"✅ Database connected, found {case_count} cases")
    conn.close()
    
    print("6. Testing case manager initialization...")
    from src.case_management.case_manager import case_manager
    print("✅ Case manager imported successfully")
    
    print("7. Testing cases retrieval...")
    cases = case_manager.list_cases()
    print(f"✅ Retrieved {len(cases)} cases")
    for case in cases[:3]:  # Show first 3 cases
        print(f"   - {case.case_number}: {case.title}")
    
    print("\n🎉 All tests passed! API should work.")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()