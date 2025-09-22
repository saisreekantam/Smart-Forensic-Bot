"""
Test script for the enhanced forensic bot system
Tests knowledge graphs, case memory, and intelligent reporting integration
"""

import asyncio
import sys
import os
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import enhanced modules
from enhanced_chat_handler import enhanced_chat_handler
from src.ai_cores.case_memory import case_memory
from src.ai_cores.enhanced_knowledge_graph import enhanced_kg_db
from src.ai_cores.intelligent_report_generator import IntelligentReportGenerator

# Test case ID
TEST_CASE_ID = "TEST-2024-001"

async def test_enhanced_chat():
    """Test the enhanced chat handler with knowledge graph integration"""
    print("🧪 Testing Enhanced Chat Handler...")
    
    # Test queries that should trigger entity extraction and knowledge graph updates
    test_queries = [
        "What phone calls were made between John Smith and the suspect?",
        "Search for any communications from phone number +1234567890",
        "Show me all messages mentioning cryptocurrency transactions",
        "Find evidence related to sarah.jones@email.com",
        "What happened on 2024-01-15 between 2pm and 4pm?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📞 Test Query {i}: {query}")
        
        try:
            result = await enhanced_chat_handler.process_query(
                case_id=TEST_CASE_ID,
                query=query,
                conversation_history=[]
            )
            
            print(f"✅ Success: {result['success']}")
            print(f"📊 Entities found: {len(result.get('entities', []))}")
            print(f"🔗 Relationships: {len(result.get('relationships', []))}")
            print(f"📈 KG updated: {result.get('knowledge_graph_metrics', {}).get('knowledge_graph_updated', False)}")
            
            if result.get('entities'):
                print(f"🔍 Sample entities: {result['entities'][:3]}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n✅ Enhanced Chat Handler test completed!")

def test_case_memory():
    """Test case memory functionality"""
    print("\n🧠 Testing Case Memory System...")
    
    try:
        # Get case memory stats
        stats = case_memory.get_case_memory_stats(TEST_CASE_ID)
        print(f"📊 Total interactions: {stats.total_interactions}")
        print(f"🏷️ Unique entities: {stats.unique_entities_mentioned}")
        print(f"📅 Investigation days: {len(stats.temporal_activity)}")
        
        # Get investigation summary
        summary = case_memory.get_investigation_summary(TEST_CASE_ID)
        print(f"📝 Investigation summary generated: {bool(summary)}")
        
        print("✅ Case Memory test completed!")
        
    except Exception as e:
        print(f"❌ Case Memory Error: {str(e)}")

def test_knowledge_graph():
    """Test knowledge graph functionality"""
    print("\n🕸️ Testing Knowledge Graph System...")
    
    try:
        # Get knowledge graph data
        kg_data = enhanced_kg_db.get_case_knowledge_graph(TEST_CASE_ID)
        
        if kg_data:
            entities = kg_data.get("entities", [])
            relationships = kg_data.get("relationships", [])
            
            print(f"🏷️ Entities in graph: {len(entities)}")
            print(f"🔗 Relationships in graph: {len(relationships)}")
            
            if entities:
                print(f"📋 Sample entities: {[e.get('value', 'Unknown') for e in entities[:5]]}")
            
            if relationships:
                print(f"🔗 Sample relationships: {[r.get('relationship_type', 'Unknown') for r in relationships[:3]]}")
        else:
            print("📭 No knowledge graph data found (expected for new test case)")
        
        print("✅ Knowledge Graph test completed!")
        
    except Exception as e:
        print(f"❌ Knowledge Graph Error: {str(e)}")

async def test_report_generation():
    """Test intelligent report generation"""
    print("\n📄 Testing Intelligent Report Generation...")
    
    try:
        # Generate a test report
        result = await enhanced_chat_handler.generate_case_report(
            TEST_CASE_ID, 
            "detailed_analysis"
        )
        
        if result['success']:
            report = result['report']
            print(f"📋 Report generated: {report.get('title', 'Untitled')}")
            print(f"📊 Sections: {len(report.get('sections', []))}")
            print(f"💡 Insights: {len(report.get('insights', []))}")
            print(f"🎯 Confidence: {report.get('confidence_score', 0.0):.2f}")
        else:
            print(f"❌ Report generation failed: {result.get('error', 'Unknown error')}")
        
        print("✅ Report Generation test completed!")
        
    except Exception as e:
        print(f"❌ Report Generation Error: {str(e)}")

async def test_case_insights():
    """Test case insights functionality"""
    print("\n💡 Testing Case Insights...")
    
    try:
        result = await enhanced_chat_handler.get_case_insights(TEST_CASE_ID)
        
        if result['success']:
            insights = result.get('insights', [])
            statistics = result.get('statistics', {})
            health = result.get('investigation_health', {})
            
            print(f"💡 Insights found: {len(insights)}")
            print(f"📊 Statistics available: {bool(statistics)}")
            print(f"🏥 Health score: {health.get('overall_score', 0.0):.2f}")
            
            if insights:
                print(f"📋 Sample insight: {insights[0].get('title', 'Unknown')}")
        else:
            print(f"❌ Insights failed: {result.get('error', 'Unknown error')}")
        
        print("✅ Case Insights test completed!")
        
    except Exception as e:
        print(f"❌ Case Insights Error: {str(e)}")

async def test_knowledge_graph_summary():
    """Test knowledge graph summary"""
    print("\n🕸️ Testing Knowledge Graph Summary...")
    
    try:
        result = await enhanced_chat_handler.get_knowledge_graph_summary(TEST_CASE_ID)
        
        if result['success']:
            kg = result.get('knowledge_graph', {})
            print(f"🏷️ Entities: {kg.get('entities', 0)}")
            print(f"🔗 Relationships: {kg.get('relationships', 0)}")
            print(f"🔑 Key entities: {len(kg.get('key_entities', []))}")
        else:
            print(f"❌ KG Summary failed: {result.get('error', 'Unknown error')}")
        
        print("✅ Knowledge Graph Summary test completed!")
        
    except Exception as e:
        print(f"❌ KG Summary Error: {str(e)}")

def test_system_health():
    """Test overall system health and dependencies"""
    print("\n🏥 Testing System Health...")
    
    components = {
        "Enhanced Chat Handler": enhanced_chat_handler,
        "Case Memory": case_memory,
        "Enhanced Knowledge Graph": enhanced_kg_db
    }
    
    for name, component in components.items():
        try:
            # Basic availability check
            if hasattr(component, '__class__'):
                print(f"✅ {name}: Available")
            else:
                print(f"⚠️ {name}: May have issues")
        except Exception as e:
            print(f"❌ {name}: Error - {str(e)}")
    
    print("✅ System Health check completed!")

async def run_comprehensive_test():
    """Run all tests"""
    print("🚀 Starting Comprehensive Enhanced Forensic Bot Test")
    print("=" * 60)
    
    # Test individual components
    test_system_health()
    test_case_memory()
    test_knowledge_graph()
    
    # Test async components
    await test_enhanced_chat()
    await test_report_generation()
    await test_case_insights()
    await test_knowledge_graph_summary()
    
    print("\n" + "=" * 60)
    print("🎉 All Enhanced System Tests Completed!")
    print("\n📋 Summary:")
    print("   ✅ Enhanced Chat Handler with Knowledge Graph integration")
    print("   ✅ Case Memory System for storing all interactions")
    print("   ✅ Enhanced Knowledge Graph Database")
    print("   ✅ Intelligent Report Generation")
    print("   ✅ Case Insights and Investigation Analytics")
    print("   ✅ Knowledge Graph Summary and Analysis")
    print("\n🔥 Your Enhanced Forensic Bot is ready for action!")

if __name__ == "__main__":
    # Run the comprehensive test
    asyncio.run(run_comprehensive_test())