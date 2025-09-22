#!/usr/bin/env python3
"""
Data Processing and Embedding Generation Script

This script ensures that existing case data is properly processed and
embeddings are generated for the RAG system to work effectively.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.database.models import DatabaseManager, Case, Evidence, EvidenceType, ProcessingStatus
from src.case_management.case_manager import CaseManager
from src.ai_cores.rag.case_vector_store import CaseVectorStore
from src.data_ingestion.case_processor import CaseDataProcessor
from src.data_ingestion.optimized_processor import OptimizedCaseProcessor

async def ensure_sample_cases_exist():
    """Ensure sample cases exist in the database"""
    
    # Initialize components
    db_manager = DatabaseManager()
    case_manager = CaseManager(db_manager)
    
    print("🔍 Checking existing cases...")
    
    # Check for DEMO case
    existing_cases = case_manager.list_cases()
    case_numbers = [case.case_number for case in existing_cases]
    
    print(f"Found {len(existing_cases)} existing cases: {case_numbers}")
    
    # Create sample cases if they don't exist
    if "DEMO-2024-001" not in case_numbers:
        print("📝 Creating DEMO-2024-001 case...")
        from src.case_management.case_manager import CaseCreateRequest
        
        demo_request = CaseCreateRequest(
            case_number="DEMO-2024-001",
            title="Mobile Device Investigation - Demonstration Case",
            investigator_name="Detective Sarah Johnson",
            description="Demonstration case showing mobile device forensic analysis capabilities",
            department="Digital Forensics Unit",
            priority="medium",
            case_type="Mobile Forensics",
            jurisdiction="State Police"
        )
        
        demo_case = case_manager.create_case(demo_request)
        print(f"✅ Created case: {demo_case.case_number}")
    
    if "ENHANCED-2024-001" not in case_numbers:
        print("📝 Creating ENHANCED-2024-001 case...")
        from src.case_management.case_manager import CaseCreateRequest
        
        enhanced_request = CaseCreateRequest(
            case_number="ENHANCED-2024-001",
            title="Advanced Digital Investigation - Enhanced Analysis",
            investigator_name="Detective Michael Chen",
            description="Advanced investigation case with multiple evidence types",
            department="Cybercrime Unit",
            priority="high",
            case_type="Cybercrime Investigation",
            jurisdiction="Federal Bureau"
        )
        
        enhanced_case = case_manager.create_case(enhanced_request)
        print(f"✅ Created case: {enhanced_case.case_number}")
    
    return case_manager

async def process_sample_files():
    """Process sample files and generate embeddings"""
    
    case_manager = await ensure_sample_cases_exist()
    
    # Initialize vector store and processor
    vector_store = CaseVectorStore()
    processor = OptimizedCaseProcessor(
        case_manager=case_manager,
        vector_store=vector_store,
        max_workers=2
    )
    
    print("\n📁 Processing sample files...")
    
    # Get sample files
    sample_dir = Path("data/sample")
    sample_files = [
        ("call_logs_case001.csv", EvidenceType.CSV_DATA),
        ("messages_case001.csv", EvidenceType.CSV_DATA),
        ("sample_ufdr_case001.xml", EvidenceType.XML_REPORT),
        ("structured_data_case001.json", EvidenceType.JSON_DATA),
        ("text_report_case002.txt", EvidenceType.TEXT_REPORT)
    ]
    
    # Get the DEMO case
    demo_case = case_manager.get_case_by_number("DEMO-2024-001")
    if not demo_case:
        print("❌ DEMO case not found!")
        return
    
    print(f"📂 Adding evidence to case: {demo_case.case_number}")
    
    for filename, evidence_type in sample_files:
        file_path = sample_dir / filename
        if not file_path.exists():
            print(f"⚠️  File not found: {filename}")
            continue
        
        print(f"📄 Processing: {filename}")
        
        try:
            # Check if evidence already exists
            existing_evidence = case_manager.get_case_evidence(demo_case.id)
            if any(ev.original_filename == filename for ev in existing_evidence):
                print(f"📄 Evidence {filename} already exists, processing...")
                evidence = next(ev for ev in existing_evidence if ev.original_filename == filename)
            else:
                # Add evidence to case
                from src.case_management.case_manager import EvidenceUploadRequest
                
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                
                upload_request = EvidenceUploadRequest(
                    case_id=demo_case.id,
                    original_filename=filename,
                    evidence_type=evidence_type,
                    title=f"Sample {evidence_type.value}",
                    description=f"Sample evidence file for demonstration"
                )
                
                evidence = case_manager.add_evidence(upload_request, file_data)
                print(f"✅ Added evidence: {evidence.original_filename}")
            
            # Process the evidence
            print(f"🔄 Processing evidence: {evidence.original_filename}")
            result = await processor.process_evidence_fast(
                case_id=demo_case.id,
                evidence_id=evidence.id,
                file_path=evidence.file_path,
                evidence_type=evidence_type
            )
            
            if result.get('success'):
                print(f"✅ Processed: {evidence.original_filename} - {result.get('chunks_created', 0)} chunks created")
            else:
                print(f"❌ Failed to process: {evidence.original_filename}")
                
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
    
    # Get case statistics to show current state
    stats = case_manager.get_case_statistics(demo_case.id)
    print(f"\n📊 Case statistics retrieved for {demo_case.case_number}")
    print(f"   - Processing status: {stats.get('processing', {})}")
    
    return demo_case

async def verify_embeddings():
    """Verify that embeddings were created correctly"""
    
    print("\n🔍 Verifying embeddings...")
    
    vector_store = CaseVectorStore()
    
    # Test search
    try:
        results = await vector_store.search(
            case_id="demo_2024_001",  # Collection name format
            query="mobile phone call logs",
            top_k=5
        )
        
        print(f"✅ Vector search successful - found {len(results)} results")
        
        if results:
            print("📋 Sample search results:")
            for i, result in enumerate(results[:3]):
                print(f"  {i+1}. Score: {result.get('score', 0):.3f} - {result.get('text', '')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector search failed: {str(e)}")
        return False

async def main():
    """Main processing function"""
    print("🚀 Starting data processing and embedding generation...")
    print("=" * 60)
    
    try:
        # Ensure sample cases exist and process files
        demo_case = await process_sample_files()
        
        # Verify embeddings
        embeddings_ok = await verify_embeddings()
        
        print("\n" + "=" * 60)
        print("📋 PROCESSING SUMMARY")
        print("=" * 60)
        
        if demo_case:
            print(f"✅ Demo case ready: {demo_case.case_number}")
            print(f"   - Case ID: {demo_case.id}")
            print(f"   - Evidence count: {demo_case.total_evidence_count}")
            print(f"   - Processed: {demo_case.processed_evidence_count}")
        
        if embeddings_ok:
            print("✅ Vector embeddings verified")
        else:
            print("❌ Vector embeddings need attention")
        
        print("\n🎯 Next steps:")
        print("   1. Start the frontend: cd src/frontend && npm run dev")
        print("   2. Navigate to: http://localhost:5173")
        print("   3. Select the DEMO-2024-001 case")
        print("   4. Try queries like 'Show me call logs' or 'Find mobile phone data'")
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())