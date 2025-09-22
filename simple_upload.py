#!/usr/bin/env python3
"""
Simple Evidence Upload and Processing Script

This script uploads sample evidence files to existing cases and processes them.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

async def upload_sample_evidence():
    """Upload sample evidence files to existing cases"""
    
    # Import after path setup
    from src.database.models import EvidenceType
    
    # Sample file with simple content for testing
    sample_content = """FORENSIC INVESTIGATION REPORT
    
Case: DEMO-2024-001
Subject: Mobile Device Analysis

CALL LOG ANALYSIS:
- Outgoing call to +1-555-0123 at 2024-09-15 14:30:00
- Incoming call from +1-555-0456 at 2024-09-15 15:45:00
- Missed call from +1-555-0789 at 2024-09-15 16:20:00

MESSAGE ANALYSIS:
- Text message to +1-555-0123: "Meeting at 7pm"
- Text message from +1-555-0456: "Documents are ready"
- Deleted message recovered: "Transfer completed"

CONTACT ANALYSIS:
- John Smith: +1-555-0123 (Work)
- Jane Doe: +1-555-0456 (Personal)
- Unknown Contact: +1-555-0789

GPS LOCATION DATA:
- 2024-09-15 14:25:00: 40.7128, -74.0060 (New York)
- 2024-09-15 15:40:00: 40.7589, -73.9851 (Manhattan)
- 2024-09-15 16:15:00: 40.7505, -73.9934 (Times Square)

FINDINGS:
- Device was active during the timeframe 14:00-17:00
- Multiple communications with known contacts
- Location data shows movement in NYC area
- No evidence of data tampering detected
"""
    
    # Create sample file
    sample_file = Path("data/sample/forensic_report_demo.txt")
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    with open(sample_file, 'w') as f:
        f.write(sample_content)
    
    print(f"✅ Created sample file: {sample_file}")
    
    # Test API upload
    try:
        import requests
        
        # Get existing cases
        response = requests.get("http://localhost:8000/cases")
        if response.status_code == 200:
            cases = response.json()
            print(f"📋 Found {len(cases)} cases")
            
            if cases:
                case = cases[0]  # Use first case
                case_id = case['id']
                print(f"📁 Using case: {case['case_number']} ({case_id})")
                
    # Upload evidence file
    files = {'file': (filename, file_content, 'text/plain')}
    data = {
        'evidence_type': 'other',  # Use 'other' type which should be supported
        'title': f'Demo Evidence - {filename}',
        'description': 'Sample evidence file for testing the forensic analysis system'
    }                    upload_response = requests.post(
                        f"http://localhost:8000/cases/{case_id}/evidence",
                        files=files,
                        data=data
                    )
                    
                    if upload_response.status_code == 200:
                        result = upload_response.json()
                        print(f"✅ Evidence uploaded successfully: {result['evidence_id']}")
                        print(f"   Status: {result['status']}")
                        print(f"   Processing: {result['processing']}")
                        
                        # Wait a bit for processing
                        await asyncio.sleep(3)
                        
                        # Check case status
                        case_response = requests.get(f"http://localhost:8000/cases/{case_id}")
                        if case_response.status_code == 200:
                            updated_case = case_response.json()
                            case_data = updated_case['case']
                            print(f"📊 Case updated:")
                            print(f"   Total evidence: {case_data['total_evidence_count']}")
                            print(f"   Processed: {case_data['processed_evidence_count']}")
                            print(f"   Progress: {case_data['processing_progress']:.1f}%")
                        
                        return case_id
                    else:
                        print(f"❌ Upload failed: {upload_response.status_code}")
                        print(f"   Error: {upload_response.text}")
            else:
                print("❌ No cases found")
        else:
            print(f"❌ Failed to get cases: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Make sure it's running on localhost:8000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    return None

async def test_chat_functionality(case_id: str):
    """Test the chat functionality with the case"""
    
    try:
        import requests
        
        test_queries = [
            "What evidence is available in this case?",
            "Show me call log information",
            "What locations were visited?",
            "Summarize the investigation findings"
        ]
        
        print(f"\n🤖 Testing chat functionality for case {case_id}...")
        
        for query in test_queries:
            print(f"\n💬 Query: {query}")
            
            response = requests.post(
                f"http://localhost:8000/cases/{case_id}/chat",
                json={
                    "message": query,
                    "case_id": case_id
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"🤖 Response: {result['response'][:200]}...")
                print(f"   Confidence: {result['confidence']:.2f}")
                print(f"   Sources: {len(result.get('sources', []))}")
            else:
                print(f"❌ Chat failed: {response.status_code}")
                print(f"   Error: {response.text}")
            
            await asyncio.sleep(1)  # Brief pause between queries
            
    except Exception as e:
        print(f"❌ Chat test error: {str(e)}")

async def main():
    """Main function"""
    print("🚀 Starting evidence upload and processing...")
    print("=" * 60)
    
    # Upload evidence
    case_id = await upload_sample_evidence()
    
    if case_id:
        # Test chat functionality
        await test_chat_functionality(case_id)
        
        print("\n" + "=" * 60)
        print("📋 SUMMARY")
        print("=" * 60)
        print("✅ Evidence uploaded and processed")
        print("✅ Chat functionality tested")
        print("\n🎯 You can now:")
        print("   1. Open the frontend at http://localhost:5173")
        print("   2. Select a case with processed evidence")
        print("   3. Try natural language queries in the chat interface")
    else:
        print("\n❌ Setup incomplete - please check API server status")

if __name__ == "__main__":
    asyncio.run(main())