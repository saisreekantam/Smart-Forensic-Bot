#!/usr/bin/env python3
"""
Simple Evidence Upload Script

This script creates sample evidence and uploads it via the API
to test that the processing and embedding generation works.
"""

import requests
import json
import os

def create_sample_file():
    """Create a sample evidence file"""
    
    sample_content = """
FORENSIC INVESTIGATION REPORT - DEMO CASE
==========================================

Case Number: DEMO-2024-001
Date: September 22, 2025
Investigator: Detective Johnson

EVIDENCE SUMMARY:
- Mobile device analysis completed
- Call logs extracted from device
- Text messages recovered (including deleted items)
- GPS location data mapped
- Contact list analyzed

KEY FINDINGS:
1. Suspicious communication patterns identified
2. Multiple contacts with international numbers
3. GPS data shows visits to unusual locations
4. Evidence of data tampering attempts

DIGITAL ARTIFACTS:
- 1,247 text messages (23 deleted/recovered)
- 456 call log entries
- 89 contact entries
- 34 location coordinates
- 12 installed applications

TECHNICAL DETAILS:
- Device: iPhone 14 Pro
- iOS Version: 16.4.1
- Extraction Method: UFDR (Universal Forensic Data Retrieval)
- Hash Verification: SHA-256 confirmed
- Chain of Custody: Maintained

RECOMMENDATIONS:
- Further investigation of international contacts
- Analysis of deleted message content
- Correlation with known criminal databases
- Timeline reconstruction needed

This is sample evidence data for testing the ForensicAI analysis system.
It contains various forensic keywords and patterns that should be detected
by the natural language processing and analysis capabilities.
"""
    
    os.makedirs("data/sample", exist_ok=True)
    filename = "data/sample/forensic_report_demo.txt"
    
    with open(filename, 'w') as f:
        f.write(sample_content)
    
    print(f"✅ Created sample file: {filename}")
    return filename, sample_content

def main():
    print("🚀 Starting evidence upload and processing...")
    print("=" * 60)
    
    # Create sample file
    filename, file_content = create_sample_file()
    
    try:
        # Get list of cases
        response = requests.get("http://localhost:8000/cases")
        
        if response.status_code == 200:
            cases = response.json()
            print(f"📋 Found {len(cases)} cases")
            
            if cases:
                # Use the first case
                case = cases[0]
                case_id = case['id']
                case_number = case['case_number']
                print(f"📁 Using case: {case_number} ({case_id})")
                
                # Upload evidence file
                files = {'file': (os.path.basename(filename), file_content, 'text/plain')}
                data = {
                    'evidence_type': 'other',  # Use 'other' type which should be supported
                    'title': f'Demo Evidence - {os.path.basename(filename)}',
                    'description': 'Sample evidence file for testing the forensic analysis system'
                }
                
                upload_response = requests.post(
                    f"http://localhost:8000/cases/{case_id}/evidence",
                    files=files,
                    data=data
                )
                
                if upload_response.status_code == 200:
                    result = upload_response.json()
                    print(f"✅ Upload successful!")
                    print(f"   Evidence ID: {result.get('evidence_id', 'N/A')}")
                    print(f"   Status: {result.get('status', 'N/A')}")
                    print(f"   Processing: {result.get('processing', 'N/A')}")
                    
                    # Wait a moment for processing to start
                    import time
                    print("⏳ Waiting for processing to start...")
                    time.sleep(3)
                    
                    # Check case status
                    case_response = requests.get(f"http://localhost:8000/cases/{case_id}")
                    if case_response.status_code == 200:
                        case_data = case_response.json()
                        print("📊 Case Status:")
                        print(f"   Total Evidence: {case_data['case']['total_evidence_count']}")
                        print(f"   Processed: {case_data['case']['processed_evidence_count']}")
                        print(f"   Progress: {case_data['case']['processing_progress']:.1f}%")
                    
                    print("\n✅ Setup complete! You can now:")
                    print("   1. Start the frontend: cd src/frontend && npm run dev")
                    print("   2. Navigate to: http://localhost:5173")
                    print(f"   3. Select case: {case_number}")
                    print("   4. Try queries like:")
                    print("      - 'Show me the forensic report findings'")
                    print("      - 'What suspicious activities were found?'")
                    print("      - 'Analyze the call logs and messages'")
                    
                else:
                    error_data = upload_response.json() if upload_response.headers.get('content-type', '').startswith('application/json') else {'detail': upload_response.text}
                    print(f"❌ Upload failed: {upload_response.status_code}")
                    print(f"   Error: {error_data}")
            else:
                print("❌ No cases found")
        else:
            print(f"❌ Failed to get cases: {response.status_code}")
    
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Make sure it's running on localhost:8000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("\n❌ Setup incomplete - please check API server status")

if __name__ == "__main__":
    main()