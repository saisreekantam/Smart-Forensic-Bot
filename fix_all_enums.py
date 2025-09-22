#!/usr/bin/env python3
"""
Comprehensive Database Enum Fix

This script fixes all enum value mismatches between the database 
and Python enum definitions.
"""

import sqlite3
import sys
import os

# Add src to path
sys.path.append('src')
from database.models import EvidenceType, CaseStatus, ProcessingStatus

def main():
    print("🔧 Comprehensive Database Enum Fix")
    print("=" * 50)
    
    # Connect to database
    db_path = 'data/forensic_cases.db'
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        print("1. Checking current database values...")
        
        # Check cases table status values
        print("\n📋 Cases table status values:")
        cursor.execute("SELECT DISTINCT status FROM cases")
        case_statuses = [row[0] for row in cursor.fetchall()]
        print(f"   Current: {case_statuses}")
        print(f"   Expected: {[s.value for s in CaseStatus]}")
        
        # Check evidence table evidence_type values
        print("\n📂 Evidence table evidence_type values:")
        cursor.execute("SELECT DISTINCT evidence_type FROM evidence")
        evidence_types = [row[0] for row in cursor.fetchall()]
        print(f"   Current: {evidence_types}")
        print(f"   Expected: {[t.value for t in EvidenceType]}")
        
        # Check evidence table processing_status values
        print("\n⚙️ Evidence table processing_status values:")
        cursor.execute("SELECT DISTINCT processing_status FROM evidence")
        processing_statuses = [row[0] for row in cursor.fetchall()]
        print(f"   Current: {processing_statuses}")
        print(f"   Expected: {[s.value for s in ProcessingStatus]}")
        
        print("\n2. Fixing enum values...")
        
        # Fix case status values (lowercase to match enum)
        print("   📋 Fixing case status values...")
        cursor.execute("UPDATE cases SET status = 'active' WHERE status = 'ACTIVE'")
        cursor.execute("UPDATE cases SET status = 'closed' WHERE status = 'CLOSED'")
        cursor.execute("UPDATE cases SET status = 'archived' WHERE status = 'ARCHIVED'")
        cursor.execute("UPDATE cases SET status = 'under_review' WHERE status = 'UNDER_REVIEW'")
        print(f"   ✅ Updated {cursor.rowcount} case status records")
        
        # Fix evidence type values (lowercase to match enum)
        print("   📂 Fixing evidence type values...")
        cursor.execute("UPDATE evidence SET evidence_type = 'chat' WHERE evidence_type = 'CHAT'")
        cursor.execute("UPDATE evidence SET evidence_type = 'call_log' WHERE evidence_type = 'CALL_LOG'")
        cursor.execute("UPDATE evidence SET evidence_type = 'contact' WHERE evidence_type = 'CONTACT'")
        cursor.execute("UPDATE evidence SET evidence_type = 'image' WHERE evidence_type = 'IMAGE'")
        cursor.execute("UPDATE evidence SET evidence_type = 'video' WHERE evidence_type = 'VIDEO'")
        cursor.execute("UPDATE evidence SET evidence_type = 'audio' WHERE evidence_type = 'AUDIO'")
        cursor.execute("UPDATE evidence SET evidence_type = 'document' WHERE evidence_type = 'DOCUMENT'")
        cursor.execute("UPDATE evidence SET evidence_type = 'xml_report' WHERE evidence_type = 'XML_REPORT'")
        cursor.execute("UPDATE evidence SET evidence_type = 'json_data' WHERE evidence_type = 'JSON_DATA'")
        cursor.execute("UPDATE evidence SET evidence_type = 'csv_data' WHERE evidence_type = 'CSV_DATA'")
        cursor.execute("UPDATE evidence SET evidence_type = 'text_report' WHERE evidence_type = 'TEXT_REPORT'")
        cursor.execute("UPDATE evidence SET evidence_type = 'other' WHERE evidence_type = 'OTHER'")
        print(f"   ✅ Fixed evidence type values")
        
        # Fix processing status values (lowercase to match enum)
        print("   ⚙️ Fixing processing status values...")
        cursor.execute("UPDATE evidence SET processing_status = 'pending' WHERE processing_status = 'PENDING'")
        cursor.execute("UPDATE evidence SET processing_status = 'processing' WHERE processing_status = 'PROCESSING'")
        cursor.execute("UPDATE evidence SET processing_status = 'completed' WHERE processing_status = 'COMPLETED'")
        cursor.execute("UPDATE evidence SET processing_status = 'failed' WHERE processing_status = 'FAILED'")
        print(f"   ✅ Fixed processing status values")
        
        # Commit changes
        conn.commit()
        
        print("\n3. Verifying fixes...")
        
        # Verify cases
        cursor.execute("SELECT DISTINCT status FROM cases")
        new_case_statuses = [row[0] for row in cursor.fetchall()]
        print(f"   📋 Cases status values: {new_case_statuses}")
        
        # Verify evidence types
        cursor.execute("SELECT DISTINCT evidence_type FROM evidence")
        new_evidence_types = [row[0] for row in cursor.fetchall()]
        print(f"   📂 Evidence types: {new_evidence_types}")
        
        # Verify processing statuses
        cursor.execute("SELECT DISTINCT processing_status FROM evidence")
        new_processing_statuses = [row[0] for row in cursor.fetchall()]
        print(f"   ⚙️ Processing statuses: {new_processing_statuses}")
        
        print("\n✅ Database enum fix completed successfully!")
        print("🚀 API server should now work without enum errors.")
        
    except Exception as e:
        print(f"❌ Error fixing database: {e}")
        conn.rollback()
        return 1
        
    finally:
        conn.close()
    
    return 0

if __name__ == "__main__":
    exit(main())