#!/usr/bin/env python3
"""
Fix SQLAlchemy Enum Values

SQLAlchemy expects enum names, not values. This script updates 
the database to use enum names instead of values.
"""

import sqlite3
import sys
import os

def main():
    print("🔧 Fixing SQLAlchemy Enum Values")
    print("=" * 50)
    
    # Connect to database
    db_path = 'data/forensic_cases.db'
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        print("1. Checking current values...")
        
        # Check current values
        cursor.execute("SELECT DISTINCT status FROM cases")
        case_statuses = [row[0] for row in cursor.fetchall()]
        print(f"   Cases status: {case_statuses}")
        
        cursor.execute("SELECT DISTINCT evidence_type FROM evidence")
        evidence_types = [row[0] for row in cursor.fetchall()]
        print(f"   Evidence types: {evidence_types}")
        
        cursor.execute("SELECT DISTINCT processing_status FROM evidence")
        processing_statuses = [row[0] for row in cursor.fetchall()]
        print(f"   Processing statuses: {processing_statuses}")
        
        print("\n2. Converting to SQLAlchemy enum names...")
        
        # Fix case status - convert to enum names
        print("   📋 Fixing case status...")
        cursor.execute("UPDATE cases SET status = 'ACTIVE' WHERE status = 'active'")
        cursor.execute("UPDATE cases SET status = 'CLOSED' WHERE status = 'closed'")
        cursor.execute("UPDATE cases SET status = 'ARCHIVED' WHERE status = 'archived'")
        cursor.execute("UPDATE cases SET status = 'UNDER_REVIEW' WHERE status = 'under_review'")
        
        # Fix evidence type - convert to enum names
        print("   📂 Fixing evidence types...")
        cursor.execute("UPDATE evidence SET evidence_type = 'CHAT' WHERE evidence_type = 'chat'")
        cursor.execute("UPDATE evidence SET evidence_type = 'CALL_LOG' WHERE evidence_type = 'call_log'")
        cursor.execute("UPDATE evidence SET evidence_type = 'CONTACT' WHERE evidence_type = 'contact'")
        cursor.execute("UPDATE evidence SET evidence_type = 'IMAGE' WHERE evidence_type = 'image'")
        cursor.execute("UPDATE evidence SET evidence_type = 'VIDEO' WHERE evidence_type = 'video'")
        cursor.execute("UPDATE evidence SET evidence_type = 'AUDIO' WHERE evidence_type = 'audio'")
        cursor.execute("UPDATE evidence SET evidence_type = 'DOCUMENT' WHERE evidence_type = 'document'")
        cursor.execute("UPDATE evidence SET evidence_type = 'XML_REPORT' WHERE evidence_type = 'xml_report'")
        cursor.execute("UPDATE evidence SET evidence_type = 'JSON_DATA' WHERE evidence_type = 'json_data'")
        cursor.execute("UPDATE evidence SET evidence_type = 'CSV_DATA' WHERE evidence_type = 'csv_data'")
        cursor.execute("UPDATE evidence SET evidence_type = 'TEXT_REPORT' WHERE evidence_type = 'text_report'")
        cursor.execute("UPDATE evidence SET evidence_type = 'OTHER' WHERE evidence_type = 'other'")
        
        # Fix processing status - convert to enum names
        print("   ⚙️ Fixing processing statuses...")
        cursor.execute("UPDATE evidence SET processing_status = 'PENDING' WHERE processing_status = 'pending'")
        cursor.execute("UPDATE evidence SET processing_status = 'PROCESSING' WHERE processing_status = 'processing'")
        cursor.execute("UPDATE evidence SET processing_status = 'COMPLETED' WHERE processing_status = 'completed'")
        cursor.execute("UPDATE evidence SET processing_status = 'FAILED' WHERE processing_status = 'failed'")
        
        # Commit changes
        conn.commit()
        
        print("\n3. Verifying fixes...")
        
        # Verify cases
        cursor.execute("SELECT DISTINCT status FROM cases")
        new_case_statuses = [row[0] for row in cursor.fetchall()]
        print(f"   📋 Cases status: {new_case_statuses}")
        
        # Verify evidence types
        cursor.execute("SELECT DISTINCT evidence_type FROM evidence")
        new_evidence_types = [row[0] for row in cursor.fetchall()]
        print(f"   📂 Evidence types: {new_evidence_types}")
        
        # Verify processing statuses
        cursor.execute("SELECT DISTINCT processing_status FROM evidence")
        new_processing_statuses = [row[0] for row in cursor.fetchall()]
        print(f"   ⚙️ Processing statuses: {new_processing_statuses}")
        
        print("\n✅ SQLAlchemy enum fix completed!")
        print("🚀 Database now uses enum names that SQLAlchemy expects.")
        
    except Exception as e:
        print(f"❌ Error fixing database: {e}")
        conn.rollback()
        return 1
        
    finally:
        conn.close()
    
    return 0

if __name__ == "__main__":
    exit(main())