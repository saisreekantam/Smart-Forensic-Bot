#!/usr/bin/env python3
"""
Database Migration Script - Fix Enum Values

This script fixes mismatches between enum values in the code and database.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import sqlite3
from pathlib import Path

def fix_database_enums():
    """Fix enum value mismatches in the database"""
    
    db_path = "data/forensic_cases.db"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found at {db_path}")
        return False
    
    print("🔧 Fixing database enum values...")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check current evidence types in database
        cursor.execute("SELECT DISTINCT evidence_type FROM evidence")
        existing_types = [row[0] for row in cursor.fetchall()]
        print(f"📋 Existing evidence types in DB: {existing_types}")
        
        # Check current processing statuses
        cursor.execute("SELECT DISTINCT processing_status FROM evidence")
        existing_statuses = [row[0] for row in cursor.fetchall()]
        print(f"📋 Existing processing statuses in DB: {existing_statuses}")
        
        # Check case statuses
        cursor.execute("SELECT DISTINCT status FROM cases")
        existing_case_statuses = [row[0] for row in cursor.fetchall()]
        print(f"📋 Existing case statuses in DB: {existing_case_statuses}")
        
        # Update processing statuses that don't match
        status_updates = {
            'PROCESSING': 'processing',
            'COMPLETED': 'completed',
            'FAILED': 'failed',
            'PENDING': 'pending'
        }
        
        for old_status, new_status in status_updates.items():
            cursor.execute(
                "UPDATE evidence SET processing_status = ? WHERE processing_status = ?",
                (new_status, old_status)
            )
            if cursor.rowcount > 0:
                print(f"✅ Updated {cursor.rowcount} records: {old_status} -> {new_status}")
        
        # Update case statuses
        case_status_updates = {
            'ACTIVE': 'active',
            'PENDING': 'pending',
            'COMPLETED': 'completed',
            'CLOSED': 'closed'
        }
        
        for old_status, new_status in case_status_updates.items():
            cursor.execute(
                "UPDATE cases SET status = ? WHERE status = ?",
                (new_status, old_status)
            )
            if cursor.rowcount > 0:
                print(f"✅ Updated {cursor.rowcount} case records: {old_status} -> {new_status}")
        
        # Update evidence types to use correct values
        evidence_type_updates = {
            'CHAT': 'chat',
            'CALL_LOG': 'call_log',
            'CONTACT': 'contact',
            'IMAGE': 'image',
            'VIDEO': 'video',
            'AUDIO': 'audio',
            'DOCUMENT': 'document',
            'XML_REPORT': 'xml_report',
            'JSON_DATA': 'json_data',
            'CSV_DATA': 'csv_data',
            'TEXT_REPORT': 'text_report',
            'OTHER': 'other',
            'UFDR': 'document'  # Map UFDR to document type
        }
        
        for old_type, new_type in evidence_type_updates.items():
            cursor.execute(
                "UPDATE evidence SET evidence_type = ? WHERE evidence_type = ?",
                (new_type, old_type)
            )
            if cursor.rowcount > 0:
                print(f"✅ Updated {cursor.rowcount} evidence records: {old_type} -> {new_type}")
        
        conn.commit()
        print("✅ Database enum values updated successfully")
        
        # Verify updates
        cursor.execute("SELECT DISTINCT evidence_type FROM evidence")
        updated_types = [row[0] for row in cursor.fetchall()]
        print(f"📋 Updated evidence types: {updated_types}")
        
        cursor.execute("SELECT DISTINCT processing_status FROM evidence")
        updated_statuses = [row[0] for row in cursor.fetchall()]
        print(f"📋 Updated processing statuses: {updated_statuses}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error fixing database: {e}")
        return False

def reset_evidence_processing():
    """Reset all evidence to pending status for reprocessing"""
    
    db_path = "data/forensic_cases.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Reset all evidence to pending status
        cursor.execute("""
            UPDATE evidence 
            SET processing_status = 'pending',
                processing_error = NULL,
                processed_at = NULL,
                has_embeddings = 0,
                embedding_count = 0,
                chunk_count = 0
        """)
        
        print(f"✅ Reset {cursor.rowcount} evidence records to pending status")
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error resetting evidence: {e}")
        return False

def main():
    print("🔧 Database Migration - Fixing Enum Values")
    print("=" * 50)
    
    # Fix enum values
    if fix_database_enums():
        print("\n🔄 Resetting evidence processing status...")
        if reset_evidence_processing():
            print("\n✅ Database migration completed successfully!")
            print("\n🎯 Next steps:")
            print("   1. Restart the API server")
            print("   2. Run simple_upload.py to test evidence processing")
            print("   3. Check that embeddings are generated properly")
        else:
            print("\n❌ Failed to reset evidence processing")
    else:
        print("\n❌ Database migration failed")

if __name__ == "__main__":
    main()