"""
Migration Scripts for Case-Based System

This module provides scripts to migrate existing sample data into the new
case-based structure, creating proper cases and evidence records.
"""

import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
import shutil

# Local imports
from src.database.models import EvidenceType, db_manager
from src.case_management.case_manager import (
    CaseManager, CaseCreateRequest, EvidenceUploadRequest, case_manager
)
from src.data_ingestion.case_processor import CaseDataProcessor
from src.ai_cores.rag.case_vector_store import case_vector_store

logger = logging.getLogger(__name__)

class DataMigrator:
    """
    Migrates existing sample data into the case-based structure
    """
    
    def __init__(self, 
                 data_root: str = "data",
                 case_manager: CaseManager = None):
        self.data_root = Path(data_root)
        self.case_manager = case_manager or case_manager
        self.sample_dir = self.data_root / "sample"
        self.migration_log = []
        
    def migrate_sample_data(self) -> Dict[str, Any]:
        """
        Migrate all sample data into case-based structure
        """
        try:
            logger.info("Starting migration of sample data to case-based structure")
            
            # Initialize database
            db_manager.init_database()
            
            # Define sample cases based on existing data
            sample_cases = self._define_sample_cases()
            
            migration_results = {
                "status": "success",
                "cases_created": [],
                "evidence_migrated": [],
                "errors": [],
                "summary": {}
            }
            
            # Create cases and migrate evidence
            for case_definition in sample_cases:
                try:
                    case_result = self._create_sample_case(case_definition)
                    migration_results["cases_created"].append(case_result)
                    
                    # Migrate evidence for this case
                    evidence_results = self._migrate_case_evidence(
                        case_definition["case_id"], 
                        case_definition["evidence_files"]
                    )
                    migration_results["evidence_migrated"].extend(evidence_results)
                    
                except Exception as e:
                    error_msg = f"Error migrating case {case_definition.get('case_number', 'unknown')}: {str(e)}"
                    logger.error(error_msg)
                    migration_results["errors"].append(error_msg)
            
            # Generate summary
            migration_results["summary"] = {
                "total_cases": len(migration_results["cases_created"]),
                "total_evidence": len(migration_results["evidence_migrated"]),
                "total_errors": len(migration_results["errors"]),
                "migration_timestamp": datetime.now().isoformat()
            }
            
            # Save migration log
            self._save_migration_log(migration_results)
            
            logger.info(f"Migration completed: {migration_results['summary']}")
            return migration_results
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _define_sample_cases(self) -> List[Dict[str, Any]]:
        """Define sample cases based on existing data structure"""
        
        cases = [
            {
                "case_number": "CASE-2024-001",
                "title": "Cryptocurrency Fraud Investigation",
                "description": "Investigation into suspicious cryptocurrency transactions and related communications",
                "investigator_name": "Detective Sarah Chen",
                "investigator_id": "DET-001",
                "department": "Cybercrime Unit",
                "case_type": "Financial Fraud",
                "priority": "high",
                "incident_date": datetime.now() - timedelta(days=30),
                "due_date": datetime.now() + timedelta(days=60),
                "tags": ["cryptocurrency", "fraud", "digital_evidence", "financial_crime"],
                "suspects": [
                    {
                        "name": "Alex Rivera",
                        "phone": "+1-555-0123",
                        "role": "Primary Suspect"
                    }
                ],
                "victims": [
                    {
                        "name": "Multiple Investors",
                        "role": "Financial Victims"
                    }
                ],
                "evidence_files": [
                    {
                        "filename": "sample_ufdr_case001.xml",
                        "evidence_type": EvidenceType.XML_REPORT,
                        "title": "Mobile Device Extraction Report",
                        "description": "Complete UFDR extraction from suspect's Samsung Galaxy S23",
                        "source_device": "Samsung Galaxy S23 (IMEI: 123456789012345)"
                    },
                    {
                        "filename": "call_logs_case001.csv",
                        "evidence_type": EvidenceType.CALL_LOG,
                        "title": "Call History Records",
                        "description": "Complete call log history from suspect's device",
                        "source_device": "Samsung Galaxy S23"
                    },
                    {
                        "filename": "messages_case001.csv",
                        "evidence_type": EvidenceType.CHAT,
                        "title": "Text Message Communications",
                        "description": "SMS and messaging app conversations",
                        "source_device": "Samsung Galaxy S23"
                    },
                    {
                        "filename": "structured_data_case001.json",
                        "evidence_type": EvidenceType.JSON_DATA,
                        "title": "Structured Application Data",
                        "description": "JSON data from various applications",
                        "source_device": "Samsung Galaxy S23"
                    }
                ]
            },
            {
                "case_number": "CASE-2024-002",
                "title": "Organized Crime Communication Analysis",
                "description": "Analysis of communication patterns in organized crime network",
                "investigator_name": "Detective Marcus Johnson",
                "investigator_id": "DET-002",
                "department": "Organized Crime Task Force",
                "case_type": "Organized Crime",
                "priority": "critical",
                "incident_date": datetime.now() - timedelta(days=45),
                "due_date": datetime.now() + timedelta(days=30),
                "tags": ["organized_crime", "network_analysis", "communications"],
                "suspects": [
                    {
                        "name": "Unknown Network",
                        "role": "Criminal Organization"
                    }
                ],
                "evidence_files": [
                    {
                        "filename": "text_report_case002.txt",
                        "evidence_type": EvidenceType.TEXT_REPORT,
                        "title": "Investigation Summary Report",
                        "description": "Detailed investigation report and findings",
                        "source_device": "Investigation Notes"
                    }
                ]
            }
        ]
        
        # Add case_id for internal tracking
        for i, case in enumerate(cases):
            case["case_id"] = f"case_{i+1}"
        
        return cases
    
    def _create_sample_case(self, case_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Create a sample case from definition"""
        
        # Create case request
        create_request = CaseCreateRequest(
            case_number=case_definition["case_number"],
            title=case_definition["title"],
            description=case_definition["description"],
            investigator_name=case_definition["investigator_name"],
            investigator_id=case_definition.get("investigator_id"),
            department=case_definition.get("department"),
            incident_date=case_definition.get("incident_date"),
            due_date=case_definition.get("due_date"),
            priority=case_definition.get("priority", "medium"),
            case_type=case_definition.get("case_type"),
            tags=case_definition.get("tags", []),
            suspects=case_definition.get("suspects", []),
            victims=case_definition.get("victims", [])
        )
        
        # Create the case
        case = self.case_manager.create_case(create_request)
        
        result = {
            "case_id": case.id,
            "case_number": case.case_number,
            "title": case.title,
            "status": "created",
            "internal_id": case_definition["case_id"]
        }
        
        # Store case_id mapping for evidence migration
        case_definition["created_case_id"] = case.id
        
        logger.info(f"Created case: {case.case_number}")
        return result
    
    def _migrate_case_evidence(self, internal_case_id: str, evidence_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Migrate evidence files for a case"""
        
        results = []
        
        # Find the created case
        case_definition = None
        for case_def in self._define_sample_cases():
            if case_def["case_id"] == internal_case_id:
                case_definition = case_def
                break
        
        if not case_definition or "created_case_id" not in case_definition:
            raise ValueError(f"Case {internal_case_id} not found or not created")
        
        case_id = case_definition["created_case_id"]
        
        for evidence_file in evidence_files:
            try:
                result = self._migrate_evidence_file(case_id, evidence_file)
                results.append(result)
            except Exception as e:
                error_result = {
                    "filename": evidence_file.get("filename", "unknown"),
                    "status": "error",
                    "error": str(e)
                }
                results.append(error_result)
                logger.error(f"Error migrating evidence {evidence_file.get('filename')}: {str(e)}")
        
        return results
    
    def _migrate_evidence_file(self, case_id: str, evidence_file: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate a single evidence file"""
        
        # Find source file
        source_file = self.sample_dir / evidence_file["filename"]
        if not source_file.exists():
            raise FileNotFoundError(f"Source file not found: {source_file}")
        
        # Read file data
        with open(source_file, 'rb') as f:
            file_data = f.read()
        
        # Create evidence upload request
        upload_request = EvidenceUploadRequest(
            case_id=case_id,
            original_filename=evidence_file["filename"],
            evidence_type=evidence_file["evidence_type"],
            title=evidence_file.get("title"),
            description=evidence_file.get("description"),
            source_device=evidence_file.get("source_device"),
            extraction_method="Migration from sample data",
            evidence_date=datetime.now() - timedelta(days=30)  # Simulate older evidence
        )
        
        # Add evidence to case
        evidence = self.case_manager.add_evidence(upload_request, file_data)
        
        logger.info(f"Migrated evidence: {evidence_file['filename']} -> {evidence.id}")
        
        return {
            "evidence_id": evidence.id,
            "filename": evidence_file["filename"],
            "evidence_type": evidence_file["evidence_type"].value,
            "status": "migrated",
            "file_size": len(file_data)
        }
    
    def process_migrated_cases(self) -> Dict[str, Any]:
        """
        Process all migrated cases to generate embeddings
        """
        try:
            logger.info("Starting processing of migrated cases")
            
            # Get all cases
            cases = self.case_manager.list_cases()
            
            # Initialize case data processor
            case_processor = CaseDataProcessor(
                case_manager=self.case_manager,
                vector_store=case_vector_store
            )
            
            processing_results = {
                "processed_cases": [],
                "errors": [],
                "summary": {}
            }
            
            for case in cases:
                try:
                    logger.info(f"Processing case: {case.case_number}")
                    
                    # Process all evidence for this case
                    result = case_processor.process_case_evidence(case.id)
                    
                    processing_results["processed_cases"].append({
                        "case_id": case.id,
                        "case_number": case.case_number,
                        "processing_result": result
                    })
                    
                except Exception as e:
                    error_msg = f"Error processing case {case.case_number}: {str(e)}"
                    logger.error(error_msg)
                    processing_results["errors"].append(error_msg)
            
            # Generate summary
            processing_results["summary"] = {
                "total_cases_processed": len(processing_results["processed_cases"]),
                "total_errors": len(processing_results["errors"]),
                "processing_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Processing completed: {processing_results['summary']}")
            return processing_results
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _save_migration_log(self, migration_results: Dict[str, Any]):
        """Save migration log to file"""
        try:
            log_file = self.data_root / "migration_log.json"
            with open(log_file, 'w') as f:
                json.dump(migration_results, f, indent=2, default=str)
            logger.info(f"Migration log saved to: {log_file}")
        except Exception as e:
            logger.error(f"Error saving migration log: {str(e)}")
    
    def reset_database(self):
        """Reset database and remove all case data (use with caution!)"""
        try:
            logger.warning("Resetting database - all case data will be lost!")
            
            # Drop and recreate tables
            db_manager.drop_tables()
            db_manager.create_tables()
            
            # Clear case directories
            cases_dir = Path("data/cases")
            if cases_dir.exists():
                shutil.rmtree(cases_dir)
                cases_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("Database reset completed")
            
        except Exception as e:
            logger.error(f"Error resetting database: {str(e)}")
            raise

# Create migration instance
data_migrator = DataMigrator()

def run_complete_migration():
    """
    Run complete migration: create cases, migrate evidence, and process data
    """
    try:
        print("🚀 Starting complete data migration...")
        
        # Step 1: Migrate sample data to case structure
        print("📁 Migrating sample data to case-based structure...")
        migration_result = data_migrator.migrate_sample_data()
        
        if migration_result["status"] != "success":
            print(f"❌ Migration failed: {migration_result.get('error', 'Unknown error')}")
            return migration_result
        
        print(f"✅ Migration completed: {migration_result['summary']['total_cases']} cases, "
              f"{migration_result['summary']['total_evidence']} evidence files")
        
        # Step 2: Process evidence and generate embeddings
        print("🧠 Processing evidence and generating embeddings...")
        processing_result = data_migrator.process_migrated_cases()
        
        print(f"✅ Processing completed: {processing_result['summary']['total_cases_processed']} cases processed")
        
        # Summary
        print("\n📊 Migration Summary:")
        print(f"   Cases Created: {migration_result['summary']['total_cases']}")
        print(f"   Evidence Files: {migration_result['summary']['total_evidence']}")
        print(f"   Cases Processed: {processing_result['summary']['total_cases_processed']}")
        print(f"   Errors: {migration_result['summary']['total_errors'] + processing_result['summary']['total_errors']}")
        
        return {
            "migration": migration_result,
            "processing": processing_result,
            "status": "complete"
        }
        
    except Exception as e:
        print(f"❌ Complete migration failed: {str(e)}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    # Run migration when script is executed directly
    result = run_complete_migration()
    print(f"\n🏁 Migration result: {result['status']}")