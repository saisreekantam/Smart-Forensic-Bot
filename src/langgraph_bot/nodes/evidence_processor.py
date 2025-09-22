
import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from langchain_core.messages import AIMessage

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ..state import ForensicBotState, Evidence, Entity, Event, add_workflow_step
from src.database.models import Case, Evidence as EvidenceModel, EvidenceType
from src.data_ingestion.case_processor import CaseDataProcessor
from src.data_ingestion.evidence_handlers import BaseEvidenceHandler, EvidenceHandlerFactory

class EvidenceProcessor:
    """Advanced evidence processing for forensic analysis"""
    
    def __init__(self):
        self.case_processor = None
        self.evidence_handler = None
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the evidence processing components"""
        try:
            # Initialize required dependencies
            from src.case_management.case_manager import CaseManager
            from src.ai_cores.rag.case_vector_store import CaseVectorStore
            from src.database.models import db_manager
            
            case_manager = CaseManager(db_manager)
            vector_store = CaseVectorStore()
            
            self.case_processor = CaseDataProcessor(case_manager, vector_store)
            self.evidence_handler = EvidenceHandlerFactory()
            self.initialized = True
            return True
        except Exception as e:
            print(f"Evidence processor initialization failed: {e}")
            return False
    
    def process_evidence_file(
        self, 
        file_path: str, 
        case_id: str,
        evidence_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single evidence file
        
        Args:
            file_path: Path to the evidence file
            case_id: Case ID for the evidence
            evidence_type: Type of evidence
            
        Returns:
            Dict containing processing results
        """
        if not self.initialized:
            if not self.initialize():
                return {
                    "success": False,
                    "error": "Failed to initialize evidence processor"
                }
        
        try:
            # Check if file exists
            if not Path(file_path).exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }
            
            # Auto-detect evidence type if not provided
            if not evidence_type:
                evidence_type = self._detect_evidence_type(file_path)
            
            # Process the file based on type
            processing_result = self.evidence_handler.process_file(
                file_path, 
                case_id, 
                evidence_type
            )
            
            return {
                "success": True,
                "file_path": file_path,
                "evidence_type": evidence_type,
                "processing_result": processing_result,
                "entities_extracted": processing_result.get("entities", []),
                "events_extracted": processing_result.get("events", []),
                "metadata": processing_result.get("metadata", {})
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Evidence processing failed: {str(e)}",
                "file_path": file_path
            }
    
    def _detect_evidence_type(self, file_path: str) -> str:
        """Detect evidence type based on file extension and content"""
        file_ext = Path(file_path).suffix.lower()
        
        type_mapping = {
            '.xml': 'xml_report',
            '.json': 'json_data',
            '.csv': 'csv_data',
            '.txt': 'text_report',
            '.pdf': 'document',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.png': 'image',
            '.mp4': 'video',
            '.avi': 'video',
            '.mp3': 'audio',
            '.wav': 'audio'
        }
        
        return type_mapping.get(file_ext, 'other')
    
    def batch_process_evidence(
        self, 
        evidence_files: List[str], 
        case_id: str
    ) -> Dict[str, Any]:
        """
        Process multiple evidence files in batch
        
        Args:
            evidence_files: List of file paths
            case_id: Case ID
            
        Returns:
            Dict containing batch processing results
        """
        results = {
            "success": True,
            "processed_files": [],
            "failed_files": [],
            "total_entities": 0,
            "total_events": 0
        }
        
        for file_path in evidence_files:
            result = self.process_evidence_file(file_path, case_id)
            
            if result["success"]:
                results["processed_files"].append(result)
                results["total_entities"] += len(result.get("entities_extracted", []))
                results["total_events"] += len(result.get("events_extracted", []))
            else:
                results["failed_files"].append(result)
        
        if results["failed_files"]:
            results["success"] = len(results["processed_files"]) > 0
        
        return results

def evidence_processor(state: ForensicBotState) -> ForensicBotState:
    """
    Process evidence files and extract forensic information
    
    This node handles evidence ingestion, processing, and extraction
    of entities, events, and metadata for forensic analysis.
    
    Args:
        state: Current forensic bot state
        
    Returns:
        ForensicBotState: Updated state with evidence processing results
    """
    start_time = datetime.now()
    
    try:
        # Get evidence processing request from the user message
        if not state["messages"]:
            raise ValueError("No messages found in state")
        
        last_message = state["messages"][-1]
        message_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # Initialize evidence processor
        if "evidence_processor_instance" not in state["tool_results"]:
            state["tool_results"]["evidence_processor_instance"] = EvidenceProcessor()
        
        processor = state["tool_results"]["evidence_processor_instance"]
        
        # Extract file paths from the message or use active evidence
        evidence_files = extract_file_paths_from_message(message_content)
        
        if not evidence_files and state["active_evidence"]:
            # Use active evidence if no specific files mentioned
            evidence_files = [ev.file_path for ev in state["active_evidence"] if not ev.processed]
        
        if not evidence_files:
            # Look for evidence in the case directory
            case_id = state["current_case_id"]
            if case_id:
                evidence_files = find_case_evidence_files(case_id)
        
        if not evidence_files:
            response = (
                "I couldn't find any evidence files to process. "
                "Please specify file paths or ensure evidence files are available in the case directory."
            )
            ai_message = AIMessage(content=response)
            state["messages"].append(ai_message)
            return state
        
        # Process the evidence files
        case_id = state["current_case_id"] or "unknown_case"
        
        if len(evidence_files) == 1:
            # Single file processing
            result = processor.process_evidence_file(evidence_files[0], case_id)
            processing_results = [result] if result["success"] else []
            failed_results = [result] if not result["success"] else []
        else:
            # Batch processing
            batch_result = processor.batch_process_evidence(evidence_files, case_id)
            processing_results = batch_result["processed_files"]
            failed_results = batch_result["failed_files"]
        
        # Update state with processing results
        total_processed = len(processing_results)
        total_failed = len(failed_results)
        total_entities = 0
        total_events = 0
        
        # Process successful results
        for result in processing_results:
            # Create Evidence objects
            evidence = Evidence(
                id=f"ev_{datetime.now().timestamp()}",
                file_path=result["file_path"],
                evidence_type=result["evidence_type"],
                case_id=case_id,
                metadata=result["metadata"],
                processed=True,
                analysis_results=result["processing_result"]
            )
            
            # Add to active evidence if not already present
            existing_paths = [ev.file_path for ev in state["active_evidence"]]
            if evidence.file_path not in existing_paths:
                state["active_evidence"].append(evidence)
            
            # Extract and store entities
            for entity_data in result.get("entities_extracted", []):
                entity = Entity(
                    id=entity_data.get("id", f"ent_{datetime.now().timestamp()}"),
                    name=entity_data.get("name", "Unknown"),
                    type=entity_data.get("type", "unknown"),
                    attributes=entity_data.get("attributes", {}),
                    confidence=entity_data.get("confidence", 0.0),
                    case_id=case_id
                )
                state["entity_memory"][entity.id] = entity
                total_entities += 1
            
            # Extract and store events
            for event_data in result.get("events_extracted", []):
                event = Event(
                    id=event_data.get("id", f"evt_{datetime.now().timestamp()}"),
                    timestamp=datetime.fromisoformat(event_data.get("timestamp", datetime.now().isoformat())),
                    event_type=event_data.get("type", "unknown"),
                    description=event_data.get("description", ""),
                    entities_involved=event_data.get("entities", []),
                    evidence_source=result["file_path"],
                    confidence=event_data.get("confidence", 0.0),
                    case_id=case_id
                )
                state["timeline_memory"].append(event)
                total_events += 1
        
        # Update evidence index
        for result in processing_results:
            evidence_id = f"ev_{hash(result['file_path'])}"
            state["evidence_index"][evidence_id] = result["file_path"]
        
        # Store processing results
        state["analysis_results"]["evidence_processing"] = {
            "timestamp": datetime.now().isoformat(),
            "total_files": len(evidence_files),
            "processed_successfully": total_processed,
            "failed": total_failed,
            "entities_extracted": total_entities,
            "events_extracted": total_events,
            "results": processing_results,
            "failures": failed_results
        }
        
        # Generate response message
        if total_processed > 0:
            response = f"Successfully processed {total_processed} evidence file(s).\n"
            response += f"Extracted {total_entities} entities and {total_events} events.\n"
            
            if total_failed > 0:
                response += f"\nNote: {total_failed} file(s) failed to process."
            
            # Add findings to recommendations
            if total_entities > 0:
                state["recommendations"].append(f"Identified {total_entities} forensic entities for further investigation")
            if total_events > 0:
                state["recommendations"].append(f"Extracted {total_events} timeline events for analysis")
        else:
            response = f"Failed to process {len(evidence_files)} evidence file(s). Please check file formats and accessibility."
        
        ai_message = AIMessage(content=response)
        state["messages"].append(ai_message)
        
        # Add to tools used
        if "evidence_processor" not in state["tools_used"]:
            state["tools_used"].append("evidence_processor")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        add_workflow_step(
            state,
            node_name="evidence_processor",
            action="process_evidence",
            input_data={
                "files_count": len(evidence_files),
                "case_id": case_id
            },
            output_data={
                "processed": total_processed,
                "failed": total_failed,
                "entities": total_entities,
                "events": total_events
            },
            execution_time=execution_time,
            success=total_processed > 0
        )
        
    except Exception as e:
        # Handle errors gracefully
        error_msg = f"Evidence processing error: {str(e)}"
        state["last_tool_error"] = error_msg
        
        fallback_response = (
            "I encountered an error while processing the evidence. "
            "Please ensure the evidence files are accessible and in a supported format."
        )
        
        ai_message = AIMessage(content=fallback_response)
        state["messages"].append(ai_message)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        add_workflow_step(
            state,
            node_name="evidence_processor",
            action="process_evidence",
            input_data={"error": str(e)},
            output_data={"fallback_response": True},
            execution_time=execution_time,
            success=False
        )
    
    return state

def extract_file_paths_from_message(message: str) -> List[str]:
    """
    Extract file paths from user message
    
    Args:
        message: User message content
        
    Returns:
        List of extracted file paths
    """
    import re
    
    # Common file path patterns
    patterns = [
        r'["\']([^"\']+\.[a-zA-Z0-9]+)["\']',  # Quoted paths
        r'(/[^\s]+\.[a-zA-Z0-9]+)',           # Unix paths
        r'([A-Za-z]:\\[^\s]+\.[a-zA-Z0-9]+)', # Windows paths
        r'(\.?/[^\s]+\.[a-zA-Z0-9]+)'         # Relative paths
    ]
    
    file_paths = []
    for pattern in patterns:
        matches = re.findall(pattern, message)
        file_paths.extend(matches)
    
    # Filter out duplicates and validate
    unique_paths = list(set(file_paths))
    valid_paths = [path for path in unique_paths if Path(path).suffix]
    
    return valid_paths

def find_case_evidence_files(case_id: str) -> List[str]:
    """
    Find evidence files for a specific case
    
    Args:
        case_id: Case identifier
        
    Returns:
        List of evidence file paths
    """
    case_dir = Path(f"data/cases/{case_id}")
    
    if not case_dir.exists():
        return []
    
    evidence_files = []
    
    # Supported evidence file extensions
    supported_extensions = {
        '.xml', '.json', '.csv', '.txt', '.pdf',
        '.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mp3', '.wav'
    }
    
    # Search recursively for evidence files
    for file_path in case_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            evidence_files.append(str(file_path))
    
    return evidence_files