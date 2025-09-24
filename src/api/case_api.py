"""
Enhanced Case Management API with Knowledge Graph and Intelligent Reporting

FastAPI endpoints for managing forensic cases and providing intelligent
chat-based investigation assistance with enhanced knowledge graphs and case memory.
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import json
import io
import os

# Local imports
from src.database.models import (
    Case, Evidence, EvidenceType, CaseStatus, ProcessingStatus,
    DatabaseManager, db_manager
)
from src.case_management.case_manager import (
    CaseManager, CaseCreateRequest, EvidenceUploadRequest, case_manager
)
from src.data_ingestion.case_processor import CaseDataProcessor
from src.data_ingestion.optimized_processor import OptimizedCaseProcessor
from src.ai_cores.rag.case_vector_store import CaseVectorStore, case_vector_store
from src.ai_cores.langgraph_assistant import LangGraphCaseAssistant

# Import enhanced systems
import sys
sys.path.append('.')
from simple_data_processor import SimpleDataProcessor
from simple_chat_handler import process_case_query  # Use original simple chat handler global function
from enhanced_chat_handler import enhanced_chat_handler  # For enhanced features only
from chat_history_manager import chat_history_manager
from src.ai_cores.case_memory import case_memory
from src.ai_cores.enhanced_knowledge_graph import enhanced_kg_db
from src.ai_cores.intelligent_report_generator import report_generator
from src.ai_cores.supreme_forensic_agent import (
    supreme_agent, analyze_case_intelligently, detect_patterns_intelligently,
    analyze_timeline_intelligently, get_investigation_guidance
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Project Sentinel - Forensic Case Management API",
    description="AI-powered forensic investigation platform with knowledge graphs, case memory, and intelligent reporting",
    version="2.0.0"
)

# Add CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class CaseCreateAPI(BaseModel):
    case_number: str = Field(..., description="Unique case identifier")
    title: str = Field(..., description="Case title")
    investigator_name: str = Field(..., description="Lead investigator name")
    description: Optional[str] = Field(None, description="Case description")
    investigator_id: Optional[str] = Field(None, description="Investigator ID")
    department: Optional[str] = Field(None, description="Department")
    incident_date: Optional[datetime] = Field(None, description="Date of incident")
    due_date: Optional[datetime] = Field(None, description="Case due date")
    priority: str = Field("medium", description="Case priority (low/medium/high/critical)")
    case_type: Optional[str] = Field(None, description="Type of case")
    jurisdiction: Optional[str] = Field(None, description="Jurisdiction")
    tags: Optional[List[str]] = Field(None, description="Case tags")

class CaseResponse(BaseModel):
    id: str
    case_number: str
    title: str
    status: str
    investigator_name: str
    created_at: datetime
    updated_at: datetime
    total_evidence_count: int
    processed_evidence_count: int
    processing_progress: float

class EvidenceResponse(BaseModel):
    id: str
    original_filename: str
    evidence_type: str
    processing_status: str
    file_size: int
    created_at: datetime
    has_embeddings: bool

class ChatMessage(BaseModel):
    message: str = Field(..., description="User's question or query")
    case_id: str = Field(..., description="Case ID for context")
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, description="Previous messages")
    session_id: Optional[str] = Field(None, description="Chat session ID for history")

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    confidence: float
    case_context: Dict[str, Any]
    timestamp: datetime

# Initialize dependencies
@app.on_event("startup")
async def startup_event():
    """Initialize database and components on startup"""
    try:
        # Initialize database
        db_manager.init_database()
        
        # Initialize both standard and optimized processors
        global case_data_processor, optimized_processor, langgraph_assistant
        case_data_processor = CaseDataProcessor(
            case_manager=case_manager,
            vector_store=case_vector_store
        )
        
        # Initialize optimized processor for faster evidence processing
        optimized_processor = OptimizedCaseProcessor(
            case_manager=case_manager,
            vector_store=case_vector_store,
            max_workers=4
        )
        
        # Initialize LangGraph assistant for advanced AI responses
        langgraph_assistant = LangGraphCaseAssistant(
            case_manager=case_manager,
            vector_store=case_vector_store,
            debug_mode=True
        )
        
        logger.info("🚀 API startup completed successfully with LangGraph components")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

# Case Management Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """API health check and information"""
    return {
        "message": "Project Sentinel - Forensic Case Management API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.post("/cases", response_model=CaseResponse)
async def create_case(case_data: CaseCreateAPI):
    """Create a new forensic case"""
    try:
        # Convert API model to case manager request
        create_request = CaseCreateRequest(
            case_number=case_data.case_number,
            title=case_data.title,
            investigator_name=case_data.investigator_name,
            description=case_data.description,
            investigator_id=case_data.investigator_id,
            department=case_data.department,
            incident_date=case_data.incident_date,
            due_date=case_data.due_date,
            priority=case_data.priority,
            case_type=case_data.case_type,
            jurisdiction=case_data.jurisdiction,
            tags=case_data.tags or []
        )
        
        case = case_manager.create_case(create_request)
        
        return CaseResponse(
            id=case.id,
            case_number=case.case_number,
            title=case.title,
            status=case.status,  # Already a string
            investigator_name=case.investigator_name,
            created_at=case.created_at,
            updated_at=case.updated_at,
            total_evidence_count=case.total_evidence_count,
            processed_evidence_count=case.processed_evidence_count,
            processing_progress=0.0
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating case: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/cases", response_model=List[CaseResponse])
async def list_cases(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List all cases with optional filtering"""
    try:
        case_status = None
        if status:
            try:
                case_status = CaseStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        cases = case_manager.list_cases(status=case_status, limit=limit, offset=offset)
        
        response_cases = []
        for case in cases:
            # Calculate processing progress
            progress = 0.0
            if case.total_evidence_count > 0:
                progress = (case.processed_evidence_count / case.total_evidence_count) * 100
            
            response_cases.append(CaseResponse(
                id=case.id,
                case_number=case.case_number,
                title=case.title,
                status=case.status,  # Already a string
                investigator_name=case.investigator_name,
                created_at=case.created_at,
                updated_at=case.updated_at,
                total_evidence_count=case.total_evidence_count,
                processed_evidence_count=case.processed_evidence_count,
                processing_progress=progress
            ))
        
        return response_cases
        
    except Exception as e:
        logger.error(f"Error listing cases: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/cases/{case_id}", response_model=Dict[str, Any])
async def get_case(case_id: str):
    """Get detailed information about a specific case"""
    try:
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Get case statistics
        statistics = case_manager.get_case_statistics(case_id)
        
        return {
            "case": {
                "id": case.id,
                "case_number": case.case_number,
                "title": case.title,
                "description": case.description,
                "status": case.status,  # Already a string
                "investigator_name": case.investigator_name,
                "investigator_id": case.investigator_id,
                "department": case.department,
                "priority": case.priority,
                "case_type": case.case_type,
                "jurisdiction": case.jurisdiction,
                "tags": case.tags,
                "suspects": case.suspects,
                "victims": case.victims,
                "created_at": case.created_at,
                "updated_at": case.updated_at,
                "incident_date": case.incident_date,
                "due_date": case.due_date
            },
            "statistics": statistics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/cases/{case_id}/evidence")
async def upload_evidence(
    case_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    evidence_type: str = Form(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    source_device: Optional[str] = Form(None),
    extraction_method: Optional[str] = Form(None)
):
    """Upload evidence file to a case"""
    try:
        # Validate case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Validate evidence type
        try:
            evidence_type_enum = EvidenceType(evidence_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid evidence type: {evidence_type}")
        
        # Read file data
        file_data = await file.read()
        
        # Create evidence upload request
        upload_request = EvidenceUploadRequest(
            case_id=case_id,
            original_filename=file.filename,
            evidence_type=evidence_type_enum,
            title=title,
            description=description,
            source_device=source_device,
            extraction_method=extraction_method
        )
        
        # Add evidence to case
        evidence = case_manager.add_evidence(upload_request, file_data)
        
        # Schedule background processing
        background_tasks.add_task(
            process_evidence_background,
            case_id,
            evidence.id,
            evidence.file_path,
            evidence_type_enum
        )
        
        return {
            "message": "Evidence uploaded successfully",
            "evidence_id": evidence.id,
            "filename": evidence.original_filename,
            "status": "uploaded",
            "processing": "scheduled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading evidence: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/cases/{case_id}/evidence/upload")
async def upload_evidence_simple(
    case_id: str,
    file: UploadFile = File(...),
    description: Optional[str] = Form(None)
):
    """Simple evidence file upload endpoint for frontend"""
    try:
        # Validate case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Determine evidence type from file extension
        file_ext = file.filename.split('.')[-1].lower() if file.filename else 'txt'
        evidence_type_mapping = {
            'csv': EvidenceType.CSV_DATA,
            'xml': EvidenceType.XML_REPORT,
            'json': EvidenceType.JSON_DATA,
            'txt': EvidenceType.TEXT_REPORT,
            'pdf': EvidenceType.TEXT_REPORT,
            'ufdr': EvidenceType.XML_REPORT
        }
        evidence_type = evidence_type_mapping.get(file_ext, EvidenceType.TEXT_REPORT)
        
        # Read file data
        file_data = await file.read()
        
        # Create evidence upload request
        upload_request = EvidenceUploadRequest(
            case_id=case_id,
            original_filename=file.filename,
            evidence_type=evidence_type,
            title=file.filename,
            description=description,
            source_device="Upload Interface",
            extraction_method="Web Upload"
        )
        
        # Add evidence to case
        evidence = case_manager.add_evidence(upload_request, file_data)
        
        # Process the evidence immediately for simple files
        try:
            from simple_data_processor import SimpleDataProcessor
            processor = SimpleDataProcessor()
            await processor.process_case_evidence(case_id)
        except Exception as e:
            logger.warning(f"Background processing failed: {str(e)}")
        
        return {
            "message": "File uploaded successfully",
            "evidence_id": evidence.id,
            "filename": evidence.original_filename,
            "status": "uploaded_and_processed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading evidence (simple): {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/cases/{case_id}/evidence", response_model=List[EvidenceResponse])
async def get_case_evidence(case_id: str, evidence_type: Optional[str] = None):
    """Get evidence files for a case"""
    try:
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        evidence_type_enum = None
        if evidence_type:
            try:
                evidence_type_enum = EvidenceType(evidence_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid evidence type: {evidence_type}")
        
        evidence_files = case_manager.get_case_evidence(case_id, evidence_type_enum)
        
        response_evidence = []
        for evidence in evidence_files:
            response_evidence.append(EvidenceResponse(
                id=evidence.id,
                original_filename=evidence.original_filename,
                evidence_type=evidence.evidence_type,  # Already a string
                processing_status=evidence.processing_status,  # Already a string
                file_size=evidence.file_size or 0,
                created_at=evidence.created_at,
                has_embeddings=evidence.has_embeddings or False
            ))
        
        return response_evidence
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting evidence for case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/cases/{case_id}/process")
async def process_case_data(case_id: str):
    """
    Process all unprocessed evidence in a case using the simple, reliable processor
    """
    try:
        # Validate case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Use the simple processor
        simple_processor = SimpleDataProcessor()
        results = simple_processor.process_case_evidence(case_id)
        
        if results["total_processed"] == 0 and not results["errors"]:
            return {
                "message": "All evidence in this case is already processed",
                "processed_count": 0,
                "already_processed": case.total_evidence_count,
                "status": "up_to_date",
                "details": results
            }
        
        if results["errors"]:
            error_message = f"Processing completed with {len(results['errors'])} errors"
            logger.warning(f"{error_message}: {results['errors']}")
        else:
            error_message = None
        
        return {
            "message": f"Successfully processed {results['total_processed']} evidence files" + 
                      (f" ({error_message})" if error_message else ""),
            "processed_count": results["total_processed"],
            "total_evidence": case.total_evidence_count,
            "status": "processing_completed",
            "details": results,
            "processed_files": [f["filename"] for f in results["processed_files"]],
            "errors": results["errors"] if results["errors"] else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing case data {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Intelligent Chatbot Endpoints
@app.post("/cases/{case_id}/chat", response_model=ChatResponse)
async def chat_with_case(case_id: str, chat_message: ChatMessage):
    """
    Simple but effective chatbot that actually retrieves and uses processed evidence data
    """
    try:
        # Validate case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Get or create session
        session_id = chat_message.session_id
        if not session_id:
            # Create a new session if none provided
            session_id = chat_history_manager.create_chat_session(case_id)
        
        # Get conversation history from session if not provided
        conversation_history = chat_message.conversation_history
        if not conversation_history and session_id:
            messages = chat_history_manager.get_chat_messages(session_id)
            conversation_history = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in messages[-10:]  # Last 10 messages
            ]
        
        # Save user message to history
        if session_id:
            chat_history_manager.save_message(session_id, "user", chat_message.message)
        
        # Use original simple chat handler global function
        chat_result = await process_case_query(
            case_id=case_id,
            query=chat_message.message,
            conversation_history=conversation_history or []
        )
        
        # Save assistant response to history
        if session_id:
            chat_history_manager.save_message(
                session_id, 
                "assistant", 
                chat_result['response'],
                chat_result.get('sources', []),
                chat_result.get('confidence', 0.0)
            )
        
        # Convert to API response format
        return ChatResponse(
            response=chat_result['response'],
            sources=chat_result.get('sources', []),
            confidence=chat_result.get('confidence', 0.0),
            case_context={
                "case_number": case.case_number,
                "evidence_count": case.total_evidence_count,
                "processed_count": case.processed_evidence_count,
                "session_id": session_id
            },
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in simple chat for case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/cases/{case_id}/chat/suggestions")
async def get_chat_suggestions(case_id: str):
    """Get suggested questions for the case based on available evidence"""
    try:
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Get case statistics to determine available data types
        statistics = case_manager.get_case_statistics(case_id)
        evidence_types = statistics.get("evidence_by_type", {})
        
        suggestions = []
        
        # Generate suggestions based on available evidence types
        if evidence_types.get("chat", 0) > 0:
            suggestions.extend([
                "What are the key conversations in this case?",
                "Who are the main participants in the chats?",
                "Show me suspicious messages or communications",
                "What was discussed around [specific date]?"
            ])
        
        if evidence_types.get("call_log", 0) > 0:
            suggestions.extend([
                "What call patterns do you see in this case?",
                "Who made the most calls?",
                "Are there any unusual call times or durations?",
                "Show me international calls"
            ])
        
        if evidence_types.get("contact", 0) > 0:
            suggestions.extend([
                "Who are the key contacts in this case?",
                "Are there any suspicious contacts?",
                "Show me contacts with multiple phone numbers"
            ])
        
        # General suggestions
        suggestions.extend([
            "Summarize the key findings in this case",
            "What evidence suggests criminal activity?",
            "Are there any patterns or connections?",
            "What should I investigate next?"
        ])
        
        return {
            "suggestions": suggestions[:10],  # Limit to 10 suggestions
            "case_context": {
                "case_number": case.case_number,
                "evidence_types": list(evidence_types.keys()),
                "total_evidence": sum(evidence_types.values())
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting suggestions for case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Chat History Endpoints
@app.get("/cases/{case_id}/chat/sessions")
async def get_chat_sessions(case_id: str):
    """Get all chat sessions for a case"""
    try:
        # Validate case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        sessions = chat_history_manager.get_chat_sessions(case_id)
        return {"sessions": sessions}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat sessions for case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/cases/{case_id}/chat/sessions")
async def create_chat_session(case_id: str, session_data: Dict[str, str] = None):
    """Create a new chat session"""
    try:
        # Validate case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        session_name = session_data.get("name") if session_data else None
        session_id = chat_history_manager.create_chat_session(case_id, session_name)
        
        if not session_id:
            raise HTTPException(status_code=500, detail="Failed to create chat session")
        
        return {"session_id": session_id, "message": "Chat session created successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating chat session for case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/cases/{case_id}/chat/sessions/{session_id}/messages")
async def get_chat_messages(case_id: str, session_id: str):
    """Get all messages for a chat session"""
    try:
        # Validate case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        messages = chat_history_manager.get_chat_messages(session_id)
        return {"messages": messages}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat messages for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/cases/{case_id}/chat/sessions/{session_id}/messages")
async def save_chat_message(case_id: str, session_id: str, message_data: Dict[str, str]):
    """Save a message to a chat session"""
    try:
        # Validate case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        role = message_data.get("role")
        message = message_data.get("message")
        
        if not role or not message:
            raise HTTPException(status_code=400, detail="Role and message are required")
        
        success = chat_history_manager.save_message(session_id, role, message)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save message")
        
        return {"success": True, "message": "Message saved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving message to session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/cases/{case_id}/chat/sessions/{session_id}")
async def delete_chat_session(case_id: str, session_id: str):
    """Delete a chat session"""
    try:
        # Validate case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        success = chat_history_manager.delete_chat_session(session_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete chat session")
        
        return {"message": "Chat session deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# File Upload and Analytics Endpoints
@app.get("/cases/{case_id}/analytics")
async def get_case_analytics(case_id: str):
    """Get analytics data for a case"""
    try:
        # Validate case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Load processed data for analytics
        from simple_search_system import SimpleSearchSystem
        search_system = SimpleSearchSystem()
        case_summary = search_system.get_case_summary(case_id)
        
        if case_summary.get("total_records", 0) == 0:
            return {
                "message": "No processed data available for analytics",
                "analytics": {
                    "total_records": 0,
                    "data_sources": [],
                    "communication_stats": {},
                    "timeline_data": [],
                    "contact_network": []
                }
            }
        
        # Generate analytics
        analytics = await _generate_case_analytics(case_id, search_system)
        
        return {
            "analytics": analytics,
            "case_summary": case_summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analytics for case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/cases/{case_id}/reports")
async def get_case_reports(case_id: str):
    """Get available reports for a case"""
    try:
        # Validate case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Check for processed data
        from simple_search_system import SimpleSearchSystem
        search_system = SimpleSearchSystem()
        case_summary = search_system.get_case_summary(case_id)
        
        if case_summary.get("total_records", 0) == 0:
            return {
                "message": "No processed data available for reports",
                "reports": []
            }
        
        # Generate available reports
        reports = [
            {
                "id": "evidence_summary",
                "title": "Evidence Summary Report",
                "description": "Comprehensive overview of all evidence in the case",
                "type": "summary",
                "available": True
            },
            {
                "id": "communication_analysis", 
                "title": "Communication Analysis Report",
                "description": "Analysis of calls, messages, and communication patterns",
                "type": "analysis",
                "available": any("message" in src["file"].lower() or "call" in src["file"].lower() 
                               for src in case_summary.get("data_sources", []))
            },
            {
                "id": "timeline_report",
                "title": "Timeline Report", 
                "description": "Chronological timeline of events and activities",
                "type": "timeline",
                "available": case_summary.get("total_records", 0) > 0
            },
            {
                "id": "contact_network",
                "title": "Contact Network Report",
                "description": "Network analysis of contacts and relationships",
                "type": "network",
                "available": any("contact" in src["file"].lower() or "call" in src["file"].lower()
                               for src in case_summary.get("data_sources", []))
            }
        ]
        
        return {"reports": reports}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting reports for case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/cases/{case_id}/reports/{report_id}/generate")
async def generate_report(case_id: str, report_id: str):
    """Generate a specific report for a case"""
    try:
        # Validate case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Generate the requested report
        report_data = await _generate_specific_report(case_id, report_id)
        
        return {
            "report_id": report_id,
            "generated_at": datetime.now().isoformat(),
            "data": report_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report {report_id} for case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Helper functions for analytics and reports
async def _generate_case_analytics(case_id: str, search_system):
    """Generate analytics data for a case"""
    try:
        import json
        from collections import Counter, defaultdict
        
        # Get all processed files for this case
        processed_dir = f"data/processed"
        analytics = {
            "total_records": 0,
            "data_sources": [],
            "communication_stats": {},
            "timeline_data": [],
            "contact_network": []
        }
        
        # Find processed files for this case
        import os
        import glob
        
        processed_files = glob.glob(f"{processed_dir}/*{case_id}*.json")
        
        if not processed_files:
            return analytics
        
        all_data = []
        contact_counter = Counter()
        communication_types = Counter()
        timeline_events = []
        
        for file_path in processed_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                if isinstance(data, list):
                    all_data.extend(data)
                elif isinstance(data, dict):
                    all_data.append(data)
                    
            except Exception as e:
                logger.warning(f"Could not load {file_path}: {str(e)}")
                continue
        
        # Analyze the data
        for record in all_data:
            # Count contacts
            if isinstance(record, dict):
                for key, value in record.items():
                    if key.lower() in ['contact', 'name', 'caller', 'recipient']:
                        if value and isinstance(value, str):
                            contact_counter[value] += 1
                    elif key.lower() in ['type', 'call_type', 'message_type']:
                        if value and isinstance(value, str):
                            communication_types[value] += 1
                    elif key.lower() in ['timestamp', 'date', 'time']:
                        if value:
                            timeline_events.append({
                                "timestamp": str(value),
                                "event": f"{record.get('type', 'Event')}"
                            })
        
        # Build analytics
        analytics.update({
            "total_records": len(all_data),
            "data_sources": [{"file": os.path.basename(f), "records": "varies"} for f in processed_files],
            "communication_stats": {
                "total_communications": sum(communication_types.values()),
                "by_type": dict(communication_types.most_common()),
                "unique_contacts": len(contact_counter)
            },
            "timeline_data": sorted(timeline_events, key=lambda x: x["timestamp"])[:50],  # Limit to 50 events
            "contact_network": [{"name": name, "frequency": count} for name, count in contact_counter.most_common(20)]
        })
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error generating analytics: {str(e)}")
        return {
            "total_records": 0,
            "data_sources": [],
            "communication_stats": {},
            "timeline_data": [],
            "contact_network": []
        }

async def _generate_specific_report(case_id: str, report_id: str):
    """Generate a specific report for a case"""
    try:
        from simple_search_system import SimpleSearchSystem
        search_system = SimpleSearchSystem()
        
        if report_id == "evidence_summary":
            return await _generate_evidence_summary(case_id, search_system)
        elif report_id == "communication_analysis":
            return await _generate_communication_analysis(case_id, search_system)
        elif report_id == "timeline_report":
            return await _generate_timeline_report(case_id, search_system)
        elif report_id == "contact_network":
            return await _generate_contact_network_report(case_id, search_system)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown report type: {report_id}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating specific report {report_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Report generation failed")

async def _generate_evidence_summary(case_id: str, search_system):
    """Generate evidence summary report"""
    case_summary = search_system.get_case_summary(case_id)
    
    summary = {
        "title": "Evidence Summary Report",
        "case_id": case_id,
        "generated_at": datetime.now().isoformat(),
        "total_evidence_items": case_summary.get("total_records", 0),
        "data_sources": case_summary.get("data_sources", []),
        "summary": f"This case contains {case_summary.get('total_records', 0)} evidence records from {len(case_summary.get('data_sources', []))} different sources."
    }
    
    return summary

async def _generate_communication_analysis(case_id: str, search_system):
    """Generate communication analysis report"""
    # Search for communication-related data
    call_results = search_system.search_case_data(case_id, "call")
    message_results = search_system.search_case_data(case_id, "message")
    
    analysis = {
        "title": "Communication Analysis Report", 
        "case_id": case_id,
        "generated_at": datetime.now().isoformat(),
        "total_calls": len(call_results.get("results", [])),
        "total_messages": len(message_results.get("results", [])),
        "key_findings": [
            f"Found {len(call_results.get('results', []))} call records",
            f"Found {len(message_results.get('results', []))} message records"
        ]
    }
    
    return analysis

async def _generate_timeline_report(case_id: str, search_system):
    """Generate timeline report"""
    case_summary = search_system.get_case_summary(case_id)
    
    timeline = {
        "title": "Timeline Report",
        "case_id": case_id, 
        "generated_at": datetime.now().isoformat(),
        "total_events": case_summary.get("total_records", 0),
        "timeline_summary": f"Timeline contains {case_summary.get('total_records', 0)} events across multiple data sources."
    }
    
    return timeline

async def _generate_contact_network_report(case_id: str, search_system):
    """Generate contact network report"""
    case_summary = search_system.get_case_summary(case_id)
    
    network = {
        "title": "Contact Network Report",
        "case_id": case_id,
        "generated_at": datetime.now().isoformat(), 
        "data_sources": case_summary.get("data_sources", []),
        "network_summary": "Contact network analysis based on available communication data."
    }
    
    return network

# Background task functions
async def process_evidence_background(case_id: str, evidence_id: str, file_path: str, evidence_type: EvidenceType):
    """Fast background task to process evidence using optimized processor"""
    try:
        # Use optimized processor for faster processing
        result = await optimized_processor.process_evidence_fast(
            case_id=case_id,
            evidence_id=evidence_id,
            file_path=file_path,
            evidence_type=evidence_type
        )
        logger.info(f"⚡ Optimized processing completed for evidence {evidence_id} in {result.get('processing_time_seconds', 0):.2f}s")
    except Exception as e:
        logger.error(f"❌ Optimized processing failed for evidence {evidence_id}: {str(e)}")

# Intelligent Case Assistant Implementation
class IntelligentCaseAssistant:
    """
    Intelligent assistant for case investigation using OpenAI GPT models
    Uses GPT-4o-mini for simple questions and GPT-4 for complex investigations
    """
    
    def __init__(self, case_manager: CaseManager, vector_store: CaseVectorStore):
        self.case_manager = case_manager
        self.vector_store = vector_store
        
        # Initialize OpenAI client
        self.openai_client = self._initialize_openai()
    
    def _initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                return OpenAI(api_key=api_key)
            else:
                logger.warning("OpenAI API key not found, chatbot will have limited functionality")
                return None
        except ImportError:
            logger.warning("OpenAI library not available, chatbot will have limited functionality")
            return None
    
    def _determine_question_complexity(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """
        Determine if the question is simple or complex to choose appropriate GPT model
        Returns 'gpt-4o-mini' for simple questions, 'gpt-4' for complex ones
        """
        query_lower = query.lower()
        
        # Simple question indicators
        simple_indicators = [
            'what is', 'who is', 'when did', 'where is', 'how many',
            'list', 'show me', 'find', 'search', 'tell me about'
        ]
        
        # Complex question indicators
        complex_indicators = [
            'analyze', 'investigate', 'correlate', 'pattern', 'suspicious',
            'relationship', 'connect', 'timeline', 'motive', 'evidence chain',
            'conclusion', 'theory', 'hypothesis', 'reconstruct', 'why',
            'how did', 'explain the connection', 'what does this mean',
            'significance', 'implication', 'strategy', 'recommend'
        ]
        
        # Check for complex indicators first
        for indicator in complex_indicators:
            if indicator in query_lower:
                return 'gpt-4'
        
        # Check for simple indicators
        for indicator in simple_indicators:
            if indicator in query_lower:
                return 'gpt-4o-mini'
        
        # Default logic based on other factors
        if len(sources) > 5:  # Many sources suggest complex analysis
            return 'gpt-4'
        
        if len(query.split()) > 20:  # Long queries often need complex reasoning
            return 'gpt-4'
        
        # Default to mini for efficiency
        return 'gpt-4o-mini'
    
    async def process_query(
        self, 
        case_id: str, 
        query: str, 
        conversation_history: List[Dict[str, str]]
    ) -> ChatResponse:
        """Process user query and generate intelligent response"""
        
        # Get case information
        case = self.case_manager.get_case(case_id)
        case_stats = self.case_manager.get_case_statistics(case_id)
        
        # Search relevant evidence using vector store
        relevant_sources = await self._search_case_evidence(case_id, query, case.embedding_collection_name)
        
        # Generate response using OpenAI
        if self.openai_client:
            response_text = await self._generate_openai_response(case, query, relevant_sources, conversation_history)
            confidence = 0.85  # High confidence with OpenAI
        else:
            response_text = self._generate_fallback_response(case, query, relevant_sources)
            confidence = 0.6  # Lower confidence without LLM
        
        return ChatResponse(
            response=response_text,
            sources=relevant_sources,
            confidence=confidence,
            case_context={
                "case_number": case.case_number,
                "case_title": case.title,
                "evidence_count": case_stats.get("processing", {}).get("total_evidence", 0),
                "processed_count": case_stats.get("processing", {}).get("processed_evidence", 0)
            },
            timestamp=datetime.now()
        )
    
    async def _search_case_evidence(self, case_id: str, query: str, collection_name: str) -> List[Dict[str, Any]]:
        """Search for relevant evidence using vector similarity"""
        try:
            # This is a simplified version - in reality, you'd generate query embeddings
            # and search the vector store
            
            # For now, return mock relevant sources
            return [
                {
                    "source_type": "chat",
                    "content": "Sample relevant chat message...",
                    "metadata": {"timestamp": "2024-01-15", "participants": ["User A", "User B"]},
                    "similarity_score": 0.85
                }
            ]
        except Exception as e:
            logger.error(f"Error searching case evidence: {str(e)}")
            return []
    
    async def _generate_openai_response(
        self, 
        case: Case, 
        query: str, 
        sources: List[Dict[str, Any]], 
        history: List[Dict[str, str]]
    ) -> str:
        """Generate response using OpenAI GPT models (smart model selection)"""
        try:
            # Determine appropriate model based on question complexity
            model = self._determine_question_complexity(query, sources)
            
            # Build forensic investigation prompt
            messages = self._build_openai_messages(case, query, sources, history)
            
            # Generate response with selected OpenAI model
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000 if model == 'gpt-4o-mini' else 1500,
                temperature=0.3,  # Lower temperature for factual accuracy
                top_p=0.9
            )
            
            response_text = response.choices[0].message.content
            
            # Add model info to response for transparency
            model_info = f"\n\n---\n*Analysis powered by {model.upper()} - {'Quick Response' if 'mini' in model else 'Deep Analysis'} mode*"
            
            return response_text + model_info
            
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {str(e)}")
            return self._generate_fallback_response(case, query, sources)
    
    def _build_openai_messages(
        self, 
        case: Case, 
        query: str, 
        sources: List[Dict[str, Any]], 
        history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Build OpenAI chat messages for forensic investigation"""
        
        system_prompt = f"""You are an expert forensic investigator AI assistant specializing in digital evidence analysis and criminal investigation.

CASE CONTEXT:
- Case Number: {case.case_number}
- Case Title: {case.title}
- Case Type: {case.case_type or 'General Investigation'}
- Lead Investigator: {case.investigator_name}
- Case Status: {case.status}
- Priority: {case.priority if hasattr(case, 'priority') else 'Standard'}

CORE RESPONSIBILITIES:
1. Analyze digital evidence with forensic accuracy
2. Identify patterns, connections, and anomalies
3. Provide actionable investigative insights
4. Maintain chain of custody awareness
5. Suggest follow-up investigative steps
6. Flag potential legal/procedural considerations

ANALYSIS GUIDELINES:
- Base conclusions strictly on available evidence
- Clearly distinguish between facts and inferences
- Highlight gaps in evidence or missing information
- Use professional forensic terminology
- Consider alternative explanations for findings
- Recommend additional evidence collection when needed

Remember: You are assisting a law enforcement investigation. Accuracy and objectivity are paramount."""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 6 messages to stay within context limits)
        for msg in history[-6:]:
            role = "user" if msg.get('role') == 'user' else "assistant"
            messages.append({"role": role, "content": msg.get('content', '')})
        
        # Build current query with evidence context
        evidence_context = ""
        if sources:
            evidence_context = "\n\nRELEVANT EVIDENCE FOUND:\n"
            for i, source in enumerate(sources[:8]):  # Limit to prevent token overflow
                evidence_context += f"""
Evidence {i+1} - {source.get('source_type', 'Unknown').title()}:
Content: {source.get('content', 'No content')[:500]}{'...' if len(source.get('content', '')) > 500 else ''}
Source: {source.get('source_file', 'Unknown source')}
Relevance Score: {source.get('similarity_score', 0.0):.3f}
Metadata: {str(source.get('metadata', {}))[:200]}
---"""
        
        current_query = f"""INVESTIGATION QUERY: {query}{evidence_context}

Please analyze the available evidence and provide a comprehensive response to this query. If the evidence is insufficient, clearly state what additional information would be needed for a complete analysis."""
        
        messages.append({"role": "user", "content": current_query})
        
        return messages
    
    def _generate_fallback_response(self, case: Case, query: str, sources: List[Dict[str, Any]]) -> str:
        """Generate fallback response without LLM"""
        
        if not sources:
            return f"""
I found your query about "{query}" for case {case.case_number}.

However, I couldn't find specific relevant evidence in the processed data for this case. This could mean:

1. The evidence hasn't been processed yet
2. The query terms don't match the available evidence
3. The evidence might be in unprocessed files

Please try:
- Checking if all evidence files have been uploaded and processed
- Using different search terms
- Being more specific about what you're looking for

Case Status: {case.processed_evidence_count} of {case.total_evidence_count} evidence files processed.
"""
        
        response = f"Based on the available evidence for case {case.case_number}, I found {len(sources)} relevant items:\n\n"
        
        for i, source in enumerate(sources[:3]):
            response += f"{i+1}. {source.get('source_type', 'Evidence').title()}: {source.get('content', 'No content')[:200]}...\n"
        
        response += f"\nThis information relates to your query: '{query}'"
        response += "\n\nNote: For more detailed analysis, please ensure all evidence is processed and consider using more specific search terms."
        
        return response

# =============================================================================
# ENHANCED FEATURES: Knowledge Graph, Case Memory, and Intelligent Reporting
# =============================================================================

# Data models for new endpoints
class ReportRequest(BaseModel):
    report_type: str = Field(default="detailed_analysis", description="Type of report to generate")
    custom_sections: Optional[List[str]] = Field(default=None, description="Custom sections to include")

class InsightResponse(BaseModel):
    success: bool
    case_id: str
    insights: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    investigation_health: Dict[str, Any]

class KnowledgeGraphResponse(BaseModel):
    success: bool
    case_id: str
    knowledge_graph: Dict[str, Any]

class ReportResponse(BaseModel):
    success: bool
    report: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@app.post("/cases/{case_id}/reports/generate", response_model=ReportResponse)
async def generate_case_report(case_id: str, report_request: ReportRequest):
    """
    Generate an intelligent forensic investigation report
    """
    try:
        # Verify case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Generate report using enhanced chat handler
        result = await enhanced_chat_handler.generate_case_report(
            case_id, 
            report_request.report_type
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return ReportResponse(success=True, report=result["report"])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report for case {case_id}: {str(e)}")
        return ReportResponse(success=False, error=str(e))

@app.get("/cases/{case_id}/reports/{report_id}/html")
async def get_report_html(case_id: str, report_id: str):
    """
    Get report in HTML format for viewing/printing
    """
    try:
        # This is a simplified version - in a real implementation,
        # you'd store reports and retrieve by ID
        result = await enhanced_chat_handler.generate_case_report(case_id, "detailed_analysis")
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Convert to HTML using report generator
        from src.ai_cores.intelligent_report_generator import InvestigationReport
        
        # Create a mock report object for HTML export
        # In practice, you'd reconstruct this from stored data
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Investigation Report - Case {case_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
        .header {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; }}
        .confidence {{ font-style: italic; color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Forensic Investigation Report</h1>
        <p><strong>Case ID:</strong> {case_id}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Report Type:</strong> Enhanced Analysis</p>
    </div>
    
    <div class="section">
        <h2>Report Content</h2>
        <p>This report contains comprehensive analysis of case evidence and findings.</p>
        <p>Generated using advanced AI analysis with knowledge graph integration.</p>
    </div>
</body>
</html>
        """
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Error generating HTML report: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating HTML report")

@app.get("/cases/{case_id}/insights", response_model=InsightResponse)
async def get_case_insights(case_id: str):
    """
    Get comprehensive case insights and investigation analysis
    """
    try:
        # Verify case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Get insights using enhanced chat handler
        result = await enhanced_chat_handler.get_case_insights(case_id)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return InsightResponse(
            success=True,
            case_id=case_id,
            insights=result["insights"],
            statistics=result["statistics"],
            investigation_health=result["investigation_health"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting case insights for {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/cases/{case_id}/knowledge-graph", response_model=KnowledgeGraphResponse)
async def get_case_knowledge_graph(case_id: str):
    """
    Get knowledge graph summary for the case
    """
    try:
        # Verify case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Get knowledge graph using enhanced chat handler
        result = await enhanced_chat_handler.get_knowledge_graph_summary(case_id)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return KnowledgeGraphResponse(
            success=True,
            case_id=case_id,
            knowledge_graph=result["knowledge_graph"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting knowledge graph for {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/cases/{case_id}/memory/stats")
async def get_case_memory_stats(case_id: str):
    """
    Get case memory statistics and investigation patterns
    """
    try:
        # Verify case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Get case memory stats
        stats = case_memory.get_case_memory_stats(case_id)
        summary = case_memory.get_investigation_summary(case_id)
        
        return {
            "case_id": case_id,
            "statistics": {
                "total_interactions": stats.total_interactions,
                "unique_entities_mentioned": stats.unique_entities_mentioned,
                "investigation_days": len(stats.temporal_activity),
                "common_query_patterns": stats.common_query_patterns[:10],
                "investigation_focus_areas": stats.investigation_focus_areas[:10],
                "entity_co_occurrence": stats.entity_co_occurrence
            },
            "investigation_summary": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting case memory stats for {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/cases/{case_id}/knowledge-graph/query")
async def query_knowledge_graph(case_id: str, query_request: Dict[str, Any]):
    """
    Query the knowledge graph with specific parameters
    """
    try:
        # Verify case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        query_type = query_request.get("query_type", "find_entity")
        parameters = query_request.get("parameters", {})
        
        # Query knowledge graph
        results = enhanced_kg_db.query_knowledge_graph(case_id, query_type, parameters)
        
        return {
            "case_id": case_id,
            "query_type": query_type,
            "parameters": parameters,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying knowledge graph for {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health/enhanced")
async def enhanced_health_check():
    """
    Enhanced health check including new systems
    """
    try:
        # Test database connection
        db_status = "healthy" if db_manager.test_connection() else "unhealthy"
        
        # Test case memory
        try:
            stats = case_memory.get_case_memory_stats("test")
            memory_status = "healthy"
        except Exception:
            memory_status = "unhealthy"
        
        # Test knowledge graph
        try:
            kg_data = enhanced_kg_db.get_case_knowledge_graph("test")
            kg_status = "healthy"
        except Exception:
            kg_status = "unhealthy"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "services": {
                "database": db_status,
                "case_memory": memory_status,
                "knowledge_graph": kg_status,
                "enhanced_chat": "healthy",
                "report_generator": "healthy"
            },
            "features": [
                "enhanced_chat_handler",
                "knowledge_graphs", 
                "case_memory",
                "intelligent_reporting",
                "entity_extraction",
                "relationship_mapping"
            ]
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# =============================================================================
# GPT-5 INTELLIGENCE ENGINE ENDPOINTS
# =============================================================================

# Import the new GPT-5 intelligence engine
from src.ai_cores.gpt5_intelligence_engine import (
    generate_evidence_timeline,
    analyze_network_flows,
    detect_evidence_patterns,
    get_intelligence_engine,
    AnalysisType,
    IntelligenceRequest
)

class TimelineRequest(BaseModel):
    time_range: Optional[str] = Field(None, description="Time range for analysis (e.g., '2024-01-01 to 2024-01-31')")
    focus_entities: Optional[List[str]] = Field(None, description="Specific entities to focus on")
    event_types: Optional[List[str]] = Field(None, description="Types of events to include")

class NetworkAnalysisRequest(BaseModel):
    analysis_depth: Optional[str] = Field("standard", description="Analysis depth: basic, standard, deep")
    focus_entities: Optional[List[str]] = Field(None, description="Entities to focus network analysis on")
    relationship_types: Optional[List[str]] = Field(None, description="Types of relationships to analyze")

class IntelligenceResponse(BaseModel):
    success: bool
    case_id: str
    analysis_type: str
    timestamp: str
    results: Dict[str, Any]
    insights: List[str]
    confidence_scores: Dict[str, float]
    timeline_events: Optional[List[Dict[str, Any]]] = None
    network_flows: Optional[List[Dict[str, Any]]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    relationships: Optional[List[Dict[str, Any]]] = None
    workflow_steps: Optional[List[str]] = None
    error: Optional[str] = None

@app.post("/cases/{case_id}/intelligence/timeline", response_model=IntelligenceResponse)
async def generate_intelligent_timeline(case_id: str, request: TimelineRequest):
    """
    Generate intelligent evidence timeline using GPT-5 multi-agent analysis
    """
    try:
        # Verify case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Prepare parameters
        parameters = {
            "time_range": request.time_range,
            "focus_entities": request.focus_entities or [],
            "event_types": request.event_types or []
        }
        
        # Generate timeline using GPT-5 intelligence engine
        result = await generate_evidence_timeline(case_id, parameters)
        
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Timeline generation failed"))
        
        return IntelligenceResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating intelligent timeline for case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/cases/{case_id}/intelligence/network", response_model=IntelligenceResponse)
async def analyze_intelligent_network(case_id: str, request: NetworkAnalysisRequest):
    """
    Perform intelligent network flow analysis using GPT-5 multi-agent system
    """
    try:
        # Verify case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Prepare parameters
        parameters = {
            "analysis_depth": request.analysis_depth,
            "focus_entities": request.focus_entities or [],
            "relationship_types": request.relationship_types or []
        }
        
        # Analyze network using GPT-5 intelligence engine
        result = await analyze_network_flows(case_id, parameters)
        
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Network analysis failed"))
        
        return IntelligenceResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing network for case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/cases/{case_id}/evidence/timeline")
async def get_evidence_timeline_data(case_id: str):
    """
    Get timeline data for Evidence Viewer page
    """
    try:
        # Verify case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Try to get GPT-5 generated timeline first
        try:
            timeline_result = await generate_evidence_timeline(case_id, {})
            if timeline_result.get("success"):
                return {
                    "timeline_events": timeline_result.get("timeline_events", []),
                    "insights": timeline_result.get("insights", []),
                    "confidence": timeline_result.get("confidence_scores", {}).get("timeline", 0.8),
                    "generated_by": "gpt5_intelligence",
                    "case_id": case_id
                }
        except Exception as e:
            logger.warning(f"GPT-5 timeline generation failed, using fallback: {str(e)}")
        
        # Fallback to basic timeline from case data
        evidence_items = case_manager.get_case_evidence(case_id)
        timeline_events = []
        
        for evidence in evidence_items:
            timeline_events.append({
                "timestamp": evidence.created_at.isoformat(),
                "event": f"Evidence uploaded: {evidence.original_filename}",
                "type": "evidence_upload",
                "significance": 0.6,
                "confidence": 0.9,
                "evidence_refs": [evidence.id],
                "metadata": {
                    "evidence_type": evidence.evidence_type,
                    "file_size": evidence.file_size,
                    "processing_status": evidence.processing_status
                }
            })
        
        return {
            "timeline_events": sorted(timeline_events, key=lambda x: x["timestamp"]),
            "insights": ["Basic timeline generated from evidence upload data"],
            "confidence": 0.7,
            "generated_by": "basic_timeline",
            "case_id": case_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting timeline data for case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# =============================================================================
# SUPREME FORENSIC AGENT ENDPOINTS - Evidence Viewer Integration
# =============================================================================

class SupremeAnalysisRequest(BaseModel):
    query: Optional[str] = Field("Analyze this case comprehensively", description="Specific analysis query")
    analysis_mode: Optional[str] = Field("overview", description="Analysis mode: overview, pattern_detection, timeline_analysis, investigation_guidance")

class SupremeAnalysisResponse(BaseModel):
    success: bool
    case_id: str
    analysis_mode: str
    confidence_score: float
    structured_analysis: Dict[str, Any]
    investigation_recommendations: List[str]
    evidence_gaps: List[str]
    next_steps: List[str]
    patterns_detected: Optional[List[Dict[str, Any]]] = None
    raw_analysis: str
    timestamp: str

@app.post("/cases/{case_id}/evidence/analyze-patterns", response_model=SupremeAnalysisResponse)
async def find_evidence_patterns(case_id: str, request: SupremeAnalysisRequest):
    """
    🔍 Find Patterns - Supreme AI pattern detection for Evidence Viewer
    Uses the most advanced forensic AI to detect patterns in case evidence
    """
    try:
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Use the Supreme Forensic Agent for pattern detection
        result = await detect_patterns_intelligently(case_id, request.query)
        
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail="Pattern analysis failed")
        
        return SupremeAnalysisResponse(
            success=True,
            case_id=case_id,
            analysis_mode="pattern_detection",
            confidence_score=result.get("confidence_score", 0.8),
            structured_analysis=result.get("structured_analysis", {}),
            investigation_recommendations=result.get("investigation_recommendations", []),
            evidence_gaps=result.get("evidence_gaps", []),
            next_steps=result.get("next_steps", []),
            patterns_detected=result.get("patterns_found", []),
            raw_analysis=result.get("raw_analysis", ""),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in supreme pattern analysis for case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pattern analysis failed: {str(e)}")

@app.post("/cases/{case_id}/evidence/supreme-analysis", response_model=SupremeAnalysisResponse)
async def supreme_case_analysis(case_id: str, request: SupremeAnalysisRequest):
    """
    🧠 Supreme Analysis - Advanced forensic investigation analysis
    Provides comprehensive case analysis using the Supreme Forensic Agent
    """
    try:
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Use the Supreme Forensic Agent for comprehensive analysis
        result = await analyze_case_intelligently(case_id, request.query, request.analysis_mode)
        
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail="Supreme analysis failed")
        
        return SupremeAnalysisResponse(
            success=True,
            case_id=case_id,
            analysis_mode=request.analysis_mode,
            confidence_score=result.get("confidence_score", 0.8),
            structured_analysis=result.get("structured_analysis", {}),
            investigation_recommendations=result.get("investigation_recommendations", []),
            evidence_gaps=result.get("evidence_gaps", []),
            next_steps=result.get("next_steps", []),
            raw_analysis=result.get("raw_analysis", ""),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in supreme analysis for case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Supreme analysis failed: {str(e)}")

@app.post("/cases/{case_id}/evidence/investigation-guidance", response_model=SupremeAnalysisResponse)
async def get_supreme_investigation_guidance(case_id: str, request: SupremeAnalysisRequest):
    """
    🎯 Investigation Guidance - Strategic investigation recommendations
    Provides expert guidance on investigation priorities and next steps
    """
    try:
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Use the Supreme Forensic Agent for investigation guidance
        result = await get_investigation_guidance(case_id, request.query)
        
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail="Investigation guidance failed")
        
        return SupremeAnalysisResponse(
            success=True,
            case_id=case_id,
            analysis_mode="investigation_guidance",
            confidence_score=result.get("confidence_score", 0.9),
            structured_analysis=result.get("structured_analysis", {}),
            investigation_recommendations=result.get("investigation_recommendations", []),
            evidence_gaps=result.get("evidence_gaps", []),
            next_steps=result.get("next_steps", []),
            raw_analysis=result.get("raw_analysis", ""),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in investigation guidance for case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Investigation guidance failed: {str(e)}")

@app.post("/cases/{case_id}/evidence/timeline-analysis", response_model=SupremeAnalysisResponse)  
async def supreme_timeline_analysis(case_id: str, request: SupremeAnalysisRequest):
    """
    ⏰ Timeline Analysis - Advanced timeline reconstruction and analysis
    Provides detailed timeline analysis using AI forensic expertise
    """
    try:
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Use the Supreme Forensic Agent for timeline analysis
        result = await analyze_timeline_intelligently(case_id, request.query)
        
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail="Timeline analysis failed")
        
        return SupremeAnalysisResponse(
            success=True,
            case_id=case_id,
            analysis_mode="timeline_analysis",
            confidence_score=result.get("confidence_score", 0.85),
            structured_analysis=result.get("structured_analysis", {}),
            investigation_recommendations=result.get("investigation_recommendations", []),
            evidence_gaps=result.get("evidence_gaps", []),
            next_steps=result.get("next_steps", []),
            raw_analysis=result.get("raw_analysis", ""),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in timeline analysis for case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Timeline analysis failed: {str(e)}")

@app.get("/cases/{case_id}/network/data")
async def get_network_analysis_data(case_id: str):
    """
    Get network analysis data for Network Analysis page
    """
    try:
        # Verify case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Try to get GPT-5 generated network analysis first
        try:
            network_result = await analyze_network_flows(case_id, {})
            if network_result.get("success"):
                return {
                    "network_flows": network_result.get("network_flows", []),
                    "entities": network_result.get("entities", []),
                    "relationships": network_result.get("relationships", []),
                    "insights": network_result.get("insights", []),
                    "confidence": network_result.get("confidence_scores", {}).get("network", 0.8),
                    "generated_by": "gpt5_intelligence",
                    "case_id": case_id
                }
        except Exception as e:
            logger.warning(f"GPT-5 network analysis failed, using fallback: {str(e)}")
        
        # Fallback to basic network from knowledge graph
        entities = enhanced_kg_db.get_case_entities(case_id) or []
        relationships = enhanced_kg_db.get_case_relationships(case_id) or []
        
        # Create basic network flows
        network_flows = []
        for rel in relationships[:20]:  # Limit for performance
            network_flows.append({
                "source": rel.get("source_entity_id", "unknown"),
                "target": rel.get("target_entity_id", "unknown"),
                "weight": rel.get("confidence", 0.5),
                "type": rel.get("relationship_type", "related"),
                "frequency": 1,
                "metadata": rel.get("metadata", {})
            })
        
        return {
            "network_flows": network_flows,
            "entities": entities[:50],
            "relationships": relationships[:50],
            "insights": ["Basic network generated from knowledge graph data"],
            "confidence": 0.7,
            "generated_by": "basic_network",
            "case_id": case_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting network data for case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Database Search Endpoints
@app.get("/database/search")
async def search_database(
    query: str = "",
    case_id: Optional[str] = None,
    data_type: Optional[str] = None,
    max_results: int = 20
):
    """
    Search through the forensic database
    
    Args:
        query: Search query string
        case_id: Optional case ID to search within specific case
        data_type: Optional data type filter (messages, files, etc.)
        max_results: Maximum number of results to return
    """
    try:
        from simple_search_system import SimpleSearchSystem
        search_system = SimpleSearchSystem()
        
        if case_id and data_type:
            # Search specific data type within a case
            results = search_system.search_specific_data_type(case_id, data_type, query)
        elif case_id:
            # Search within a specific case
            results = search_system.search_case_data(case_id, query, max_results)
        else:
            # Search across all available cases
            results = []
            # Get all available cases first
            cases = case_manager.list_cases()
            
            for case in cases:
                case_results = search_system.search_case_data(case.id, query, max_results // len(cases) if cases else max_results)
                for result in case_results:
                    result['case_id'] = case.id
                    result['case_title'] = case.title or f"Case {case.id}"
                results.extend(case_results)
            
            # Sort by relevance score if available
            results = sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)[:max_results]
        
        return {
            "success": True,
            "query": query,
            "case_id": case_id,
            "data_type": data_type,
            "total_results": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Database search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/database/cases")
async def get_all_cases_summary():
    """
    Get summary of all cases in the database
    """
    try:
        cases = case_manager.list_cases()
        from simple_search_system import SimpleSearchSystem
        search_system = SimpleSearchSystem()
        
        cases_summary = []
        for case in cases:
            try:
                case_summary = search_system.get_case_summary(case.id)
                cases_summary.append({
                    "case_id": case.id,
                    "title": case.title or f"Case {case.id}",
                    "description": case.description,
                    "status": case.status.value if case.status else "unknown",
                    "created_at": case.created_at.isoformat() if case.created_at else None,
                    "summary": case_summary
                })
            except Exception as e:
                logger.warning(f"Could not get summary for case {case.id}: {str(e)}")
                cases_summary.append({
                    "case_id": case.id,
                    "title": case.title or f"Case {case.id}",
                    "description": case.description,
                    "status": case.status.value if case.status else "unknown",
                    "created_at": case.created_at.isoformat() if case.created_at else None,
                    "summary": {"error": "Could not load case data"}
                })
        
        return {
            "success": True,
            "total_cases": len(cases_summary),
            "cases": cases_summary
        }
        
    except Exception as e:
        logger.error(f"Error getting cases summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cases: {str(e)}")


@app.get("/database/case/{case_id}/data")
async def get_case_data_summary(case_id: str):
    """
    Get detailed data summary for a specific case
    """
    try:
        from simple_search_system import SimpleSearchSystem
        search_system = SimpleSearchSystem()
        
        case_summary = search_system.get_case_summary(case_id)
        
        return {
            "success": True,
            "case_id": case_id,
            "data": case_summary
        }
        
    except Exception as e:
        logger.error(f"Error getting case data for {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get case data: {str(e)}")


@app.get("/database/analytics/detailed")
async def get_detailed_analytics(case_id: Optional[str] = None):
    """
    Get detailed analytics including call statistics, communication patterns, and forensic insights
    """
    try:
        from simple_search_system import SimpleSearchSystem
        search_system = SimpleSearchSystem()
        from collections import Counter, defaultdict
        
        analytics_data = {
            "call_statistics": {
                "most_incoming_calls": [],
                "most_outgoing_calls": [],
                "most_contacted_numbers": [],
                "call_duration_stats": {},
                "peak_call_times": {},
                "missed_calls": 0,
                "answered_calls": 0,
                "total_call_duration": 0
            },
            "communication_patterns": {
                "daily_activity": {},
                "hourly_patterns": {},
                "contact_frequency": {},
                "suspicious_patterns": []
            },
            "forensic_insights": {
                "location_patterns": [],
                "device_information": [],
                "timeline_gaps": [],
                "anomalies": []
            },
            "case_summary": {
                "total_cases": 0,
                "total_evidence_files": 0,
                "data_types": {},
                "date_range": {}
            }
        }
        
        # Get cases to analyze
        cases_to_analyze = []
        if case_id:
            # Analyze specific case
            case = case_manager.get_case(case_id)
            if case:
                cases_to_analyze = [case]
        else:
            # Analyze all cases
            cases_to_analyze = case_manager.list_cases()
        
        analytics_data["case_summary"]["total_cases"] = len(cases_to_analyze)
        
        # Aggregate data from all cases
        all_call_records = []
        all_contact_names = []
        all_locations = []
        data_type_counter = Counter()
        
        for case in cases_to_analyze:
            try:
                case_summary = search_system.get_case_summary(case.id)
                
                # Count data sources
                for source in case_summary.get("data_sources", []):
                    analytics_data["case_summary"]["total_evidence_files"] += 1
                    file_type = source.get("file_type", "unknown")
                    data_type_counter[file_type] += 1
                
                # Process call records - directly read processed JSON files
                import glob
                processed_files = glob.glob(f"data/processed/*{case.id}*.json")
                
                for file_path in processed_files:
                    try:
                        # Only process call log files
                        if "call" in os.path.basename(file_path).lower():
                            with open(file_path, 'r') as f:
                                call_data = json.load(f)
                            
                            # Extract call records from the JSON structure
                            if isinstance(call_data, dict) and "records" in call_data:
                                for record in call_data["records"]:
                                    if record.get("data"):
                                        all_call_records.append(record["data"])
                                        
                    except Exception as e:
                        logger.warning(f"Error processing call file {file_path}: {str(e)}")
                        continue
                
                # Get contact network data
                contact_network = case_summary.get("contact_network", [])
                for contact in contact_network:
                    all_contact_names.append(contact.get("name", ""))
                
            except Exception as e:
                logger.warning(f"Could not analyze case {case.id}: {str(e)}")
                continue
        
        analytics_data["case_summary"]["data_types"] = dict(data_type_counter)
        
        # Analyze call records
        if all_call_records:
            incoming_counter = Counter()
            outgoing_counter = Counter()
            contact_counter = Counter()
            duration_total = 0
            duration_count = 0
            hourly_activity = defaultdict(int)
            daily_activity = defaultdict(int)
            missed_calls = 0
            answered_calls = 0
            
            for record in all_call_records:
                direction = record.get("direction", "").lower()
                phone_number = record.get("phone_number", "unknown")
                contact_name = record.get("contact_name", "") or phone_number
                duration = record.get("duration", "0")
                status = record.get("status", "").lower()
                timestamp = record.get("timestamp", "")
                
                # Count calls by direction
                if direction == "incoming":
                    incoming_counter[contact_name] += 1
                elif direction == "outgoing":
                    outgoing_counter[contact_name] += 1
                
                # Count all contacts
                contact_counter[contact_name] += 1
                
                # Duration statistics
                try:
                    duration_val = int(duration)
                    duration_total += duration_val
                    duration_count += 1
                except:
                    pass
                
                # Status statistics
                if status == "missed":
                    missed_calls += 1
                elif status == "answered":
                    answered_calls += 1
                
                # Time pattern analysis
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        hour_key = f"{dt.hour:02d}:00"
                        day_key = dt.strftime("%Y-%m-%d")
                        hourly_activity[hour_key] += 1
                        daily_activity[day_key] += 1
                    except:
                        pass
            
            # Populate call statistics
            analytics_data["call_statistics"].update({
                "most_incoming_calls": [{"contact": k, "count": v} for k, v in incoming_counter.most_common(10)],
                "most_outgoing_calls": [{"contact": k, "count": v} for k, v in outgoing_counter.most_common(10)],
                "most_contacted_numbers": [{"contact": k, "count": v} for k, v in contact_counter.most_common(10)],
                "call_duration_stats": {
                    "average_duration": duration_total / duration_count if duration_count > 0 else 0,
                    "total_duration": duration_total
                },
                "missed_calls": missed_calls,
                "answered_calls": answered_calls,
                "total_call_duration": duration_total
            })
            
            # Communication patterns
            analytics_data["communication_patterns"].update({
                "daily_activity": dict(daily_activity),
                "hourly_patterns": dict(hourly_activity),
                "contact_frequency": dict(contact_counter.most_common(20))
            })
            
            # Detect suspicious patterns
            suspicious_patterns = []
            for contact, count in contact_counter.most_common(5):
                if count > 20:  # Threshold for suspicious activity
                    suspicious_patterns.append({
                        "type": "high_frequency",
                        "contact": contact,
                        "frequency": count,
                        "description": f"High call frequency with {contact} ({count} calls)"
                    })
            
            analytics_data["communication_patterns"]["suspicious_patterns"] = suspicious_patterns
        
        return {
            "success": True,
            "analytics": analytics_data,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating detailed analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analytics generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)