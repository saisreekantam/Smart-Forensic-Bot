"""
Case Management API with Intelligent Chatbot

FastAPI endpoints for managing forensic cases and providing intelligent
chat-based investigation assistance using OpenAI GPT models (GPT-4o-mini and GPT-4).
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
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
from src.ai_cores.enhanced_assistant import EnhancedCaseAssistant

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Project Sentinel - Forensic Case Management API",
    description="AI-powered forensic investigation platform with intelligent chatbot",
    version="1.0.0"
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
        global case_data_processor, optimized_processor, enhanced_assistant
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
        
        # Initialize enhanced assistant for sophisticated AI responses
        enhanced_assistant = EnhancedCaseAssistant(
            case_manager=case_manager,
            vector_store=case_vector_store
        )
        
        logger.info("🚀 API startup completed successfully with enhanced components")
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
            status=case.status.value,
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
                status=case.status.value,
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
                "status": case.status.value,
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
                evidence_type=evidence.evidence_type.value,
                processing_status=evidence.processing_status.value,
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

# Intelligent Chatbot Endpoints
@app.post("/cases/{case_id}/chat", response_model=ChatResponse)
async def chat_with_case(case_id: str, chat_message: ChatMessage):
    """
    Enhanced intelligent chatbot for case investigation
    Uses GPT-4/GPT-4o-mini with sophisticated reasoning and RAG integration
    """
    try:
        # Validate case exists
        case = case_manager.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Check if case has processed evidence
        if case.processed_evidence_count == 0:
            return ChatResponse(
                response="⚠️ This case doesn't have any processed evidence yet. Please upload and process evidence files first to enable AI analysis.",
                sources=[],
                confidence=1.0,
                case_context={"case_number": case.case_number, "evidence_count": 0},
                timestamp=datetime.now()
            )
        
        # Use enhanced assistant for sophisticated analysis
        investigation_response = await enhanced_assistant.process_investigation_query(
            case_id=case_id,
            query=chat_message.message,
            conversation_history=chat_message.conversation_history or []
        )
        
        # Convert to API response format
        return ChatResponse(
            response=f"{investigation_response.response}\n\n🤖 *Analysis powered by {investigation_response.model_used.upper()}* | Complexity: {investigation_response.complexity_analysis.complexity_score:.2f} | Confidence: {investigation_response.confidence:.1%}",
            sources=investigation_response.evidence_sources,
            confidence=investigation_response.confidence,
            case_context={
                "case_number": case.case_number,
                "evidence_count": case.total_evidence_count,
                "processed_count": case.processed_evidence_count,
                "reasoning_type": investigation_response.complexity_analysis.reasoning_type,
                "model_used": investigation_response.model_used
            },
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat for case {case_id}: {str(e)}")
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
- Case Status: {case.status.value}
- Priority: {case.priority.value if hasattr(case, 'priority') else 'Standard'}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)