"""
Forensic Bot State Management

This module defines the comprehensive state structure for the LangGraph-based
forensic investigation bot, including conversation context, memory, evidence tracking,
and investigation workflow state.
"""

from typing import TypedDict, Annotated, Sequence, Optional, List, Dict, Any, Union
from langchain_core.messages import BaseMessage
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import operator

class InvestigationPhase(Enum):
    """Investigation phases for the forensic bot"""
    INTAKE = "intake"                   # Initial case setup and evidence ingestion
    ANALYSIS = "analysis"               # Active evidence analysis and processing
    SYNTHESIS = "synthesis"             # Cross-evidence correlation and pattern recognition
    REPORTING = "reporting"             # Report generation and findings compilation
    CONVERSATION = "conversation"       # General conversation and query handling

class ConfidenceLevel(Enum):
    """Confidence levels for findings and analysis"""
    VERY_HIGH = "very_high"    # 90-100%
    HIGH = "high"              # 70-89%
    MEDIUM = "medium"          # 50-69%
    LOW = "low"                # 30-49%
    VERY_LOW = "very_low"      # 0-29%

@dataclass
class Entity:
    """Represents a forensic entity (person, place, object, etc.)"""
    id: str
    name: str
    type: str  # person, phone, email, location, organization, etc.
    attributes: Dict[str, Any] = field(default_factory=dict)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    confidence: float = 0.0
    case_id: Optional[str] = None

@dataclass
class Event:
    """Represents a temporal event in the investigation"""
    id: str
    timestamp: datetime
    event_type: str
    description: str
    entities_involved: List[str] = field(default_factory=list)
    evidence_source: Optional[str] = None
    confidence: float = 0.0
    case_id: Optional[str] = None

@dataclass
class Pattern:
    """Represents a detected pattern in the evidence"""
    id: str
    pattern_type: str  # communication, behavioral, temporal, financial, etc.
    description: str
    entities: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    confidence: float = 0.0
    significance: str = "medium"  # low, medium, high, critical
    first_detected: datetime = field(default_factory=datetime.now)

@dataclass
class Hypothesis:
    """Represents an investigative hypothesis or theory"""
    id: str
    description: str
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    status: str = "active"  # active, validated, refuted, pending
    created: datetime = field(default_factory=datetime.now)
    updated: datetime = field(default_factory=datetime.now)

@dataclass
class Evidence:
    """Represents a piece of evidence in the investigation"""
    id: str
    file_path: str
    evidence_type: str
    case_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False
    embeddings_generated: bool = False
    analysis_results: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Action:
    """Represents a pending or completed action"""
    id: str
    action_type: str
    description: str
    priority: str = "medium"  # low, medium, high, urgent
    status: str = "pending"  # pending, in_progress, completed, failed
    created: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None

@dataclass
class WorkflowStep:
    """Represents a step in the investigation workflow"""
    timestamp: datetime
    node_name: str
    action: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    execution_time: float
    success: bool

class ForensicBotState(TypedDict):
    """
    Comprehensive state for the LangGraph forensic investigation bot
    
    This state maintains all context, memory, and workflow information
    needed for intelligent forensic investigation assistance.
    """
    
    # === CONVERSATION CONTEXT ===
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_case_id: Optional[str]
    investigation_phase: str  # Current phase of investigation
    user_intent: Optional[str]  # Last detected user intent
    conversation_history: List[Dict[str, Any]]
    
    # === MEMORY & KNOWLEDGE ===
    entity_memory: Dict[str, Entity]  # People, places, objects, etc.
    timeline_memory: List[Event]  # Chronological events
    pattern_memory: Dict[str, Pattern]  # Detected patterns
    hypothesis_tracker: List[Hypothesis]  # Working theories
    relationship_graph: Dict[str, List[str]]  # Entity relationships
    
    # === EVIDENCE CONTEXT ===
    active_evidence: List[Evidence]  # Currently relevant evidence
    evidence_index: Dict[str, str]  # Evidence ID to file path mapping
    analysis_results: Dict[str, Any]  # Cached analysis results
    cross_references: Dict[str, List[str]]  # Cross-evidence references
    embeddings_cache: Dict[str, Any]  # Cached embeddings and retrievals
    
    # === INVESTIGATION STATE ===
    current_focus: Optional[str]  # What we're currently investigating
    pending_actions: List[Action]  # Next steps to take
    completed_actions: List[Action]  # Completed actions history
    confidence_scores: Dict[str, float]  # Confidence in various findings
    investigation_goals: List[str]  # Current investigation objectives
    
    # === TOOL USAGE & WORKFLOW ===
    tools_used: List[str]  # History of tools used
    tool_results: Dict[str, Any]  # Results from tool executions
    workflow_history: List[WorkflowStep]  # Complete workflow history
    last_tool_error: Optional[str]  # Last tool error for debugging
    
    # === RAG & KNOWLEDGE GRAPH STATE ===
    rag_context: Dict[str, Any]  # RAG system context and settings
    kg_query_results: Dict[str, Any]  # Knowledge graph query results
    semantic_search_cache: Dict[str, Any]  # Cached semantic search results
    
    # === ANALYSIS & SYNTHESIS ===
    analysis_summary: Dict[str, Any]  # Current analysis summary
    key_findings: List[str]  # Key findings from investigation
    anomalies_detected: List[Dict[str, Any]]  # Detected anomalies
    recommendations: List[str]  # Current recommendations
    
    # === REPORTING STATE ===
    report_sections: Dict[str, Any]  # Generated report sections
    visualization_data: Dict[str, Any]  # Data for visualizations
    export_format: Optional[str]  # Desired export format
    
    # === SESSION MANAGEMENT ===
    session_id: str  # Unique session identifier
    session_start: datetime  # Session start time
    last_activity: datetime  # Last activity timestamp
    debug_mode: bool  # Whether debug mode is enabled

def create_initial_state(
    case_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False
) -> ForensicBotState:
    """
    Create an initial state for the forensic bot
    
    Args:
        case_id: Optional case ID to start with
        session_id: Optional session ID
        debug_mode: Whether to enable debug mode
        
    Returns:
        ForensicBotState: Initial state configuration
    """
    import uuid
    from datetime import datetime
    
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    now = datetime.now()
    
    return ForensicBotState(
        # Conversation Context
        messages=[],
        current_case_id=case_id,
        investigation_phase=InvestigationPhase.INTAKE.value,
        user_intent=None,
        conversation_history=[],
        
        # Memory & Knowledge
        entity_memory={},
        timeline_memory=[],
        pattern_memory={},
        hypothesis_tracker=[],
        relationship_graph={},
        
        # Evidence Context
        active_evidence=[],
        evidence_index={},
        analysis_results={},
        cross_references={},
        embeddings_cache={},
        
        # Investigation State
        current_focus=None,
        pending_actions=[],
        completed_actions=[],
        confidence_scores={},
        investigation_goals=[],
        
        # Tool Usage & Workflow
        tools_used=[],
        tool_results={},
        workflow_history=[],
        last_tool_error=None,
        
        # RAG & Knowledge Graph State
        rag_context={},
        kg_query_results={},
        semantic_search_cache={},
        
        # Analysis & Synthesis
        analysis_summary={},
        key_findings=[],
        anomalies_detected=[],
        recommendations=[],
        
        # Reporting State
        report_sections={},
        visualization_data={},
        export_format=None,
        
        # Session Management
        session_id=session_id,
        session_start=now,
        last_activity=now,
        debug_mode=debug_mode
    )

def update_state_activity(state: ForensicBotState) -> ForensicBotState:
    """Update the last activity timestamp in the state"""
    state["last_activity"] = datetime.now()
    return state

def add_workflow_step(
    state: ForensicBotState,
    node_name: str,
    action: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    execution_time: float,
    success: bool = True
) -> ForensicBotState:
    """Add a workflow step to the state history"""
    step = WorkflowStep(
        timestamp=datetime.now(),
        node_name=node_name,
        action=action,
        input_data=input_data,
        output_data=output_data,
        execution_time=execution_time,
        success=success
    )
    state["workflow_history"].append(step)
    return update_state_activity(state)