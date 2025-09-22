"""
Conversation Router Node

This module implements the conversation routing logic for the forensic bot,
determining the appropriate workflow path based on user intent and context.
"""

from typing import Dict, Any, List
import re
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

from ..state import ForensicBotState, InvestigationPhase, add_workflow_step

class IntentClassifier:
    """Classifies user intents for forensic investigations"""
    
    def __init__(self):
        self.intent_patterns = {
            "evidence_analysis": [
                r"analyze.*evidence",
                r"process.*file",
                r"examine.*data",
                r"look at.*evidence",
                r"investigate.*file",
                r"what.*in.*evidence",
                r"parse.*report",
                r"extract.*from"
            ],
            "question_answering": [
                r"what.*happened",
                r"who.*involved",
                r"when.*occur",
                r"where.*located",
                r"how.*connect",
                r"tell me about",
                r"explain.*",
                r"describe.*",
                r"show me.*",
                r"find.*information"
            ],
            "relationship_analysis": [
                r"relationship.*between",
                r"connection.*between",
                r"link.*between",
                r"who.*communicat",
                r"network.*analysis",
                r"social.*graph",
                r"contact.*pattern",
                r"interaction.*between"
            ],
            "pattern_detection": [
                r"pattern.*in",
                r"anomal.*in",
                r"unusual.*activit",
                r"suspicious.*behav",
                r"detect.*pattern",
                r"find.*trend",
                r"identify.*pattern",
                r"behavioral.*analysis"
            ],
            "timeline_analysis": [
                r"timeline.*of",
                r"chronolog.*order",
                r"sequence.*of.*event",
                r"when.*happen",
                r"temporal.*analysis",
                r"time.*series",
                r"order.*of.*event",
                r"reconstruct.*timeline"
            ],
            "synthesis": [
                r"summarize.*case",
                r"overall.*picture",
                r"compile.*finding",
                r"cross.*reference",
                r"correlate.*evidence",
                r"synthesis.*of",
                r"comprehensive.*analysis",
                r"big.*picture"
            ],
            "reporting": [
                r"generate.*report",
                r"create.*summary",
                r"export.*findings",
                r"prepare.*report",
                r"document.*findings",
                r"formal.*report",
                r"investigation.*report",
                r"case.*summary"
            ],
            "case_management": [
                r"new.*case",
                r"switch.*case",
                r"load.*case",
                r"case.*details",
                r"case.*status",
                r"manage.*case",
                r"create.*case",
                r"open.*case"
            ],
            "greeting": [
                r"hello",
                r"hi",
                r"good.*morning",
                r"good.*afternoon",
                r"good.*evening",
                r"hey",
                r"help",
                r"start",
                r"who.*are.*you",
                r"what.*are.*you",
                r"introduce",
                r"tell.*me.*about",
                r"what.*is.*digital.*forensics",
                r"what.*are.*the.*principles",
                r"explain.*forensics",
                r"key.*principles",
                r"digital.*forensics.*principles",
                r"forensic.*methodology",
                r"best.*practices",
                r"how.*does.*forensics.*work"
            ]
        }
    
    def classify_intent(self, message: str) -> str:
        """
        Classify the user's intent based on their message
        
        Args:
            message: User's message text
            
        Returns:
            str: Classified intent
        """
        message_lower = message.lower()
        
        # Score each intent
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    score += 1
            intent_scores[intent] = score
        
        # Return the highest scoring intent
        if max(intent_scores.values()) > 0:
            return max(intent_scores, key=intent_scores.get)
        
        # Default to general conversation if no clear intent
        return "greeting"

def conversation_router(state: ForensicBotState) -> ForensicBotState:
    """
    Route conversations based on user intent and current context
    
    This node analyzes the user's message and current investigation state
    to determine the most appropriate next action.
    
    Args:
        state: Current forensic bot state
        
    Returns:
        ForensicBotState: Updated state with routing decision
    """
    start_time = datetime.now()
    
    try:
        # Get the last user message
        if not state["messages"]:
            # No messages yet, set up initial greeting
            state["user_intent"] = "greeting"
            state["investigation_phase"] = InvestigationPhase.CONVERSATION.value
            
            # Add workflow step
            execution_time = (datetime.now() - start_time).total_seconds()
            add_workflow_step(
                state,
                node_name="conversation_router",
                action="initial_setup",
                input_data={"messages_count": 0},
                output_data={"intent": "greeting", "phase": "conversation"},
                execution_time=execution_time,
                success=True
            )
            
            return state
        
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            message_text = last_message.content
        else:
            # If last message is not from human, default to conversation
            state["user_intent"] = "question_answering"
            return state
        
        # Initialize intent classifier
        classifier = IntentClassifier()
        
        # Classify the intent
        intent = classifier.classify_intent(message_text)
        state["user_intent"] = intent
        
        # Determine investigation phase based on intent and context
        phase_mapping = {
            "evidence_analysis": InvestigationPhase.ANALYSIS,
            "question_answering": InvestigationPhase.CONVERSATION,
            "relationship_analysis": InvestigationPhase.ANALYSIS,
            "pattern_detection": InvestigationPhase.ANALYSIS,
            "timeline_analysis": InvestigationPhase.ANALYSIS,
            "synthesis": InvestigationPhase.SYNTHESIS,
            "reporting": InvestigationPhase.REPORTING,
            "case_management": InvestigationPhase.INTAKE,
            "greeting": InvestigationPhase.CONVERSATION
        }
        
        state["investigation_phase"] = phase_mapping.get(
            intent, InvestigationPhase.CONVERSATION
        ).value
        
        # Update conversation history
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message_text,
            "intent": intent,
            "phase": state["investigation_phase"]
        }
        state["conversation_history"].append(conversation_entry)
        
        # Set current focus based on intent
        if intent in ["evidence_analysis", "pattern_detection"]:
            state["current_focus"] = "evidence_processing"
        elif intent in ["relationship_analysis", "timeline_analysis"]:
            state["current_focus"] = "relationship_mapping"
        elif intent == "synthesis":
            state["current_focus"] = "evidence_correlation"
        elif intent == "reporting":
            state["current_focus"] = "report_generation"
        else:
            state["current_focus"] = "conversation"
        
        # Determine next node based on intent
        next_node_mapping = {
            "evidence_analysis": "evidence_processor",
            "question_answering": "rag_analyzer",
            "relationship_analysis": "knowledge_graph_reasoner",
            "pattern_detection": "pattern_detector",
            "timeline_analysis": "knowledge_graph_reasoner",
            "synthesis": "synthesis_engine",
            "reporting": "report_generator",
            "case_management": "case_manager",
            "greeting": "conversation_handler"
        }
        
        next_node = next_node_mapping.get(intent, "conversation_handler")
        state["_next_node"] = next_node  # Store next node for conditional routing
        
        # Add contextual information for downstream nodes
        state["routing_context"] = {
            "original_message": message_text,
            "classified_intent": intent,
            "next_node": next_node,
            "reasoning": f"Classified as '{intent}' based on message content analysis"
        }
        
        # Log the routing decision
        execution_time = (datetime.now() - start_time).total_seconds()
        add_workflow_step(
            state,
            node_name="conversation_router",
            action="route_conversation",
            input_data={
                "message": message_text,
                "message_length": len(message_text)
            },
            output_data={
                "intent": intent,
                "phase": state["investigation_phase"],
                "next_node": next_node,
                "focus": state["current_focus"]
            },
            execution_time=execution_time,
            success=True
        )
        
    except Exception as e:
        # Handle errors gracefully
        state["last_tool_error"] = f"Conversation routing error: {str(e)}"
        state["user_intent"] = "question_answering"  # Default fallback
        state["_next_node"] = "rag_analyzer"
        
        execution_time = (datetime.now() - start_time).total_seconds()
        add_workflow_step(
            state,
            node_name="conversation_router",
            action="route_conversation",
            input_data={"error": str(e)},
            output_data={"fallback_intent": "question_answering"},
            execution_time=execution_time,
            success=False
        )
    
    return state

def route_conversation(state: ForensicBotState) -> str:
    """
    Conditional routing function for LangGraph
    
    Args:
        state: Current forensic bot state
        
    Returns:
        str: Next node name
    """
    # Get the next node from routing context
    next_node = state.get("_next_node", "conversation_handler")
    
    # Remove the temporary routing variable
    if "_next_node" in state:
        del state["_next_node"]
    
    return next_node

def conversation_handler(state: ForensicBotState) -> ForensicBotState:
    """
    Handle general conversation and forensic expertise questions using OpenAI
    
    Args:
        state: Current forensic bot state
        
    Returns:
        ForensicBotState: Updated state with response
    """
    start_time = datetime.now()
    
    try:
        from openai import OpenAI
        client = OpenAI()
        
        # Get the user's message
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        if user_messages:
            user_message = user_messages[-1].content
        else:
            user_message = "Hello"
        
        # Create forensic-specialized prompt
        forensic_prompt = f"""You are an expert digital forensics investigator and AI assistant. 
You specialize in cybercrime investigation, digital evidence analysis, and forensic methodology.

Key capabilities:
- Digital evidence analysis (mobile devices, computers, networks)
- Cybercrime investigation techniques
- Data recovery and preservation
- Timeline reconstruction
- Pattern detection in digital artifacts
- Legal considerations in digital forensics
- Chain of custody procedures
- Forensic tools and methodologies

User question: {user_message}

Provide a helpful, accurate, and professional response based on your forensic expertise. 
If the question is not forensics-related, politely redirect to forensic topics while still being helpful.
Keep responses concise but informative."""

        # Get response from OpenAI
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": forensic_prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        response = completion.choices[0].message.content
        
        # Add the AI response to messages
        ai_message = AIMessage(content=response)
        state["messages"].append(ai_message)
        
        # Update conversation history
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": response,
            "type": "ai_response",
            "intent": state.get("user_intent", "conversation")
        }
        state["conversation_history"].append(conversation_entry)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        add_workflow_step(
            state,
            node_name="conversation_handler",
            action="openai_conversation",
            input_data={"user_message": user_message},
            output_data={"response_length": len(response)},
            execution_time=execution_time,
            success=True
        )
        
    except Exception as e:
        state["last_tool_error"] = f"Conversation handler error: {str(e)}"
        
        # Fallback response
        fallback_response = (
            "I'm your AI forensic investigation assistant. "
            "I can help you analyze evidence, answer questions about digital forensics, "
            "detect patterns, and generate comprehensive reports. "
            "How can I assist with your investigation today?"
        )
        ai_message = AIMessage(content=fallback_response)
        state["messages"].append(ai_message)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        add_workflow_step(
            state,
            node_name="conversation_handler",
            action="fallback_conversation",
            input_data={"error": str(e)},
            output_data={"fallback_response": True},
            execution_time=execution_time,
            success=False
        )
    
    return state