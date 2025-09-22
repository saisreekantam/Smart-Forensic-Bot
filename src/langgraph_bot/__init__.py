"""
LangGraph Forensic Bot Module

This module provides a comprehensive forensic investigation assistant
built using LangGraph for intelligent workflow orchestration.
"""

from .forensic_bot import ForensicBot, create_forensic_bot
from .state import (
    ForensicBotState, 
    Entity, 
    Event, 
    Pattern, 
    Hypothesis,
    create_initial_state,
    update_state_activity
)

__version__ = "1.0.0"
__author__ = "Forensic Investigation Team"

__all__ = [
    "ForensicBot",
    "create_forensic_bot", 
    "ForensicBotState",
    "Entity",
    "Event", 
    "Pattern",
    "Hypothesis",
    "create_initial_state",
    "update_state_activity"
]