"""
Pattern Detector Node

This module implements pattern detection capabilities for forensic investigations,
including behavioral, communication, temporal, and anomaly detection.
"""

import sys
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
import numpy as np
from langchain_core.messages import AIMessage

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ..state import ForensicBotState, Pattern, Entity, Event, add_workflow_step

class PatternDetector:
    """Advanced pattern detection for forensic analysis"""
    
    def __init__(self):
        self.pattern_types = {
            "communication": self.detect_communication_patterns,
            "temporal": self.detect_temporal_patterns,
            "behavioral": self.detect_behavioral_patterns,
            "financial": self.detect_financial_patterns,
            "geographical": self.detect_geographical_patterns,
            "anomaly": self.detect_anomalies
        }
    
    def detect_patterns(
        self, 
        entities: Dict[str, Entity], 
        events: List[Event],
        focus_type: Optional[str] = None
    ) -> List[Pattern]:
        """
        Detect patterns in forensic data
        
        Args:
            entities: Dictionary of entities
            events: List of events
            focus_type: Optional specific pattern type to focus on
            
        Returns:
            List of detected patterns
        """
        detected_patterns = []
        
        # Determine which pattern types to run
        pattern_types_to_run = [focus_type] if focus_type and focus_type in self.pattern_types else self.pattern_types.keys()
        
        for pattern_type in pattern_types_to_run:
            try:
                patterns = self.pattern_types[pattern_type](entities, events)
                detected_patterns.extend(patterns)
            except Exception as e:
                print(f"Error detecting {pattern_type} patterns: {e}")
        
        # Sort patterns by significance and confidence
        detected_patterns.sort(key=lambda p: (
            {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(p.significance, 0),
            p.confidence
        ), reverse=True)
        
        return detected_patterns
    
    def detect_communication_patterns(
        self, 
        entities: Dict[str, Entity], 
        events: List[Event]
    ) -> List[Pattern]:
        """Detect communication patterns"""
        patterns = []
        
        # Communication frequency analysis
        communication_events = [e for e in events if "communication" in e.event_type.lower() or "call" in e.event_type.lower() or "message" in e.event_type.lower()]
        
        if len(communication_events) < 2:
            return patterns
        
        # Analyze communication frequency by entity pairs
        entity_pair_communications = defaultdict(list)
        
        for event in communication_events:
            entities_in_event = event.entities_involved
            if len(entities_in_event) >= 2:
                # Create pairs
                for i in range(len(entities_in_event)):
                    for j in range(i + 1, len(entities_in_event)):
                        pair = tuple(sorted([entities_in_event[i], entities_in_event[j]]))
                        entity_pair_communications[pair].append(event)
        
        # Detect high-frequency communication patterns
        for pair, communications in entity_pair_communications.items():
            if len(communications) > 5:  # Threshold for frequent communication
                # Analyze timing patterns
                timestamps = [event.timestamp for event in communications]
                timestamps.sort()
                
                # Check for burst patterns
                time_gaps = [(timestamps[i+1] - timestamps[i]).total_seconds() / 60 for i in range(len(timestamps)-1)]  # minutes
                
                avg_gap = np.mean(time_gaps) if time_gaps else 0
                
                # Detect communication bursts (multiple communications in short time)
                burst_threshold = 10  # 10 minutes
                bursts = sum(1 for gap in time_gaps if gap < burst_threshold)
                
                significance = "high" if bursts > len(time_gaps) * 0.5 else "medium"
                confidence = min(1.0, len(communications) / 20)  # Normalize by expected max
                
                pattern = Pattern(
                    id=f"comm_pattern_{hash(pair)}",
                    pattern_type="communication",
                    description=f"High-frequency communication between {pair[0]} and {pair[1]} ({len(communications)} instances, {bursts} bursts)",
                    entities=list(pair),
                    events=[e.id for e in communications],
                    confidence=confidence,
                    significance=significance
                )
                patterns.append(pattern)
        
        # Detect unusual communication timing
        for pair, communications in entity_pair_communications.items():
            if len(communications) >= 3:
                timestamps = [event.timestamp for event in communications]
                
                # Check for communications at unusual hours (e.g., late night)
                unusual_hours = sum(1 for ts in timestamps if ts.hour < 6 or ts.hour > 22)
                
                if unusual_hours > len(timestamps) * 0.3:  # More than 30% at unusual hours
                    pattern = Pattern(
                        id=f"unusual_timing_{hash(pair)}",
                        pattern_type="communication",
                        description=f"Unusual timing communication pattern between {pair[0]} and {pair[1]} ({unusual_hours}/{len(timestamps)} at unusual hours)",
                        entities=list(pair),
                        events=[e.id for e in communications],
                        confidence=0.7,
                        significance="medium"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def detect_temporal_patterns(
        self, 
        entities: Dict[str, Entity], 
        events: List[Event]
    ) -> List[Pattern]:
        """Detect temporal patterns"""
        patterns = []
        
        if len(events) < 3:
            return patterns
        
        # Analyze activity by time periods
        hourly_activity = defaultdict(int)
        daily_activity = defaultdict(int)
        entity_activity_by_hour = defaultdict(lambda: defaultdict(int))
        
        for event in events:
            hour = event.timestamp.hour
            day = event.timestamp.date()
            
            hourly_activity[hour] += 1
            daily_activity[day] += 1
            
            for entity_id in event.entities_involved:
                entity_activity_by_hour[entity_id][hour] += 1
        
        # Detect unusual activity periods
        avg_hourly_activity = np.mean(list(hourly_activity.values()))
        high_activity_hours = [hour for hour, count in hourly_activity.items() if count > avg_hourly_activity * 2]
        
        if high_activity_hours:
            pattern = Pattern(
                id=f"high_activity_hours_{datetime.now().timestamp()}",
                pattern_type="temporal",
                description=f"High activity during hours: {', '.join(map(str, high_activity_hours))}",
                entities=list(entities.keys()),
                events=[e.id for e in events if e.timestamp.hour in high_activity_hours],
                confidence=0.8,
                significance="medium"
            )
            patterns.append(pattern)
        
        # Detect entity-specific temporal patterns
        for entity_id, hourly_counts in entity_activity_by_hour.items():
            if sum(hourly_counts.values()) >= 5:  # Minimum activity threshold
                # Find peak activity hours for this entity
                max_count = max(hourly_counts.values())
                peak_hours = [hour for hour, count in hourly_counts.items() if count == max_count]
                
                if max_count >= 3 and len(peak_hours) <= 3:  # Concentrated activity
                    pattern = Pattern(
                        id=f"entity_temporal_{entity_id}_{datetime.now().timestamp()}",
                        pattern_type="temporal",
                        description=f"Entity {entity_id} shows concentrated activity during hours: {', '.join(map(str, peak_hours))}",
                        entities=[entity_id],
                        events=[e.id for e in events if entity_id in e.entities_involved and e.timestamp.hour in peak_hours],
                        confidence=0.7,
                        significance="low"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def detect_behavioral_patterns(
        self, 
        entities: Dict[str, Entity], 
        events: List[Event]
    ) -> List[Pattern]:
        """Detect behavioral patterns"""
        patterns = []
        
        # Analyze entity behavior patterns
        entity_event_types = defaultdict(lambda: defaultdict(int))
        entity_event_sequences = defaultdict(list)
        
        for event in events:
            for entity_id in event.entities_involved:
                entity_event_types[entity_id][event.event_type] += 1
                entity_event_sequences[entity_id].append((event.timestamp, event.event_type))
        
        # Detect entities with unusual behavior diversity
        for entity_id, event_types in entity_event_types.items():
            if len(event_types) >= 3:  # Entity involved in multiple types of events
                total_events = sum(event_types.values())
                dominant_event_type = max(event_types, key=event_types.get)
                dominant_ratio = event_types[dominant_event_type] / total_events
                
                if dominant_ratio < 0.5:  # No single dominant behavior
                    pattern = Pattern(
                        id=f"diverse_behavior_{entity_id}",
                        pattern_type="behavioral",
                        description=f"Entity {entity_id} shows diverse behavioral patterns across {len(event_types)} event types",
                        entities=[entity_id],
                        events=[e.id for e in events if entity_id in e.entities_involved],
                        confidence=0.6,
                        significance="medium"
                    )
                    patterns.append(pattern)
        
        # Detect behavioral sequences
        for entity_id, event_sequence in entity_event_sequences.items():
            if len(event_sequence) >= 4:
                # Sort by timestamp
                event_sequence.sort(key=lambda x: x[0])
                
                # Look for repeating sequences
                event_types_only = [event_type for _, event_type in event_sequence]
                
                # Simple sequence pattern detection (consecutive identical events)
                consecutive_patterns = []
                current_pattern = [event_types_only[0]]
                
                for i in range(1, len(event_types_only)):
                    if event_types_only[i] == event_types_only[i-1]:
                        current_pattern.append(event_types_only[i])
                    else:
                        if len(current_pattern) >= 3:  # Sequence of 3+ consecutive same events
                            consecutive_patterns.append(current_pattern.copy())
                        current_pattern = [event_types_only[i]]
                
                if len(current_pattern) >= 3:
                    consecutive_patterns.append(current_pattern)
                
                for pattern_seq in consecutive_patterns:
                    pattern = Pattern(
                        id=f"sequence_{entity_id}_{hash(tuple(pattern_seq))}",
                        pattern_type="behavioral",
                        description=f"Entity {entity_id} shows repetitive sequence: {len(pattern_seq)} consecutive {pattern_seq[0]} events",
                        entities=[entity_id],
                        events=[],  # Would need more complex logic to map back to specific events
                        confidence=0.7,
                        significance="medium"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def detect_financial_patterns(
        self, 
        entities: Dict[str, Entity], 
        events: List[Event]
    ) -> List[Pattern]:
        """Detect financial patterns"""
        patterns = []
        
        # Look for financial-related events
        financial_events = [
            e for e in events 
            if any(keyword in e.event_type.lower() for keyword in ['transaction', 'payment', 'transfer', 'financial'])
        ]
        
        if len(financial_events) < 2:
            return patterns
        
        # Analyze transaction patterns
        entity_transactions = defaultdict(list)
        
        for event in financial_events:
            for entity_id in event.entities_involved:
                # Extract amount if available (simplified)
                amount = self._extract_amount_from_event(event)
                if amount:
                    entity_transactions[entity_id].append((event.timestamp, amount, event))
        
        # Detect suspicious financial patterns
        for entity_id, transactions in entity_transactions.items():
            if len(transactions) >= 3:
                amounts = [amount for _, amount, _ in transactions]
                
                # Round number pattern (many amounts ending in 00)
                round_amounts = sum(1 for amount in amounts if amount % 100 == 0)
                
                if round_amounts > len(amounts) * 0.7:  # More than 70% round amounts
                    pattern = Pattern(
                        id=f"round_amounts_{entity_id}",
                        pattern_type="financial",
                        description=f"Entity {entity_id} shows suspicious round amount pattern ({round_amounts}/{len(amounts)} transactions)",
                        entities=[entity_id],
                        events=[e.id for _, _, e in transactions],
                        confidence=0.8,
                        significance="high"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def detect_geographical_patterns(
        self, 
        entities: Dict[str, Entity], 
        events: List[Event]
    ) -> List[Pattern]:
        """Detect geographical patterns"""
        patterns = []
        
        # Look for location-related entities and events
        location_entities = {
            entity_id: entity for entity_id, entity in entities.items()
            if entity.type.lower() in ['location', 'address', 'place']
        }
        
        if len(location_entities) < 2:
            return patterns
        
        # Analyze location patterns (simplified - would need proper geospatial analysis)
        entity_locations = defaultdict(set)
        
        for event in events:
            for entity_id in event.entities_involved:
                if entity_id in location_entities:
                    for other_entity in event.entities_involved:
                        if other_entity != entity_id and other_entity not in location_entities:
                            entity_locations[other_entity].add(entity_id)
        
        # Detect entities associated with multiple locations
        for entity_id, locations in entity_locations.items():
            if len(locations) >= 3:
                pattern = Pattern(
                    id=f"multi_location_{entity_id}",
                    pattern_type="geographical",
                    description=f"Entity {entity_id} associated with multiple locations: {list(locations)}",
                    entities=[entity_id] + list(locations),
                    events=[e.id for e in events if entity_id in e.entities_involved],
                    confidence=0.6,
                    significance="medium"
                )
                patterns.append(pattern)
        
        return patterns
    
    def detect_anomalies(
        self, 
        entities: Dict[str, Entity], 
        events: List[Event]
    ) -> List[Pattern]:
        """Detect anomalous patterns"""
        patterns = []
        
        if len(events) < 5:
            return patterns
        
        # Analyze event frequency by type
        event_type_counts = Counter(event.event_type for event in events)
        
        # Detect rare event types (potentially suspicious)
        total_events = len(events)
        rare_threshold = max(1, total_events * 0.05)  # Less than 5% of all events
        
        rare_event_types = [
            event_type for event_type, count in event_type_counts.items()
            if count <= rare_threshold and count >= 1
        ]
        
        if rare_event_types:
            rare_events = [e for e in events if e.event_type in rare_event_types]
            
            pattern = Pattern(
                id=f"rare_events_{datetime.now().timestamp()}",
                pattern_type="anomaly",
                description=f"Rare event types detected: {', '.join(rare_event_types)}",
                entities=list(set(entity for event in rare_events for entity in event.entities_involved)),
                events=[e.id for e in rare_events],
                confidence=0.7,
                significance="medium"
            )
            patterns.append(pattern)
        
        # Detect isolated entities (entities with very few connections)
        entity_connection_counts = defaultdict(set)
        
        for event in events:
            entities_in_event = event.entities_involved
            for entity in entities_in_event:
                entity_connection_counts[entity].update(entities_in_event)
                entity_connection_counts[entity].discard(entity)  # Remove self
        
        isolated_entities = [
            entity_id for entity_id, connections in entity_connection_counts.items()
            if len(connections) <= 1 and entity_id in entities
        ]
        
        if isolated_entities:
            pattern = Pattern(
                id=f"isolated_entities_{datetime.now().timestamp()}",
                pattern_type="anomaly",
                description=f"Isolated entities with minimal connections: {isolated_entities}",
                entities=isolated_entities,
                events=[e.id for e in events if any(entity in e.entities_involved for entity in isolated_entities)],
                confidence=0.5,
                significance="low"
            )
            patterns.append(pattern)
        
        return patterns
    
    def _extract_amount_from_event(self, event: Event) -> Optional[float]:
        """Extract monetary amount from event description (simplified)"""
        # Simple regex to find currency amounts
        amount_patterns = [
            r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',  # $1,000.00
            r'(\d+(?:,\d{3})*(?:\.\d{2})?) USD',  # 1000.00 USD
            r'amount:?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'  # amount: 1000.00
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, event.description, re.IGNORECASE)
            if match:
                try:
                    amount_str = match.group(1).replace(',', '')
                    return float(amount_str)
                except (ValueError, IndexError):
                    continue
        
        return None

def pattern_detector(state: ForensicBotState) -> ForensicBotState:
    """
    Detect patterns in forensic evidence and entities
    
    This node analyzes evidence patterns including communication, temporal,
    behavioral, financial, and anomaly patterns.
    
    Args:
        state: Current forensic bot state
        
    Returns:
        ForensicBotState: Updated state with detected patterns
    """
    start_time = datetime.now()
    
    try:
        # Get the user's query to determine pattern focus
        if state["messages"]:
            last_message = state["messages"][-1]
            query = last_message.content if hasattr(last_message, 'content') else str(last_message)
            focus_type = determine_pattern_focus(query)
        else:
            query = "general pattern detection"
            focus_type = None
        
        # Check if we have enough data for pattern detection
        if not state["entity_memory"] and not state["timeline_memory"]:
            response = (
                "I don't have enough processed evidence to detect patterns. "
                "Please ensure evidence has been processed and entities have been extracted first."
            )
            ai_message = AIMessage(content=response)
            state["messages"].append(ai_message)
            return state
        
        # Initialize pattern detector
        if "pattern_detector_instance" not in state["tool_results"]:
            state["tool_results"]["pattern_detector_instance"] = PatternDetector()
        
        detector = state["tool_results"]["pattern_detector_instance"]
        
        # Detect patterns
        detected_patterns = detector.detect_patterns(
            entities=state["entity_memory"],
            events=state["timeline_memory"],
            focus_type=focus_type
        )
        
        # Update state with detected patterns
        for pattern in detected_patterns:
            state["pattern_memory"][pattern.id] = pattern
        
        # Store pattern detection results
        state["analysis_results"]["pattern_detection"] = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "focus_type": focus_type,
            "patterns_detected": len(detected_patterns),
            "pattern_types": list(set(p.pattern_type for p in detected_patterns)),
            "high_significance_patterns": len([p for p in detected_patterns if p.significance in ["high", "critical"]])
        }
        
        # Generate response
        if detected_patterns:
            response = f"## Pattern Detection Results\n\n"
            response += f"Detected {len(detected_patterns)} patterns in the evidence.\n\n"
            
            # Group patterns by significance
            critical_patterns = [p for p in detected_patterns if p.significance == "critical"]
            high_patterns = [p for p in detected_patterns if p.significance == "high"]
            medium_patterns = [p for p in detected_patterns if p.significance == "medium"]
            
            if critical_patterns:
                response += f"**Critical Patterns ({len(critical_patterns)}):**\n"
                for pattern in critical_patterns[:3]:  # Show top 3
                    response += f"• {pattern.description} (Confidence: {pattern.confidence:.2f})\n"
                response += "\n"
            
            if high_patterns:
                response += f"**High Significance Patterns ({len(high_patterns)}):**\n"
                for pattern in high_patterns[:3]:  # Show top 3
                    response += f"• {pattern.description} (Confidence: {pattern.confidence:.2f})\n"
                response += "\n"
            
            if medium_patterns:
                response += f"**Medium Significance Patterns ({len(medium_patterns)}):**\n"
                for pattern in medium_patterns[:2]:  # Show top 2
                    response += f"• {pattern.description} (Confidence: {pattern.confidence:.2f})\n"
                response += "\n"
            
            # Add pattern type summary
            pattern_types = Counter(p.pattern_type for p in detected_patterns)
            response += f"**Pattern Types:** {', '.join(f'{ptype}({count})' for ptype, count in pattern_types.items())}"
            
            # Add to key findings
            if critical_patterns or high_patterns:
                significant_patterns = critical_patterns + high_patterns
                for pattern in significant_patterns[:3]:
                    state["key_findings"].append(pattern.description)
        else:
            response = "No significant patterns detected in the current evidence. This could indicate normal activity or insufficient data for pattern analysis."
        
        ai_message = AIMessage(content=response)
        state["messages"].append(ai_message)
        
        # Update recommendations
        if detected_patterns:
            recommendations = [
                f"Investigate {len([p for p in detected_patterns if p.significance in ['high', 'critical']])} high-priority patterns",
                f"Focus on {Counter(p.pattern_type for p in detected_patterns).most_common(1)[0][0]} patterns which are most prevalent"
            ]
            state["recommendations"].extend(recommendations)
        
        # Add to tools used
        if "pattern_detector" not in state["tools_used"]:
            state["tools_used"].append("pattern_detector")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        add_workflow_step(
            state,
            node_name="pattern_detector",
            action="detect_patterns",
            input_data={
                "query": query,
                "focus_type": focus_type,
                "entities_count": len(state["entity_memory"]),
                "events_count": len(state["timeline_memory"])
            },
            output_data={
                "patterns_detected": len(detected_patterns),
                "high_significance": len([p for p in detected_patterns if p.significance in ["high", "critical"]]),
                "pattern_types": list(set(p.pattern_type for p in detected_patterns))
            },
            execution_time=execution_time,
            success=True
        )
        
    except Exception as e:
        # Handle errors gracefully
        error_msg = f"Pattern detection error: {str(e)}"
        state["last_tool_error"] = error_msg
        
        fallback_response = (
            "I encountered an issue while detecting patterns in the evidence. "
            "This might be due to insufficient processed data or a configuration issue. "
            "Please ensure evidence has been properly analyzed first."
        )
        
        ai_message = AIMessage(content=fallback_response)
        state["messages"].append(ai_message)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        add_workflow_step(
            state,
            node_name="pattern_detector",
            action="detect_patterns",
            input_data={"error": str(e)},
            output_data={"fallback_response": True},
            execution_time=execution_time,
            success=False
        )
    
    return state

def determine_pattern_focus(query: str) -> Optional[str]:
    """Determine the specific pattern type to focus on based on query"""
    query_lower = query.lower()
    
    pattern_keywords = {
        "communication": ["communication", "call", "message", "contact", "phone", "email"],
        "temporal": ["time", "temporal", "timeline", "when", "frequency", "schedule"],
        "behavioral": ["behavior", "behav", "activity", "action", "pattern"],
        "financial": ["financial", "money", "transaction", "payment", "transfer", "bank"],
        "geographical": ["location", "place", "address", "geographic", "spatial", "where"],
        "anomaly": ["anomaly", "unusual", "suspicious", "abnormal", "outlier", "strange"]
    }
    
    for pattern_type, keywords in pattern_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            return pattern_type
    
    return None