"""
Synthesis Engine Node

This module implements the synthesis engine for combining evidence analysis,
correlating findings, and generating comprehensive insights.
"""

import sys
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict, Counter
from langchain_core.messages import AIMessage

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ..state import ForensicBotState, Entity, Event, Pattern, add_workflow_step

class SynthesisEngine:
    """Advanced evidence synthesis and correlation engine"""
    
    def __init__(self):
        self.correlation_methods = {
            "entity_correlation": self.correlate_entities,
            "temporal_correlation": self.correlate_temporal_evidence,
            "pattern_correlation": self.correlate_patterns,
            "cross_evidence_correlation": self.correlate_cross_evidence
        }
    
    def synthesize_investigation(
        self, 
        entities: Dict[str, Entity],
        events: List[Event],
        patterns: Dict[str, Pattern],
        evidence: List[Any],
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive synthesis of investigation data
        
        Args:
            entities: Dictionary of extracted entities
            events: List of temporal events
            patterns: Dictionary of detected patterns
            evidence: List of evidence objects
            analysis_results: Previous analysis results
            
        Returns:
            Dict containing synthesis results
        """
        synthesis_results = {
            "timestamp": datetime.now().isoformat(),
            "entity_analysis": self.analyze_entities(entities),
            "temporal_analysis": self.analyze_timeline(events),
            "pattern_analysis": self.analyze_patterns(patterns),
            "correlations": {},
            "key_insights": [],
            "confidence_assessment": {},
            "investigation_gaps": [],
            "recommendations": [],
            "summary": ""
        }
        
        # Perform correlations
        for correlation_type, correlation_method in self.correlation_methods.items():
            try:
                correlation_result = correlation_method(entities, events, patterns, evidence)
                synthesis_results["correlations"][correlation_type] = correlation_result
            except Exception as e:
                synthesis_results["correlations"][correlation_type] = {
                    "error": str(e),
                    "success": False
                }
        
        # Generate key insights
        synthesis_results["key_insights"] = self.generate_key_insights(
            entities, events, patterns, synthesis_results["correlations"]
        )
        
        # Assess confidence
        synthesis_results["confidence_assessment"] = self.assess_confidence(
            entities, events, patterns, analysis_results
        )
        
        # Identify investigation gaps
        synthesis_results["investigation_gaps"] = self.identify_gaps(
            entities, events, patterns, analysis_results
        )
        
        # Generate recommendations
        synthesis_results["recommendations"] = self.generate_recommendations(
            synthesis_results
        )
        
        # Create summary
        synthesis_results["summary"] = self.create_investigation_summary(
            synthesis_results
        )
        
        return synthesis_results
    
    def analyze_entities(self, entities: Dict[str, Entity]) -> Dict[str, Any]:
        """Analyze entity characteristics and relationships"""
        if not entities:
            return {"total": 0, "types": {}, "confidence_distribution": {}}
        
        entity_types = Counter(entity.type for entity in entities.values())
        confidence_scores = [entity.confidence for entity in entities.values()]
        
        # Categorize by confidence
        high_confidence = sum(1 for score in confidence_scores if score >= 0.8)
        medium_confidence = sum(1 for score in confidence_scores if 0.5 <= score < 0.8)
        low_confidence = sum(1 for score in confidence_scores if score < 0.5)
        
        # Find most connected entities (simplified - would use actual graph analysis)
        entity_connections = defaultdict(int)
        for entity in entities.values():
            # Count attributes as a proxy for connectivity
            entity_connections[entity.id] = len(entity.attributes)
        
        most_connected = sorted(
            entity_connections.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "total": len(entities),
            "types": dict(entity_types),
            "confidence_distribution": {
                "high": high_confidence,
                "medium": medium_confidence,
                "low": low_confidence
            },
            "most_connected": most_connected,
            "avg_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        }
    
    def analyze_timeline(self, events: List[Event]) -> Dict[str, Any]:
        """Analyze temporal characteristics of events"""
        if not events:
            return {"total": 0, "timespan": None, "event_types": {}}
        
        timestamps = [event.timestamp for event in events]
        timestamps.sort()
        
        event_types = Counter(event.event_type for event in events)
        
        # Calculate activity intensity over time
        timespan = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else None
        
        # Group events by day to find peak activity
        daily_activity = defaultdict(int)
        for event in events:
            day = event.timestamp.date()
            daily_activity[day] += 1
        
        peak_day = max(daily_activity.items(), key=lambda x: x[1]) if daily_activity else None
        
        return {
            "total": len(events),
            "timespan": {
                "start": timestamps[0].isoformat(),
                "end": timestamps[-1].isoformat(),
                "duration_days": timespan.days if timespan else 0
            },
            "event_types": dict(event_types),
            "peak_activity_day": {
                "date": peak_day[0].isoformat() if peak_day else None,
                "event_count": peak_day[1] if peak_day else 0
            },
            "avg_events_per_day": len(events) / max(1, timespan.days) if timespan else len(events)
        }
    
    def analyze_patterns(self, patterns: Dict[str, Pattern]) -> Dict[str, Any]:
        """Analyze detected patterns"""
        if not patterns:
            return {"total": 0, "types": {}, "significance_distribution": {}}
        
        pattern_types = Counter(pattern.pattern_type for pattern in patterns.values())
        significance_levels = Counter(pattern.significance for pattern in patterns.values())
        
        # Find highest confidence patterns
        high_confidence_patterns = [
            (pattern.id, pattern.confidence, pattern.description)
            for pattern in patterns.values()
            if pattern.confidence >= 0.7
        ]
        high_confidence_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "total": len(patterns),
            "types": dict(pattern_types),
            "significance_distribution": dict(significance_levels),
            "high_confidence_patterns": high_confidence_patterns[:5],
            "avg_confidence": sum(p.confidence for p in patterns.values()) / len(patterns)
        }
    
    def correlate_entities(
        self, 
        entities: Dict[str, Entity], 
        events: List[Event], 
        patterns: Dict[str, Pattern],
        evidence: List[Any]
    ) -> Dict[str, Any]:
        """Correlate entities across different evidence sources"""
        correlations = {
            "entity_co_occurrences": defaultdict(int),
            "entity_event_associations": defaultdict(list),
            "cross_evidence_entities": defaultdict(set)
        }
        
        # Analyze entity co-occurrences in events
        for event in events:
            entities_in_event = event.entities_involved
            for i, entity1 in enumerate(entities_in_event):
                for entity2 in entities_in_event[i+1:]:
                    pair = tuple(sorted([entity1, entity2]))
                    correlations["entity_co_occurrences"][pair] += 1
        
        # Associate entities with event types
        for event in events:
            for entity_id in event.entities_involved:
                correlations["entity_event_associations"][entity_id].append(event.event_type)
        
        # Find entities that appear across multiple evidence sources
        for evidence_item in evidence:
            evidence_source = getattr(evidence_item, 'file_path', 'unknown')
            # This would need to be implemented based on actual evidence structure
            # For now, use a simplified approach
            for entity_id in entities.keys():
                if entity_id in str(evidence_item):  # Simplified check
                    correlations["cross_evidence_entities"][entity_id].add(evidence_source)
        
        # Find most correlated entity pairs
        top_correlations = sorted(
            correlations["entity_co_occurrences"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "top_entity_correlations": top_correlations,
            "multi_source_entities": {
                entity_id: len(sources) 
                for entity_id, sources in correlations["cross_evidence_entities"].items()
                if len(sources) > 1
            },
            "entity_event_diversity": {
                entity_id: len(set(event_types))
                for entity_id, event_types in correlations["entity_event_associations"].items()
            }
        }
    
    def correlate_temporal_evidence(
        self, 
        entities: Dict[str, Entity], 
        events: List[Event], 
        patterns: Dict[str, Pattern],
        evidence: List[Any]
    ) -> Dict[str, Any]:
        """Correlate evidence based on temporal relationships"""
        if not events:
            return {"error": "No events available for temporal correlation"}
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Find temporal clusters (events happening close in time)
        temporal_clusters = []
        current_cluster = [sorted_events[0]]
        
        for i in range(1, len(sorted_events)):
            time_diff = (sorted_events[i].timestamp - sorted_events[i-1].timestamp).total_seconds() / 3600  # hours
            
            if time_diff <= 6:  # Events within 6 hours are considered clustered
                current_cluster.append(sorted_events[i])
            else:
                if len(current_cluster) > 1:
                    temporal_clusters.append(current_cluster)
                current_cluster = [sorted_events[i]]
        
        if len(current_cluster) > 1:
            temporal_clusters.append(current_cluster)
        
        # Analyze temporal patterns
        sequential_patterns = []
        for i in range(len(sorted_events) - 1):
            event1 = sorted_events[i]
            event2 = sorted_events[i + 1]
            
            # Check for entities involved in sequential events
            common_entities = set(event1.entities_involved) & set(event2.entities_involved)
            if common_entities:
                time_gap = (event2.timestamp - event1.timestamp).total_seconds() / 60  # minutes
                sequential_patterns.append({
                    "event1": event1.id,
                    "event2": event2.id,
                    "common_entities": list(common_entities),
                    "time_gap_minutes": time_gap
                })
        
        return {
            "temporal_clusters": [
                {
                    "cluster_id": i,
                    "event_count": len(cluster),
                    "timespan": {
                        "start": cluster[0].timestamp.isoformat(),
                        "end": cluster[-1].timestamp.isoformat()
                    },
                    "entities_involved": list(set(entity for event in cluster for entity in event.entities_involved))
                }
                for i, cluster in enumerate(temporal_clusters)
            ],
            "sequential_patterns": sequential_patterns[:10],  # Top 10 sequential patterns
            "total_clusters": len(temporal_clusters)
        }
    
    def correlate_patterns(
        self, 
        entities: Dict[str, Entity], 
        events: List[Event], 
        patterns: Dict[str, Pattern],
        evidence: List[Any]
    ) -> Dict[str, Any]:
        """Correlate detected patterns to find meta-patterns"""
        if not patterns:
            return {"error": "No patterns available for correlation"}
        
        # Group patterns by entities involved
        entity_pattern_map = defaultdict(list)
        for pattern in patterns.values():
            for entity_id in pattern.entities:
                entity_pattern_map[entity_id].append(pattern)
        
        # Find entities involved in multiple patterns
        multi_pattern_entities = {
            entity_id: len(entity_patterns)
            for entity_id, entity_patterns in entity_pattern_map.items()
            if len(entity_patterns) > 1
        }
        
        # Find pattern type correlations
        pattern_type_correlations = defaultdict(list)
        for entity_id, entity_patterns in entity_pattern_map.items():
            if len(entity_patterns) > 1:
                pattern_types = [p.pattern_type for p in entity_patterns]
                for i, ptype1 in enumerate(pattern_types):
                    for ptype2 in pattern_types[i+1:]:
                        pair = tuple(sorted([ptype1, ptype2]))
                        pattern_type_correlations[pair].append(entity_id)
        
        return {
            "multi_pattern_entities": multi_pattern_entities,
            "pattern_type_correlations": dict(pattern_type_correlations),
            "pattern_overlap_analysis": {
                pair: len(entities)
                for pair, entities in pattern_type_correlations.items()
            }
        }
    
    def correlate_cross_evidence(
        self, 
        entities: Dict[str, Entity], 
        events: List[Event], 
        patterns: Dict[str, Pattern],
        evidence: List[Any]
    ) -> Dict[str, Any]:
        """Correlate findings across different types of evidence"""
        # This is a simplified implementation
        # In a real system, this would perform sophisticated cross-evidence analysis
        
        evidence_types = defaultdict(list)
        
        for evidence_item in evidence:
            evidence_type = getattr(evidence_item, 'evidence_type', 'unknown')
            evidence_types[evidence_type].append(evidence_item)
        
        cross_evidence_correlations = {}
        
        # Find entities that appear in multiple evidence types
        entity_evidence_types = defaultdict(set)
        for entity_id, entity in entities.items():
            case_id = entity.case_id
            # This would need more sophisticated mapping in a real implementation
            for evidence_type in evidence_types.keys():
                # Simplified check - in reality, would check actual evidence content
                entity_evidence_types[entity_id].add(evidence_type)
        
        cross_evidence_entities = {
            entity_id: list(types)
            for entity_id, types in entity_evidence_types.items()
            if len(types) > 1
        }
        
        return {
            "evidence_types_available": list(evidence_types.keys()),
            "cross_evidence_entities": cross_evidence_entities,
            "evidence_type_count": {etype: len(items) for etype, items in evidence_types.items()}
        }
    
    def generate_key_insights(
        self, 
        entities: Dict[str, Entity], 
        events: List[Event], 
        patterns: Dict[str, Pattern],
        correlations: Dict[str, Any]
    ) -> List[str]:
        """Generate key insights from the analysis"""
        insights = []
        
        # Entity insights
        if entities:
            entity_analysis = self.analyze_entities(entities)
            most_common_entity_type = max(entity_analysis["types"].items(), key=lambda x: x[1])[0] if entity_analysis["types"] else None
            if most_common_entity_type:
                insights.append(f"Investigation primarily involves {most_common_entity_type} entities ({entity_analysis['types'][most_common_entity_type]} instances)")
        
        # Pattern insights
        if patterns:
            pattern_analysis = self.analyze_patterns(patterns)
            high_confidence_patterns = pattern_analysis.get("high_confidence_patterns", [])
            if high_confidence_patterns:
                insights.append(f"High-confidence pattern detected: {high_confidence_patterns[0][2]}")
        
        # Temporal insights
        if events:
            timeline_analysis = self.analyze_timeline(events)
            if timeline_analysis["peak_activity_day"]["event_count"] > 3:
                insights.append(f"Peak activity occurred on {timeline_analysis['peak_activity_day']['date']} with {timeline_analysis['peak_activity_day']['event_count']} events")
        
        # Correlation insights
        entity_correlations = correlations.get("entity_correlation", {})
        top_correlations = entity_correlations.get("top_entity_correlations", [])
        if top_correlations and top_correlations[0][1] > 3:
            pair, count = top_correlations[0]
            insights.append(f"Strong correlation found between entities {pair[0]} and {pair[1]} (co-occurred {count} times)")
        
        return insights
    
    def assess_confidence(
        self, 
        entities: Dict[str, Entity], 
        events: List[Event], 
        patterns: Dict[str, Pattern],
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess overall confidence in the investigation findings"""
        confidence_factors = {
            "entity_confidence": 0.0,
            "pattern_confidence": 0.0,
            "temporal_confidence": 0.0,
            "evidence_completeness": 0.0
        }
        
        # Entity confidence
        if entities:
            entity_confidences = [entity.confidence for entity in entities.values()]
            confidence_factors["entity_confidence"] = sum(entity_confidences) / len(entity_confidences)
        
        # Pattern confidence
        if patterns:
            pattern_confidences = [pattern.confidence for pattern in patterns.values()]
            confidence_factors["pattern_confidence"] = sum(pattern_confidences) / len(pattern_confidences)
        
        # Temporal confidence (based on event coverage)
        if events:
            # Simple metric: more events = higher temporal confidence
            confidence_factors["temporal_confidence"] = min(1.0, len(events) / 10)  # Normalize to 10 events
        
        # Evidence completeness (simplified)
        evidence_types_needed = ["communication", "financial", "location", "document"]
        evidence_types_found = set()
        
        for entity in entities.values():
            if any(etype in entity.type.lower() for etype in evidence_types_needed):
                evidence_types_found.add(entity.type.lower())
        
        confidence_factors["evidence_completeness"] = len(evidence_types_found) / len(evidence_types_needed)
        
        # Overall confidence (weighted average)
        overall_confidence = (
            confidence_factors["entity_confidence"] * 0.3 +
            confidence_factors["pattern_confidence"] * 0.3 +
            confidence_factors["temporal_confidence"] * 0.2 +
            confidence_factors["evidence_completeness"] * 0.2
        )
        
        return {
            "overall_confidence": overall_confidence,
            "confidence_factors": confidence_factors,
            "confidence_level": self._categorize_confidence(overall_confidence)
        }
    
    def identify_gaps(
        self, 
        entities: Dict[str, Entity], 
        events: List[Event], 
        patterns: Dict[str, Pattern],
        analysis_results: Dict[str, Any]
    ) -> List[str]:
        """Identify gaps in the investigation"""
        gaps = []
        
        # Check for entity type gaps
        entity_types = set(entity.type for entity in entities.values()) if entities else set()
        expected_types = {"person", "phone", "email", "location", "organization"}
        missing_types = expected_types - entity_types
        
        if missing_types:
            gaps.append(f"Missing entity types: {', '.join(missing_types)}")
        
        # Check for temporal gaps
        if events:
            sorted_events = sorted(events, key=lambda e: e.timestamp)
            for i in range(len(sorted_events) - 1):
                time_gap = (sorted_events[i+1].timestamp - sorted_events[i].timestamp).total_seconds() / 3600  # hours
                if time_gap > 72:  # Gap longer than 3 days
                    gaps.append(f"Significant temporal gap detected: {time_gap:.1f} hours between events")
                    break  # Only report the first major gap
        
        # Check for pattern coverage
        pattern_types = set(pattern.pattern_type for pattern in patterns.values()) if patterns else set()
        expected_pattern_types = {"communication", "temporal", "behavioral"}
        missing_pattern_types = expected_pattern_types - pattern_types
        
        if missing_pattern_types:
            gaps.append(f"Missing pattern analysis: {', '.join(missing_pattern_types)}")
        
        # Check for low confidence areas
        if entities:
            low_confidence_entities = [
                entity.id for entity in entities.values()
                if entity.confidence < 0.5
            ]
            if len(low_confidence_entities) > len(entities) * 0.3:
                gaps.append(f"High number of low-confidence entities ({len(low_confidence_entities)}) may indicate data quality issues")
        
        return gaps
    
    def generate_recommendations(self, synthesis_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on synthesis"""
        recommendations = []
        
        # Confidence-based recommendations
        confidence_assessment = synthesis_results.get("confidence_assessment", {})
        overall_confidence = confidence_assessment.get("overall_confidence", 0)
        
        if overall_confidence < 0.5:
            recommendations.append("Consider gathering additional evidence to improve analysis confidence")
        
        # Gap-based recommendations
        gaps = synthesis_results.get("investigation_gaps", [])
        if gaps:
            recommendations.append(f"Address investigation gaps: {gaps[0]}")  # Address the first gap
        
        # Pattern-based recommendations
        pattern_analysis = synthesis_results.get("pattern_analysis", {})
        high_confidence_patterns = pattern_analysis.get("high_confidence_patterns", [])
        
        if high_confidence_patterns:
            recommendations.append(f"Investigate high-confidence pattern: {high_confidence_patterns[0][2]}")
        
        # Correlation-based recommendations
        correlations = synthesis_results.get("correlations", {})
        entity_correlations = correlations.get("entity_correlation", {})
        top_correlations = entity_correlations.get("top_entity_correlations", [])
        
        if top_correlations and top_correlations[0][1] > 5:
            pair, count = top_correlations[0]
            recommendations.append(f"Focus investigation on highly correlated entities: {pair[0]} and {pair[1]}")
        
        return recommendations
    
    def create_investigation_summary(self, synthesis_results: Dict[str, Any]) -> str:
        """Create a comprehensive investigation summary"""
        summary_parts = []
        
        # Overview
        entity_analysis = synthesis_results.get("entity_analysis", {})
        temporal_analysis = synthesis_results.get("temporal_analysis", {})
        pattern_analysis = synthesis_results.get("pattern_analysis", {})
        
        summary_parts.append(f"Investigation involves {entity_analysis.get('total', 0)} entities across {temporal_analysis.get('total', 0)} events.")
        
        if temporal_analysis.get("timespan"):
            timespan = temporal_analysis["timespan"]
            summary_parts.append(f"Timeline spans {timespan['duration_days']} days from {timespan['start'][:10]} to {timespan['end'][:10]}.")
        
        # Key findings
        key_insights = synthesis_results.get("key_insights", [])
        if key_insights:
            summary_parts.append(f"Key insight: {key_insights[0]}")
        
        # Confidence assessment
        confidence_assessment = synthesis_results.get("confidence_assessment", {})
        confidence_level = confidence_assessment.get("confidence_level", "unknown")
        summary_parts.append(f"Overall analysis confidence: {confidence_level}")
        
        return " ".join(summary_parts)
    
    def _categorize_confidence(self, confidence_score: float) -> str:
        """Categorize confidence score into levels"""
        if confidence_score >= 0.8:
            return "High"
        elif confidence_score >= 0.6:
            return "Medium"
        elif confidence_score >= 0.4:
            return "Low"
        else:
            return "Very Low"

def synthesis_engine(state: ForensicBotState) -> ForensicBotState:
    """
    Synthesize and correlate evidence findings
    
    This node performs comprehensive analysis of all collected evidence,
    entities, patterns, and events to generate unified insights.
    
    Args:
        state: Current forensic bot state
        
    Returns:
        ForensicBotState: Updated state with synthesis results
    """
    start_time = datetime.now()
    
    try:
        # Check if we have enough data for synthesis
        if not state["entity_memory"] and not state["timeline_memory"] and not state["pattern_memory"]:
            response = (
                "I don't have enough analyzed evidence to perform synthesis. "
                "Please ensure evidence has been processed, entities extracted, and patterns detected first."
            )
            ai_message = AIMessage(content=response)
            state["messages"].append(ai_message)
            return state
        
        # Initialize synthesis engine
        if "synthesis_engine_instance" not in state["tool_results"]:
            state["tool_results"]["synthesis_engine_instance"] = SynthesisEngine()
        
        engine = state["tool_results"]["synthesis_engine_instance"]
        
        # Perform synthesis
        synthesis_results = engine.synthesize_investigation(
            entities=state["entity_memory"],
            events=state["timeline_memory"],
            patterns=state["pattern_memory"],
            evidence=state["active_evidence"],
            analysis_results=state["analysis_results"]
        )
        
        # Update state with synthesis results
        state["analysis_results"]["synthesis"] = synthesis_results
        
        # Update key findings
        new_insights = synthesis_results.get("key_insights", [])
        for insight in new_insights:
            if insight not in state["key_findings"]:
                state["key_findings"].append(insight)
        
        # Update recommendations
        new_recommendations = synthesis_results.get("recommendations", [])
        for recommendation in new_recommendations:
            if recommendation not in state["recommendations"]:
                state["recommendations"].append(recommendation)
        
        # Update confidence scores
        confidence_assessment = synthesis_results.get("confidence_assessment", {})
        state["confidence_scores"]["overall"] = confidence_assessment.get("overall_confidence", 0)
        state["confidence_scores"]["entity_analysis"] = confidence_assessment.get("confidence_factors", {}).get("entity_confidence", 0)
        state["confidence_scores"]["pattern_analysis"] = confidence_assessment.get("confidence_factors", {}).get("pattern_confidence", 0)
        
        # Generate comprehensive response
        response = generate_synthesis_response(synthesis_results)
        ai_message = AIMessage(content=response)
        state["messages"].append(ai_message)
        
        # Add to tools used
        if "synthesis_engine" not in state["tools_used"]:
            state["tools_used"].append("synthesis_engine")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        add_workflow_step(
            state,
            node_name="synthesis_engine",
            action="synthesize_investigation",
            input_data={
                "entities_count": len(state["entity_memory"]),
                "events_count": len(state["timeline_memory"]),
                "patterns_count": len(state["pattern_memory"]),
                "evidence_count": len(state["active_evidence"])
            },
            output_data={
                "insights_generated": len(synthesis_results.get("key_insights", [])),
                "correlations_found": len(synthesis_results.get("correlations", {})),
                "overall_confidence": synthesis_results.get("confidence_assessment", {}).get("overall_confidence", 0),
                "gaps_identified": len(synthesis_results.get("investigation_gaps", []))
            },
            execution_time=execution_time,
            success=True
        )
        
    except Exception as e:
        # Handle errors gracefully
        error_msg = f"Synthesis engine error: {str(e)}"
        state["last_tool_error"] = error_msg
        
        fallback_response = (
            "I encountered an issue while synthesizing the investigation findings. "
            "This might be due to incomplete analysis or data inconsistencies. "
            "Please ensure all previous analysis steps have completed successfully."
        )
        
        ai_message = AIMessage(content=fallback_response)
        state["messages"].append(ai_message)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        add_workflow_step(
            state,
            node_name="synthesis_engine",
            action="synthesize_investigation",
            input_data={"error": str(e)},
            output_data={"fallback_response": True},
            execution_time=execution_time,
            success=False
        )
    
    return state

def generate_synthesis_response(synthesis_results: Dict[str, Any]) -> str:
    """Generate a comprehensive response from synthesis results"""
    response = "## Investigation Synthesis Report\n\n"
    
    # Summary
    summary = synthesis_results.get("summary", "")
    if summary:
        response += f"**Executive Summary:** {summary}\n\n"
    
    # Key insights
    insights = synthesis_results.get("key_insights", [])
    if insights:
        response += "**Key Insights:**\n"
        for i, insight in enumerate(insights[:5], 1):  # Top 5 insights
            response += f"{i}. {insight}\n"
        response += "\n"
    
    # Confidence assessment
    confidence_assessment = synthesis_results.get("confidence_assessment", {})
    if confidence_assessment:
        confidence_level = confidence_assessment.get("confidence_level", "Unknown")
        overall_confidence = confidence_assessment.get("overall_confidence", 0)
        response += f"**Analysis Confidence:** {confidence_level} ({overall_confidence:.1%})\n\n"
    
    # Entity analysis
    entity_analysis = synthesis_results.get("entity_analysis", {})
    if entity_analysis.get("total", 0) > 0:
        response += f"**Entity Analysis:** {entity_analysis['total']} entities identified across {len(entity_analysis.get('types', {}))} types. "
        if entity_analysis.get("most_connected"):
            most_connected = entity_analysis["most_connected"][0]
            response += f"Most connected entity: {most_connected[0]}.\n\n"
    
    # Pattern analysis
    pattern_analysis = synthesis_results.get("pattern_analysis", {})
    if pattern_analysis.get("total", 0) > 0:
        response += f"**Pattern Analysis:** {pattern_analysis['total']} patterns detected. "
        high_confidence = pattern_analysis.get("high_confidence_patterns", [])
        if high_confidence:
            response += f"Highest confidence pattern: {high_confidence[0][2]} ({high_confidence[0][1]:.2f})\n\n"
    
    # Correlations
    correlations = synthesis_results.get("correlations", {})
    entity_correlations = correlations.get("entity_correlation", {})
    top_correlations = entity_correlations.get("top_entity_correlations", [])
    if top_correlations:
        top_pair, count = top_correlations[0]
        response += f"**Key Correlations:** Strongest entity correlation between {top_pair[0]} and {top_pair[1]} ({count} co-occurrences)\n\n"
    
    # Investigation gaps
    gaps = synthesis_results.get("investigation_gaps", [])
    if gaps:
        response += "**Investigation Gaps:**\n"
        for gap in gaps[:3]:  # Top 3 gaps
            response += f"• {gap}\n"
        response += "\n"
    
    # Recommendations
    recommendations = synthesis_results.get("recommendations", [])
    if recommendations:
        response += "**Recommendations:**\n"
        for i, recommendation in enumerate(recommendations[:3], 1):  # Top 3 recommendations
            response += f"{i}. {recommendation}\n"
    
    return response