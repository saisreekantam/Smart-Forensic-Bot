"""
Report Generator Node

This module implements comprehensive report generation for forensic investigations,
creating structured reports with findings, evidence, and recommendations.
"""

import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from langchain_core.messages import AIMessage

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ..state import ForensicBotState, Entity, Event, Pattern, add_workflow_step

class ReportGenerator:
    """Advanced forensic report generation system"""
    
    def __init__(self):
        self.report_templates = {
            "executive_summary": self.generate_executive_summary,
            "detailed_analysis": self.generate_detailed_analysis,
            "evidence_inventory": self.generate_evidence_inventory,
            "timeline_report": self.generate_timeline_report,
            "entity_analysis": self.generate_entity_analysis,
            "pattern_report": self.generate_pattern_report,
            "recommendations": self.generate_recommendations_report
        }
    
    def generate_comprehensive_report(
        self,
        case_id: str,
        entities: Dict[str, Entity],
        events: List[Event],
        patterns: Dict[str, Pattern],
        evidence: List[Any],
        analysis_results: Dict[str, Any],
        synthesis_results: Dict[str, Any],
        key_findings: List[str],
        recommendations: List[str],
        report_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive forensic investigation report
        
        Args:
            case_id: Case identifier
            entities: Dictionary of entities
            events: List of events
            patterns: Dictionary of patterns
            evidence: List of evidence
            analysis_results: Analysis results
            synthesis_results: Synthesis results
            key_findings: List of key findings
            recommendations: List of recommendations
            report_type: Type of report to generate
            
        Returns:
            Dict containing the complete report
        """
        report = {
            "metadata": {
                "case_id": case_id,
                "generated_at": datetime.now().isoformat(),
                "report_type": report_type,
                "generator": "Forensic AI Assistant",
                "version": "1.0"
            },
            "sections": {}
        }
        
        # Generate different sections based on report type
        if report_type == "comprehensive":
            sections_to_generate = list(self.report_templates.keys())
        elif report_type == "summary":
            sections_to_generate = ["executive_summary", "key_findings", "recommendations"]
        elif report_type == "technical":
            sections_to_generate = ["detailed_analysis", "evidence_inventory", "pattern_report"]
        else:
            sections_to_generate = ["executive_summary"]
        
        # Generate each section
        for section_name in sections_to_generate:
            if section_name in self.report_templates:
                try:
                    section_content = self.report_templates[section_name](
                        case_id, entities, events, patterns, evidence,
                        analysis_results, synthesis_results, key_findings, recommendations
                    )
                    report["sections"][section_name] = section_content
                except Exception as e:
                    report["sections"][section_name] = {
                        "error": f"Failed to generate section: {str(e)}",
                        "content": "Section could not be generated due to an error."
                    }
        
        # Add summary statistics
        report["statistics"] = self.generate_report_statistics(
            entities, events, patterns, evidence, analysis_results
        )
        
        return report
    
    def generate_executive_summary(
        self, case_id: str, entities: Dict[str, Entity], events: List[Event],
        patterns: Dict[str, Pattern], evidence: List[Any], analysis_results: Dict[str, Any],
        synthesis_results: Dict[str, Any], key_findings: List[str], recommendations: List[str]
    ) -> Dict[str, Any]:
        """Generate executive summary section"""
        
        # Get confidence assessment
        confidence_assessment = synthesis_results.get("confidence_assessment", {})
        overall_confidence = confidence_assessment.get("overall_confidence", 0)
        confidence_level = confidence_assessment.get("confidence_level", "Unknown")
        
        # Get timeline information
        timeline_info = ""
        if events:
            sorted_events = sorted(events, key=lambda e: e.timestamp)
            start_date = sorted_events[0].timestamp.strftime("%Y-%m-%d")
            end_date = sorted_events[-1].timestamp.strftime("%Y-%m-%d")
            timeline_info = f"spanning from {start_date} to {end_date}"
        
        # Get top entity types
        entity_types = {}
        for entity in entities.values():
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
        
        top_entity_type = max(entity_types.items(), key=lambda x: x[1])[0] if entity_types else "unknown"
        
        summary_text = f"""
        Case {case_id} investigation analysis has been completed with {confidence_level.lower()} confidence ({overall_confidence:.1%}).
        
        The investigation processed {len(evidence)} pieces of evidence, identifying {len(entities)} entities 
        and {len(events)} events {timeline_info}. Analysis revealed {len(patterns)} distinct patterns 
        with primary focus on {top_entity_type} entities.
        
        {len(key_findings)} key findings have been identified, leading to {len(recommendations)} 
        actionable recommendations for further investigation.
        """.strip()
        
        return {
            "title": "Executive Summary",
            "content": summary_text,
            "key_metrics": {
                "confidence_level": confidence_level,
                "confidence_score": overall_confidence,
                "entities_identified": len(entities),
                "events_analyzed": len(events),
                "patterns_detected": len(patterns),
                "evidence_processed": len(evidence)
            }
        }
    
    def generate_detailed_analysis(
        self, case_id: str, entities: Dict[str, Entity], events: List[Event],
        patterns: Dict[str, Pattern], evidence: List[Any], analysis_results: Dict[str, Any],
        synthesis_results: Dict[str, Any], key_findings: List[str], recommendations: List[str]
    ) -> Dict[str, Any]:
        """Generate detailed analysis section"""
        
        analysis_content = []
        
        # Entity analysis
        if entities:
            entity_analysis = synthesis_results.get("entity_analysis", {})
            analysis_content.append({
                "subsection": "Entity Analysis",
                "content": f"""
                Identified {len(entities)} entities across {len(entity_analysis.get('types', {}))} different types.
                Average entity confidence: {entity_analysis.get('avg_confidence', 0):.2f}
                
                Entity type distribution:
                {self._format_dict_as_list(entity_analysis.get('types', {}))}
                
                Confidence distribution:
                - High confidence (≥0.8): {entity_analysis.get('confidence_distribution', {}).get('high', 0)} entities
                - Medium confidence (0.5-0.8): {entity_analysis.get('confidence_distribution', {}).get('medium', 0)} entities
                - Low confidence (<0.5): {entity_analysis.get('confidence_distribution', {}).get('low', 0)} entities
                """.strip()
            })
        
        # Temporal analysis
        if events:
            temporal_analysis = synthesis_results.get("temporal_analysis", {})
            analysis_content.append({
                "subsection": "Temporal Analysis",
                "content": f"""
                Analyzed {len(events)} events over {temporal_analysis.get('timespan', {}).get('duration_days', 0)} days.
                Average events per day: {temporal_analysis.get('avg_events_per_day', 0):.1f}
                
                Peak activity: {temporal_analysis.get('peak_activity_day', {}).get('event_count', 0)} events 
                on {temporal_analysis.get('peak_activity_day', {}).get('date', 'unknown')}
                
                Event type distribution:
                {self._format_dict_as_list(temporal_analysis.get('event_types', {}))}
                """.strip()
            })
        
        # Pattern analysis
        if patterns:
            pattern_analysis = synthesis_results.get("pattern_analysis", {})
            high_confidence_patterns = pattern_analysis.get("high_confidence_patterns", [])
            
            pattern_content = f"""
            Detected {len(patterns)} patterns with average confidence {pattern_analysis.get('avg_confidence', 0):.2f}.
            
            Pattern type distribution:
            {self._format_dict_as_list(pattern_analysis.get('types', {}))}
            
            Significance distribution:
            {self._format_dict_as_list(pattern_analysis.get('significance_distribution', {}))}
            """
            
            if high_confidence_patterns:
                pattern_content += f"\n\nTop high-confidence patterns:\n"
                for i, (pattern_id, confidence, description) in enumerate(high_confidence_patterns[:3], 1):
                    pattern_content += f"{i}. {description} (Confidence: {confidence:.2f})\n"
            
            analysis_content.append({
                "subsection": "Pattern Analysis",
                "content": pattern_content.strip()
            })
        
        # Correlation analysis
        correlations = synthesis_results.get("correlations", {})
        if correlations:
            correlation_content = "Cross-evidence correlation analysis:\n\n"
            
            # Entity correlations
            entity_correlations = correlations.get("entity_correlation", {})
            top_correlations = entity_correlations.get("top_entity_correlations", [])
            if top_correlations:
                correlation_content += "Top entity correlations:\n"
                for i, (pair, count) in enumerate(top_correlations[:5], 1):
                    correlation_content += f"{i}. {pair[0]} ↔ {pair[1]} ({count} co-occurrences)\n"
            
            # Temporal correlations
            temporal_correlations = correlations.get("temporal_correlation", {})
            if temporal_correlations.get("total_clusters", 0) > 0:
                correlation_content += f"\nTemporal clustering: {temporal_correlations['total_clusters']} event clusters identified\n"
            
            analysis_content.append({
                "subsection": "Correlation Analysis",
                "content": correlation_content.strip()
            })
        
        return {
            "title": "Detailed Analysis",
            "content": analysis_content
        }
    
    def generate_evidence_inventory(
        self, case_id: str, entities: Dict[str, Entity], events: List[Event],
        patterns: Dict[str, Pattern], evidence: List[Any], analysis_results: Dict[str, Any],
        synthesis_results: Dict[str, Any], key_findings: List[str], recommendations: List[str]
    ) -> Dict[str, Any]:
        """Generate evidence inventory section"""
        
        evidence_items = []
        evidence_types = {}
        
        for i, evidence_item in enumerate(evidence, 1):
            evidence_type = getattr(evidence_item, 'evidence_type', 'unknown')
            file_path = getattr(evidence_item, 'file_path', 'unknown')
            processed = getattr(evidence_item, 'processed', False)
            
            evidence_types[evidence_type] = evidence_types.get(evidence_type, 0) + 1
            
            evidence_items.append({
                "item_number": i,
                "file_path": file_path,
                "evidence_type": evidence_type,
                "processed": processed,
                "metadata": getattr(evidence_item, 'metadata', {})
            })
        
        inventory_summary = f"""
        Total evidence items: {len(evidence)}
        
        Evidence type breakdown:
        {self._format_dict_as_list(evidence_types)}
        
        Processing status:
        - Processed: {sum(1 for item in evidence_items if item['processed'])} items
        - Pending: {sum(1 for item in evidence_items if not item['processed'])} items
        """
        
        return {
            "title": "Evidence Inventory",
            "summary": inventory_summary.strip(),
            "evidence_items": evidence_items,
            "evidence_types": evidence_types
        }
    
    def generate_timeline_report(
        self, case_id: str, entities: Dict[str, Entity], events: List[Event],
        patterns: Dict[str, Pattern], evidence: List[Any], analysis_results: Dict[str, Any],
        synthesis_results: Dict[str, Any], key_findings: List[str], recommendations: List[str]
    ) -> Dict[str, Any]:
        """Generate timeline report section"""
        
        if not events:
            return {
                "title": "Timeline Report",
                "content": "No events available for timeline analysis."
            }
        
        # Sort events chronologically
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        timeline_entries = []
        for event in sorted_events:
            timeline_entries.append({
                "timestamp": event.timestamp.isoformat(),
                "date": event.timestamp.strftime("%Y-%m-%d"),
                "time": event.timestamp.strftime("%H:%M:%S"),
                "event_type": event.event_type,
                "description": event.description,
                "entities_involved": event.entities_involved,
                "confidence": event.confidence,
                "evidence_source": event.evidence_source
            })
        
        # Generate timeline summary
        start_date = sorted_events[0].timestamp.strftime("%Y-%m-%d %H:%M")
        end_date = sorted_events[-1].timestamp.strftime("%Y-%m-%d %H:%M")
        duration = (sorted_events[-1].timestamp - sorted_events[0].timestamp).days
        
        timeline_summary = f"""
        Timeline Analysis: {len(events)} events
        Timespan: {start_date} to {end_date} ({duration} days)
        
        Event frequency by type:
        {self._format_dict_as_list(dict(Counter(event.event_type for event in events)))}
        """
        
        return {
            "title": "Timeline Report",
            "summary": timeline_summary.strip(),
            "timeline_entries": timeline_entries,
            "total_events": len(events),
            "timespan_days": duration
        }
    
    def generate_entity_analysis(
        self, case_id: str, entities: Dict[str, Entity], events: List[Event],
        patterns: Dict[str, Pattern], evidence: List[Any], analysis_results: Dict[str, Any],
        synthesis_results: Dict[str, Any], key_findings: List[str], recommendations: List[str]
    ) -> Dict[str, Any]:
        """Generate entity analysis section"""
        
        if not entities:
            return {
                "title": "Entity Analysis",
                "content": "No entities identified for analysis."
            }
        
        entity_details = []
        for entity_id, entity in entities.items():
            entity_details.append({
                "entity_id": entity_id,
                "name": entity.name,
                "type": entity.type,
                "confidence": entity.confidence,
                "attributes": entity.attributes,
                "first_seen": entity.first_seen.isoformat() if entity.first_seen else None,
                "last_seen": entity.last_seen.isoformat() if entity.last_seen else None,
                "case_id": entity.case_id
            })
        
        # Sort by confidence (highest first)
        entity_details.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Generate entity summary
        entity_types = {}
        confidence_levels = {"high": 0, "medium": 0, "low": 0}
        
        for entity in entities.values():
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
            
            if entity.confidence >= 0.8:
                confidence_levels["high"] += 1
            elif entity.confidence >= 0.5:
                confidence_levels["medium"] += 1
            else:
                confidence_levels["low"] += 1
        
        entity_summary = f"""
        Total entities identified: {len(entities)}
        
        Entity types:
        {self._format_dict_as_list(entity_types)}
        
        Confidence distribution:
        - High confidence (≥0.8): {confidence_levels['high']} entities
        - Medium confidence (0.5-0.8): {confidence_levels['medium']} entities  
        - Low confidence (<0.5): {confidence_levels['low']} entities
        
        Top 5 highest confidence entities:
        {self._format_top_entities(entity_details[:5])}
        """
        
        return {
            "title": "Entity Analysis",
            "summary": entity_summary.strip(),
            "entity_details": entity_details,
            "entity_types": entity_types,
            "confidence_distribution": confidence_levels
        }
    
    def generate_pattern_report(
        self, case_id: str, entities: Dict[str, Entity], events: List[Event],
        patterns: Dict[str, Pattern], evidence: List[Any], analysis_results: Dict[str, Any],
        synthesis_results: Dict[str, Any], key_findings: List[str], recommendations: List[str]
    ) -> Dict[str, Any]:
        """Generate pattern analysis report section"""
        
        if not patterns:
            return {
                "title": "Pattern Analysis Report",
                "content": "No patterns detected in the current analysis."
            }
        
        pattern_details = []
        for pattern_id, pattern in patterns.items():
            pattern_details.append({
                "pattern_id": pattern_id,
                "pattern_type": pattern.pattern_type,
                "description": pattern.description,
                "entities": pattern.entities,
                "events": pattern.events,
                "confidence": pattern.confidence,
                "significance": pattern.significance,
                "first_detected": pattern.first_detected.isoformat()
            })
        
        # Sort by significance and confidence
        significance_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        pattern_details.sort(
            key=lambda x: (significance_order.get(x["significance"], 0), x["confidence"]),
            reverse=True
        )
        
        # Generate pattern summary
        pattern_types = {}
        significance_levels = {}
        
        for pattern in patterns.values():
            pattern_types[pattern.pattern_type] = pattern_types.get(pattern.pattern_type, 0) + 1
            significance_levels[pattern.significance] = significance_levels.get(pattern.significance, 0) + 1
        
        critical_patterns = [p for p in patterns.values() if p.significance == "critical"]
        high_patterns = [p for p in patterns.values() if p.significance == "high"]
        
        pattern_summary = f"""
        Total patterns detected: {len(patterns)}
        
        Pattern types:
        {self._format_dict_as_list(pattern_types)}
        
        Significance distribution:
        {self._format_dict_as_list(significance_levels)}
        
        Priority patterns requiring immediate attention:
        - Critical: {len(critical_patterns)} patterns
        - High: {len(high_patterns)} patterns
        """
        
        if critical_patterns or high_patterns:
            priority_patterns = critical_patterns + high_patterns
            pattern_summary += f"\n\nTop priority patterns:\n"
            for i, pattern in enumerate(priority_patterns[:5], 1):
                pattern_summary += f"{i}. [{pattern.significance.upper()}] {pattern.description} (Confidence: {pattern.confidence:.2f})\n"
        
        return {
            "title": "Pattern Analysis Report",
            "summary": pattern_summary.strip(),
            "pattern_details": pattern_details,
            "pattern_types": pattern_types,
            "significance_distribution": significance_levels
        }
    
    def generate_recommendations_report(
        self, case_id: str, entities: Dict[str, Entity], events: List[Event],
        patterns: Dict[str, Pattern], evidence: List[Any], analysis_results: Dict[str, Any],
        synthesis_results: Dict[str, Any], key_findings: List[str], recommendations: List[str]
    ) -> Dict[str, Any]:
        """Generate recommendations report section"""
        
        # Combine recommendations from synthesis and current state
        all_recommendations = recommendations.copy()
        
        # Add synthesis recommendations
        synthesis_recommendations = synthesis_results.get("recommendations", [])
        for rec in synthesis_recommendations:
            if rec not in all_recommendations:
                all_recommendations.append(rec)
        
        # Categorize recommendations
        categorized_recommendations = {
            "immediate_actions": [],
            "investigative_steps": [],
            "evidence_collection": [],
            "analysis_improvements": [],
            "general": []
        }
        
        for recommendation in all_recommendations:
            rec_lower = recommendation.lower()
            if any(word in rec_lower for word in ["urgent", "immediate", "critical"]):
                categorized_recommendations["immediate_actions"].append(recommendation)
            elif any(word in rec_lower for word in ["investigate", "examine", "analyze"]):
                categorized_recommendations["investigative_steps"].append(recommendation)
            elif any(word in rec_lower for word in ["evidence", "collect", "gather"]):
                categorized_recommendations["evidence_collection"].append(recommendation)
            elif any(word in rec_lower for word in ["improve", "enhance", "confidence"]):
                categorized_recommendations["analysis_improvements"].append(recommendation)
            else:
                categorized_recommendations["general"].append(recommendation)
        
        # Generate investigation gaps section
        gaps = synthesis_results.get("investigation_gaps", [])
        
        recommendations_content = f"""
        Total recommendations generated: {len(all_recommendations)}
        Investigation gaps identified: {len(gaps)}
        
        IMMEDIATE ACTIONS REQUIRED ({len(categorized_recommendations['immediate_actions'])}):
        {self._format_list_with_numbers(categorized_recommendations['immediate_actions'])}
        
        INVESTIGATIVE STEPS ({len(categorized_recommendations['investigative_steps'])}):
        {self._format_list_with_numbers(categorized_recommendations['investigative_steps'])}
        
        EVIDENCE COLLECTION ({len(categorized_recommendations['evidence_collection'])}):
        {self._format_list_with_numbers(categorized_recommendations['evidence_collection'])}
        
        ANALYSIS IMPROVEMENTS ({len(categorized_recommendations['analysis_improvements'])}):
        {self._format_list_with_numbers(categorized_recommendations['analysis_improvements'])}
        """
        
        if gaps:
            recommendations_content += f"\n\nINVESTIGATION GAPS TO ADDRESS:\n"
            for i, gap in enumerate(gaps, 1):
                recommendations_content += f"{i}. {gap}\n"
        
        return {
            "title": "Recommendations and Next Steps",
            "content": recommendations_content.strip(),
            "categorized_recommendations": categorized_recommendations,
            "investigation_gaps": gaps,
            "total_recommendations": len(all_recommendations)
        }
    
    def generate_report_statistics(
        self, entities: Dict[str, Entity], events: List[Event],
        patterns: Dict[str, Pattern], evidence: List[Any], analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate overall report statistics"""
        
        return {
            "total_entities": len(entities),
            "total_events": len(events),
            "total_patterns": len(patterns),
            "total_evidence": len(evidence),
            "entity_types_count": len(set(entity.type for entity in entities.values())) if entities else 0,
            "event_types_count": len(set(event.event_type for event in events)) if events else 0,
            "pattern_types_count": len(set(pattern.pattern_type for pattern in patterns.values())) if patterns else 0,
            "high_confidence_entities": sum(1 for entity in entities.values() if entity.confidence >= 0.8) if entities else 0,
            "high_significance_patterns": sum(1 for pattern in patterns.values() if pattern.significance in ["high", "critical"]) if patterns else 0,
            "analysis_completeness": self._calculate_analysis_completeness(entities, events, patterns, evidence)
        }
    
    def _calculate_analysis_completeness(
        self, entities: Dict[str, Entity], events: List[Event],
        patterns: Dict[str, Pattern], evidence: List[Any]
    ) -> float:
        """Calculate analysis completeness score"""
        completeness_factors = []
        
        # Entity analysis completeness
        if evidence:
            entity_ratio = len(entities) / len(evidence)
            completeness_factors.append(min(1.0, entity_ratio))  # Normalize to 1.0
        
        # Event analysis completeness
        if entities:
            event_ratio = len(events) / len(entities)
            completeness_factors.append(min(1.0, event_ratio * 0.5))  # Expect fewer events than entities
        
        # Pattern analysis completeness
        if events:
            pattern_ratio = len(patterns) / max(1, len(events) // 5)  # Expect 1 pattern per 5 events
            completeness_factors.append(min(1.0, pattern_ratio))
        
        return sum(completeness_factors) / max(1, len(completeness_factors))
    
    def _format_dict_as_list(self, data_dict: Dict[str, Any]) -> str:
        """Format dictionary as a numbered list"""
        if not data_dict:
            return "- None"
        
        formatted_items = []
        for key, value in sorted(data_dict.items(), key=lambda x: x[1], reverse=True):
            formatted_items.append(f"- {key}: {value}")
        
        return "\n".join(formatted_items)
    
    def _format_list_with_numbers(self, items: List[str]) -> str:
        """Format list with numbers"""
        if not items:
            return "- None"
        
        formatted_items = []
        for i, item in enumerate(items, 1):
            formatted_items.append(f"{i}. {item}")
        
        return "\n".join(formatted_items)
    
    def _format_top_entities(self, entity_details: List[Dict[str, Any]]) -> str:
        """Format top entities list"""
        if not entity_details:
            return "- None"
        
        formatted_entities = []
        for i, entity in enumerate(entity_details, 1):
            formatted_entities.append(f"{i}. {entity['name']} ({entity['type']}) - Confidence: {entity['confidence']:.2f}")
        
        return "\n".join(formatted_entities)

def report_generator(state: ForensicBotState) -> ForensicBotState:
    """
    Generate comprehensive forensic investigation reports
    
    This node creates detailed reports based on all analysis results,
    including executive summaries, detailed findings, and recommendations.
    
    Args:
        state: Current forensic bot state
        
    Returns:
        ForensicBotState: Updated state with generated report
    """
    start_time = datetime.now()
    
    try:
        # Determine report type from user query
        if state["messages"]:
            last_message = state["messages"][-1]
            query = last_message.content if hasattr(last_message, 'content') else str(last_message)
            report_type = determine_report_type(query)
        else:
            report_type = "comprehensive"
        
        # Check if we have enough data for report generation
        if not state["entity_memory"] and not state["timeline_memory"] and not state["analysis_results"]:
            response = (
                "I don't have enough analyzed data to generate a meaningful report. "
                "Please ensure evidence has been processed and analysis has been completed first."
            )
            ai_message = AIMessage(content=response)
            state["messages"].append(ai_message)
            return state
        
        # Initialize report generator
        if "report_generator_instance" not in state["tool_results"]:
            state["tool_results"]["report_generator_instance"] = ReportGenerator()
        
        generator = state["tool_results"]["report_generator_instance"]
        
        # Get synthesis results
        synthesis_results = state["analysis_results"].get("synthesis", {})
        
        # Generate the report
        case_id = state["current_case_id"] or "unknown_case"
        
        report = generator.generate_comprehensive_report(
            case_id=case_id,
            entities=state["entity_memory"],
            events=state["timeline_memory"],
            patterns=state["pattern_memory"],
            evidence=state["active_evidence"],
            analysis_results=state["analysis_results"],
            synthesis_results=synthesis_results,
            key_findings=state["key_findings"],
            recommendations=state["recommendations"],
            report_type=report_type
        )
        
        # Store the report
        state["report_sections"] = report["sections"]
        state["analysis_results"]["generated_report"] = report
        
        # Generate response based on report type
        response = generate_report_response(report, report_type)
        ai_message = AIMessage(content=response)
        state["messages"].append(ai_message)
        
        # Update recommendations with report-specific ones
        if "recommendations" in report["sections"]:
            report_recommendations = report["sections"]["recommendations"].get("categorized_recommendations", {})
            immediate_actions = report_recommendations.get("immediate_actions", [])
            for action in immediate_actions:
                if action not in state["recommendations"]:
                    state["recommendations"].append(action)
        
        # Add to tools used
        if "report_generator" not in state["tools_used"]:
            state["tools_used"].append("report_generator")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        add_workflow_step(
            state,
            node_name="report_generator",
            action=f"generate_{report_type}_report",
            input_data={
                "case_id": case_id,
                "report_type": report_type,
                "entities_count": len(state["entity_memory"]),
                "events_count": len(state["timeline_memory"]),
                "patterns_count": len(state["pattern_memory"])
            },
            output_data={
                "sections_generated": len(report["sections"]),
                "report_statistics": report["statistics"],
                "total_recommendations": report["statistics"].get("total_recommendations", 0)
            },
            execution_time=execution_time,
            success=True
        )
        
    except Exception as e:
        # Handle errors gracefully
        error_msg = f"Report generation error: {str(e)}"
        state["last_tool_error"] = error_msg
        
        fallback_response = (
            "I encountered an issue while generating the investigation report. "
            "This might be due to incomplete analysis or data formatting issues. "
            "Please ensure all analysis steps have been completed successfully."
        )
        
        ai_message = AIMessage(content=fallback_response)
        state["messages"].append(ai_message)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        add_workflow_step(
            state,
            node_name="report_generator",
            action="generate_report",
            input_data={"error": str(e)},
            output_data={"fallback_response": True},
            execution_time=execution_time,
            success=False
        )
    
    return state

def determine_report_type(query: str) -> str:
    """Determine the type of report to generate based on user query"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["summary", "executive", "brief"]):
        return "summary"
    elif any(word in query_lower for word in ["technical", "detailed", "full", "comprehensive"]):
        return "comprehensive"
    elif any(word in query_lower for word in ["timeline", "chronological"]):
        return "timeline"
    elif any(word in query_lower for word in ["entity", "entities"]):
        return "entity_focused"
    elif any(word in query_lower for word in ["pattern", "patterns"]):
        return "pattern_focused"
    else:
        return "comprehensive"

def generate_report_response(report: Dict[str, Any], report_type: str) -> str:
    """Generate a user-friendly response about the generated report"""
    
    metadata = report.get("metadata", {})
    statistics = report.get("statistics", {})
    sections = report.get("sections", {})
    
    response = f"## Forensic Investigation Report Generated\n\n"
    response += f"**Report Type:** {report_type.title()}\n"
    response += f"**Case ID:** {metadata.get('case_id', 'Unknown')}\n"
    response += f"**Generated:** {metadata.get('generated_at', '')[:19].replace('T', ' ')}\n\n"
    
    # Report statistics
    response += f"**Report Statistics:**\n"
    response += f"- Entities analyzed: {statistics.get('total_entities', 0)}\n"
    response += f"- Events processed: {statistics.get('total_events', 0)}\n"
    response += f"- Patterns detected: {statistics.get('total_patterns', 0)}\n"
    response += f"- Evidence items: {statistics.get('total_evidence', 0)}\n"
    response += f"- Analysis completeness: {statistics.get('analysis_completeness', 0):.1%}\n\n"
    
    # Show executive summary if available
    if "executive_summary" in sections:
        exec_summary = sections["executive_summary"]
        response += f"**Executive Summary:**\n{exec_summary.get('content', '')}\n\n"
    
    # Show key sections generated
    response += f"**Report Sections Generated ({len(sections)}):**\n"
    for section_name, section_data in sections.items():
        section_title = section_data.get("title", section_name.replace("_", " ").title())
        response += f"- {section_title}\n"
    
    # Show immediate actions if available
    if "recommendations" in sections:
        recommendations_section = sections["recommendations"]
        immediate_actions = recommendations_section.get("categorized_recommendations", {}).get("immediate_actions", [])
        if immediate_actions:
            response += f"\n**Immediate Actions Required:**\n"
            for i, action in enumerate(immediate_actions[:3], 1):  # Top 3 immediate actions
                response += f"{i}. {action}\n"
    
    # Add access information
    response += f"\n*The complete {report_type} report has been generated and stored in the case analysis results. "
    response += f"All sections and detailed findings are available for review and export.*"
    
    return response

# Import Counter for use in timeline report
from collections import Counter