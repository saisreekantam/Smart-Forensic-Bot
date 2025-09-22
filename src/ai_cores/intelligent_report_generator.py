"""
Intelligent Report Generator

This module creates comprehensive forensic investigation reports using
knowledge graphs, case memory, and advanced analytics.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import logging
from dataclasses import dataclass
from collections import Counter, defaultdict
import uuid

from .case_memory import CaseMemoryManager, CaseMemoryStats, CaseInsight
from .enhanced_knowledge_graph import EnhancedKnowledgeGraphDB

logger = logging.getLogger(__name__)

@dataclass
class ReportSection:
    """Represents a section of the investigation report"""
    title: str
    content: str
    section_type: str  # summary, analysis, findings, recommendations, etc.
    confidence: float
    supporting_evidence: List[str]
    metadata: Dict[str, Any]

@dataclass
class InvestigationReport:
    """Complete investigation report"""
    report_id: str
    case_id: str
    title: str
    generated_at: datetime
    report_type: str  # summary, detailed, executive, technical
    sections: List[ReportSection]
    statistics: Dict[str, Any]
    insights: List[Dict[str, Any]]
    recommendations: List[str]
    confidence_score: float
    metadata: Dict[str, Any]

class IntelligentReportGenerator:
    """Generates comprehensive forensic investigation reports"""
    
    def __init__(self, 
                 case_memory: CaseMemoryManager,
                 knowledge_graph: EnhancedKnowledgeGraphDB):
        """Initialize report generator"""
        self.case_memory = case_memory
        self.knowledge_graph = knowledge_graph
        self.report_templates = self._load_report_templates()
        
    def _load_report_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load report templates for different report types"""
        return {
            "executive_summary": {
                "sections": ["case_overview", "key_findings", "executive_summary", "recommendations"],
                "max_length": 2000,
                "focus": "high_level"
            },
            "detailed_analysis": {
                "sections": ["case_overview", "evidence_analysis", "entity_network", "timeline_analysis", 
                           "relationship_analysis", "technical_findings", "conclusions", "recommendations"],
                "max_length": 10000,
                "focus": "comprehensive"
            },
            "technical_report": {
                "sections": ["technical_summary", "data_sources", "methodology", "entity_extraction",
                           "network_analysis", "temporal_analysis", "anomaly_detection", "technical_conclusions"],
                "max_length": 8000,
                "focus": "technical"
            },
            "investigation_summary": {
                "sections": ["investigation_progress", "key_discoveries", "entity_mapping", 
                           "investigation_gaps", "next_steps"],
                "max_length": 5000,
                "focus": "investigative"
            }
        }
    
    def generate_report(self, 
                       case_id: str, 
                       report_type: str = "detailed_analysis",
                       custom_sections: Optional[List[str]] = None) -> InvestigationReport:
        """Generate a comprehensive investigation report"""
        
        # Get case data
        case_stats = self.case_memory.get_case_memory_stats(case_id)
        case_insights = self.case_memory.get_case_insights(case_id)
        knowledge_graph = self.knowledge_graph.get_case_knowledge_graph(case_id)
        interactions = self.case_memory.get_case_interactions(case_id)
        
        # Determine sections to include
        template = self.report_templates.get(report_type, self.report_templates["detailed_analysis"])
        sections_to_generate = custom_sections or template["sections"]
        
        # Generate report sections
        sections = []
        for section_name in sections_to_generate:
            section = self._generate_section(
                section_name, case_id, case_stats, case_insights, 
                knowledge_graph, interactions, template
            )
            if section:
                sections.append(section)
        
        # Calculate overall confidence
        confidence_score = self._calculate_report_confidence(sections, case_stats, knowledge_graph)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(case_stats, case_insights, knowledge_graph)
        
        # Create report
        report = InvestigationReport(
            report_id=str(uuid.uuid4()),
            case_id=case_id,
            title=f"{report_type.replace('_', ' ').title()} - Case {case_id}",
            generated_at=datetime.now(),
            report_type=report_type,
            sections=sections,
            statistics=self._compile_statistics(case_stats, knowledge_graph),
            insights=[self._insight_to_dict(insight) for insight in case_insights],
            recommendations=recommendations,
            confidence_score=confidence_score,
            metadata={
                "total_interactions": case_stats.total_interactions,
                "entities_analyzed": knowledge_graph.get("summary", {}).get("total_entities", 0),
                "relationships_found": knowledge_graph.get("summary", {}).get("total_relationships", 0),
                "template_used": report_type,
                "generation_method": "ai_enhanced"
            }
        )
        
        return report
    
    def _generate_section(self, 
                         section_name: str, 
                         case_id: str,
                         case_stats: CaseMemoryStats,
                         case_insights: List[CaseInsight],
                         knowledge_graph: Dict[str, Any],
                         interactions: List,
                         template: Dict[str, Any]) -> Optional[ReportSection]:
        """Generate a specific section of the report"""
        
        section_generators = {
            "case_overview": self._generate_case_overview,
            "key_findings": self._generate_key_findings,
            "executive_summary": self._generate_executive_summary,
            "evidence_analysis": self._generate_evidence_analysis,
            "entity_network": self._generate_entity_network_analysis,
            "timeline_analysis": self._generate_timeline_analysis,
            "relationship_analysis": self._generate_relationship_analysis,
            "technical_findings": self._generate_technical_findings,
            "conclusions": self._generate_conclusions,
            "recommendations": self._generate_recommendations_section,
            "investigation_progress": self._generate_investigation_progress,
            "key_discoveries": self._generate_key_discoveries,
            "entity_mapping": self._generate_entity_mapping,
            "investigation_gaps": self._generate_investigation_gaps,
            "next_steps": self._generate_next_steps,
            "technical_summary": self._generate_technical_summary,
            "data_sources": self._generate_data_sources,
            "methodology": self._generate_methodology,
            "entity_extraction": self._generate_entity_extraction_section,
            "network_analysis": self._generate_network_analysis,
            "temporal_analysis": self._generate_temporal_analysis,
            "anomaly_detection": self._generate_anomaly_detection,
            "technical_conclusions": self._generate_technical_conclusions
        }
        
        generator = section_generators.get(section_name)
        if not generator:
            logger.warning(f"No generator found for section: {section_name}")
            return None
        
        try:
            return generator(case_id, case_stats, case_insights, knowledge_graph, interactions, template)
        except Exception as e:
            logger.error(f"Error generating section {section_name}: {e}")
            return ReportSection(
                title=section_name.replace('_', ' ').title(),
                content=f"Error generating section: {str(e)}",
                section_type="error",
                confidence=0.0,
                supporting_evidence=[],
                metadata={"error": str(e)}
            )
    
    def _generate_case_overview(self, case_id, case_stats, case_insights, knowledge_graph, interactions, template) -> ReportSection:
        """Generate case overview section"""
        
        # Get basic statistics
        total_entities = knowledge_graph.get("summary", {}).get("total_entities", 0)
        total_relationships = knowledge_graph.get("summary", {}).get("total_relationships", 0)
        investigation_days = len(case_stats.temporal_activity) if case_stats.temporal_activity else 0
        
        # Get key entities
        key_entities = knowledge_graph.get("summary", {}).get("key_entities", [])[:5]
        
        content = f"""
## Case Overview

**Case ID:** {case_id}
**Investigation Period:** {investigation_days} days of active investigation
**Total Bot Interactions:** {case_stats.total_interactions}
**Unique Entities Identified:** {total_entities}
**Entity Relationships Mapped:** {total_relationships}

### Investigation Scope

This forensic investigation has analyzed evidence through {case_stats.total_interactions} intelligent queries, 
resulting in the identification of {total_entities} unique entities and {total_relationships} relationships 
between them.

### Key Entities Identified

"""
        
        for entity in key_entities:
            content += f"- **{entity.get('type', 'Unknown').title()}**: {entity.get('value', 'N/A')} "
            content += f"(Importance: {entity.get('importance', 0.0):.2f}, Mentions: {entity.get('mentions', 0)})\n"
        
        if case_stats.investigation_focus_areas:
            content += "\n### Investigation Focus Areas\n\n"
            for area, score in case_stats.investigation_focus_areas[:5]:
                percentage = score * 100
                content += f"- **{area.title()}**: {percentage:.1f}% of investigation effort\n"
        
        return ReportSection(
            title="Case Overview",
            content=content.strip(),
            section_type="overview",
            confidence=0.9,
            supporting_evidence=[f"interaction_count_{case_stats.total_interactions}"],
            metadata={
                "entities_analyzed": total_entities,
                "relationships_found": total_relationships,
                "investigation_days": investigation_days
            }
        )
    
    def _generate_key_findings(self, case_id, case_stats, case_insights, knowledge_graph, interactions, template) -> ReportSection:
        """Generate key findings section"""
        
        # High-priority insights
        high_priority_insights = [insight for insight in case_insights if insight.priority == "high"]
        
        # Most important entities
        key_entities = knowledge_graph.get("summary", {}).get("key_entities", [])[:10]
        
        # Strongest relationships
        important_relationships = knowledge_graph.get("summary", {}).get("important_relationships", [])[:5]
        
        content = "## Key Findings\n\n"
        
        if high_priority_insights:
            content += "### High-Priority Discoveries\n\n"
            for insight in high_priority_insights:
                content += f"**{insight.title}**\n"
                content += f"{insight.description}\n"
                content += f"*Confidence: {insight.confidence:.2f}*\n\n"
        
        if key_entities:
            content += "### Critical Entities\n\n"
            for entity in key_entities:
                content += f"- **{entity.get('value', 'N/A')}** ({entity.get('type', 'unknown')}): "
                content += f"Mentioned {entity.get('mentions', 0)} times, "
                content += f"Importance score: {entity.get('importance', 0.0):.2f}\n"
        
        if important_relationships:
            content += "\n### Significant Relationships\n\n"
            for rel in important_relationships:
                content += f"- **{rel.get('entity1', 'Entity1')}** → **{rel.get('entity2', 'Entity2')}** "
                content += f"({rel.get('type', 'related')})\n"
                content += f"  Strength: {rel.get('strength', 0.0):.2f}, "
                content += f"Confidence: {rel.get('confidence', 0.0):.2f}\n"
        
        # Network analysis highlights
        network_metrics = knowledge_graph.get("summary", {}).get("detailed_summary", {}).get("network_metrics", {})
        if network_metrics:
            content += "\n### Network Analysis Highlights\n\n"
            content += f"- Network density: {network_metrics.get('density', 0.0):.3f}\n"
            content += f"- Connected components: {network_metrics.get('connected_components', 0)}\n"
            content += f"- Largest component size: {network_metrics.get('largest_component_size', 0)} entities\n"
        
        return ReportSection(
            title="Key Findings",
            content=content.strip(),
            section_type="findings",
            confidence=0.85,
            supporting_evidence=[insight.id for insight in high_priority_insights],
            metadata={
                "high_priority_insights": len(high_priority_insights),
                "key_entities_count": len(key_entities),
                "relationships_analyzed": len(important_relationships)
            }
        )
    
    def _generate_entity_network_analysis(self, case_id, case_stats, case_insights, knowledge_graph, interactions, template) -> ReportSection:
        """Generate entity network analysis section"""
        
        entities = knowledge_graph.get("entities", [])
        relationships = knowledge_graph.get("relationships", [])
        network_metrics = knowledge_graph.get("summary", {}).get("detailed_summary", {}).get("network_metrics", {})
        
        content = "## Entity Network Analysis\n\n"
        
        # Entity distribution
        entity_types = {}
        for entity in entities:
            entity_type = entity.get("type", "unknown")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        content += "### Entity Distribution\n\n"
        for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
            content += f"- **{entity_type.title()}**: {count} entities\n"
        
        # Relationship analysis
        if relationships:
            relationship_types = {}
            for rel in relationships:
                rel_type = rel.get("relationship_type", "unknown")
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
            
            content += "\n### Relationship Types\n\n"
            for rel_type, count in sorted(relationship_types.items(), key=lambda x: x[1], reverse=True):
                content += f"- **{rel_type.replace('_', ' ').title()}**: {count} relationships\n"
        
        # Network structure analysis
        if network_metrics:
            content += "\n### Network Structure\n\n"
            content += f"**Network Density**: {network_metrics.get('density', 0.0):.3f}\n"
            content += f"Network density measures how interconnected the entities are. "
            content += f"A density of {network_metrics.get('density', 0.0):.3f} indicates "
            
            density = network_metrics.get('density', 0.0)
            if density < 0.1:
                content += "a sparse network with few connections between entities.\n\n"
            elif density < 0.3:
                content += "a moderately connected network.\n\n"
            else:
                content += "a highly interconnected network with many relationships.\n\n"
            
            components = network_metrics.get('connected_components', 0)
            largest_component = network_metrics.get('largest_component_size', 0)
            
            content += f"**Connected Components**: {components}\n"
            content += f"**Largest Component Size**: {largest_component} entities\n\n"
            
            if components > 1:
                content += f"The network contains {components} separate groups of connected entities, "
                content += f"with the largest group containing {largest_component} entities. "
                content += "This suggests multiple distinct clusters of activity or investigation areas.\n\n"
        
        # Central entities
        if entities:
            # Sort by importance score
            central_entities = sorted(entities, key=lambda x: x.get("importance_score", 0.0), reverse=True)[:5]
            
            content += "### Most Important Entities\n\n"
            for entity in central_entities:
                content += f"**{entity.get('value', 'N/A')}** ({entity.get('type', 'unknown')})\n"
                content += f"- Importance Score: {entity.get('importance_score', 0.0):.3f}\n"
                content += f"- Mentions: {entity.get('mention_count', 0)}\n"
                content += f"- First Seen: {entity.get('first_seen', 'Unknown')}\n\n"
        
        return ReportSection(
            title="Entity Network Analysis",
            content=content.strip(),
            section_type="analysis",
            confidence=0.8,
            supporting_evidence=[f"entity_{e['id']}" for e in entities[:10]],
            metadata={
                "total_entities": len(entities),
                "total_relationships": len(relationships),
                "entity_types": len(entity_types),
                "network_metrics": network_metrics
            }
        )
    
    def _generate_timeline_analysis(self, case_id, case_stats, case_insights, knowledge_graph, interactions, template) -> ReportSection:
        """Generate timeline analysis section"""
        
        content = "## Timeline Analysis\n\n"
        
        # Investigation activity timeline
        if case_stats.temporal_activity:
            content += "### Investigation Activity Timeline\n\n"
            sorted_dates = sorted(case_stats.temporal_activity.items())
            
            for date, activity_count in sorted_dates:
                content += f"- **{date}**: {activity_count} investigative queries\n"
            
            # Analysis of activity patterns
            total_days = len(sorted_dates)
            avg_activity = sum(case_stats.temporal_activity.values()) / total_days if total_days > 0 else 0
            
            content += f"\n**Total Investigation Days**: {total_days}\n"
            content += f"**Average Daily Activity**: {avg_activity:.1f} queries per day\n\n"
            
            # Find peak activity days
            if case_stats.temporal_activity:
                peak_day = max(case_stats.temporal_activity.items(), key=lambda x: x[1])
                content += f"**Peak Investigation Day**: {peak_day[0]} with {peak_day[1]} queries\n\n"
        
        # Entity discovery timeline
        temporal_analysis = knowledge_graph.get("summary", {}).get("detailed_summary", {}).get("temporal_analysis", {})
        
        if temporal_analysis.get("entity_timeline"):
            content += "### Entity Discovery Timeline\n\n"
            entity_timeline = temporal_analysis["entity_timeline"]
            
            for entry in entity_timeline:
                content += f"- **{entry.get('date', 'Unknown')}**: {entry.get('count', 0)} new entities discovered\n"
            
            total_discovery_days = len(entity_timeline)
            total_entities_discovered = sum(entry.get('count', 0) for entry in entity_timeline)
            
            content += f"\n**Total Entity Discovery Period**: {total_discovery_days} days\n"
            content += f"**Total Entities Discovered**: {total_entities_discovered}\n\n"
        
        # Relationship discovery timeline
        if temporal_analysis.get("relationship_timeline"):
            content += "### Relationship Discovery Timeline\n\n"
            relationship_timeline = temporal_analysis["relationship_timeline"]
            
            for entry in relationship_timeline:
                content += f"- **{entry.get('date', 'Unknown')}**: {entry.get('count', 0)} new relationships identified\n"
            
            total_relationships_discovered = sum(entry.get('count', 0) for entry in relationship_timeline)
            content += f"\n**Total Relationships Discovered**: {total_relationships_discovered}\n\n"
        
        # Investigation patterns
        content += "### Investigation Patterns\n\n"
        
        if case_stats.common_query_patterns:
            content += "**Query Pattern Analysis**:\n"
            for pattern, frequency in case_stats.common_query_patterns[:5]:
                percentage = (frequency / case_stats.total_interactions * 100) if case_stats.total_interactions > 0 else 0
                content += f"- {pattern.replace('_', ' ').title()}: {frequency} queries ({percentage:.1f}%)\n"
            content += "\n"
        
        # Temporal insights from case insights
        temporal_insights = [insight for insight in case_insights if "temporal" in insight.insight_type.lower() or "time" in insight.insight_type.lower()]
        
        if temporal_insights:
            content += "### Temporal Insights\n\n"
            for insight in temporal_insights:
                content += f"**{insight.title}**\n"
                content += f"{insight.description}\n"
                content += f"*Discovered: {insight.discovered_at.strftime('%Y-%m-%d %H:%M')}*\n\n"
        
        return ReportSection(
            title="Timeline Analysis",
            content=content.strip(),
            section_type="analysis",
            confidence=0.75,
            supporting_evidence=list(case_stats.temporal_activity.keys()),
            metadata={
                "investigation_days": len(case_stats.temporal_activity),
                "total_queries": case_stats.total_interactions,
                "temporal_insights": len(temporal_insights)
            }
        )
    
    def _generate_executive_summary(self, case_id, case_stats, case_insights, knowledge_graph, interactions, template) -> ReportSection:
        """Generate executive summary section"""
        
        # Get key metrics
        total_entities = knowledge_graph.get("summary", {}).get("total_entities", 0)
        total_relationships = knowledge_graph.get("summary", {}).get("total_relationships", 0)
        high_priority_insights = [insight for insight in case_insights if insight.priority == "high"]
        key_entities = knowledge_graph.get("summary", {}).get("key_entities", [])[:3]
        
        content = "## Executive Summary\n\n"
        
        # Investigation overview
        content += f"This forensic investigation of Case {case_id} has conducted comprehensive analysis "
        content += f"through {case_stats.total_interactions} intelligent queries over "
        content += f"{len(case_stats.temporal_activity)} days of active investigation.\n\n"
        
        # Key discoveries
        content += "### Key Discoveries\n\n"
        content += f"The investigation has identified **{total_entities} unique entities** and mapped "
        content += f"**{total_relationships} relationships** between them, providing a comprehensive "
        content += "view of the case landscape.\n\n"
        
        if high_priority_insights:
            content += f"**{len(high_priority_insights)} high-priority insights** have been discovered:\n"
            for insight in high_priority_insights[:3]:
                content += f"- {insight.title}\n"
            if len(high_priority_insights) > 3:
                content += f"- ...and {len(high_priority_insights) - 3} additional high-priority findings\n"
            content += "\n"
        
        # Critical entities
        if key_entities:
            content += "### Critical Entities\n\n"
            content += "The most significant entities identified in this investigation are:\n"
            for entity in key_entities:
                content += f"- **{entity.get('value', 'N/A')}** ({entity.get('type', 'unknown')}): "
                content += f"High importance entity mentioned {entity.get('mentions', 0)} times\n"
            content += "\n"
        
        # Investigation focus
        if case_stats.investigation_focus_areas:
            content += "### Investigation Focus\n\n"
            top_focus = case_stats.investigation_focus_areas[0]
            content += f"The investigation has primarily focused on **{top_focus[0]}** "
            content += f"({top_focus[1]*100:.1f}% of queries), indicating this as the main area of concern.\n\n"
        
        # Network analysis summary
        network_metrics = knowledge_graph.get("summary", {}).get("detailed_summary", {}).get("network_metrics", {})
        if network_metrics:
            density = network_metrics.get('density', 0.0)
            components = network_metrics.get('connected_components', 0)
            
            content += "### Network Structure\n\n"
            content += f"The entity network shows a density of {density:.3f}, indicating "
            
            if density < 0.1:
                content += "a sparse network with focused, specific connections. "
            elif density < 0.3:
                content += "moderate interconnectedness between entities. "
            else:
                content += "high interconnectedness suggesting complex relationships. "
            
            content += f"The network contains {components} distinct groups of connected entities.\n\n"
        
        # Recommendations preview
        content += "### Immediate Actions\n\n"
        recommendations = self._generate_recommendations(case_stats, case_insights, knowledge_graph)
        for rec in recommendations[:3]:
            content += f"- {rec}\n"
        
        if len(recommendations) > 3:
            content += f"- ...and {len(recommendations) - 3} additional recommendations (see full report)\n"
        
        return ReportSection(
            title="Executive Summary",
            content=content.strip(),
            section_type="summary",
            confidence=0.9,
            supporting_evidence=[f"case_overview_{case_id}"],
            metadata={
                "summary_type": "executive",
                "key_metrics": {
                    "entities": total_entities,
                    "relationships": total_relationships,
                    "insights": len(case_insights),
                    "high_priority_insights": len(high_priority_insights)
                }
            }
        )
    
    def _generate_recommendations(self, case_stats: CaseMemoryStats, case_insights: List[CaseInsight], knowledge_graph: Dict[str, Any]) -> List[str]:
        """Generate investigation recommendations"""
        recommendations = []
        
        # Entity co-occurrence analysis
        if case_stats.entity_co_occurrence:
            recommendations.append(
                f"Investigate relationships between {len(case_stats.entity_co_occurrence)} entities "
                "that frequently appear together in the evidence."
            )
        
        # High-priority insights
        high_priority_insights = [insight for insight in case_insights if insight.priority == "high"]
        if high_priority_insights:
            recommendations.append(
                f"Prioritize investigation of {len(high_priority_insights)} high-priority insights "
                "that require immediate attention."
            )
        
        # Network analysis recommendations
        network_metrics = knowledge_graph.get("summary", {}).get("detailed_summary", {}).get("network_metrics", {})
        if network_metrics:
            components = network_metrics.get('connected_components', 0)
            if components > 1:
                recommendations.append(
                    f"Analyze {components} separate entity groups to identify potential "
                    "connections or distinct investigation tracks."
                )
            
            density = network_metrics.get('density', 0.0)
            if density < 0.1:
                recommendations.append(
                    "Low network density suggests opportunities to discover additional "
                    "relationships through expanded evidence analysis."
                )
        
        # Investigation diversity
        if len(case_stats.investigation_focus_areas) < 3:
            recommendations.append(
                "Expand investigation scope to cover additional areas (financial, communication, "
                "technical, timeline) for comprehensive analysis."
            )
        
        # Entity importance analysis
        key_entities = knowledge_graph.get("summary", {}).get("key_entities", [])
        if key_entities:
            top_entity = key_entities[0]
            if top_entity.get('importance', 0.0) > 0.8:
                recommendations.append(
                    f"Deep-dive analysis of critical entity '{top_entity.get('value', 'N/A')}' "
                    "which shows high importance in the investigation."
                )
        
        # Temporal analysis
        if len(case_stats.temporal_activity) > 7:
            recent_activity = list(case_stats.temporal_activity.values())[-3:]
            if all(activity < 2 for activity in recent_activity):
                recommendations.append(
                    "Recent investigation activity is low. Consider conducting additional "
                    "evidence analysis or exploring new investigation areas."
                )
        
        return recommendations
    
    def _generate_recommendations_section(self, case_id, case_stats, case_insights, knowledge_graph, interactions, template) -> ReportSection:
        """Generate recommendations section"""
        
        recommendations = self._generate_recommendations(case_stats, case_insights, knowledge_graph)
        
        content = "## Recommendations\n\n"
        content += "Based on the comprehensive analysis of case evidence and investigation patterns, "
        content += "the following recommendations are provided:\n\n"
        
        for i, recommendation in enumerate(recommendations, 1):
            content += f"{i}. {recommendation}\n\n"
        
        # Investigation health assessment
        investigation_summary = self.case_memory.get_investigation_summary(case_id)
        health_metrics = investigation_summary.get("investigation_health", {})
        
        if health_metrics:
            content += "### Investigation Health Assessment\n\n"
            overall_score = health_metrics.get("overall_score", 0.0)
            
            if overall_score >= 0.8:
                health_status = "Excellent"
                health_color = "🟢"
            elif overall_score >= 0.6:
                health_status = "Good"
                health_color = "🟡"
            else:
                health_status = "Needs Improvement"
                health_color = "🔴"
            
            content += f"{health_color} **Investigation Health**: {health_status} ({overall_score:.2f}/1.0)\n\n"
            
            content += "**Health Breakdown**:\n"
            content += f"- Activity Level: {health_metrics.get('activity_level', 0.0):.2f}\n"
            content += f"- Investigation Diversity: {health_metrics.get('investigation_diversity', 0.0):.2f}\n"
            content += f"- Insight Discovery: {health_metrics.get('insight_discovery', 0.0):.2f}\n\n"
            
            # Additional health-based recommendations
            health_recommendations = health_metrics.get("recommendations", [])
            if health_recommendations:
                content += "**Health-Based Recommendations**:\n"
                for rec in health_recommendations:
                    content += f"- {rec}\n"
        
        return ReportSection(
            title="Recommendations",
            content=content.strip(),
            section_type="recommendations",
            confidence=0.85,
            supporting_evidence=["investigation_health", "case_analysis"],
            metadata={
                "total_recommendations": len(recommendations),
                "health_score": health_metrics.get("overall_score", 0.0) if health_metrics else 0.0
            }
        )
    
    def _calculate_report_confidence(self, sections: List[ReportSection], case_stats: CaseMemoryStats, knowledge_graph: Dict[str, Any]) -> float:
        """Calculate overall report confidence score"""
        
        # Base confidence from sections
        if sections:
            section_confidence = sum(section.confidence for section in sections) / len(sections)
        else:
            section_confidence = 0.5
        
        # Data quality factors
        data_quality = 0.0
        
        # Interaction volume boost
        if case_stats.total_interactions >= 20:
            data_quality += 0.2
        elif case_stats.total_interactions >= 10:
            data_quality += 0.1
        
        # Entity diversity boost
        total_entities = knowledge_graph.get("summary", {}).get("total_entities", 0)
        if total_entities >= 20:
            data_quality += 0.2
        elif total_entities >= 10:
            data_quality += 0.1
        
        # Relationship density boost
        total_relationships = knowledge_graph.get("summary", {}).get("total_relationships", 0)
        if total_relationships >= 15:
            data_quality += 0.2
        elif total_relationships >= 5:
            data_quality += 0.1
        
        # Investigation diversity boost
        if len(case_stats.investigation_focus_areas) >= 4:
            data_quality += 0.1
        
        # Combine confidences
        final_confidence = (section_confidence * 0.7) + (data_quality * 0.3)
        
        return min(1.0, max(0.0, final_confidence))
    
    def _compile_statistics(self, case_stats: CaseMemoryStats, knowledge_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Compile comprehensive statistics for the report"""
        
        return {
            "investigation_metrics": {
                "total_interactions": case_stats.total_interactions,
                "unique_entities_mentioned": case_stats.unique_entities_mentioned,
                "investigation_days": len(case_stats.temporal_activity),
                "avg_daily_activity": (sum(case_stats.temporal_activity.values()) / 
                                     len(case_stats.temporal_activity)) if case_stats.temporal_activity else 0
            },
            "knowledge_graph_metrics": {
                "total_entities": knowledge_graph.get("summary", {}).get("total_entities", 0),
                "total_relationships": knowledge_graph.get("summary", {}).get("total_relationships", 0),
                "knowledge_density": knowledge_graph.get("summary", {}).get("knowledge_density", 0.0),
                "network_metrics": knowledge_graph.get("summary", {}).get("detailed_summary", {}).get("network_metrics", {})
            },
            "investigation_focus": {
                "top_focus_areas": case_stats.investigation_focus_areas[:5],
                "query_patterns": case_stats.common_query_patterns[:10]
            },
            "entity_analysis": {
                "entity_co_occurrence_count": len(case_stats.entity_co_occurrence),
                "key_entities": knowledge_graph.get("summary", {}).get("key_entities", [])[:10]
            }
        }
    
    def _insight_to_dict(self, insight: CaseInsight) -> Dict[str, Any]:
        """Convert CaseInsight to dictionary"""
        return {
            "id": insight.id,
            "type": insight.insight_type,
            "title": insight.title,
            "description": insight.description,
            "confidence": insight.confidence,
            "priority": insight.priority,
            "discovered_at": insight.discovered_at.isoformat(),
            "evidence_count": len(insight.evidence)
        }
    
    def export_report_to_html(self, report: InvestigationReport) -> str:
        """Export report to HTML format"""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
        h3 {{ color: #34495e; }}
        .header {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; }}
        .confidence {{ font-style: italic; color: #7f8c8d; }}
        .statistics {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
        .recommendation {{ background-color: #e8f5e8; padding: 10px; border-left: 4px solid #27ae60; margin: 10px 0; }}
        .insight {{ background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report.title}</h1>
        <p><strong>Report ID:</strong> {report.report_id}</p>
        <p><strong>Case ID:</strong> {report.case_id}</p>
        <p><strong>Generated:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Report Type:</strong> {report.report_type.replace('_', ' ').title()}</p>
        <p><strong>Overall Confidence:</strong> <span class="confidence">{report.confidence_score:.2f}</span></p>
    </div>
"""
        
        # Add sections
        for section in report.sections:
            html += f"""
    <div class="section">
        <div class="confidence">Section Confidence: {section.confidence:.2f}</div>
        {self._markdown_to_html(section.content)}
    </div>
"""
        
        # Add insights
        if report.insights:
            html += """
    <div class="section">
        <h2>Investigation Insights</h2>
"""
            for insight in report.insights:
                html += f"""
        <div class="insight">
            <h4>{insight['title']}</h4>
            <p>{insight['description']}</p>
            <small>Priority: {insight['priority']} | Confidence: {insight['confidence']:.2f} | 
                   Discovered: {insight['discovered_at']}</small>
        </div>
"""
            html += "    </div>\n"
        
        # Add recommendations
        if report.recommendations:
            html += """
    <div class="section">
        <h2>Recommendations</h2>
"""
            for i, rec in enumerate(report.recommendations, 1):
                html += f"""
        <div class="recommendation">
            <strong>{i}.</strong> {rec}
        </div>
"""
            html += "    </div>\n"
        
        html += """
</body>
</html>
"""
        
        return html
    
    def _markdown_to_html(self, markdown_text: str) -> str:
        """Simple markdown to HTML conversion"""
        html = markdown_text
        
        # Convert headers
        html = html.replace('### ', '<h3>').replace('\n\n', '</h3>\n\n')
        html = html.replace('## ', '<h2>').replace('\n\n', '</h2>\n\n')
        
        # Convert bold
        html = html.replace('**', '<strong>').replace('**', '</strong>')
        
        # Convert italic
        html = html.replace('*', '<em>').replace('*', '</em>')
        
        # Convert lists
        lines = html.split('\n')
        in_list = False
        result_lines = []
        
        for line in lines:
            if line.strip().startswith('- '):
                if not in_list:
                    result_lines.append('<ul>')
                    in_list = True
                result_lines.append(f"  <li>{line.strip()[2:]}</li>")
            else:
                if in_list:
                    result_lines.append('</ul>')
                    in_list = False
                result_lines.append(line)
        
        if in_list:
            result_lines.append('</ul>')
        
        # Convert paragraphs
        html = '\n'.join(result_lines)
        paragraphs = html.split('\n\n')
        html = '</p>\n<p>'.join(paragraphs)
        html = '<p>' + html + '</p>'
        
        return html
    
    def export_report_to_json(self, report: InvestigationReport) -> str:
        """Export report to JSON format"""
        
        report_dict = {
            "report_id": report.report_id,
            "case_id": report.case_id,
            "title": report.title,
            "generated_at": report.generated_at.isoformat(),
            "report_type": report.report_type,
            "confidence_score": report.confidence_score,
            "sections": [
                {
                    "title": section.title,
                    "content": section.content,
                    "section_type": section.section_type,
                    "confidence": section.confidence,
                    "supporting_evidence": section.supporting_evidence,
                    "metadata": section.metadata
                }
                for section in report.sections
            ],
            "statistics": report.statistics,
            "insights": report.insights,
            "recommendations": report.recommendations,
            "metadata": report.metadata
        }
        
        return json.dumps(report_dict, indent=2, ensure_ascii=False)

    # Additional section generators for comprehensive reporting
    def _generate_investigation_progress(self, case_id, case_stats, case_insights, knowledge_graph, interactions, template) -> ReportSection:
        """Generate investigation progress section"""
        
        content = "## Investigation Progress\n\n"
        
        # Progress metrics
        total_days = len(case_stats.temporal_activity)
        total_queries = case_stats.total_interactions
        avg_daily = total_queries / total_days if total_days > 0 else 0
        
        content += f"**Investigation Duration**: {total_days} days\n"
        content += f"**Total Investigative Queries**: {total_queries}\n"
        content += f"**Average Daily Activity**: {avg_daily:.1f} queries/day\n\n"
        
        # Discovery rate
        entities_discovered = knowledge_graph.get("summary", {}).get("total_entities", 0)
        relationships_discovered = knowledge_graph.get("summary", {}).get("total_relationships", 0)
        
        content += f"**Entities Discovered**: {entities_discovered}\n"
        content += f"**Relationships Mapped**: {relationships_discovered}\n"
        content += f"**Knowledge Discovery Rate**: {(entities_discovered + relationships_discovered) / total_days:.1f} discoveries/day\n\n"
        
        # Investigation phases
        if case_stats.temporal_activity:
            sorted_dates = sorted(case_stats.temporal_activity.items())
            
            # Early phase (first 25%)
            early_phase_end = len(sorted_dates) // 4
            early_activity = sum(activity for _, activity in sorted_dates[:early_phase_end])
            
            # Recent phase (last 25%)
            recent_phase_start = len(sorted_dates) * 3 // 4
            recent_activity = sum(activity for _, activity in sorted_dates[recent_phase_start:])
            
            content += "### Investigation Phases\n\n"
            content += f"**Early Phase Activity**: {early_activity} queries\n"
            content += f"**Recent Phase Activity**: {recent_activity} queries\n"
            
            if recent_activity > early_activity:
                content += "📈 Investigation intensity has increased over time.\n\n"
            elif recent_activity < early_activity * 0.7:
                content += "📉 Investigation activity has decreased recently.\n\n"
            else:
                content += "📊 Investigation activity has remained consistent.\n\n"
        
        return ReportSection(
            title="Investigation Progress",
            content=content.strip(),
            section_type="progress",
            confidence=0.9,
            supporting_evidence=["temporal_activity"],
            metadata={
                "total_days": total_days,
                "total_queries": total_queries,
                "discovery_rate": (entities_discovered + relationships_discovered) / total_days if total_days > 0 else 0
            }
        )

# Global instance for easy access (import after definition)
def get_report_generator():
    """Get a report generator instance with proper imports"""
    from .case_memory import case_memory
    from .enhanced_knowledge_graph import enhanced_kg_db
    return IntelligentReportGenerator(
        case_memory=case_memory,
        knowledge_graph=enhanced_kg_db
    )

# Create global instance
try:
    from .case_memory import case_memory
    from .enhanced_knowledge_graph import enhanced_kg_db
    report_generator = IntelligentReportGenerator(
        case_memory=case_memory,
        knowledge_graph=enhanced_kg_db
    )
except ImportError:
    # Create placeholder that will be initialized later
    report_generator = None