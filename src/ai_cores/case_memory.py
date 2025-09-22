"""
Case Memory System

This module implements a comprehensive case memory system that stores, analyzes,
and learns from all bot interactions within each forensic case.
"""

import json
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from collections import defaultdict, Counter
import re

logger = logging.getLogger(__name__)

@dataclass
class BotInteraction:
    """Represents a single bot interaction"""
    id: str
    case_id: str
    session_id: str
    timestamp: datetime
    user_query: str
    bot_response: str
    entities_found: List[Dict[str, Any]]
    relationships_discovered: List[Dict[str, Any]]
    evidence_sources: List[str]
    query_type: str  # forensic, general, evidence_search, etc.
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]

@dataclass
class CaseInsight:
    """Represents an insight discovered from case interactions"""
    id: str
    case_id: str
    insight_type: str  # pattern, anomaly, connection, timeline_gap, etc.
    title: str
    description: str
    evidence: List[str]
    confidence: float
    discovered_at: datetime
    related_interactions: List[str]
    priority: str  # high, medium, low

@dataclass
class CaseMemoryStats:
    """Statistics about case memory and interactions"""
    total_interactions: int
    unique_entities_mentioned: int
    common_query_patterns: List[Tuple[str, int]]
    investigation_focus_areas: List[Tuple[str, float]]
    temporal_activity: Dict[str, int]
    entity_co_occurrence: Dict[str, List[str]]

class CaseMemoryManager:
    """Manages case memory, storing and analyzing all bot interactions"""
    
    def __init__(self, db_path: str = "data/case_memory.db"):
        """Initialize case memory manager"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize SQLite database for case memory"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bot_interactions (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    user_query TEXT NOT NULL,
                    bot_response TEXT NOT NULL,
                    entities_found TEXT,  -- JSON string
                    relationships_discovered TEXT,  -- JSON string
                    evidence_sources TEXT,  -- JSON string
                    query_type TEXT NOT NULL,
                    confidence_score REAL,
                    processing_time REAL,
                    metadata TEXT,  -- JSON string
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS case_insights (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    insight_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    evidence TEXT,  -- JSON string
                    confidence REAL,
                    discovered_at TIMESTAMP NOT NULL,
                    related_interactions TEXT,  -- JSON string
                    priority TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_interactions_case_id 
                ON bot_interactions(case_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_interactions_timestamp 
                ON bot_interactions(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_insights_case_id 
                ON case_insights(case_id)
            """)

    def store_interaction(self, interaction: BotInteraction) -> bool:
        """Store a bot interaction in case memory"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO bot_interactions 
                    (id, case_id, session_id, timestamp, user_query, bot_response,
                     entities_found, relationships_discovered, evidence_sources,
                     query_type, confidence_score, processing_time, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    interaction.id,
                    interaction.case_id,
                    interaction.session_id,
                    interaction.timestamp,
                    interaction.user_query,
                    interaction.bot_response,
                    json.dumps(interaction.entities_found),
                    json.dumps(interaction.relationships_discovered),
                    json.dumps(interaction.evidence_sources),
                    interaction.query_type,
                    interaction.confidence_score,
                    interaction.processing_time,
                    json.dumps(interaction.metadata)
                ))
                
            # Analyze for insights after storing
            self._analyze_for_insights(interaction.case_id)
            return True
            
        except Exception as e:
            logger.error(f"Error storing interaction: {e}")
            return False

    def get_case_interactions(self, case_id: str, limit: Optional[int] = None) -> List[BotInteraction]:
        """Get all interactions for a specific case"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM bot_interactions 
                    WHERE case_id = ? 
                    ORDER BY timestamp DESC
                """
                if limit:
                    query += f" LIMIT {limit}"
                    
                cursor = conn.execute(query, (case_id,))
                rows = cursor.fetchall()
                
                interactions = []
                for row in rows:
                    interaction = BotInteraction(
                        id=row[0],
                        case_id=row[1],
                        session_id=row[2],
                        timestamp=datetime.fromisoformat(row[3]),
                        user_query=row[4],
                        bot_response=row[5],
                        entities_found=json.loads(row[6] or "[]"),
                        relationships_discovered=json.loads(row[7] or "[]"),
                        evidence_sources=json.loads(row[8] or "[]"),
                        query_type=row[9],
                        confidence_score=row[10] or 0.0,
                        processing_time=row[11] or 0.0,
                        metadata=json.loads(row[12] or "{}")
                    )
                    interactions.append(interaction)
                    
                return interactions
                
        except Exception as e:
            logger.error(f"Error getting case interactions: {e}")
            return []

    def get_case_memory_stats(self, case_id: str) -> CaseMemoryStats:
        """Get comprehensive statistics about case memory"""
        interactions = self.get_case_interactions(case_id)
        
        if not interactions:
            return CaseMemoryStats(
                total_interactions=0,
                unique_entities_mentioned=0,
                common_query_patterns=[],
                investigation_focus_areas=[],
                temporal_activity={},
                entity_co_occurrence={}
            )
        
        # Analyze query patterns
        query_types = Counter(i.query_type for i in interactions)
        common_patterns = query_types.most_common(10)
        
        # Analyze entities
        all_entities = []
        for interaction in interactions:
            all_entities.extend([e.get('value', '') for e in interaction.entities_found])
        unique_entities = len(set(all_entities))
        
        # Analyze focus areas based on query content
        focus_areas = self._analyze_investigation_focus(interactions)
        
        # Temporal activity
        temporal_activity = defaultdict(int)
        for interaction in interactions:
            day = interaction.timestamp.strftime('%Y-%m-%d')
            temporal_activity[day] += 1
        
        # Entity co-occurrence
        entity_cooccurrence = self._analyze_entity_cooccurrence(interactions)
        
        return CaseMemoryStats(
            total_interactions=len(interactions),
            unique_entities_mentioned=unique_entities,
            common_query_patterns=common_patterns,
            investigation_focus_areas=focus_areas,
            temporal_activity=dict(temporal_activity),
            entity_co_occurrence=entity_cooccurrence
        )

    def _analyze_investigation_focus(self, interactions: List[BotInteraction]) -> List[Tuple[str, float]]:
        """Analyze what areas the investigation has been focusing on"""
        focus_keywords = {
            'communication': ['message', 'chat', 'call', 'contact', 'communication'],
            'financial': ['money', 'payment', 'transaction', 'crypto', 'bitcoin', 'bank'],
            'timeline': ['when', 'time', 'date', 'timeline', 'sequence', 'chronological'],
            'relationships': ['relationship', 'connection', 'network', 'contact', 'know'],
            'location': ['where', 'location', 'place', 'address', 'GPS', 'coordinate'],
            'technical': ['device', 'IMEI', 'IP', 'technical', 'system', 'data'],
            'suspicious': ['suspicious', 'anomaly', 'unusual', 'weird', 'strange', 'odd']
        }
        
        focus_scores = defaultdict(float)
        total_queries = len(interactions)
        
        for interaction in interactions:
            query_lower = interaction.user_query.lower()
            for focus_area, keywords in focus_keywords.items():
                score = sum(1 for keyword in keywords if keyword in query_lower)
                if score > 0:
                    focus_scores[focus_area] += score / len(keywords)
        
        # Normalize scores
        for area in focus_scores:
            focus_scores[area] = focus_scores[area] / total_queries
        
        return sorted(focus_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    def _analyze_entity_cooccurrence(self, interactions: List[BotInteraction]) -> Dict[str, List[str]]:
        """Analyze which entities frequently appear together"""
        entity_pairs = defaultdict(list)
        
        for interaction in interactions:
            entities = [e.get('value', '') for e in interaction.entities_found if e.get('value')]
            
            # Find entities that appear together
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    if entity1 != entity2:
                        entity_pairs[entity1].append(entity2)
                        entity_pairs[entity2].append(entity1)
        
        # Keep only frequently co-occurring entities
        result = {}
        for entity, cooccurring in entity_pairs.items():
            if len(cooccurring) >= 2:  # Appeared together at least twice
                # Get unique co-occurring entities with counts
                counter = Counter(cooccurring)
                result[entity] = [entity for entity, count in counter.most_common(5)]
        
        return result

    def _analyze_for_insights(self, case_id: str):
        """Analyze recent interactions to discover new insights"""
        recent_interactions = self.get_case_interactions(case_id, limit=20)
        
        if len(recent_interactions) < 3:
            return  # Need minimum interactions for analysis
        
        insights = []
        
        # Pattern: Repeated queries about same entity
        entity_queries = defaultdict(list)
        for interaction in recent_interactions:
            for entity in interaction.entities_found:
                entity_value = entity.get('value', '')
                if entity_value:
                    entity_queries[entity_value].append(interaction)
        
        for entity, interactions in entity_queries.items():
            if len(interactions) >= 3:
                insight = CaseInsight(
                    id=f"insight_{case_id}_{datetime.now().timestamp()}",
                    case_id=case_id,
                    insight_type="repeated_focus",
                    title=f"High Interest in Entity: {entity}",
                    description=f"The entity '{entity}' has been queried {len(interactions)} times recently, indicating high investigative interest.",
                    evidence=[i.id for i in interactions],
                    confidence=0.8,
                    discovered_at=datetime.now(),
                    related_interactions=[i.id for i in interactions],
                    priority="medium"
                )
                insights.append(insight)
        
        # Pattern: Time-based query clusters
        time_clusters = self._find_temporal_clusters(recent_interactions)
        for cluster in time_clusters:
            if len(cluster) >= 3:
                insight = CaseInsight(
                    id=f"insight_cluster_{case_id}_{datetime.now().timestamp()}",
                    case_id=case_id,
                    insight_type="investigation_burst",
                    title="Intense Investigation Period",
                    description=f"High activity period with {len(cluster)} queries in short timeframe, indicating active investigation phase.",
                    evidence=[i.id for i in cluster],
                    confidence=0.7,
                    discovered_at=datetime.now(),
                    related_interactions=[i.id for i in cluster],
                    priority="low"
                )
                insights.append(insight)
        
        # Store insights
        for insight in insights:
            self.store_insight(insight)

    def _find_temporal_clusters(self, interactions: List[BotInteraction]) -> List[List[BotInteraction]]:
        """Find clusters of interactions that happened close in time"""
        if not interactions:
            return []
        
        # Sort by timestamp
        sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)
        clusters = []
        current_cluster = [sorted_interactions[0]]
        
        for i in range(1, len(sorted_interactions)):
            time_diff = sorted_interactions[i].timestamp - sorted_interactions[i-1].timestamp
            
            # If less than 30 minutes apart, add to current cluster
            if time_diff < timedelta(minutes=30):
                current_cluster.append(sorted_interactions[i])
            else:
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = [sorted_interactions[i]]
        
        # Don't forget the last cluster
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
        
        return clusters

    def store_insight(self, insight: CaseInsight) -> bool:
        """Store a discovered insight"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO case_insights 
                    (id, case_id, insight_type, title, description, evidence,
                     confidence, discovered_at, related_interactions, priority)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    insight.id,
                    insight.case_id,
                    insight.insight_type,
                    insight.title,
                    insight.description,
                    json.dumps(insight.evidence),
                    insight.confidence,
                    insight.discovered_at,
                    json.dumps(insight.related_interactions),
                    insight.priority
                ))
            return True
            
        except Exception as e:
            logger.error(f"Error storing insight: {e}")
            return False

    def get_case_insights(self, case_id: str) -> List[CaseInsight]:
        """Get all insights for a specific case"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM case_insights 
                    WHERE case_id = ? 
                    ORDER BY discovered_at DESC
                """, (case_id,))
                rows = cursor.fetchall()
                
                insights = []
                for row in rows:
                    insight = CaseInsight(
                        id=row[0],
                        case_id=row[1],
                        insight_type=row[2],
                        title=row[3],
                        description=row[4],
                        evidence=json.loads(row[5] or "[]"),
                        confidence=row[6] or 0.0,
                        discovered_at=datetime.fromisoformat(row[7]),
                        related_interactions=json.loads(row[8] or "[]"),
                        priority=row[9]
                    )
                    insights.append(insight)
                    
                return insights
                
        except Exception as e:
            logger.error(f"Error getting case insights: {e}")
            return []

    def get_investigation_summary(self, case_id: str) -> Dict[str, Any]:
        """Get a comprehensive investigation summary"""
        stats = self.get_case_memory_stats(case_id)
        insights = self.get_case_insights(case_id)
        recent_interactions = self.get_case_interactions(case_id, limit=10)
        
        return {
            "case_id": case_id,
            "generated_at": datetime.now().isoformat(),
            "statistics": asdict(stats),
            "insights": [asdict(insight) for insight in insights],
            "recent_activity": [
                {
                    "timestamp": i.timestamp.isoformat(),
                    "query": i.user_query,
                    "query_type": i.query_type,
                    "entities_found": len(i.entities_found)
                }
                for i in recent_interactions
            ],
            "investigation_health": self._calculate_investigation_health(stats, insights)
        }

    def _calculate_investigation_health(self, stats: CaseMemoryStats, insights: List[CaseInsight]) -> Dict[str, Any]:
        """Calculate investigation health metrics"""
        # Activity level
        activity_score = min(1.0, stats.total_interactions / 50)  # Normalize to 50 interactions
        
        # Diversity of investigation
        diversity_score = min(1.0, len(stats.investigation_focus_areas) / 5)  # Normalize to 5 areas
        
        # Insight discovery rate
        insight_score = min(1.0, len(insights) / 10)  # Normalize to 10 insights
        
        # Overall health (weighted average)
        overall_health = (activity_score * 0.4 + diversity_score * 0.3 + insight_score * 0.3)
        
        return {
            "overall_score": round(overall_health, 2),
            "activity_level": round(activity_score, 2),
            "investigation_diversity": round(diversity_score, 2),
            "insight_discovery": round(insight_score, 2),
            "recommendations": self._generate_recommendations(stats, insights)
        }

    def _generate_recommendations(self, stats: CaseMemoryStats, insights: List[CaseInsight]) -> List[str]:
        """Generate investigation recommendations based on memory analysis"""
        recommendations = []
        
        if stats.total_interactions < 10:
            recommendations.append("Consider conducting more comprehensive evidence analysis to gather additional insights.")
        
        if len(stats.investigation_focus_areas) < 3:
            recommendations.append("Expand investigation scope to cover more areas (communication, financial, timeline, etc.).")
        
        if stats.unique_entities_mentioned < 5:
            recommendations.append("Look for additional entities and connections in the evidence.")
        
        # Check for co-occurring entities
        if len(stats.entity_co_occurrence) > 0:
            recommendations.append("Investigate relationships between frequently co-occurring entities.")
        
        # Check for high-priority insights
        high_priority_insights = [i for i in insights if i.priority == "high"]
        if high_priority_insights:
            recommendations.append(f"Address {len(high_priority_insights)} high-priority insights discovered during investigation.")
        
        if not recommendations:
            recommendations.append("Investigation appears comprehensive. Consider reviewing findings and preparing final report.")
        
        return recommendations

    def clear_case_memory(self, case_id: str) -> bool:
        """Clear all memory for a specific case"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM bot_interactions WHERE case_id = ?", (case_id,))
                conn.execute("DELETE FROM case_insights WHERE case_id = ?", (case_id,))
            return True
            
        except Exception as e:
            logger.error(f"Error clearing case memory: {e}")
            return False

# Global instance
case_memory = CaseMemoryManager()