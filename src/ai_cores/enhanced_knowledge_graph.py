"""
Enhanced Knowledge Graph Database Layer

Integrates knowledge graph functionality with case memory and provides
intelligent entity relationship storage and querying capabilities.
"""

import json
import sqlite3
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from pathlib import Path
import logging
from collections import defaultdict
import networkx as nx
import uuid

from .knowledge_graph.entity_extractor import ForensicEntity, ForensicEntityExtractor
from .knowledge_graph.relationship_extractor import ForensicRelationship, ForensicRelationshipExtractor
from .knowledge_graph.graph_store import BaseGraphStore
from .case_memory import BotInteraction, case_memory

logger = logging.getLogger(__name__)

class EnhancedKnowledgeGraphDB:
    """Enhanced knowledge graph database with case memory integration"""
    
    def __init__(self, db_path: str = "data/enhanced_knowledge_graph.db"):
        """Initialize enhanced knowledge graph database"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.entity_extractor = ForensicEntityExtractor()
        self.relationship_extractor = ForensicRelationshipExtractor()
        self._initialize_database()
    
    def _get_connection(self):
        """Get a properly configured database connection"""
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")  # 30 seconds
        return conn
        
    def _initialize_database(self):
        """Initialize comprehensive knowledge graph database"""
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA cache_size=10000;")
            conn.execute("PRAGMA temp_store=memory;")
            
            # Enhanced entities table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kg_entities (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    context TEXT,
                    metadata TEXT,  -- JSON
                    source_document TEXT,
                    first_seen TIMESTAMP NOT NULL,
                    last_seen TIMESTAMP NOT NULL,
                    mention_count INTEGER DEFAULT 1,
                    importance_score REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Enhanced relationships table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kg_relationships (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    entity1_id TEXT NOT NULL,
                    entity2_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    context TEXT,
                    evidence TEXT,  -- JSON list of evidence
                    metadata TEXT,  -- JSON
                    first_seen TIMESTAMP NOT NULL,
                    last_seen TIMESTAMP NOT NULL,
                    strength REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (entity1_id) REFERENCES kg_entities (id),
                    FOREIGN KEY (entity2_id) REFERENCES kg_entities (id)
                )
            """)
            
            # Entity interactions tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_interactions (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    interaction_id TEXT NOT NULL,
                    query_context TEXT,
                    mentioned_at TIMESTAMP NOT NULL,
                    relevance_score REAL DEFAULT 0.5,
                    FOREIGN KEY (entity_id) REFERENCES kg_entities (id)
                )
            """)
            
            # Case knowledge summaries
            conn.execute("""
                CREATE TABLE IF NOT EXISTS case_knowledge_summaries (
                    case_id TEXT PRIMARY KEY,
                    total_entities INTEGER DEFAULT 0,
                    total_relationships INTEGER DEFAULT 0,
                    key_entities TEXT,  -- JSON list
                    important_relationships TEXT,  -- JSON list
                    knowledge_density REAL DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    summary_data TEXT  -- JSON comprehensive summary
                )
            """)
            
            # Create indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_kg_entities_case_id ON kg_entities(case_id)",
                "CREATE INDEX IF NOT EXISTS idx_kg_entities_type ON kg_entities(entity_type)",
                "CREATE INDEX IF NOT EXISTS idx_kg_entities_value ON kg_entities(entity_value)",
                "CREATE INDEX IF NOT EXISTS idx_kg_relationships_case_id ON kg_relationships(case_id)",
                "CREATE INDEX IF NOT EXISTS idx_kg_relationships_entities ON kg_relationships(entity1_id, entity2_id)",
                "CREATE INDEX IF NOT EXISTS idx_entity_interactions_case_id ON entity_interactions(case_id)",
                "CREATE INDEX IF NOT EXISTS idx_entity_interactions_entity_id ON entity_interactions(entity_id)"
            ]
            
            for index in indexes:
                conn.execute(index)

    def process_interaction_for_knowledge(self, interaction: BotInteraction) -> Dict[str, Any]:
        """Process a bot interaction to extract and update knowledge graph"""
        try:
            # Extract entities from both query and response
            query_entities = self._extract_entities_from_text(
                interaction.user_query, 
                interaction.case_id,
                f"query_{interaction.id}"
            )
            
            response_entities = self._extract_entities_from_text(
                interaction.bot_response,
                interaction.case_id, 
                f"response_{interaction.id}"
            )
            
            all_entities = query_entities + response_entities
            
            # Store/update entities
            stored_entities = []
            for entity in all_entities:
                stored_entity = self._store_or_update_entity(entity, interaction)
                if stored_entity:
                    stored_entities.append(stored_entity)
            
            # Extract and store relationships
            relationships = self._extract_relationships_from_interaction(
                interaction, stored_entities
            )
            
            stored_relationships = []
            for relationship in relationships:
                stored_rel = self._store_or_update_relationship(relationship, interaction)
                if stored_rel:
                    stored_relationships.append(stored_rel)
            
            # Update case knowledge summary
            self._update_case_knowledge_summary(interaction.case_id)
            
            return {
                "entities_processed": len(stored_entities),
                "relationships_processed": len(stored_relationships),
                "entities": [self._entity_to_dict(e) for e in stored_entities],
                "relationships": [self._relationship_to_dict(r) for r in stored_relationships]
            }
            
        except Exception as e:
            logger.error(f"Error processing interaction for knowledge: {e}")
            return {"error": str(e)}

    def _extract_entities_from_text(self, text: str, case_id: str, source_id: str) -> List[ForensicEntity]:
        """Extract entities from text using our enhanced extractor"""
        # Create a simple document structure for the extractor
        document = {
            "text_content": text,
            "source_id": source_id,
            "case_id": case_id
        }
        
        return self.entity_extractor.extract_entities_from_document(document, source_id)

    def _store_or_update_entity(self, entity: ForensicEntity, interaction: BotInteraction) -> Optional[Dict[str, Any]]:
        """Store a new entity or update existing one"""
        try:
            with self._get_connection() as conn:
                # Check if entity already exists
                cursor = conn.execute("""
                    SELECT id, mention_count, importance_score, last_seen 
                    FROM kg_entities 
                    WHERE case_id = ? AND entity_value = ? AND entity_type = ?
                """, (interaction.case_id, entity.value, entity.type))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing entity
                    entity_id, mention_count, importance_score, last_seen = existing
                    new_mention_count = mention_count + 1
                    new_importance = self._calculate_entity_importance(
                        entity, new_mention_count, importance_score
                    )
                    
                    conn.execute("""
                        UPDATE kg_entities 
                        SET mention_count = ?, importance_score = ?, last_seen = ?, updated_at = ?
                        WHERE id = ?
                    """, (new_mention_count, new_importance, interaction.timestamp, datetime.now(), entity_id))
                    
                    entity.id = entity_id
                    
                else:
                    # Insert new entity
                    entity_id = str(uuid.uuid4())
                    entity.id = entity_id
                    importance_score = self._calculate_entity_importance(entity, 1, 0.5)
                    
                    conn.execute("""
                        INSERT INTO kg_entities 
                        (id, case_id, entity_type, entity_value, confidence, context, 
                         metadata, source_document, first_seen, last_seen, mention_count, importance_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entity_id, interaction.case_id, entity.type, entity.value,
                        entity.confidence, entity.context, json.dumps(entity.metadata),
                        entity.source_document, entity.first_seen, entity.last_seen,
                        1, importance_score
                    ))
                
                # Track entity interaction
                self._track_entity_interaction(entity_id, interaction)
                
                return self._entity_to_dict(entity)
                
        except Exception as e:
            logger.error(f"Error storing entity: {e}")
            return None

    def _calculate_entity_importance(self, entity: ForensicEntity, mention_count: int, current_score: float) -> float:
        """Calculate entity importance based on various factors"""
        # Base importance from entity type
        type_importance = {
            'person': 0.9,
            'phone': 0.8,
            'email': 0.7,
            'crypto_address': 0.9,
            'organization': 0.8,
            'location': 0.6,
            'ip_address': 0.7,
            'imei': 0.8
        }
        
        base_score = type_importance.get(entity.type, 0.5)
        
        # Frequency boost
        frequency_boost = min(0.3, mention_count * 0.05)
        
        # Confidence boost
        confidence_boost = (entity.confidence - 0.5) * 0.2
        
        # Calculate new importance
        new_score = base_score + frequency_boost + confidence_boost
        
        # Smooth update with existing score
        final_score = (current_score * 0.3) + (new_score * 0.7)
        
        return min(1.0, max(0.0, final_score))

    def _track_entity_interaction(self, entity_id: str, interaction: BotInteraction):
        """Track when an entity is mentioned in an interaction"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with self._get_connection() as conn:
                    interaction_track_id = str(uuid.uuid4())
                    conn.execute("""
                        INSERT INTO entity_interactions 
                        (id, case_id, entity_id, interaction_id, query_context, mentioned_at, relevance_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        interaction_track_id, interaction.case_id, entity_id, interaction.id,
                        interaction.user_query, interaction.timestamp, interaction.confidence_score
                    ))
                    conn.commit()
                    break  # Success, exit retry loop
                    
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    import time
                    wait_time = 0.1 * (2 ** attempt)  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Error tracking entity interaction after {attempt + 1} attempts: {e}")
                    break
            except Exception as e:
                logger.error(f"Error tracking entity interaction: {e}")
                break

    def _extract_relationships_from_interaction(self, interaction: BotInteraction, entities: List[Dict[str, Any]]) -> List[ForensicRelationship]:
        """Extract relationships from interaction context"""
        relationships = []
        
        if len(entities) < 2:
            return relationships
        
        # Create entity pairs for relationship analysis
        entity_values = [e['value'] for e in entities]
        
        # Use relationship extractor with interaction context
        mock_documents = [{
            "text_content": interaction.user_query + " " + interaction.bot_response,
            "entities": entity_values,
            "timestamp": interaction.timestamp,
            "source": f"interaction_{interaction.id}"
        }]
        
        # Convert our entities to the format expected by relationship extractor
        forensic_entities = []
        for entity_dict in entities:
            forensic_entity = ForensicEntity(
                id=entity_dict['id'],
                type=entity_dict['type'],
                value=entity_dict['value'],
                confidence=entity_dict['confidence'],
                context=entity_dict.get('context', ''),
                metadata=entity_dict.get('metadata', {}),
                source_document=f"interaction_{interaction.id}",
                first_seen=interaction.timestamp,
                last_seen=interaction.timestamp
            )
            forensic_entities.append(forensic_entity)
        
        # Extract relationships
        try:
            relationships = self.relationship_extractor.extract_relationships(
                forensic_entities, mock_documents
            )
        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
            relationships = []
        
        return relationships

    def _store_or_update_relationship(self, relationship: ForensicRelationship, interaction: BotInteraction) -> Optional[Dict[str, Any]]:
        """Store a new relationship or update existing one"""
        try:
            with self._get_connection() as conn:
                # Check if relationship already exists
                cursor = conn.execute("""
                    SELECT id, strength, evidence, last_seen 
                    FROM kg_relationships 
                    WHERE case_id = ? AND entity1_id = ? AND entity2_id = ? AND relationship_type = ?
                """, (
                    interaction.case_id, relationship.source_entity_id, 
                    relationship.target_entity_id, relationship.relationship_type
                ))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing relationship
                    rel_id, strength, evidence_json, last_seen = existing
                    evidence = json.loads(evidence_json or "[]")
                    evidence.append({
                        "interaction_id": interaction.id,
                        "timestamp": interaction.timestamp.isoformat(),
                        "context": str(relationship.metadata)
                    })
                    
                    new_strength = min(1.0, strength + 0.1)  # Increase strength
                    
                    conn.execute("""
                        UPDATE kg_relationships 
                        SET strength = ?, evidence = ?, last_seen = ?, updated_at = ?
                        WHERE id = ?
                    """, (new_strength, json.dumps(evidence), interaction.timestamp, datetime.now(), rel_id))
                    
                    relationship.id = rel_id
                    
                else:
                    # Insert new relationship
                    rel_id = str(uuid.uuid4())
                    relationship.id = rel_id
                    evidence = [{
                        "interaction_id": interaction.id,
                        "timestamp": interaction.timestamp.isoformat(),
                        "context": str(relationship.metadata)
                    }]
                    
                    conn.execute("""
                        INSERT INTO kg_relationships 
                        (id, case_id, entity1_id, entity2_id, relationship_type, confidence,
                         context, evidence, metadata, first_seen, last_seen, strength)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        rel_id, interaction.case_id, relationship.source_entity_id, relationship.target_entity_id,
                        relationship.relationship_type, relationship.confidence, str(relationship.metadata),
                        json.dumps(evidence), json.dumps(relationship.metadata),
                        relationship.first_observed, relationship.last_observed, 0.5
                    ))
                
                return self._relationship_to_dict(relationship)
                
        except Exception as e:
            logger.error(f"Error storing relationship: {e}")
            return None

    def _update_case_knowledge_summary(self, case_id: str):
        """Update comprehensive knowledge summary for a case"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get entity and relationship counts
                entity_count = conn.execute(
                    "SELECT COUNT(*) FROM kg_entities WHERE case_id = ?", (case_id,)
                ).fetchone()[0]
                
                relationship_count = conn.execute(
                    "SELECT COUNT(*) FROM kg_relationships WHERE case_id = ?", (case_id,)
                ).fetchone()[0]
                
                # Get key entities (top 10 by importance)
                key_entities_cursor = conn.execute("""
                    SELECT entity_type, entity_value, importance_score, mention_count
                    FROM kg_entities 
                    WHERE case_id = ? 
                    ORDER BY importance_score DESC, mention_count DESC 
                    LIMIT 10
                """, (case_id,))
                
                key_entities = [
                    {
                        "type": row[0],
                        "value": row[1], 
                        "importance": row[2],
                        "mentions": row[3]
                    }
                    for row in key_entities_cursor.fetchall()
                ]
                
                # Get important relationships (top 10 by strength)
                important_relationships_cursor = conn.execute("""
                    SELECT r.relationship_type, e1.entity_value, e2.entity_value, r.strength, r.confidence
                    FROM kg_relationships r
                    JOIN kg_entities e1 ON r.entity1_id = e1.id
                    JOIN kg_entities e2 ON r.entity2_id = e2.id
                    WHERE r.case_id = ?
                    ORDER BY r.strength DESC, r.confidence DESC
                    LIMIT 10
                """, (case_id,))
                
                important_relationships = [
                    {
                        "type": row[0],
                        "entity1": row[1],
                        "entity2": row[2],
                        "strength": row[3],
                        "confidence": row[4]
                    }
                    for row in important_relationships_cursor.fetchall()
                ]
                
                # Calculate knowledge density
                max_possible_relationships = entity_count * (entity_count - 1) / 2 if entity_count > 1 else 1
                knowledge_density = relationship_count / max_possible_relationships if max_possible_relationships > 0 else 0.0
                
                # Create comprehensive summary
                summary_data = {
                    "entity_breakdown": self._get_entity_type_breakdown(case_id),
                    "relationship_breakdown": self._get_relationship_type_breakdown(case_id),
                    "temporal_analysis": self._get_temporal_knowledge_analysis(case_id),
                    "network_metrics": self._calculate_network_metrics(case_id),
                    "last_updated": datetime.now().isoformat()
                }
                
                # Store or update summary
                conn.execute("""
                    INSERT OR REPLACE INTO case_knowledge_summaries
                    (case_id, total_entities, total_relationships, key_entities,
                     important_relationships, knowledge_density, last_updated, summary_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    case_id, entity_count, relationship_count,
                    json.dumps(key_entities), json.dumps(important_relationships),
                    knowledge_density, datetime.now(), json.dumps(summary_data)
                ))
                
        except Exception as e:
            logger.error(f"Error updating case knowledge summary: {e}")

    def _get_entity_type_breakdown(self, case_id: str) -> Dict[str, int]:
        """Get breakdown of entities by type"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT entity_type, COUNT(*) 
                    FROM kg_entities 
                    WHERE case_id = ? 
                    GROUP BY entity_type
                """, (case_id,))
                
                return dict(cursor.fetchall())
        except Exception as e:
            logger.error(f"Error getting entity breakdown: {e}")
            return {}

    def _get_relationship_type_breakdown(self, case_id: str) -> Dict[str, int]:
        """Get breakdown of relationships by type"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT relationship_type, COUNT(*) 
                    FROM kg_relationships 
                    WHERE case_id = ? 
                    GROUP BY relationship_type
                """, (case_id,))
                
                return dict(cursor.fetchall())
        except Exception as e:
            logger.error(f"Error getting relationship breakdown: {e}")
            return {}

    def _get_temporal_knowledge_analysis(self, case_id: str) -> Dict[str, Any]:
        """Analyze knowledge discovery over time"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Entity discovery timeline
                entity_timeline = conn.execute("""
                    SELECT DATE(first_seen) as date, COUNT(*) as count
                    FROM kg_entities 
                    WHERE case_id = ?
                    GROUP BY DATE(first_seen)
                    ORDER BY date
                """, (case_id,)).fetchall()
                
                # Relationship discovery timeline
                relationship_timeline = conn.execute("""
                    SELECT DATE(first_seen) as date, COUNT(*) as count
                    FROM kg_relationships 
                    WHERE case_id = ?
                    GROUP BY DATE(first_seen)
                    ORDER BY date
                """, (case_id,)).fetchall()
                
                return {
                    "entity_timeline": [{"date": row[0], "count": row[1]} for row in entity_timeline],
                    "relationship_timeline": [{"date": row[0], "count": row[1]} for row in relationship_timeline]
                }
        except Exception as e:
            logger.error(f"Error getting temporal analysis: {e}")
            return {}

    def _calculate_network_metrics(self, case_id: str) -> Dict[str, Any]:
        """Calculate network analysis metrics"""
        try:
            # Create NetworkX graph for analysis
            graph = nx.Graph()
            
            with sqlite3.connect(self.db_path) as conn:
                # Add nodes (entities)
                entities = conn.execute("""
                    SELECT id, entity_value, importance_score 
                    FROM kg_entities 
                    WHERE case_id = ?
                """, (case_id,)).fetchall()
                
                for entity_id, value, importance in entities:
                    graph.add_node(entity_id, value=value, importance=importance)
                
                # Add edges (relationships)
                relationships = conn.execute("""
                    SELECT entity1_id, entity2_id, strength 
                    FROM kg_relationships 
                    WHERE case_id = ?
                """, (case_id,)).fetchall()
                
                for entity1, entity2, strength in relationships:
                    graph.add_edge(entity1, entity2, weight=strength)
            
            if len(graph.nodes()) == 0:
                return {"message": "No entities found for network analysis"}
            
            # Calculate network metrics
            metrics = {}
            
            # Basic metrics
            metrics["node_count"] = len(graph.nodes())
            metrics["edge_count"] = len(graph.edges())
            metrics["density"] = nx.density(graph)
            
            # Connected components
            components = list(nx.connected_components(graph))
            metrics["connected_components"] = len(components)
            metrics["largest_component_size"] = len(max(components, key=len)) if components else 0
            
            # Centrality measures (for largest component)
            if len(graph.nodes()) > 1:
                largest_component = max(components, key=len) if components else set()
                if len(largest_component) > 1:
                    subgraph = graph.subgraph(largest_component)
                    
                    # Degree centrality
                    degree_centrality = nx.degree_centrality(subgraph)
                    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    # Betweenness centrality
                    betweenness_centrality = nx.betweenness_centrality(subgraph)
                    top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    metrics["top_degree_centrality"] = [
                        {"entity_id": node, "centrality": score} for node, score in top_degree
                    ]
                    metrics["top_betweenness_centrality"] = [
                        {"entity_id": node, "centrality": score} for node, score in top_betweenness
                    ]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating network metrics: {e}")
            return {"error": str(e)}

    def get_case_knowledge_graph(self, case_id: str) -> Dict[str, Any]:
        """Get complete knowledge graph for a case"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get entities
                entities = conn.execute("""
                    SELECT id, entity_type, entity_value, confidence, context, 
                           mention_count, importance_score, first_seen, last_seen
                    FROM kg_entities 
                    WHERE case_id = ?
                    ORDER BY importance_score DESC
                """, (case_id,)).fetchall()
                
                # Get relationships
                relationships = conn.execute("""
                    SELECT r.id, r.entity1_id, r.entity2_id, r.relationship_type,
                           r.confidence, r.context, r.strength, r.first_seen, r.last_seen,
                           e1.entity_value as entity1_value, e2.entity_value as entity2_value
                    FROM kg_relationships r
                    JOIN kg_entities e1 ON r.entity1_id = e1.id
                    JOIN kg_entities e2 ON r.entity2_id = e2.id
                    WHERE r.case_id = ?
                    ORDER BY r.strength DESC
                """, (case_id,)).fetchall()
                
                # Get case summary
                summary = conn.execute("""
                    SELECT * FROM case_knowledge_summaries WHERE case_id = ?
                """, (case_id,)).fetchone()
                
                return {
                    "case_id": case_id,
                    "entities": [
                        {
                            "id": row[0],
                            "type": row[1],
                            "value": row[2],
                            "confidence": row[3],
                            "context": row[4],
                            "mention_count": row[5],
                            "importance_score": row[6],
                            "first_seen": row[7],
                            "last_seen": row[8]
                        }
                        for row in entities
                    ],
                    "relationships": [
                        {
                            "id": row[0],
                            "entity1_id": row[1],
                            "entity2_id": row[2],
                            "relationship_type": row[3],
                            "confidence": row[4],
                            "context": row[5],
                            "strength": row[6],
                            "first_seen": row[7],
                            "last_seen": row[8],
                            "entity1_value": row[9],
                            "entity2_value": row[10]
                        }
                        for row in relationships
                    ],
                    "summary": {
                        "total_entities": summary[1] if summary else 0,
                        "total_relationships": summary[2] if summary else 0,
                        "key_entities": json.loads(summary[3]) if summary and summary[3] else [],
                        "important_relationships": json.loads(summary[4]) if summary and summary[4] else [],
                        "knowledge_density": summary[5] if summary else 0.0,
                        "detailed_summary": json.loads(summary[7]) if summary and summary[7] else {}
                    } if summary else None
                }
                
        except Exception as e:
            logger.error(f"Error getting case knowledge graph: {e}")
            return {"error": str(e)}

    def get_case_entities(self, case_id: str) -> List[Dict[str, Any]]:
        """Get all entities for a specific case"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, entity_value, entity_type, context, 
                           importance_score, confidence, metadata, created_at, updated_at
                    FROM kg_entities 
                    WHERE case_id = ?
                    ORDER BY importance_score DESC, entity_value
                """, (case_id,))
                
                entities = []
                for row in cursor.fetchall():
                    entities.append({
                        "entity_id": row[0],
                        "entity_name": row[1],
                        "entity_type": row[2],
                        "description": row[3],
                        "importance_score": row[4],
                        "confidence": row[5],
                        "metadata": json.loads(row[6]) if row[6] else {},
                        "created_at": row[7],
                        "updated_at": row[8]
                    })
                
                return entities
                
        except Exception as e:
            logger.error(f"Error getting case entities: {e}")
            return []

    def get_case_relationships(self, case_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for a specific case"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, entity1_id, entity2_id, 
                           relationship_type, context, confidence, metadata, 
                           created_at, updated_at
                    FROM kg_relationships 
                    WHERE case_id = ?
                    ORDER BY confidence DESC, relationship_type
                """, (case_id,))
                
                relationships = []
                for row in cursor.fetchall():
                    relationships.append({
                        "relationship_id": row[0],
                        "source_entity_id": row[1],
                        "target_entity_id": row[2],
                        "relationship_type": row[3],
                        "description": row[4],
                        "confidence": row[5],
                        "metadata": json.loads(row[6]) if row[6] else {},
                        "created_at": row[7],
                        "updated_at": row[8]
                    })
                
                return relationships
                
        except Exception as e:
            logger.error(f"Error getting case relationships: {e}")
            return []

    def query_knowledge_graph(self, case_id: str, query_type: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Query the knowledge graph with various query types"""
        parameters = parameters or {}
        
        try:
            if query_type == "find_entity":
                return self._query_find_entity(case_id, parameters)
            elif query_type == "find_relationships":
                return self._query_find_relationships(case_id, parameters)
            elif query_type == "entity_network":
                return self._query_entity_network(case_id, parameters)
            elif query_type == "temporal_analysis":
                return self._query_temporal_analysis(case_id, parameters)
            elif query_type == "importance_ranking":
                return self._query_importance_ranking(case_id, parameters)
            else:
                return [{"error": f"Unknown query type: {query_type}"}]
                
        except Exception as e:
            logger.error(f"Error querying knowledge graph: {e}")
            return [{"error": str(e)}]

    def _query_find_entity(self, case_id: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find entities matching criteria"""
        entity_value = parameters.get("value", "")
        entity_type = parameters.get("type", "")
        
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM kg_entities WHERE case_id = ?"
            params = [case_id]
            
            if entity_value:
                query += " AND entity_value LIKE ?"
                params.append(f"%{entity_value}%")
            
            if entity_type:
                query += " AND entity_type = ?"
                params.append(entity_type)
            
            query += " ORDER BY importance_score DESC"
            
            results = conn.execute(query, params).fetchall()
            
            return [self._row_to_entity_dict(row) for row in results]

    def _query_find_relationships(self, case_id: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find relationships involving specific entities"""
        entity_value = parameters.get("entity_value", "")
        relationship_type = parameters.get("relationship_type", "")
        
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT r.*, e1.entity_value as entity1_value, e2.entity_value as entity2_value
                FROM kg_relationships r
                JOIN kg_entities e1 ON r.entity1_id = e1.id
                JOIN kg_entities e2 ON r.entity2_id = e2.id
                WHERE r.case_id = ?
            """
            params = [case_id]
            
            if entity_value:
                query += " AND (e1.entity_value LIKE ? OR e2.entity_value LIKE ?)"
                params.extend([f"%{entity_value}%", f"%{entity_value}%"])
            
            if relationship_type:
                query += " AND r.relationship_type = ?"
                params.append(relationship_type)
            
            query += " ORDER BY r.strength DESC"
            
            results = conn.execute(query, params).fetchall()
            
            return [self._row_to_relationship_dict(row) for row in results]

    def _entity_to_dict(self, entity: ForensicEntity) -> Dict[str, Any]:
        """Convert ForensicEntity to dictionary"""
        return {
            "id": entity.id,
            "type": entity.type,
            "value": entity.value,
            "confidence": entity.confidence,
            "context": entity.context,
            "metadata": entity.metadata,
            "source_document": entity.source_document,
            "first_seen": entity.first_seen.isoformat() if entity.first_seen else None,
            "last_seen": entity.last_seen.isoformat() if entity.last_seen else None
        }

    def _relationship_to_dict(self, relationship: ForensicRelationship) -> Dict[str, Any]:
        """Convert ForensicRelationship to dictionary"""
        return {
            "id": relationship.id,
            "source_entity_id": relationship.source_entity_id,
            "target_entity_id": relationship.target_entity_id,
            "relationship_type": relationship.relationship_type,
            "confidence": relationship.confidence,
            "evidence": relationship.evidence,
            "metadata": relationship.metadata,
            "first_observed": relationship.first_observed.isoformat() if relationship.first_observed else None,
            "last_observed": relationship.last_observed.isoformat() if relationship.last_observed else None,
            "frequency": relationship.frequency
        }

    def _row_to_entity_dict(self, row) -> Dict[str, Any]:
        """Convert database row to entity dictionary"""
        return {
            "id": row[0],
            "case_id": row[1],
            "type": row[2],
            "value": row[3],
            "confidence": row[4],
            "context": row[5],
            "metadata": json.loads(row[6] or "{}"),
            "source_document": row[7],
            "first_seen": row[8],
            "last_seen": row[9],
            "mention_count": row[10],
            "importance_score": row[11]
        }

    def _row_to_relationship_dict(self, row) -> Dict[str, Any]:
        """Convert database row to relationship dictionary"""
        return {
            "id": row[0],
            "case_id": row[1],
            "entity1_id": row[2],
            "entity2_id": row[3],
            "relationship_type": row[4],
            "confidence": row[5],
            "context": row[6],
            "evidence": json.loads(row[7] or "[]"),
            "metadata": json.loads(row[8] or "{}"),
            "first_seen": row[9],
            "last_seen": row[10],
            "strength": row[11],
            "entity1_value": row[14] if len(row) > 14 else "",
            "entity2_value": row[15] if len(row) > 15 else ""
        }

# Global instance
enhanced_kg_db = EnhancedKnowledgeGraphDB()