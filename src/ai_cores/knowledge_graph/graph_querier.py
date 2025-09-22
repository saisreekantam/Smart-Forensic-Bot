"""
Graph Querier for Forensic Knowledge Graph

Provides high-level querying capabilities for the forensic knowledge graph.
"""

from typing import List, Dict, Any, Optional, Set
import logging
from datetime import datetime

from .graph_store import BaseGraphStore
from .entity_extractor import ForensicEntity
from .relationship_extractor import ForensicRelationship

logger = logging.getLogger(__name__)

class ForensicGraphQuerier:
    """High-level querying interface for forensic knowledge graphs"""
    
    def __init__(self, graph_store: BaseGraphStore):
        """Initialize the graph querier"""
        self.graph_store = graph_store
        
    def find_communications(self, entity_value: str) -> List[Dict[str, Any]]:
        """Find all communications involving a specific entity (phone, email, etc.)"""
        entity = self._find_entity_by_value(entity_value)
        if not entity:
            return []
        
        relationships = self.graph_store.get_relationships(entity.id)
        communications = []
        
        for rel in relationships:
            if rel.relationship_type == "communicates_with":
                other_entity_id = rel.target_entity_id if rel.source_entity_id == entity.id else rel.source_entity_id
                other_entity = self.graph_store.get_entity(other_entity_id)
                
                if other_entity:
                    communications.append({
                        "participant": other_entity,
                        "relationship": rel,
                        "direction": "outgoing" if rel.source_entity_id == entity.id else "incoming"
                    })
        
        return communications
    
    def find_ownership_network(self, person_name: str) -> Dict[str, Any]:
        """Find all devices/accounts owned by a person"""
        person_entity = self._find_entity_by_value(person_name, "person")
        if not person_entity:
            return {"error": f"Person '{person_name}' not found"}
        
        relationships = self.graph_store.get_relationships(person_entity.id)
        owned_items = []
        
        for rel in relationships:
            if rel.relationship_type == "owns" and rel.source_entity_id == person_entity.id:
                owned_entity = self.graph_store.get_entity(rel.target_entity_id)
                if owned_entity:
                    owned_items.append({
                        "item": owned_entity,
                        "relationship": rel
                    })
        
        return {
            "person": person_entity,
            "owned_items": owned_items,
            "total_owned": len(owned_items)
        }
    
    def find_crypto_network(self) -> List[Dict[str, Any]]:
        """Find all cryptocurrency-related entities and their connections"""
        crypto_entities = self.graph_store.get_entities_by_type("crypto_address")
        crypto_network = []
        
        for crypto_entity in crypto_entities:
            relationships = self.graph_store.get_relationships(crypto_entity.id)
            connections = []
            
            for rel in relationships:
                other_entity_id = rel.target_entity_id if rel.source_entity_id == crypto_entity.id else rel.source_entity_id
                other_entity = self.graph_store.get_entity(other_entity_id)
                
                if other_entity:
                    connections.append({
                        "entity": other_entity,
                        "relationship": rel
                    })
            
            crypto_network.append({
                "crypto_address": crypto_entity,
                "connections": connections,
                "connection_count": len(connections)
            })
        
        return crypto_network
    
    def find_communication_patterns(self, time_window_hours: int = 24) -> List[Dict[str, Any]]:
        """Analyze communication patterns within a time window"""
        communication_rels = []
        
        # Get all communication relationships
        stats = self.graph_store.get_statistics()
        for entity_type in stats.get("entity_types", {}):
            entities = self.graph_store.get_entities_by_type(entity_type)
            for entity in entities:
                relationships = self.graph_store.get_relationships(entity.id)
                communication_rels.extend([r for r in relationships if r.relationship_type == "communicates_with"])
        
        # Group by frequency and analyze patterns
        communication_patterns = []
        entity_pairs = {}
        
        for rel in communication_rels:
            pair_key = tuple(sorted([rel.source_entity_id, rel.target_entity_id]))
            if pair_key not in entity_pairs:
                entity_pairs[pair_key] = []
            entity_pairs[pair_key].append(rel)
        
        for pair, rels in entity_pairs.items():
            if len(rels) > 1:  # Multiple communications
                entity1 = self.graph_store.get_entity(pair[0])
                entity2 = self.graph_store.get_entity(pair[1])
                
                communication_patterns.append({
                    "entity1": entity1,
                    "entity2": entity2,
                    "frequency": len(rels),
                    "relationships": rels,
                    "pattern_type": "frequent_communication"
                })
        
        return sorted(communication_patterns, key=lambda x: x["frequency"], reverse=True)
    
    def find_suspicious_patterns(self) -> List[Dict[str, Any]]:
        """Find potentially suspicious patterns in the data"""
        suspicious_patterns = []
        
        # Pattern 1: Crypto addresses with multiple connections
        crypto_entities = self.graph_store.get_entities_by_type("crypto_address")
        for crypto in crypto_entities:
            relationships = self.graph_store.get_relationships(crypto.id)
            if len(relationships) >= 3:  # Highly connected crypto address
                suspicious_patterns.append({
                    "pattern_type": "highly_connected_crypto",
                    "entity": crypto,
                    "connection_count": len(relationships),
                    "suspicion_level": "medium" if len(relationships) < 5 else "high"
                })
        
        # Pattern 2: Entities with many different relationship types
        stats = self.graph_store.get_statistics()
        for entity_type in stats.get("entity_types", {}):
            entities = self.graph_store.get_entities_by_type(entity_type)
            for entity in entities:
                relationships = self.graph_store.get_relationships(entity.id)
                rel_types = set(r.relationship_type for r in relationships)
                
                if len(rel_types) >= 3:  # Multiple relationship types
                    suspicious_patterns.append({
                        "pattern_type": "diverse_relationships",
                        "entity": entity,
                        "relationship_types": list(rel_types),
                        "suspicion_level": "low"
                    })
        
        # Pattern 3: High-frequency communication patterns
        comm_patterns = self.find_communication_patterns()
        for pattern in comm_patterns[:3]:  # Top 3 most frequent
            if pattern["frequency"] >= 5:
                suspicious_patterns.append({
                    "pattern_type": "high_frequency_communication",
                    "entities": [pattern["entity1"], pattern["entity2"]],
                    "frequency": pattern["frequency"],
                    "suspicion_level": "medium"
                })
        
        return suspicious_patterns
    
    def trace_entity_path(self, start_entity_value: str, end_entity_value: str, max_hops: int = 3) -> List[List[Dict[str, Any]]]:
        """Find paths between two entities in the graph"""
        start_entity = self._find_entity_by_value(start_entity_value)
        end_entity = self._find_entity_by_value(end_entity_value)
        
        if not start_entity or not end_entity:
            return []
        
        paths = []
        visited = set()
        
        def dfs_path(current_entity_id: str, target_entity_id: str, current_path: List[Dict[str, Any]], depth: int):
            if depth > max_hops or current_entity_id in visited:
                return
            
            if current_entity_id == target_entity_id:
                paths.append(current_path.copy())
                return
            
            visited.add(current_entity_id)
            relationships = self.graph_store.get_relationships(current_entity_id)
            
            for rel in relationships:
                next_entity_id = rel.target_entity_id if rel.source_entity_id == current_entity_id else rel.source_entity_id
                next_entity = self.graph_store.get_entity(next_entity_id)
                
                if next_entity and next_entity_id not in visited:
                    current_path.append({
                        "entity": next_entity,
                        "relationship": rel,
                        "hop": depth + 1
                    })
                    dfs_path(next_entity_id, target_entity_id, current_path, depth + 1)
                    current_path.pop()
            
            visited.remove(current_entity_id)
        
        dfs_path(start_entity.id, end_entity.id, [{"entity": start_entity, "relationship": None, "hop": 0}], 0)
        
        return paths
    
    def get_entity_timeline(self, entity_value: str) -> Dict[str, Any]:
        """Get timeline of activities for an entity"""
        entity = self._find_entity_by_value(entity_value)
        if not entity:
            return {"error": f"Entity '{entity_value}' not found"}
        
        relationships = self.graph_store.get_relationships(entity.id)
        
        timeline_events = []
        timeline_events.append({
            "timestamp": entity.first_seen,
            "event_type": "entity_first_seen",
            "description": f"{entity.type} '{entity.value}' first observed"
        })
        
        for rel in relationships:
            timeline_events.append({
                "timestamp": rel.first_observed,
                "event_type": "relationship_established",
                "description": f"{rel.relationship_type} relationship with {rel.target_entity_id if rel.source_entity_id == entity.id else rel.source_entity_id}",
                "relationship": rel
            })
        
        # Sort by timestamp
        timeline_events.sort(key=lambda x: x["timestamp"])
        
        return {
            "entity": entity,
            "timeline": timeline_events,
            "total_events": len(timeline_events)
        }
    
    def search_entities(self, search_term: str, entity_types: List[str] = None) -> List[ForensicEntity]:
        """Search for entities by value or partial match"""
        matching_entities = []
        search_term_lower = search_term.lower()
        
        # Determine which entity types to search
        stats = self.graph_store.get_statistics()
        types_to_search = entity_types or list(stats.get("entity_types", {}).keys())
        
        for entity_type in types_to_search:
            entities = self.graph_store.get_entities_by_type(entity_type)
            for entity in entities:
                if search_term_lower in entity.value.lower():
                    matching_entities.append(entity)
        
        return matching_entities
    
    def _find_entity_by_value(self, value: str, entity_type: str = None) -> Optional[ForensicEntity]:
        """Helper method to find entity by value"""
        if entity_type:
            entities = self.graph_store.get_entities_by_type(entity_type)
        else:
            # Search all entity types
            stats = self.graph_store.get_statistics()
            entities = []
            for etype in stats.get("entity_types", {}):
                entities.extend(self.graph_store.get_entities_by_type(etype))
        
        for entity in entities:
            if entity.value.lower() == value.lower():
                return entity
        
        return None
    
    def get_graph_insights(self) -> Dict[str, Any]:
        """Generate insights about the knowledge graph"""
        stats = self.graph_store.get_statistics()
        
        insights = {
            "graph_size": {
                "entities": stats.get("total_entities", 0),
                "relationships": stats.get("total_relationships", 0)
            },
            "entity_distribution": stats.get("entity_types", {}),
            "relationship_distribution": stats.get("relationship_types", {}),
            "suspicious_patterns": len(self.find_suspicious_patterns()),
            "crypto_addresses": len(self.graph_store.get_entities_by_type("crypto_address")),
            "communication_patterns": len(self.find_communication_patterns()[:5])
        }
        
        # Calculate connectivity metrics
        total_entities = stats.get("total_entities", 0)
        total_relationships = stats.get("total_relationships", 0)
        
        if total_entities > 0:
            insights["connectivity"] = {
                "average_connections": (total_relationships * 2) / total_entities,
                "graph_density": total_relationships / (total_entities * (total_entities - 1) / 2) if total_entities > 1 else 0
            }
        
        return insights