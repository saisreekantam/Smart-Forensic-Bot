"""
Graph Storage for Forensic Knowledge Graph

Provides storage backends for the knowledge graph including Neo4j and in-memory storage.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple
import logging
from datetime import datetime
import json

from .entity_extractor import ForensicEntity
from .relationship_extractor import ForensicRelationship

logger = logging.getLogger(__name__)

class BaseGraphStore(ABC):
    """Abstract base class for graph storage backends"""
    
    @abstractmethod
    def add_entity(self, entity: ForensicEntity) -> bool:
        """Add an entity to the graph"""
        pass
    
    @abstractmethod
    def add_relationship(self, relationship: ForensicRelationship) -> bool:
        """Add a relationship to the graph"""
        pass
    
    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[ForensicEntity]:
        """Get an entity by ID"""
        pass
    
    @abstractmethod
    def get_entities_by_type(self, entity_type: str) -> List[ForensicEntity]:
        """Get all entities of a specific type"""
        pass
    
    @abstractmethod
    def get_relationships(self, entity_id: str) -> List[ForensicRelationship]:
        """Get all relationships for an entity"""
        pass
    
    @abstractmethod
    def query_graph(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a graph query"""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        pass

class InMemoryGraphStore(BaseGraphStore):
    """In-memory graph storage for testing and development"""
    
    def __init__(self):
        """Initialize in-memory storage"""
        self.entities: Dict[str, ForensicEntity] = {}
        self.relationships: Dict[str, ForensicRelationship] = {}
        self.entity_relationships: Dict[str, Set[str]] = {}  # entity_id -> relationship_ids
        
    def add_entity(self, entity: ForensicEntity) -> bool:
        """Add an entity to the graph"""
        try:
            self.entities[entity.id] = entity
            if entity.id not in self.entity_relationships:
                self.entity_relationships[entity.id] = set()
            logger.debug(f"Added entity: {entity.id} ({entity.type})")
            return True
        except Exception as e:
            logger.error(f"Failed to add entity {entity.id}: {e}")
            return False
    
    def add_relationship(self, relationship: ForensicRelationship) -> bool:
        """Add a relationship to the graph"""
        try:
            self.relationships[relationship.id] = relationship
            
            # Update entity-relationship mappings
            if relationship.source_entity_id not in self.entity_relationships:
                self.entity_relationships[relationship.source_entity_id] = set()
            if relationship.target_entity_id not in self.entity_relationships:
                self.entity_relationships[relationship.target_entity_id] = set()
            
            self.entity_relationships[relationship.source_entity_id].add(relationship.id)
            self.entity_relationships[relationship.target_entity_id].add(relationship.id)
            
            logger.debug(f"Added relationship: {relationship.id} ({relationship.relationship_type})")
            return True
        except Exception as e:
            logger.error(f"Failed to add relationship {relationship.id}: {e}")
            return False
    
    def get_entity(self, entity_id: str) -> Optional[ForensicEntity]:
        """Get an entity by ID"""
        return self.entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: str) -> List[ForensicEntity]:
        """Get all entities of a specific type"""
        return [entity for entity in self.entities.values() if entity.type == entity_type]
    
    def get_relationships(self, entity_id: str) -> List[ForensicRelationship]:
        """Get all relationships for an entity"""
        if entity_id not in self.entity_relationships:
            return []
        
        relationship_ids = self.entity_relationships[entity_id]
        return [self.relationships[rel_id] for rel_id in relationship_ids if rel_id in self.relationships]
    
    def query_graph(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a simple graph query (limited functionality for in-memory store)"""
        results = []
        
        # Simple query patterns
        if query.lower().startswith("find entities"):
            if "type" in (parameters or {}):
                entities = self.get_entities_by_type(parameters["type"])
                results = [{"entity": entity} for entity in entities]
        
        elif query.lower().startswith("find relationships"):
            if "entity_id" in (parameters or {}):
                relationships = self.get_relationships(parameters["entity_id"])
                results = [{"relationship": rel} for rel in relationships]
        
        elif query.lower().startswith("find connected"):
            if "entity_id" in (parameters or {}):
                connected = self._find_connected_entities(parameters["entity_id"])
                results = [{"connected_entity": entity} for entity in connected]
        
        return results
    
    def _find_connected_entities(self, entity_id: str, max_depth: int = 2) -> List[ForensicEntity]:
        """Find entities connected to the given entity"""
        connected = set()
        visited = set()
        to_visit = [(entity_id, 0)]
        
        while to_visit:
            current_id, depth = to_visit.pop(0)
            
            if current_id in visited or depth >= max_depth:
                continue
            
            visited.add(current_id)
            relationships = self.get_relationships(current_id)
            
            for rel in relationships:
                other_id = rel.target_entity_id if rel.source_entity_id == current_id else rel.source_entity_id
                
                if other_id not in visited:
                    connected.add(other_id)
                    to_visit.append((other_id, depth + 1))
        
        return [self.entities[entity_id] for entity_id in connected if entity_id in self.entities]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        entity_types = {}
        relationship_types = {}
        
        for entity in self.entities.values():
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
        
        for relationship in self.relationships.values():
            relationship_types[relationship.relationship_type] = relationship_types.get(relationship.relationship_type, 0) + 1
        
        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "storage_type": "in_memory"
        }
    
    def export_graph(self, format: str = "json") -> str:
        """Export the graph in the specified format"""
        if format.lower() == "json":
            graph_data = {
                "entities": [
                    {
                        "id": entity.id,
                        "type": entity.type,
                        "value": entity.value,
                        "confidence": entity.confidence,
                        "metadata": entity.metadata
                    }
                    for entity in self.entities.values()
                ],
                "relationships": [
                    {
                        "id": rel.id,
                        "source": rel.source_entity_id,
                        "target": rel.target_entity_id,
                        "type": rel.relationship_type,
                        "confidence": rel.confidence,
                        "metadata": rel.metadata
                    }
                    for rel in self.relationships.values()
                ]
            }
            return json.dumps(graph_data, indent=2, default=str)
        
        return "Unsupported format"

class Neo4jForensicStore(BaseGraphStore):
    """Neo4j graph storage backend (requires Neo4j installation)"""
    
    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j connection"""
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Connect to Neo4j database"""
        try:
            # This would require neo4j package: pip install neo4j
            # from neo4j import GraphDatabase
            # self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            logger.warning("Neo4j driver not implemented. Install neo4j package and uncomment code.")
            self.driver = None
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None
    
    def add_entity(self, entity: ForensicEntity) -> bool:
        """Add an entity to Neo4j"""
        if not self.driver:
            logger.error("Neo4j driver not available")
            return False
        
        # Implementation would go here
        logger.warning("Neo4j add_entity not implemented")
        return False
    
    def add_relationship(self, relationship: ForensicRelationship) -> bool:
        """Add a relationship to Neo4j"""
        if not self.driver:
            logger.error("Neo4j driver not available")
            return False
        
        # Implementation would go here
        logger.warning("Neo4j add_relationship not implemented")
        return False
    
    def get_entity(self, entity_id: str) -> Optional[ForensicEntity]:
        """Get an entity from Neo4j"""
        if not self.driver:
            return None
        
        # Implementation would go here
        logger.warning("Neo4j get_entity not implemented")
        return None
    
    def get_entities_by_type(self, entity_type: str) -> List[ForensicEntity]:
        """Get entities by type from Neo4j"""
        if not self.driver:
            return []
        
        # Implementation would go here
        logger.warning("Neo4j get_entities_by_type not implemented")
        return []
    
    def get_relationships(self, entity_id: str) -> List[ForensicRelationship]:
        """Get relationships from Neo4j"""
        if not self.driver:
            return []
        
        # Implementation would go here
        logger.warning("Neo4j get_relationships not implemented")
        return []
    
    def query_graph(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute Cypher query on Neo4j"""
        if not self.driver:
            return []
        
        # Implementation would go here
        logger.warning("Neo4j query_graph not implemented")
        return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Neo4j graph statistics"""
        if not self.driver:
            return {"error": "Neo4j not connected"}
        
        # Implementation would go here
        logger.warning("Neo4j get_statistics not implemented")
        return {"storage_type": "neo4j", "status": "not_implemented"}
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()

def create_graph_store(store_type: str = "memory", **kwargs) -> BaseGraphStore:
    """Factory function to create graph store"""
    if store_type.lower() == "memory":
        return InMemoryGraphStore()
    elif store_type.lower() == "neo4j":
        return Neo4jForensicStore(
            uri=kwargs.get("uri", "bolt://localhost:7687"),
            username=kwargs.get("username", "neo4j"),
            password=kwargs.get("password", "")
        )
    else:
        raise ValueError(f"Unknown store type: {store_type}")