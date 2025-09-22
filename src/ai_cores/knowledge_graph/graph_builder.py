"""
Knowledge Graph Builder for Forensic Analysis

Orchestrates the construction of knowledge graphs from processed forensic data.
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from pathlib import Path
import json

from .entity_extractor import ForensicEntityExtractor, ForensicEntity
from .relationship_extractor import ForensicRelationshipExtractor, ForensicRelationship
from .graph_store import BaseGraphStore, create_graph_store

logger = logging.getLogger(__name__)

class ForensicGraphBuilder:
    """Builds knowledge graphs from forensic data"""
    
    def __init__(self, store_type: str = "memory", store_config: Dict[str, Any] = None):
        """Initialize the graph builder"""
        self.entity_extractor = ForensicEntityExtractor()
        self.relationship_extractor = ForensicRelationshipExtractor()
        self.graph_store = create_graph_store(store_type, **(store_config or {}))
        self.build_statistics = {}
        
    def build_graph_from_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build knowledge graph from a list of processed documents"""
        logger.info(f"Building knowledge graph from {len(documents)} documents")
        
        start_time = datetime.now()
        all_entities = []
        all_relationships = []
        
        try:
            # Extract entities from all documents
            for i, doc in enumerate(documents):
                source_id = doc.get("source_id", f"doc_{i}")
                logger.debug(f"Processing document {source_id}")
                
                entities = self.entity_extractor.extract_entities_from_document(doc, source_id)
                all_entities.extend(entities)
                
                # Add entities to graph store
                for entity in entities:
                    self.graph_store.add_entity(entity)
            
            # Extract relationships between entities
            logger.info(f"Extracting relationships between {len(all_entities)} entities")
            all_relationships = self.relationship_extractor.extract_relationships(all_entities, documents)
            
            # Add relationships to graph store
            for relationship in all_relationships:
                self.graph_store.add_relationship(relationship)
            
            # Calculate statistics
            build_time = (datetime.now() - start_time).total_seconds()
            
            self.build_statistics = {
                "total_documents": len(documents),
                "total_entities": len(all_entities),
                "total_relationships": len(all_relationships),
                "build_time_seconds": build_time,
                "entity_stats": self.entity_extractor.get_entity_statistics(all_entities),
                "relationship_stats": self.relationship_extractor.get_relationship_statistics(all_relationships),
                "graph_stats": self.graph_store.get_statistics()
            }
            
            logger.info(f"✅ Knowledge graph built successfully in {build_time:.2f}s")
            logger.info(f"   • {len(all_entities)} entities")
            logger.info(f"   • {len(all_relationships)} relationships")
            
            return {
                "status": "success",
                "statistics": self.build_statistics,
                "entities": all_entities,
                "relationships": all_relationships
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to build knowledge graph: {e}")
            return {
                "status": "error",
                "error": str(e),
                "statistics": self.build_statistics
            }
    
    def build_graph_from_processed_file(self, file_path: str) -> Dict[str, Any]:
        """Build knowledge graph from a processed JSON file"""
        logger.info(f"Building knowledge graph from file: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract documents from the processed file structure
            documents = []
            
            if isinstance(data, list):
                # If it's a list of processing results
                for item in data:
                    if "ufdr_document" in item:
                        doc = item["ufdr_document"]
                        doc["source_id"] = item.get("file_path", "unknown")
                        documents.append(doc)
            elif isinstance(data, dict):
                # If it's a single processing result
                if "ufdr_document" in data:
                    doc = data["ufdr_document"]
                    doc["source_id"] = data.get("file_path", "unknown")
                    documents.append(doc)
                else:
                    # Treat the whole thing as a document
                    data["source_id"] = file_path
                    documents.append(data)
            
            if not documents:
                return {
                    "status": "error",
                    "error": "No valid documents found in file"
                }
            
            return self.build_graph_from_documents(documents)
            
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            return {
                "status": "error",
                "error": f"Failed to load file: {e}"
            }
    
    def query_graph(self, query_type: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Query the knowledge graph"""
        if query_type == "find_entities":
            entity_type = parameters.get("type") if parameters else None
            if entity_type:
                entities = self.graph_store.get_entities_by_type(entity_type)
                return [{"entity": entity} for entity in entities]
            else:
                # Return all entities
                stats = self.graph_store.get_statistics()
                return [{"total_entities": stats.get("total_entities", 0)}]
        
        elif query_type == "find_relationships":
            entity_id = parameters.get("entity_id") if parameters else None
            if entity_id:
                relationships = self.graph_store.get_relationships(entity_id)
                return [{"relationship": rel} for rel in relationships]
        
        elif query_type == "find_connected":
            entity_id = parameters.get("entity_id") if parameters else None
            if entity_id:
                return self.graph_store.query_graph("find connected", {"entity_id": entity_id})
        
        elif query_type == "statistics":
            return [{"statistics": self.graph_store.get_statistics()}]
        
        # Fallback to graph store query
        return self.graph_store.query_graph(query_type, parameters)
    
    def get_entity_by_value(self, value: str, entity_type: str = None) -> Optional[ForensicEntity]:
        """Find entity by its value"""
        entities = self.graph_store.get_entities_by_type(entity_type) if entity_type else []
        
        if not entity_type:
            # Search all entity types
            stats = self.graph_store.get_statistics()
            for etype in stats.get("entity_types", {}):
                entities.extend(self.graph_store.get_entities_by_type(etype))
        
        for entity in entities:
            if entity.value.lower() == value.lower():
                return entity
        
        return None
    
    def analyze_entity_connections(self, entity_value: str) -> Dict[str, Any]:
        """Analyze connections for a specific entity"""
        entity = self.get_entity_by_value(entity_value)
        if not entity:
            return {"error": f"Entity '{entity_value}' not found"}
        
        relationships = self.graph_store.get_relationships(entity.id)
        
        # Categorize relationships
        connections = {
            "entity": entity,
            "total_connections": len(relationships),
            "outgoing": [],
            "incoming": [],
            "by_type": {}
        }
        
        for rel in relationships:
            if rel.source_entity_id == entity.id:
                target = self.graph_store.get_entity(rel.target_entity_id)
                connections["outgoing"].append({"relationship": rel, "target": target})
            else:
                source = self.graph_store.get_entity(rel.source_entity_id)
                connections["incoming"].append({"relationship": rel, "source": source})
            
            rel_type = rel.relationship_type
            if rel_type not in connections["by_type"]:
                connections["by_type"][rel_type] = []
            connections["by_type"][rel_type].append(rel)
        
        return connections
    
    def export_graph(self, format: str = "json") -> str:
        """Export the knowledge graph"""
        if hasattr(self.graph_store, 'export_graph'):
            return self.graph_store.export_graph(format)
        else:
            return "Export not supported for this graph store"
    
    def get_graph_summary(self) -> Dict[str, Any]:
        """Get a summary of the knowledge graph"""
        stats = self.graph_store.get_statistics()
        
        # Get top entity types
        entity_types = stats.get("entity_types", {})
        top_entities = sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get top relationship types
        relationship_types = stats.get("relationship_types", {})
        top_relationships = sorted(relationship_types.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_entities": stats.get("total_entities", 0),
            "total_relationships": stats.get("total_relationships", 0),
            "top_entity_types": top_entities,
            "top_relationship_types": top_relationships,
            "build_statistics": self.build_statistics,
            "storage_type": stats.get("storage_type", "unknown")
        }