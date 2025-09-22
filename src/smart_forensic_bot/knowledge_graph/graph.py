"""
Knowledge graph implementation for forensic investigations
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import networkx as nx
from pathlib import Path

from ..core.config import config

class ForensicKnowledgeGraph:
    """Knowledge graph for forensic investigations and evidence relationships"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.graph_path = config.vectordb_path / "knowledge_graph.json"
        self.load_graph()
    
    def add_entity(self, entity_id: str, entity_type: str, properties: Dict[str, Any] = None):
        """
        Add an entity to the knowledge graph
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of entity (e.g., 'file', 'process', 'network', 'user')
            properties: Additional properties for the entity
        """
        if properties is None:
            properties = {}
        
        properties.update({
            'type': entity_type,
            'created_at': datetime.now().isoformat(),
            'id': entity_id
        })
        
        self.graph.add_node(entity_id, **properties)
        self.save_graph()
    
    def add_relationship(self, from_entity: str, to_entity: str, 
                        relationship_type: str, properties: Dict[str, Any] = None):
        """
        Add a relationship between entities
        
        Args:
            from_entity: Source entity ID
            to_entity: Target entity ID
            relationship_type: Type of relationship
            properties: Additional properties for the relationship
        """
        if properties is None:
            properties = {}
        
        properties.update({
            'type': relationship_type,
            'created_at': datetime.now().isoformat()
        })
        
        self.graph.add_edge(from_entity, to_entity, **properties)
        self.save_graph()
    
    def add_forensic_evidence(self, evidence_data: Dict[str, Any]):
        """
        Add forensic evidence to the knowledge graph
        
        Args:
            evidence_data: Dictionary containing evidence information
        """
        evidence_id = evidence_data.get('id', f"evidence_{datetime.now().timestamp()}")
        
        # Add evidence entity
        self.add_entity(
            entity_id=evidence_id,
            entity_type='evidence',
            properties={
                'name': evidence_data.get('name', 'Unknown'),
                'file_path': evidence_data.get('file_path', ''),
                'hash_md5': evidence_data.get('hash_md5', ''),
                'hash_sha256': evidence_data.get('hash_sha256', ''),
                'file_type': evidence_data.get('file_type', ''),
                'size': evidence_data.get('size', 0),
                'created_time': evidence_data.get('created_time', ''),
                'modified_time': evidence_data.get('modified_time', ''),
                'case_id': evidence_data.get('case_id', '')
            }
        )
        
        # Add relationships to other entities
        case_id = evidence_data.get('case_id')
        if case_id:
            # Add case entity if it doesn't exist
            if not self.graph.has_node(case_id):
                self.add_entity(
                    entity_id=case_id,
                    entity_type='case',
                    properties={'name': f'Case {case_id}'}
                )
            
            # Link evidence to case
            self.add_relationship(
                from_entity=case_id,
                to_entity=evidence_id,
                relationship_type='contains'
            )
        
        # Add file system relationships
        file_path = evidence_data.get('file_path', '')
        if file_path:
            directory_path = str(Path(file_path).parent)
            directory_id = f"dir_{hash(directory_path)}"
            
            # Add directory entity
            if not self.graph.has_node(directory_id):
                self.add_entity(
                    entity_id=directory_id,
                    entity_type='directory',
                    properties={'path': directory_path}
                )
            
            # Link file to directory
            self.add_relationship(
                from_entity=directory_id,
                to_entity=evidence_id,
                relationship_type='contains'
            )
    
    def add_network_activity(self, network_data: Dict[str, Any]):
        """
        Add network activity to the knowledge graph
        
        Args:
            network_data: Dictionary containing network activity information
        """
        activity_id = f"network_{datetime.now().timestamp()}"
        
        # Add network activity entity
        self.add_entity(
            entity_id=activity_id,
            entity_type='network_activity',
            properties={
                'source_ip': network_data.get('source_ip', ''),
                'destination_ip': network_data.get('destination_ip', ''),
                'source_port': network_data.get('source_port', 0),
                'destination_port': network_data.get('destination_port', 0),
                'protocol': network_data.get('protocol', ''),
                'timestamp': network_data.get('timestamp', ''),
                'bytes_transferred': network_data.get('bytes_transferred', 0)
            }
        )
        
        # Add IP entities and relationships
        source_ip = network_data.get('source_ip')
        dest_ip = network_data.get('destination_ip')
        
        if source_ip:
            source_id = f"ip_{source_ip}"
            if not self.graph.has_node(source_id):
                self.add_entity(
                    entity_id=source_id,
                    entity_type='ip_address',
                    properties={'address': source_ip}
                )
            
            self.add_relationship(
                from_entity=source_id,
                to_entity=activity_id,
                relationship_type='initiated'
            )
        
        if dest_ip:
            dest_id = f"ip_{dest_ip}"
            if not self.graph.has_node(dest_id):
                self.add_entity(
                    entity_id=dest_id,
                    entity_type='ip_address',
                    properties={'address': dest_ip}
                )
            
            self.add_relationship(
                from_entity=activity_id,
                to_entity=dest_id,
                relationship_type='targeted'
            )
    
    def find_related_entities(self, entity_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Find entities related to a given entity
        
        Args:
            entity_id: The entity to find relationships for
            max_depth: Maximum depth to search
            
        Returns:
            List of related entities with their relationships
        """
        if not self.graph.has_node(entity_id):
            return []
        
        related = []
        visited = set()
        
        def dfs(current_id: str, depth: int, path: List[str]):
            if depth > max_depth or current_id in visited:
                return
            
            visited.add(current_id)
            
            # Get all neighbors
            neighbors = list(self.graph.successors(current_id)) + list(self.graph.predecessors(current_id))
            
            for neighbor in neighbors:
                if neighbor not in path:  # Avoid cycles
                    # Get edge data
                    edge_data = {}
                    if self.graph.has_edge(current_id, neighbor):
                        edge_data = dict(self.graph[current_id][neighbor])
                    elif self.graph.has_edge(neighbor, current_id):
                        edge_data = dict(self.graph[neighbor][current_id])
                    
                    related.append({
                        'entity_id': neighbor,
                        'properties': dict(self.graph.nodes[neighbor]),
                        'relationship': edge_data,
                        'depth': depth + 1,
                        'path': path + [neighbor]
                    })
                    
                    dfs(neighbor, depth + 1, path + [neighbor])
        
        dfs(entity_id, 0, [entity_id])
        return related
    
    def query_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """
        Query entities by type
        
        Args:
            entity_type: Type of entities to retrieve
            
        Returns:
            List of entities of the specified type
        """
        entities = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == entity_type:
                entities.append({
                    'entity_id': node_id,
                    'properties': data
                })
        return entities
    
    def get_timeline(self, entity_ids: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get timeline of events for specific entities or all entities
        
        Args:
            entity_ids: Optional list of entity IDs to filter by
            
        Returns:
            Chronologically sorted list of events
        """
        events = []
        
        nodes_to_check = entity_ids if entity_ids else list(self.graph.nodes())
        
        for node_id in nodes_to_check:
            if self.graph.has_node(node_id):
                data = self.graph.nodes[node_id]
                created_at = data.get('created_at')
                if created_at:
                    events.append({
                        'timestamp': created_at,
                        'entity_id': node_id,
                        'entity_type': data.get('type', 'unknown'),
                        'event_type': 'entity_created',
                        'properties': data
                    })
        
        # Sort by timestamp
        events.sort(key=lambda x: x['timestamp'])
        return events
    
    def save_graph(self):
        """Save the knowledge graph to file"""
        config.vectordb_path.mkdir(parents=True, exist_ok=True)
        
        # Convert graph to JSON serializable format
        graph_data = {
            'nodes': {
                node_id: data for node_id, data in self.graph.nodes(data=True)
            },
            'edges': [
                {
                    'from': from_node,
                    'to': to_node,
                    'properties': data
                }
                for from_node, to_node, data in self.graph.edges(data=True)
            ]
        }
        
        with open(self.graph_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
    
    def load_graph(self):
        """Load the knowledge graph from file"""
        if self.graph_path.exists():
            try:
                with open(self.graph_path, 'r') as f:
                    graph_data = json.load(f)
                
                # Reconstruct graph
                self.graph.clear()
                
                # Add nodes
                for node_id, data in graph_data.get('nodes', {}).items():
                    self.graph.add_node(node_id, **data)
                
                # Add edges
                for edge in graph_data.get('edges', []):
                    self.graph.add_edge(
                        edge['from'], 
                        edge['to'], 
                        **edge.get('properties', {})
                    )
                    
            except Exception as e:
                print(f"Error loading knowledge graph: {e}")
                self.graph = nx.MultiDiGraph()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        node_types = {}
        edge_types = {}
        
        for _, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': node_types,
            'edge_types': edge_types,
            'is_connected': nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        }