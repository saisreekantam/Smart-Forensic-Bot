"""
Knowledge Graph Reasoner Node

This module implements knowledge graph-based reasoning for forensic investigations,
including entity relationship mapping, graph traversal, and temporal analysis.
"""

import sys
import os
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict, deque
import networkx as nx
from langchain_core.messages import AIMessage

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ..state import ForensicBotState, Entity, Event, add_workflow_step

class ForensicKnowledgeGraph:
    """Knowledge graph for forensic entity relationships and reasoning"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()  # Directed multigraph for complex relationships
        self.entity_index = {}          # Entity ID to attributes mapping
        self.temporal_index = {}        # Time-based event indexing
        self.relationship_types = set() # Track all relationship types
    
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the knowledge graph"""
        # Add node with entity attributes
        self.graph.add_node(
            entity.id,
            name=entity.name,
            type=entity.type,
            attributes=entity.attributes,
            confidence=entity.confidence,
            case_id=entity.case_id,
            first_seen=entity.first_seen,
            last_seen=entity.last_seen
        )
        
        # Update entity index
        self.entity_index[entity.id] = entity
    
    def add_event(self, event: Event) -> None:
        """Add an event and create relationships between involved entities"""
        # Add event to temporal index
        timestamp_key = event.timestamp.isoformat()
        if timestamp_key not in self.temporal_index:
            self.temporal_index[timestamp_key] = []
        self.temporal_index[timestamp_key].append(event)
        
        # Create relationships between entities involved in the event
        entities = event.entities_involved
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                self.add_relationship(
                    entity1, 
                    entity2, 
                    relationship_type=f"co_occurred_in_{event.event_type}",
                    metadata={
                        "event_id": event.id,
                        "timestamp": event.timestamp,
                        "event_type": event.event_type,
                        "description": event.description,
                        "confidence": event.confidence
                    }
                )
    
    def add_relationship(
        self, 
        entity1_id: str, 
        entity2_id: str, 
        relationship_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a relationship between two entities"""
        # Ensure both entities exist in graph
        if entity1_id not in self.graph:
            self.graph.add_node(entity1_id, name=entity1_id, type="unknown")
        if entity2_id not in self.graph:
            self.graph.add_node(entity2_id, name=entity2_id, type="unknown")
        
        # Add the relationship edge
        self.graph.add_edge(
            entity1_id,
            entity2_id,
            relationship_type=relationship_type,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        
        self.relationship_types.add(relationship_type)
    
    def find_shortest_path(self, entity1_id: str, entity2_id: str) -> Optional[List[str]]:
        """Find shortest path between two entities"""
        try:
            return nx.shortest_path(self.graph, entity1_id, entity2_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def get_neighbors(self, entity_id: str, max_distance: int = 1) -> Dict[str, Any]:
        """Get neighboring entities within specified distance"""
        if entity_id not in self.graph:
            return {}
        
        neighbors = {}
        
        for distance in range(1, max_distance + 1):
            # Get nodes at exact distance
            nodes_at_distance = []
            
            # BFS to find nodes at specific distance
            visited = {entity_id}
            queue = deque([(entity_id, 0)])
            
            while queue:
                current_node, current_distance = queue.popleft()
                
                if current_distance == distance:
                    nodes_at_distance.append(current_node)
                elif current_distance < distance:
                    for neighbor in self.graph.neighbors(current_node):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, current_distance + 1))
            
            if nodes_at_distance:
                neighbors[f"distance_{distance}"] = nodes_at_distance
        
        return neighbors
    
    def analyze_centrality(self) -> Dict[str, float]:
        """Analyze entity centrality (importance) in the graph"""
        try:
            # Convert to undirected for centrality analysis
            undirected_graph = self.graph.to_undirected()
            
            # Calculate different centrality measures
            degree_centrality = nx.degree_centrality(undirected_graph)
            betweenness_centrality = nx.betweenness_centrality(undirected_graph)
            
            # Combine centrality measures
            combined_centrality = {}
            for node in undirected_graph.nodes():
                combined_centrality[node] = (
                    degree_centrality.get(node, 0) * 0.6 + 
                    betweenness_centrality.get(node, 0) * 0.4
                )
            
            return combined_centrality
        except:
            return {}
    
    def find_communities(self) -> List[Set[str]]:
        """Find communities/clusters in the graph"""
        try:
            undirected_graph = self.graph.to_undirected()
            communities = nx.community.greedy_modularity_communities(undirected_graph)
            return [set(community) for community in communities]
        except:
            return []
    
    def temporal_analysis(
        self, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in the graph"""
        # Filter events by time range if specified
        relevant_events = []
        
        for timestamp_str, events in self.temporal_index.items():
            timestamp = datetime.fromisoformat(timestamp_str)
            
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue
            
            relevant_events.extend(events)
        
        # Analyze patterns
        event_types = defaultdict(int)
        entity_activity = defaultdict(int)
        hourly_activity = defaultdict(int)
        
        for event in relevant_events:
            event_types[event.event_type] += 1
            
            for entity_id in event.entities_involved:
                entity_activity[entity_id] += 1
            
            hour = event.timestamp.hour
            hourly_activity[hour] += 1
        
        return {
            "total_events": len(relevant_events),
            "event_types": dict(event_types),
            "most_active_entities": sorted(
                entity_activity.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
            "hourly_activity": dict(hourly_activity),
            "time_range": {
                "start": min(event.timestamp for event in relevant_events) if relevant_events else None,
                "end": max(event.timestamp for event in relevant_events) if relevant_events else None
            }
        }
    
    def query_relationships(
        self, 
        entity_id: str, 
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Query relationships for a specific entity"""
        if entity_id not in self.graph:
            return []
        
        relationships = []
        
        # Outgoing relationships
        for neighbor in self.graph.neighbors(entity_id):
            edges = self.graph[entity_id][neighbor]
            for edge_key, edge_data in edges.items():
                if not relationship_types or edge_data.get("relationship_type") in relationship_types:
                    relationships.append({
                        "source": entity_id,
                        "target": neighbor,
                        "relationship_type": edge_data.get("relationship_type"),
                        "metadata": edge_data.get("metadata", {}),
                        "direction": "outgoing"
                    })
        
        # Incoming relationships
        for predecessor in self.graph.predecessors(entity_id):
            edges = self.graph[predecessor][entity_id]
            for edge_key, edge_data in edges.items():
                if not relationship_types or edge_data.get("relationship_type") in relationship_types:
                    relationships.append({
                        "source": predecessor,
                        "target": entity_id,
                        "relationship_type": edge_data.get("relationship_type"),
                        "metadata": edge_data.get("metadata", {}),
                        "direction": "incoming"
                    })
        
        return relationships

class KnowledgeGraphReasoner:
    """Forensic knowledge graph reasoning coordinator"""
    
    def __init__(self):
        self.kg = ForensicKnowledgeGraph()
        self.initialized = False
    
    def build_graph_from_state(self, state: ForensicBotState) -> bool:
        """Build knowledge graph from current state"""
        try:
            # Add entities
            for entity in state["entity_memory"].values():
                self.kg.add_entity(entity)
            
            # Add events
            for event in state["timeline_memory"]:
                self.kg.add_event(event)
            
            # Add explicit relationships from relationship_graph
            for entity_id, related_entities in state["relationship_graph"].items():
                for related_entity in related_entities:
                    self.kg.add_relationship(
                        entity_id, 
                        related_entity, 
                        "related_to"
                    )
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"Knowledge graph building failed: {e}")
            return False

def knowledge_graph_reasoner(state: ForensicBotState) -> ForensicBotState:
    """
    Perform knowledge graph-based reasoning and analysis
    
    This node uses graph algorithms to analyze entity relationships,
    find patterns, and answer complex queries about connections.
    
    Args:
        state: Current forensic bot state
        
    Returns:
        ForensicBotState: Updated state with KG reasoning results
    """
    start_time = datetime.now()
    
    try:
        # Get the user's query
        if not state["messages"]:
            raise ValueError("No messages found in state")
        
        last_message = state["messages"][-1]
        query = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # Initialize knowledge graph reasoner
        if "kg_reasoner_instance" not in state["tool_results"]:
            state["tool_results"]["kg_reasoner_instance"] = KnowledgeGraphReasoner()
        
        reasoner = state["tool_results"]["kg_reasoner_instance"]
        
        # Build/update the knowledge graph
        if not reasoner.initialized:
            if not reasoner.build_graph_from_state(state):
                raise Exception("Failed to build knowledge graph")
        
        # Determine the type of analysis needed
        analysis_type = determine_kg_analysis_type(query)
        
        # Perform the analysis
        if analysis_type == "relationship_analysis":
            result = perform_relationship_analysis(reasoner, query, state)
        elif analysis_type == "path_analysis":
            result = perform_path_analysis(reasoner, query, state)
        elif analysis_type == "centrality_analysis":
            result = perform_centrality_analysis(reasoner, query, state)
        elif analysis_type == "temporal_analysis":
            result = perform_temporal_analysis(reasoner, query, state)
        elif analysis_type == "community_analysis":
            result = perform_community_analysis(reasoner, query, state)
        else:
            result = perform_general_graph_analysis(reasoner, query, state)
        
        # Update state with results
        state["kg_query_results"][f"kg_{datetime.now().timestamp()}"] = result
        
        # Generate response
        response = generate_kg_response(result, analysis_type)
        ai_message = AIMessage(content=response)
        state["messages"].append(ai_message)
        
        # Update recommendations
        if result.get("key_findings"):
            state["recommendations"].extend(result["key_findings"][:3])  # Top 3 findings
        
        # Add to tools used
        if "knowledge_graph_reasoner" not in state["tools_used"]:
            state["tools_used"].append("knowledge_graph_reasoner")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        add_workflow_step(
            state,
            node_name="knowledge_graph_reasoner",
            action=f"kg_{analysis_type}",
            input_data={
                "query": query,
                "analysis_type": analysis_type,
                "entities_count": len(state["entity_memory"]),
                "events_count": len(state["timeline_memory"])
            },
            output_data={
                "success": result.get("success", False),
                "findings_count": len(result.get("key_findings", [])),
                "relationships_analyzed": result.get("relationships_count", 0)
            },
            execution_time=execution_time,
            success=result.get("success", False)
        )
        
    except Exception as e:
        # Handle errors gracefully
        error_msg = f"Knowledge graph reasoning error: {str(e)}"
        state["last_tool_error"] = error_msg
        
        fallback_response = (
            "I encountered an issue while analyzing entity relationships. "
            "This might be because insufficient evidence has been processed, "
            "or the entities haven't been properly extracted. "
            "Please ensure evidence has been analyzed first."
        )
        
        ai_message = AIMessage(content=fallback_response)
        state["messages"].append(ai_message)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        add_workflow_step(
            state,
            node_name="knowledge_graph_reasoner",
            action="kg_analysis",
            input_data={"error": str(e)},
            output_data={"fallback_response": True},
            execution_time=execution_time,
            success=False
        )
    
    return state

def determine_kg_analysis_type(query: str) -> str:
    """Determine the type of knowledge graph analysis needed"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["relationship", "connection", "link", "relate"]):
        return "relationship_analysis"
    elif any(word in query_lower for word in ["path", "route", "connect", "between"]):
        return "path_analysis"
    elif any(word in query_lower for word in ["important", "central", "key", "influential"]):
        return "centrality_analysis"
    elif any(word in query_lower for word in ["timeline", "temporal", "time", "when", "chronological"]):
        return "temporal_analysis"
    elif any(word in query_lower for word in ["group", "cluster", "community", "network"]):
        return "community_analysis"
    else:
        return "general_analysis"

def perform_relationship_analysis(reasoner, query: str, state: ForensicBotState) -> Dict[str, Any]:
    """Perform relationship analysis"""
    try:
        # Extract entity names from query
        entity_names = extract_entity_names_from_query(query, state)
        
        if not entity_names:
            return {
                "success": False,
                "error": "No entities found in query"
            }
        
        relationships = []
        for entity_name in entity_names:
            # Find entity ID
            entity_id = find_entity_id_by_name(entity_name, state)
            if entity_id:
                entity_relationships = reasoner.kg.query_relationships(entity_id)
                relationships.extend(entity_relationships)
        
        return {
            "success": True,
            "analysis_type": "relationship_analysis",
            "relationships": relationships,
            "relationships_count": len(relationships),
            "key_findings": [
                f"Found {len(relationships)} relationships for specified entities",
                f"Relationship types: {list(set(r['relationship_type'] for r in relationships))}"
            ]
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def perform_centrality_analysis(reasoner, query: str, state: ForensicBotState) -> Dict[str, Any]:
    """Perform centrality analysis"""
    try:
        centrality_scores = reasoner.kg.analyze_centrality()
        
        # Get top entities by centrality
        top_entities = sorted(
            centrality_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "success": True,
            "analysis_type": "centrality_analysis",
            "centrality_scores": centrality_scores,
            "top_entities": top_entities,
            "key_findings": [
                f"Most central entity: {top_entities[0][0] if top_entities else 'None'}",
                f"Analyzed {len(centrality_scores)} entities for importance"
            ]
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def perform_temporal_analysis(reasoner, query: str, state: ForensicBotState) -> Dict[str, Any]:
    """Perform temporal analysis"""
    try:
        temporal_analysis = reasoner.kg.temporal_analysis()
        
        return {
            "success": True,
            "analysis_type": "temporal_analysis",
            "temporal_data": temporal_analysis,
            "key_findings": [
                f"Analyzed {temporal_analysis['total_events']} events",
                f"Most common event type: {max(temporal_analysis['event_types'].items(), key=lambda x: x[1])[0] if temporal_analysis['event_types'] else 'None'}"
            ]
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def perform_community_analysis(reasoner, query: str, state: ForensicBotState) -> Dict[str, Any]:
    """Perform community analysis"""
    try:
        communities = reasoner.kg.find_communities()
        
        return {
            "success": True,
            "analysis_type": "community_analysis",
            "communities": [list(community) for community in communities],
            "communities_count": len(communities),
            "key_findings": [
                f"Found {len(communities)} distinct groups/communities",
                f"Largest group has {max(len(c) for c in communities) if communities else 0} entities"
            ]
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def perform_general_graph_analysis(reasoner, query: str, state: ForensicBotState) -> Dict[str, Any]:
    """Perform general graph analysis"""
    try:
        # General graph statistics
        graph = reasoner.kg.graph
        
        return {
            "success": True,
            "analysis_type": "general_analysis",
            "graph_stats": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "relationship_types": list(reasoner.kg.relationship_types)
            },
            "key_findings": [
                f"Knowledge graph contains {graph.number_of_nodes()} entities",
                f"Found {graph.number_of_edges()} relationships",
                f"Relationship types: {len(reasoner.kg.relationship_types)}"
            ]
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def generate_kg_response(result: Dict[str, Any], analysis_type: str) -> str:
    """Generate human-readable response from KG analysis results"""
    if not result.get("success", False):
        return f"Knowledge graph analysis failed: {result.get('error', 'Unknown error')}"
    
    response = f"## Knowledge Graph Analysis Results\n\n"
    
    if analysis_type == "relationship_analysis":
        relationships = result.get("relationships", [])
        response += f"Found {len(relationships)} relationships.\n\n"
        
        if relationships:
            response += "Key relationships:\n"
            for rel in relationships[:5]:  # Show top 5
                response += f"- {rel['source']} → {rel['target']} ({rel['relationship_type']})\n"
    
    elif analysis_type == "centrality_analysis":
        top_entities = result.get("top_entities", [])
        response += "Most important entities in the investigation:\n\n"
        
        for i, (entity_id, score) in enumerate(top_entities[:5], 1):
            response += f"{i}. {entity_id} (importance: {score:.3f})\n"
    
    elif analysis_type == "temporal_analysis":
        temporal_data = result.get("temporal_data", {})
        response += f"Temporal analysis of {temporal_data.get('total_events', 0)} events:\n\n"
        
        event_types = temporal_data.get("event_types", {})
        if event_types:
            response += "Event types:\n"
            for event_type, count in event_types.items():
                response += f"- {event_type}: {count}\n"
    
    # Add key findings
    key_findings = result.get("key_findings", [])
    if key_findings:
        response += f"\n**Key Findings:**\n"
        for finding in key_findings:
            response += f"• {finding}\n"
    
    return response

def extract_entity_names_from_query(query: str, state: ForensicBotState) -> List[str]:
    """Extract entity names mentioned in the query"""
    # Simple approach: look for entity names in the query
    entity_names = []
    
    for entity in state["entity_memory"].values():
        if entity.name.lower() in query.lower():
            entity_names.append(entity.name)
    
    return entity_names

def find_entity_id_by_name(name: str, state: ForensicBotState) -> Optional[str]:
    """Find entity ID by name"""
    for entity_id, entity in state["entity_memory"].items():
        if entity.name.lower() == name.lower():
            return entity_id
    return None