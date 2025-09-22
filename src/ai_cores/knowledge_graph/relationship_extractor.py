"""
Relationship Extractor for Knowledge Graph

Extracts relationships between forensic entities based on communication patterns,
temporal proximity, and semantic analysis.
"""

from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import re

from .entity_extractor import ForensicEntity

logger = logging.getLogger(__name__)

@dataclass
class ForensicRelationship:
    """Represents a relationship between two forensic entities"""
    id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str  # 'communicates_with', 'owns', 'located_at', 'associated_with'
    confidence: float
    evidence: List[str]  # Supporting evidence for the relationship
    metadata: Dict[str, Any]
    first_observed: datetime
    last_observed: datetime
    frequency: int  # How many times this relationship was observed

class ForensicRelationshipExtractor:
    """Extracts relationships between forensic entities"""
    
    def __init__(self):
        """Initialize relationship extractor"""
        self.relationship_patterns = self._initialize_relationship_patterns()
        self.temporal_threshold = timedelta(hours=24)  # Max time gap for temporal relationships
        
    def _initialize_relationship_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for relationship extraction"""
        return {
            "communication_indicators": [
                "called", "texted", "messaged", "sent", "received", "contacted",
                "talked to", "spoke with", "conversation", "chat", "reply"
            ],
            "ownership_indicators": [
                "owns", "belongs to", "registered to", "owned by", "device of",
                "phone of", "wallet of", "account of"
            ],
            "location_indicators": [
                "at", "in", "near", "located", "position", "address", "place"
            ],
            "association_indicators": [
                "with", "together", "associated", "linked", "connected", "related",
                "friend", "family", "colleague", "partner"
            ]
        }
    
    def extract_relationships(self, entities: List[ForensicEntity], documents: List[Dict[str, Any]]) -> List[ForensicRelationship]:
        """Extract relationships between entities from documents"""
        relationships = []
        
        # Extract different types of relationships
        relationships.extend(self._extract_communication_relationships(entities, documents))
        relationships.extend(self._extract_ownership_relationships(entities, documents))
        relationships.extend(self._extract_temporal_relationships(entities, documents))
        relationships.extend(self._extract_location_relationships(entities, documents))
        relationships.extend(self._extract_semantic_relationships(entities, documents))
        
        # Deduplicate and consolidate relationships
        relationships = self._consolidate_relationships(relationships)
        
        logger.info(f"Extracted {len(relationships)} relationships between {len(entities)} entities")
        return relationships
    
    def _extract_communication_relationships(self, entities: List[ForensicEntity], documents: List[Dict[str, Any]]) -> List[ForensicRelationship]:
        """Extract communication relationships from call logs and messages"""
        relationships = []
        entity_lookup = {entity.value: entity for entity in entities if entity.type in ["phone", "email"]}
        
        for doc in documents:
            # Process communications
            if "communications" in doc:
                for comm in doc["communications"]:
                    from_entity = entity_lookup.get(comm.get("from_number"))
                    to_entity = entity_lookup.get(comm.get("to_number"))
                    
                    if from_entity and to_entity and from_entity.id != to_entity.id:
                        rel = ForensicRelationship(
                            id=f"comm_{hash(f'{from_entity.id}_{to_entity.id}')}",
                            source_entity_id=from_entity.id,
                            target_entity_id=to_entity.id,
                            relationship_type="communicates_with",
                            confidence=0.9,
                            evidence=[f"Communication on {comm.get('timestamp', 'unknown time')}"],
                            metadata={
                                "communication_type": comm.get("type", "unknown"),
                                "direction": "outgoing",
                                "content": comm.get("content", "")[:100]
                            },
                            first_observed=datetime.now(),
                            last_observed=datetime.now(),
                            frequency=1
                        )
                        relationships.append(rel)
            
            # Process calls
            if "calls" in doc:
                for call in doc["calls"]:
                    caller = entity_lookup.get(call.get("caller"))
                    callee = entity_lookup.get(call.get("callee"))
                    
                    if caller and callee and caller.id != callee.id:
                        rel = ForensicRelationship(
                            id=f"call_{hash(f'{caller.id}_{callee.id}')}",
                            source_entity_id=caller.id,
                            target_entity_id=callee.id,
                            relationship_type="communicates_with",
                            confidence=0.95,
                            evidence=[f"Call on {call.get('timestamp', 'unknown time')}"],
                            metadata={
                                "communication_type": "call",
                                "direction": call.get("direction", "unknown"),
                                "duration": call.get("duration", 0)
                            },
                            first_observed=datetime.now(),
                            last_observed=datetime.now(),
                            frequency=1
                        )
                        relationships.append(rel)
        
        return relationships
    
    def _extract_ownership_relationships(self, entities: List[ForensicEntity], documents: List[Dict[str, Any]]) -> List[ForensicRelationship]:
        """Extract ownership relationships between people and devices/accounts"""
        relationships = []
        
        # Group entities by type
        people = [e for e in entities if e.type == "person"]
        devices = [e for e in entities if e.type in ["phone", "imei", "email", "crypto_address"]]
        
        for doc in documents:
            # Check contacts for ownership relationships
            if "contacts" in doc:
                for contact in doc["contacts"]:
                    person_name = contact.get("name")
                    person_phone = contact.get("phone")
                    
                    if person_name and person_phone:
                        # Find matching entities
                        person_entity = next((e for e in people if e.value.lower() == person_name.lower()), None)
                        phone_entity = next((e for e in devices if e.value == person_phone), None)
                        
                        if person_entity and phone_entity:
                            rel = ForensicRelationship(
                                id=f"owns_{hash(f'{person_entity.id}_{phone_entity.id}')}",
                                source_entity_id=person_entity.id,
                                target_entity_id=phone_entity.id,
                                relationship_type="owns",
                                confidence=0.8,
                                evidence=[f"Contact entry: {person_name} - {person_phone}"],
                                metadata={"source": "contact_list"},
                                first_observed=datetime.now(),
                                last_observed=datetime.now(),
                                frequency=1
                            )
                            relationships.append(rel)
            
            # Check device info for ownership
            if "device_info" in doc:
                device_info = doc["device_info"]
                if "owner" in device_info:
                    owner_name = device_info["owner"]
                    person_entity = next((e for e in people if e.value.lower() == owner_name.lower()), None)
                    
                    # Find device entities from this device
                    device_entities = [e for e in devices if e.source_document == doc.get("source_id")]
                    
                    if person_entity:
                        for device_entity in device_entities:
                            rel = ForensicRelationship(
                                id=f"owns_{hash(f'{person_entity.id}_{device_entity.id}')}",
                                source_entity_id=person_entity.id,
                                target_entity_id=device_entity.id,
                                relationship_type="owns",
                                confidence=0.9,
                                evidence=[f"Device registered to {owner_name}"],
                                metadata={"source": "device_registration"},
                                first_observed=datetime.now(),
                                last_observed=datetime.now(),
                                frequency=1
                            )
                            relationships.append(rel)
        
        return relationships
    
    def _extract_temporal_relationships(self, entities: List[ForensicEntity], documents: List[Dict[str, Any]]) -> List[ForensicRelationship]:
        """Extract relationships based on temporal proximity"""
        relationships = []
        
        # Group entities by time windows
        time_windows = defaultdict(list)
        
        for entity in entities:
            if hasattr(entity, 'first_seen') and entity.first_seen:
                # Round to hour for grouping
                time_key = entity.first_seen.replace(minute=0, second=0, microsecond=0)
                time_windows[time_key].append(entity)
        
        # Find entities that appear in the same time window
        for time_key, time_entities in time_windows.items():
            if len(time_entities) > 1:
                for i, entity1 in enumerate(time_entities):
                    for entity2 in time_entities[i+1:]:
                        if entity1.type != entity2.type and entity1.source_document == entity2.source_document:
                            rel = ForensicRelationship(
                                id=f"temporal_{hash(f'{entity1.id}_{entity2.id}')}",
                                source_entity_id=entity1.id,
                                target_entity_id=entity2.id,
                                relationship_type="temporally_associated",
                                confidence=0.6,
                                evidence=[f"Both appeared around {time_key}"],
                                metadata={"time_window": str(time_key)},
                                first_observed=time_key,
                                last_observed=time_key,
                                frequency=1
                            )
                            relationships.append(rel)
        
        return relationships
    
    def _extract_location_relationships(self, entities: List[ForensicEntity], documents: List[Dict[str, Any]]) -> List[ForensicRelationship]:
        """Extract location-based relationships"""
        relationships = []
        
        location_entities = [e for e in entities if e.type == "location"]
        other_entities = [e for e in entities if e.type != "location"]
        
        for doc in documents:
            # Check for location mentions in context
            for location_entity in location_entities:
                for other_entity in other_entities:
                    if (location_entity.source_document == other_entity.source_document and
                        location_entity.value.lower() in other_entity.context.lower()):
                        
                        rel = ForensicRelationship(
                            id=f"located_{hash(f'{other_entity.id}_{location_entity.id}')}",
                            source_entity_id=other_entity.id,
                            target_entity_id=location_entity.id,
                            relationship_type="located_at",
                            confidence=0.7,
                            evidence=[f"Mentioned together in context"],
                            metadata={"detection_method": "context_analysis"},
                            first_observed=datetime.now(),
                            last_observed=datetime.now(),
                            frequency=1
                        )
                        relationships.append(rel)
        
        return relationships
    
    def _extract_semantic_relationships(self, entities: List[ForensicEntity], documents: List[Dict[str, Any]]) -> List[ForensicRelationship]:
        """Extract relationships based on semantic patterns"""
        relationships = []
        
        for doc in documents:
            text_content = self._get_document_text(doc)
            
            # Look for semantic relationship patterns
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    if entity1.source_document == entity2.source_document:
                        relationship_type = self._detect_semantic_relationship(
                            entity1, entity2, text_content
                        )
                        
                        if relationship_type:
                            rel = ForensicRelationship(
                                id=f"semantic_{hash(f'{entity1.id}_{entity2.id}')}",
                                source_entity_id=entity1.id,
                                target_entity_id=entity2.id,
                                relationship_type=relationship_type,
                                confidence=0.65,
                                evidence=[f"Semantic pattern detected in text"],
                                metadata={"detection_method": "semantic_analysis"},
                                first_observed=datetime.now(),
                                last_observed=datetime.now(),
                                frequency=1
                            )
                            relationships.append(rel)
        
        return relationships
    
    def _detect_semantic_relationship(self, entity1: ForensicEntity, entity2: ForensicEntity, text: str) -> str:
        """Detect semantic relationships between entities"""
        text_lower = text.lower()
        entity1_value = entity1.value.lower()
        entity2_value = entity2.value.lower()
        
        # Find sentences containing both entities
        sentences = re.split(r'[.!?]+', text)
        relevant_sentences = [s for s in sentences 
                            if entity1_value in s.lower() and entity2_value in s.lower()]
        
        if not relevant_sentences:
            return None
        
        sentence_text = " ".join(relevant_sentences).lower()
        
        # Check for communication patterns
        for indicator in self.relationship_patterns["communication_indicators"]:
            if indicator in sentence_text:
                return "communicates_with"
        
        # Check for ownership patterns
        for indicator in self.relationship_patterns["ownership_indicators"]:
            if indicator in sentence_text:
                return "owns"
        
        # Check for location patterns
        for indicator in self.relationship_patterns["location_indicators"]:
            if indicator in sentence_text:
                return "located_at"
        
        # Check for association patterns
        for indicator in self.relationship_patterns["association_indicators"]:
            if indicator in sentence_text:
                return "associated_with"
        
        return None
    
    def _get_document_text(self, document: Dict[str, Any]) -> str:
        """Extract all text from a document"""
        text_parts = []
        
        def extract_text_recursive(obj):
            if isinstance(obj, str):
                text_parts.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_text_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_text_recursive(item)
        
        extract_text_recursive(document)
        return " ".join(text_parts)
    
    def _consolidate_relationships(self, relationships: List[ForensicRelationship]) -> List[ForensicRelationship]:
        """Consolidate duplicate relationships and merge evidence"""
        relationship_map = {}
        
        for rel in relationships:
            # Create a key for deduplication
            key = f"{rel.source_entity_id}_{rel.target_entity_id}_{rel.relationship_type}"
            reverse_key = f"{rel.target_entity_id}_{rel.source_entity_id}_{rel.relationship_type}"
            
            # Check if this relationship or its reverse already exists
            existing_key = key if key in relationship_map else reverse_key if reverse_key in relationship_map else None
            
            if existing_key:
                # Merge with existing relationship
                existing = relationship_map[existing_key]
                existing.frequency += rel.frequency
                existing.confidence = max(existing.confidence, rel.confidence)
                existing.evidence.extend(rel.evidence)
                existing.last_observed = max(existing.last_observed, rel.last_observed)
                existing.metadata.update(rel.metadata)
            else:
                relationship_map[key] = rel
        
        return list(relationship_map.values())
    
    def get_relationship_statistics(self, relationships: List[ForensicRelationship]) -> Dict[str, Any]:
        """Get statistics about extracted relationships"""
        if not relationships:
            return {}
        
        relationship_types = {}
        total_confidence = 0
        total_frequency = 0
        
        for rel in relationships:
            relationship_types[rel.relationship_type] = relationship_types.get(rel.relationship_type, 0) + 1
            total_confidence += rel.confidence
            total_frequency += rel.frequency
        
        return {
            "total_relationships": len(relationships),
            "relationship_types": relationship_types,
            "average_confidence": total_confidence / len(relationships),
            "average_frequency": total_frequency / len(relationships),
            "unique_types": len(relationship_types)
        }