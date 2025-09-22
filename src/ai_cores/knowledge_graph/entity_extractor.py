"""
Entity Extractor for Knowledge Graph

Extracts forensic entities (people, phones, crypto addresses, etc.)
from processed documents and creates structured entity representations.
"""

import re
import spacy
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ForensicEntity:
    """Represents a forensic entity extracted from data"""
    id: str
    type: str  # 'person', 'phone', 'crypto_address', 'email', 'device', 'location'
    value: str
    confidence: float
    context: str
    metadata: Dict[str, Any]
    source_document: str
    first_seen: datetime
    last_seen: datetime

class ForensicEntityExtractor:
    """Extracts forensic entities from documents for knowledge graph construction"""
    
    def __init__(self):
        """Initialize entity extractor with patterns and NLP models"""
        self.nlp = self._load_nlp_model()
        self.entity_patterns = self._initialize_patterns()
        self.extracted_entities = {}
        
    def _load_nlp_model(self):
        """Load spaCy NLP model"""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            return None
    
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for entity extraction"""
        return {
            "phone_patterns": [
                r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
                r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
                r'\+\d{1,3}\s?\d{3,4}\s?\d{3,4}\s?\d{3,4}'
            ],
            "crypto_patterns": [
                r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',  # Bitcoin
                r'\b0x[a-fA-F0-9]{40}\b',  # Ethereum
                r'\b4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}\b',  # Monero
                r'\bbc1[a-z0-9]{39,59}\b',  # Bitcoin Bech32
            ],
            "email_patterns": [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            "imei_patterns": [
                r'\b\d{15}\b',  # IMEI
                r'\b\d{14}\b'   # IMEI without check digit
            ],
            "ip_patterns": [
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            ]
        }
    
    def extract_entities_from_document(self, document: Dict[str, Any], source_id: str) -> List[ForensicEntity]:
        """Extract all entities from a processed document"""
        entities = []
        current_time = datetime.now()
        
        # Extract from different parts of the document
        text_content = self._extract_text_from_document(document)
        
        # Extract different entity types
        entities.extend(self._extract_phone_numbers(text_content, source_id, current_time))
        entities.extend(self._extract_crypto_addresses(text_content, source_id, current_time))
        entities.extend(self._extract_emails(text_content, source_id, current_time))
        entities.extend(self._extract_imeis(text_content, source_id, current_time))
        entities.extend(self._extract_ip_addresses(text_content, source_id, current_time))
        
        # Extract named entities using spaCy
        if self.nlp:
            entities.extend(self._extract_named_entities(text_content, source_id, current_time))
        
        # Extract forensic-specific entities
        entities.extend(self._extract_forensic_entities(document, source_id, current_time))
        
        # Deduplicate and merge entities
        entities = self._deduplicate_entities(entities)
        
        logger.info(f"Extracted {len(entities)} entities from document {source_id}")
        return entities
    
    def _extract_text_from_document(self, document: Dict[str, Any]) -> str:
        """Extract all text content from document"""
        text_parts = []
        
        if isinstance(document, dict):
            for key, value in document.items():
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, (list, dict)):
                    text_parts.append(str(value))
        elif isinstance(document, str):
            text_parts.append(document)
        
        return " ".join(text_parts)
    
    def _extract_phone_numbers(self, text: str, source_id: str, timestamp: datetime) -> List[ForensicEntity]:
        """Extract phone numbers from text"""
        entities = []
        for pattern in self.entity_patterns["phone_patterns"]:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                phone = match.group().strip()
                if len(phone) >= 7:  # Minimum phone number length
                    entity = ForensicEntity(
                        id=f"phone_{hash(phone)}",
                        type="phone",
                        value=phone,
                        confidence=0.9,
                        context=self._get_context(text, match.start(), match.end()),
                        metadata={"pattern_matched": pattern},
                        source_document=source_id,
                        first_seen=timestamp,
                        last_seen=timestamp
                    )
                    entities.append(entity)
        return entities
    
    def _extract_crypto_addresses(self, text: str, source_id: str, timestamp: datetime) -> List[ForensicEntity]:
        """Extract cryptocurrency addresses from text"""
        entities = []
        crypto_types = ["bitcoin", "ethereum", "monero", "bitcoin_bech32"]
        
        for i, pattern in enumerate(self.entity_patterns["crypto_patterns"]):
            matches = re.finditer(pattern, text)
            for match in matches:
                address = match.group().strip()
                entity = ForensicEntity(
                    id=f"crypto_{hash(address)}",
                    type="crypto_address",
                    value=address,
                    confidence=0.95,
                    context=self._get_context(text, match.start(), match.end()),
                    metadata={
                        "crypto_type": crypto_types[i],
                        "pattern_matched": pattern
                    },
                    source_document=source_id,
                    first_seen=timestamp,
                    last_seen=timestamp
                )
                entities.append(entity)
        return entities
    
    def _extract_emails(self, text: str, source_id: str, timestamp: datetime) -> List[ForensicEntity]:
        """Extract email addresses from text"""
        entities = []
        for pattern in self.entity_patterns["email_patterns"]:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                email = match.group().strip().lower()
                entity = ForensicEntity(
                    id=f"email_{hash(email)}",
                    type="email",
                    value=email,
                    confidence=0.9,
                    context=self._get_context(text, match.start(), match.end()),
                    metadata={"domain": email.split('@')[1] if '@' in email else None},
                    source_document=source_id,
                    first_seen=timestamp,
                    last_seen=timestamp
                )
                entities.append(entity)
        return entities
    
    def _extract_imeis(self, text: str, source_id: str, timestamp: datetime) -> List[ForensicEntity]:
        """Extract IMEI numbers from text"""
        entities = []
        for pattern in self.entity_patterns["imei_patterns"]:
            matches = re.finditer(pattern, text)
            for match in matches:
                imei = match.group().strip()
                entity = ForensicEntity(
                    id=f"imei_{hash(imei)}",
                    type="imei",
                    value=imei,
                    confidence=0.8,
                    context=self._get_context(text, match.start(), match.end()),
                    metadata={"length": len(imei)},
                    source_document=source_id,
                    first_seen=timestamp,
                    last_seen=timestamp
                )
                entities.append(entity)
        return entities
    
    def _extract_ip_addresses(self, text: str, source_id: str, timestamp: datetime) -> List[ForensicEntity]:
        """Extract IP addresses from text"""
        entities = []
        for pattern in self.entity_patterns["ip_patterns"]:
            matches = re.finditer(pattern, text)
            for match in matches:
                ip = match.group().strip()
                # Basic IP validation
                octets = ip.split('.')
                if all(0 <= int(octet) <= 255 for octet in octets):
                    entity = ForensicEntity(
                        id=f"ip_{hash(ip)}",
                        type="ip_address",
                        value=ip,
                        confidence=0.85,
                        context=self._get_context(text, match.start(), match.end()),
                        metadata={"octets": octets},
                        source_document=source_id,
                        first_seen=timestamp,
                        last_seen=timestamp
                    )
                    entities.append(entity)
        return entities
    
    def _extract_named_entities(self, text: str, source_id: str, timestamp: datetime) -> List[ForensicEntity]:
        """Extract named entities using spaCy"""
        entities = []
        if not self.nlp:
            return entities
        
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
                entity_type = {
                    "PERSON": "person",
                    "ORG": "organization", 
                    "GPE": "location",
                    "LOC": "location"
                }.get(ent.label_, "unknown")
                
                entity = ForensicEntity(
                    id=f"{entity_type}_{hash(ent.text)}",
                    type=entity_type,
                    value=ent.text.strip(),
                    confidence=0.7,
                    context=self._get_context(text, ent.start_char, ent.end_char),
                    metadata={"spacy_label": ent.label_},
                    source_document=source_id,
                    first_seen=timestamp,
                    last_seen=timestamp
                )
                entities.append(entity)
        
        return entities
    
    def _extract_forensic_entities(self, document: Dict[str, Any], source_id: str, timestamp: datetime) -> List[ForensicEntity]:
        """Extract forensic-specific entities from structured data"""
        entities = []
        
        # Extract from communications
        if "communications" in document:
            entities.extend(self._extract_from_communications(document["communications"], source_id, timestamp))
        
        # Extract from calls
        if "calls" in document:
            entities.extend(self._extract_from_calls(document["calls"], source_id, timestamp))
        
        # Extract from contacts
        if "contacts" in document:
            entities.extend(self._extract_from_contacts(document["contacts"], source_id, timestamp))
        
        # Extract from device info
        if "device_info" in document:
            entities.extend(self._extract_from_device_info(document["device_info"], source_id, timestamp))
        
        return entities
    
    def _extract_from_communications(self, communications: List[Dict], source_id: str, timestamp: datetime) -> List[ForensicEntity]:
        """Extract entities from communication logs"""
        entities = []
        for comm in communications:
            if "from_number" in comm:
                entities.append(ForensicEntity(
                    id=f"phone_{hash(comm['from_number'])}",
                    type="phone",
                    value=comm["from_number"],
                    confidence=0.95,
                    context=f"Communication from {comm.get('from_number')} to {comm.get('to_number')}",
                    metadata={"role": "sender", "comm_type": comm.get("type", "unknown")},
                    source_document=source_id,
                    first_seen=timestamp,
                    last_seen=timestamp
                ))
            
            if "to_number" in comm:
                entities.append(ForensicEntity(
                    id=f"phone_{hash(comm['to_number'])}",
                    type="phone", 
                    value=comm["to_number"],
                    confidence=0.95,
                    context=f"Communication from {comm.get('from_number')} to {comm.get('to_number')}",
                    metadata={"role": "receiver", "comm_type": comm.get("type", "unknown")},
                    source_document=source_id,
                    first_seen=timestamp,
                    last_seen=timestamp
                ))
        
        return entities
    
    def _extract_from_calls(self, calls: List[Dict], source_id: str, timestamp: datetime) -> List[ForensicEntity]:
        """Extract entities from call logs"""
        entities = []
        for call in calls:
            for field in ["caller", "callee", "phone_number"]:
                if field in call and call[field]:
                    entities.append(ForensicEntity(
                        id=f"phone_{hash(call[field])}",
                        type="phone",
                        value=call[field],
                        confidence=0.95,
                        context=f"Call log entry: {call.get('direction', 'unknown')} call",
                        metadata={"call_type": call.get("direction"), "duration": call.get("duration")},
                        source_document=source_id,
                        first_seen=timestamp,
                        last_seen=timestamp
                    ))
        
        return entities
    
    def _extract_from_contacts(self, contacts: List[Dict], source_id: str, timestamp: datetime) -> List[ForensicEntity]:
        """Extract entities from contact lists"""
        entities = []
        for contact in contacts:
            if "name" in contact:
                entities.append(ForensicEntity(
                    id=f"person_{hash(contact['name'])}",
                    type="person",
                    value=contact["name"],
                    confidence=0.8,
                    context=f"Contact entry with phone: {contact.get('phone', 'unknown')}",
                    metadata={"contact_type": "stored_contact"},
                    source_document=source_id,
                    first_seen=timestamp,
                    last_seen=timestamp
                ))
            
            if "phone" in contact:
                entities.append(ForensicEntity(
                    id=f"phone_{hash(contact['phone'])}",
                    type="phone",
                    value=contact["phone"],
                    confidence=0.9,
                    context=f"Contact phone for: {contact.get('name', 'unknown')}",
                    metadata={"owner": contact.get("name")},
                    source_document=source_id,
                    first_seen=timestamp,
                    last_seen=timestamp
                ))
        
        return entities
    
    def _extract_from_device_info(self, device_info: Dict, source_id: str, timestamp: datetime) -> List[ForensicEntity]:
        """Extract entities from device information"""
        entities = []
        
        for field in ["imei", "serial_number", "phone_number"]:
            if field in device_info and device_info[field]:
                entity_type = "imei" if field == "imei" else "phone" if field == "phone_number" else "device_serial"
                entities.append(ForensicEntity(
                    id=f"{entity_type}_{hash(device_info[field])}",
                    type=entity_type,
                    value=device_info[field],
                    confidence=0.95,
                    context=f"Device {field}: {device_info.get('make')} {device_info.get('model')}",
                    metadata={"device_make": device_info.get("make"), "device_model": device_info.get("model")},
                    source_document=source_id,
                    first_seen=timestamp,
                    last_seen=timestamp
                ))
        
        return entities
    
    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get context around an entity match"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def _deduplicate_entities(self, entities: List[ForensicEntity]) -> List[ForensicEntity]:
        """Remove duplicate entities and merge information"""
        unique_entities = {}
        
        for entity in entities:
            key = f"{entity.type}_{entity.value.lower()}"
            
            if key in unique_entities:
                # Merge with existing entity
                existing = unique_entities[key]
                existing.confidence = max(existing.confidence, entity.confidence)
                existing.last_seen = max(existing.last_seen, entity.last_seen)
                existing.first_seen = min(existing.first_seen, entity.first_seen)
                
                # Merge metadata
                existing.metadata.update(entity.metadata)
                
                # Append context if different
                if entity.context not in existing.context:
                    existing.context += f" | {entity.context}"
            else:
                unique_entities[key] = entity
        
        return list(unique_entities.values())
    
    def get_entity_statistics(self, entities: List[ForensicEntity]) -> Dict[str, Any]:
        """Get statistics about extracted entities"""
        if not entities:
            return {}
        
        entity_types = {}
        total_confidence = 0
        
        for entity in entities:
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
            total_confidence += entity.confidence
        
        return {
            "total_entities": len(entities),
            "entity_types": entity_types,
            "average_confidence": total_confidence / len(entities),
            "unique_types": len(entity_types)
        }