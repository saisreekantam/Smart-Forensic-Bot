"""
Evidence Type Handlers for Different Forensic Data Types

This module provides specialized handlers for different types of forensic evidence,
preparing the structure for future multi-modal embeddings (images, videos, audio).
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import json
import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from src.database.models import EvidenceType

logger = logging.getLogger(__name__)

@dataclass
class ProcessedEvidence:
    """Container for processed evidence data"""
    evidence_type: EvidenceType
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    entities: Dict[str, List[str]]
    searchable_text: str
    structured_data: Optional[Dict[str, Any]] = None

class BaseEvidenceHandler:
    """Base class for evidence type handlers"""
    
    def __init__(self, evidence_type: EvidenceType):
        self.evidence_type = evidence_type
    
    def process(self, file_path: str, metadata: Dict[str, Any] = None) -> ProcessedEvidence:
        """Process evidence file and extract relevant information"""
        raise NotImplementedError("Subclasses must implement process method")
    
    def extract_entities(self, content: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract forensic entities from content"""
        entities = {
            "phone_numbers": [],
            "emails": [],
            "names": [],
            "locations": [],
            "dates": [],
            "crypto_addresses": [],
            "ip_addresses": []
        }
        return entities
    
    def create_searchable_text(self, content: Dict[str, Any]) -> str:
        """Create searchable text representation of the evidence"""
        return ""

class ChatEvidenceHandler(BaseEvidenceHandler):
    """Handler for chat/messaging evidence"""
    
    def __init__(self):
        super().__init__(EvidenceType.CHAT)
    
    def process(self, file_path: str, metadata: Dict[str, Any] = None) -> ProcessedEvidence:
        """Process chat/messaging data"""
        try:
            content = self._parse_chat_file(file_path)
            entities = self.extract_entities(content)
            searchable_text = self.create_searchable_text(content)
            
            # Extract structured data for chat analysis
            structured_data = {
                "participants": content.get("participants", []),
                "total_messages": len(content.get("messages", [])),
                "date_range": self._get_date_range(content.get("messages", [])),
                "message_types": self._analyze_message_types(content.get("messages", [])),
                "conversation_summary": self._create_conversation_summary(content.get("messages", []))
            }
            
            return ProcessedEvidence(
                evidence_type=self.evidence_type,
                content=content,
                metadata=metadata or {},
                entities=entities,
                searchable_text=searchable_text,
                structured_data=structured_data
            )
            
        except Exception as e:
            logger.error(f"Error processing chat evidence {file_path}: {str(e)}")
            raise
    
    def _parse_chat_file(self, file_path: str) -> Dict[str, Any]:
        """Parse chat file (CSV, JSON, or XML format)"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            return self._parse_csv_chat(file_path)
        elif file_path.suffix.lower() == '.json':
            return self._parse_json_chat(file_path)
        elif file_path.suffix.lower() == '.xml':
            return self._parse_xml_chat(file_path)
        else:
            raise ValueError(f"Unsupported chat file format: {file_path.suffix}")
    
    def _parse_csv_chat(self, file_path: Path) -> Dict[str, Any]:
        """Parse CSV chat file"""
        messages = []
        participants = set()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                message = {
                    "timestamp": row.get("timestamp", row.get("date", "")),
                    "sender": row.get("sender", row.get("from", "")),
                    "message": row.get("message", row.get("content", "")),
                    "message_type": row.get("type", "text")
                }
                messages.append(message)
                if message["sender"]:
                    participants.add(message["sender"])
        
        return {
            "messages": messages,
            "participants": list(participants),
            "platform": "unknown",
            "total_messages": len(messages)
        }
    
    def _parse_json_chat(self, file_path: Path) -> Dict[str, Any]:
        """Parse JSON chat file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            messages = data
            participants = set(msg.get("sender", "") for msg in messages if msg.get("sender"))
        else:
            messages = data.get("messages", [])
            participants = set(data.get("participants", []))
        
        return {
            "messages": messages,
            "participants": list(participants),
            "platform": data.get("platform", "unknown") if isinstance(data, dict) else "unknown",
            "total_messages": len(messages)
        }
    
    def _parse_xml_chat(self, file_path: Path) -> Dict[str, Any]:
        """Parse XML chat file"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        messages = []
        participants = set()
        
        # Handle different XML structures
        for msg_elem in root.findall(".//message"):
            message = {
                "timestamp": msg_elem.get("timestamp", msg_elem.find("timestamp").text if msg_elem.find("timestamp") is not None else ""),
                "sender": msg_elem.get("sender", msg_elem.find("sender").text if msg_elem.find("sender") is not None else ""),
                "message": msg_elem.get("content", msg_elem.text or ""),
                "message_type": msg_elem.get("type", "text")
            }
            messages.append(message)
            if message["sender"]:
                participants.add(message["sender"])
        
        return {
            "messages": messages,
            "participants": list(participants),
            "platform": root.get("platform", "unknown"),
            "total_messages": len(messages)
        }
    
    def _get_date_range(self, messages: List[Dict[str, Any]]) -> Dict[str, str]:
        """Get date range of messages"""
        if not messages:
            return {"start": "", "end": ""}
        
        timestamps = [msg.get("timestamp", "") for msg in messages if msg.get("timestamp")]
        if not timestamps:
            return {"start": "", "end": ""}
        
        return {"start": min(timestamps), "end": max(timestamps)}
    
    def _analyze_message_types(self, messages: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze types of messages"""
        types = {}
        for msg in messages:
            msg_type = msg.get("message_type", "text")
            types[msg_type] = types.get(msg_type, 0) + 1
        return types
    
    def _create_conversation_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Create a summary of the conversation for search"""
        all_text = " ".join([msg.get("message", "") for msg in messages if msg.get("message")])
        return all_text[:1000]  # Truncate for summary
    
    def create_searchable_text(self, content: Dict[str, Any]) -> str:
        """Create searchable text from chat content"""
        searchable_parts = []
        
        # Add participants
        participants = content.get("participants", [])
        if participants:
            searchable_parts.append(f"Participants: {', '.join(participants)}")
        
        # Add all message content
        messages = content.get("messages", [])
        for msg in messages:
            if msg.get("message"):
                searchable_parts.append(f"{msg.get('sender', 'Unknown')}: {msg['message']}")
        
        return "\n".join(searchable_parts)

class CallLogEvidenceHandler(BaseEvidenceHandler):
    """Handler for call log evidence"""
    
    def __init__(self):
        super().__init__(EvidenceType.CALL_LOG)
    
    def process(self, file_path: str, metadata: Dict[str, Any] = None) -> ProcessedEvidence:
        """Process call log data"""
        try:
            content = self._parse_call_log_file(file_path)
            entities = self.extract_entities(content)
            searchable_text = self.create_searchable_text(content)
            
            # Extract structured data for call analysis
            structured_data = {
                "total_calls": len(content.get("calls", [])),
                "unique_numbers": len(set(call.get("number", "") for call in content.get("calls", []))),
                "call_types": self._analyze_call_types(content.get("calls", [])),
                "date_range": self._get_call_date_range(content.get("calls", [])),
                "duration_stats": self._calculate_duration_stats(content.get("calls", []))
            }
            
            return ProcessedEvidence(
                evidence_type=self.evidence_type,
                content=content,
                metadata=metadata or {},
                entities=entities,
                searchable_text=searchable_text,
                structured_data=structured_data
            )
            
        except Exception as e:
            logger.error(f"Error processing call log evidence {file_path}: {str(e)}")
            raise
    
    def _parse_call_log_file(self, file_path: str) -> Dict[str, Any]:
        """Parse call log file"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            return self._parse_csv_calls(file_path)
        elif file_path.suffix.lower() == '.json':
            return self._parse_json_calls(file_path)
        else:
            raise ValueError(f"Unsupported call log format: {file_path.suffix}")
    
    def _parse_csv_calls(self, file_path: Path) -> Dict[str, Any]:
        """Parse CSV call log"""
        calls = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                call = {
                    "timestamp": row.get("timestamp", row.get("date", "")),
                    "number": row.get("number", row.get("phone_number", "")),
                    "duration": row.get("duration", "0"),
                    "call_type": row.get("type", row.get("call_type", "unknown")),
                    "contact_name": row.get("contact", row.get("name", ""))
                }
                calls.append(call)
        
        return {
            "calls": calls,
            "total_calls": len(calls)
        }
    
    def _parse_json_calls(self, file_path: Path) -> Dict[str, Any]:
        """Parse JSON call log"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            calls = data
        else:
            calls = data.get("calls", [])
        
        return {
            "calls": calls,
            "total_calls": len(calls)
        }
    
    def _analyze_call_types(self, calls: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze call types"""
        types = {}
        for call in calls:
            call_type = call.get("call_type", "unknown")
            types[call_type] = types.get(call_type, 0) + 1
        return types
    
    def _get_call_date_range(self, calls: List[Dict[str, Any]]) -> Dict[str, str]:
        """Get date range of calls"""
        if not calls:
            return {"start": "", "end": ""}
        
        timestamps = [call.get("timestamp", "") for call in calls if call.get("timestamp")]
        if not timestamps:
            return {"start": "", "end": ""}
        
        return {"start": min(timestamps), "end": max(timestamps)}
    
    def _calculate_duration_stats(self, calls: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate call duration statistics"""
        durations = []
        for call in calls:
            try:
                duration = float(call.get("duration", 0))
                durations.append(duration)
            except (ValueError, TypeError):
                continue
        
        if not durations:
            return {"total": 0, "average": 0, "max": 0, "min": 0}
        
        return {
            "total": sum(durations),
            "average": sum(durations) / len(durations),
            "max": max(durations),
            "min": min(durations)
        }
    
    def create_searchable_text(self, content: Dict[str, Any]) -> str:
        """Create searchable text from call log content"""
        searchable_parts = []
        
        calls = content.get("calls", [])
        for call in calls:
            parts = []
            if call.get("number"):
                parts.append(f"Number: {call['number']}")
            if call.get("contact_name"):
                parts.append(f"Contact: {call['contact_name']}")
            if call.get("call_type"):
                parts.append(f"Type: {call['call_type']}")
            if call.get("timestamp"):
                parts.append(f"Time: {call['timestamp']}")
            
            if parts:
                searchable_parts.append(" | ".join(parts))
        
        return "\n".join(searchable_parts)

class ContactEvidenceHandler(BaseEvidenceHandler):
    """Handler for contact/address book evidence"""
    
    def __init__(self):
        super().__init__(EvidenceType.CONTACT)
    
    def process(self, file_path: str, metadata: Dict[str, Any] = None) -> ProcessedEvidence:
        """Process contact data"""
        try:
            content = self._parse_contact_file(file_path)
            entities = self.extract_entities(content)
            searchable_text = self.create_searchable_text(content)
            
            structured_data = {
                "total_contacts": len(content.get("contacts", [])),
                "contacts_with_phone": len([c for c in content.get("contacts", []) if c.get("phone")]),
                "contacts_with_email": len([c for c in content.get("contacts", []) if c.get("email")]),
                "contact_groups": self._analyze_contact_groups(content.get("contacts", []))
            }
            
            return ProcessedEvidence(
                evidence_type=self.evidence_type,
                content=content,
                metadata=metadata or {},
                entities=entities,
                searchable_text=searchable_text,
                structured_data=structured_data
            )
            
        except Exception as e:
            logger.error(f"Error processing contact evidence {file_path}: {str(e)}")
            raise
    
    def _parse_contact_file(self, file_path: str) -> Dict[str, Any]:
        """Parse contact file"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            return self._parse_csv_contacts(file_path)
        elif file_path.suffix.lower() == '.json':
            return self._parse_json_contacts(file_path)
        else:
            raise ValueError(f"Unsupported contact format: {file_path.suffix}")
    
    def _parse_csv_contacts(self, file_path: Path) -> Dict[str, Any]:
        """Parse CSV contact file"""
        contacts = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                contact = {
                    "name": row.get("name", ""),
                    "phone": row.get("phone", row.get("phone_number", "")),
                    "email": row.get("email", ""),
                    "organization": row.get("organization", row.get("company", "")),
                    "notes": row.get("notes", "")
                }
                contacts.append(contact)
        
        return {
            "contacts": contacts,
            "total_contacts": len(contacts)
        }
    
    def _parse_json_contacts(self, file_path: Path) -> Dict[str, Any]:
        """Parse JSON contact file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            contacts = data
        else:
            contacts = data.get("contacts", [])
        
        return {
            "contacts": contacts,
            "total_contacts": len(contacts)
        }
    
    def _analyze_contact_groups(self, contacts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze contact groups/organizations"""
        groups = {}
        for contact in contacts:
            org = contact.get("organization", "Unknown")
            groups[org] = groups.get(org, 0) + 1
        return groups
    
    def create_searchable_text(self, content: Dict[str, Any]) -> str:
        """Create searchable text from contact content"""
        searchable_parts = []
        
        contacts = content.get("contacts", [])
        for contact in contacts:
            parts = []
            if contact.get("name"):
                parts.append(f"Name: {contact['name']}")
            if contact.get("phone"):
                parts.append(f"Phone: {contact['phone']}")
            if contact.get("email"):
                parts.append(f"Email: {contact['email']}")
            if contact.get("organization"):
                parts.append(f"Organization: {contact['organization']}")
            
            if parts:
                searchable_parts.append(" | ".join(parts))
        
        return "\n".join(searchable_parts)

class DocumentEvidenceHandler(BaseEvidenceHandler):
    """Handler for document evidence (text reports, PDFs, etc.)"""
    
    def __init__(self):
        super().__init__(EvidenceType.DOCUMENT)
    
    def process(self, file_path: str, metadata: Dict[str, Any] = None) -> ProcessedEvidence:
        """Process document evidence"""
        try:
            content = self._parse_document_file(file_path)
            entities = self.extract_entities(content)
            searchable_text = self.create_searchable_text(content)
            
            structured_data = {
                "document_type": content.get("document_type", "unknown"),
                "word_count": len(content.get("text", "").split()),
                "page_count": content.get("page_count", 1),
                "language": content.get("language", "en")
            }
            
            return ProcessedEvidence(
                evidence_type=self.evidence_type,
                content=content,
                metadata=metadata or {},
                entities=entities,
                searchable_text=searchable_text,
                structured_data=structured_data
            )
            
        except Exception as e:
            logger.error(f"Error processing document evidence {file_path}: {str(e)}")
            raise
    
    def _parse_document_file(self, file_path: str) -> Dict[str, Any]:
        """Parse document file"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return {
                "document_type": "text",
                "text": text,
                "filename": file_path.name
            }
        else:
            # Placeholder for future PDF, DOCX parsing
            return {
                "document_type": "unknown",
                "text": f"Document file: {file_path.name}",
                "filename": file_path.name,
                "note": "Full document parsing not yet implemented"
            }
    
    def create_searchable_text(self, content: Dict[str, Any]) -> str:
        """Create searchable text from document content"""
        return content.get("text", "")

# Placeholder handlers for future multi-modal evidence
class ImageEvidenceHandler(BaseEvidenceHandler):
    """Handler for image evidence (will use CLIP embeddings in future)"""
    
    def __init__(self):
        super().__init__(EvidenceType.IMAGE)
    
    def process(self, file_path: str, metadata: Dict[str, Any] = None) -> ProcessedEvidence:
        """Process image evidence (placeholder for CLIP implementation)"""
        file_path = Path(file_path)
        
        content = {
            "image_path": str(file_path),
            "filename": file_path.name,
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "note": "Image embeddings with CLIP will be implemented later"
        }
        
        return ProcessedEvidence(
            evidence_type=self.evidence_type,
            content=content,
            metadata=metadata or {},
            entities={},
            searchable_text=f"Image file: {file_path.name}",
            structured_data={"image_format": file_path.suffix}
        )

class VideoEvidenceHandler(BaseEvidenceHandler):
    """Handler for video evidence (placeholder for future implementation)"""
    
    def __init__(self):
        super().__init__(EvidenceType.VIDEO)
    
    def process(self, file_path: str, metadata: Dict[str, Any] = None) -> ProcessedEvidence:
        """Process video evidence (placeholder)"""
        file_path = Path(file_path)
        
        content = {
            "video_path": str(file_path),
            "filename": file_path.name,
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "note": "Video processing will be implemented later"
        }
        
        return ProcessedEvidence(
            evidence_type=self.evidence_type,
            content=content,
            metadata=metadata or {},
            entities={},
            searchable_text=f"Video file: {file_path.name}",
            structured_data={"video_format": file_path.suffix}
        )

class AudioEvidenceHandler(BaseEvidenceHandler):
    """Handler for audio evidence (placeholder for future implementation)"""
    
    def __init__(self):
        super().__init__(EvidenceType.AUDIO)
    
    def process(self, file_path: str, metadata: Dict[str, Any] = None) -> ProcessedEvidence:
        """Process audio evidence (placeholder)"""
        file_path = Path(file_path)
        
        content = {
            "audio_path": str(file_path),
            "filename": file_path.name,
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "note": "Audio processing and transcription will be implemented later"
        }
        
        return ProcessedEvidence(
            evidence_type=self.evidence_type,
            content=content,
            metadata=metadata or {},
            entities={},
            searchable_text=f"Audio file: {file_path.name}",
            structured_data={"audio_format": file_path.suffix}
        )

class EvidenceHandlerFactory:
    """Factory for creating appropriate evidence handlers"""
    
    _handlers = {
        EvidenceType.CHAT: ChatEvidenceHandler,
        EvidenceType.CALL_LOG: CallLogEvidenceHandler,
        EvidenceType.CONTACT: ContactEvidenceHandler,
        EvidenceType.DOCUMENT: DocumentEvidenceHandler,
        EvidenceType.TEXT_REPORT: DocumentEvidenceHandler,
        EvidenceType.IMAGE: ImageEvidenceHandler,
        EvidenceType.VIDEO: VideoEvidenceHandler,
        EvidenceType.AUDIO: AudioEvidenceHandler,
        # XML, JSON, CSV use existing parsers
        EvidenceType.XML_REPORT: DocumentEvidenceHandler,
        EvidenceType.JSON_DATA: DocumentEvidenceHandler,
        EvidenceType.CSV_DATA: DocumentEvidenceHandler,
    }
    
    @classmethod
    def get_handler(cls, evidence_type: EvidenceType) -> BaseEvidenceHandler:
        """Get appropriate handler for evidence type"""
        handler_class = cls._handlers.get(evidence_type)
        if not handler_class:
            raise ValueError(f"No handler available for evidence type: {evidence_type}")
        
        return handler_class()
    
    @classmethod
    def get_supported_types(cls) -> List[EvidenceType]:
        """Get list of supported evidence types"""
        return list(cls._handlers.keys())

# Default instance
evidence_handler_factory = EvidenceHandlerFactory()