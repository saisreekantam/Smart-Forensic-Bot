"""
Data Ingestion Module for Project Sentinel
Handles parsing and preprocessing of UFDR files (XML, CSV, TXT, JSON formats)
"""

import os
import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
try:
    import xmltodict
except ImportError:
    xmltodict = None
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
import re
from datetime import datetime
import logging
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, ForensicEntities

# Set up logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

@dataclass
class UFDRDocument:
    """Represents a processed UFDR document"""
    file_path: str
    file_type: str
    case_id: Optional[str]
    device_info: Dict[str, Any]
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    extracted_at: datetime
    
class UFDRParser:
    """Base parser for UFDR files"""
    
    def __init__(self):
        self.supported_formats = settings.supported_formats
        self.max_file_size = settings.max_file_size_mb * 1024 * 1024  # Convert to bytes
        
    def validate_file(self, file_path: str) -> bool:
        """Validate file size and format"""
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return False
            
        # Check file size
        if path.stat().st_size > self.max_file_size:
            logger.error(f"File too large: {file_path} ({path.stat().st_size} bytes)")
            return False
            
        # Check file extension
        if path.suffix.lower().lstrip('.') not in self.supported_formats:
            logger.error(f"Unsupported file format: {path.suffix}")
            return False
            
        return True
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract basic file metadata"""
        path = Path(file_path)
        stat = path.stat()
        
        return {
            "filename": path.name,
            "file_size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime),
            "modified_at": datetime.fromtimestamp(stat.st_mtime),
            "file_extension": path.suffix.lower()
        }

class XMLParser(UFDRParser):
    """Parser for XML UFDR files"""
    
    def parse(self, file_path: str) -> UFDRDocument:
        """Parse XML UFDR file"""
        if not self.validate_file(file_path):
            raise ValueError(f"Invalid file: {file_path}")
            
        logger.info(f"Parsing XML file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Parse with xmltodict for easier handling, fallback to ET if not available
            if xmltodict:
                parsed_data = xmltodict.parse(content)
            else:
                # Fallback to ElementTree parsing
                root = ET.fromstring(content)
                parsed_data = self._et_to_dict(root)
            
            # Extract device information
            device_info = self._extract_device_info(parsed_data)
            
            # Extract case information
            case_id = self._extract_case_id(parsed_data)
            
            # Extract communication data
            communications = self._extract_communications(parsed_data)
            
            # Extract contacts
            contacts = self._extract_contacts(parsed_data)
            
            # Extract call logs
            call_logs = self._extract_call_logs(parsed_data)
            
            # Extract media files info
            media_files = self._extract_media_files(parsed_data)
            
            content_data = {
                "communications": communications,
                "contacts": contacts,
                "call_logs": call_logs,
                "media_files": media_files,
                "raw_data": parsed_data
            }
            
            metadata = self.extract_metadata(file_path)
            
            return UFDRDocument(
                file_path=file_path,
                file_type="xml",
                case_id=case_id,
                device_info=device_info,
                content=content_data,
                metadata=metadata,
                extracted_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error parsing XML file {file_path}: {str(e)}")
            raise
    
    def _et_to_dict(self, element):
        """Convert ElementTree element to dictionary (fallback for xmltodict)"""
        result = {}
        
        # Add attributes
        if element.attrib:
            result.update(element.attrib)
        
        # Add text content
        if element.text and element.text.strip():
            if len(element) == 0:  # Leaf node
                return element.text.strip()
            else:
                result['text'] = element.text.strip()
        
        # Add children
        children = {}
        for child in element:
            child_result = self._et_to_dict(child)
            if child.tag in children:
                # Convert to list if multiple children with same tag
                if not isinstance(children[child.tag], list):
                    children[child.tag] = [children[child.tag]]
                children[child.tag].append(child_result)
            else:
                children[child.tag] = child_result
        
        result.update(children)
        return result
    
    def _extract_device_info(self, data: Dict) -> Dict[str, Any]:
        """Extract device information from XML data"""
        device_info = {}
        
        # Common UFDR XML structures for device info
        paths_to_check = [
            "report.device",
            "extraction.device",
            "deviceInfo",
            "device"
        ]
        
        for path in paths_to_check:
            try:
                current = data
                for key in path.split('.'):
                    current = current.get(key, {})
                if current:
                    device_info.update(current if isinstance(current, dict) else {"value": current})
            except:
                continue
                
        return device_info
    
    def _extract_case_id(self, data: Dict) -> Optional[str]:
        """Extract case ID from XML data"""
        paths_to_check = [
            "report.case.id",
            "case.id",
            "caseId",
            "case_id"
        ]
        
        for path in paths_to_check:
            try:
                current = data
                for key in path.split('.'):
                    current = current.get(key)
                if current:
                    return str(current)
            except:
                continue
                
        return None
    
    def _extract_communications(self, data: Dict) -> List[Dict]:
        """Extract chat/message communications"""
        communications = []
        
        # Common paths for communications in UFDR XML
        comm_paths = [
            "report.communications.messages",
            "messages",
            "chats",
            "conversations"
        ]
        
        for path in comm_paths:
            try:
                current = data
                for key in path.split('.'):
                    current = current.get(key, {})
                
                if isinstance(current, list):
                    communications.extend(current)
                elif isinstance(current, dict):
                    # If it's a dict, it might contain a list of messages
                    for key, value in current.items():
                        if isinstance(value, list):
                            communications.extend(value)
                            
            except:
                continue
                
        return communications
    
    def _extract_contacts(self, data: Dict) -> List[Dict]:
        """Extract contacts information"""
        contacts = []
        
        contact_paths = [
            "report.contacts",
            "contacts",
            "phonebook"
        ]
        
        for path in contact_paths:
            try:
                current = data
                for key in path.split('.'):
                    current = current.get(key, {})
                
                if isinstance(current, list):
                    contacts.extend(current)
                elif isinstance(current, dict):
                    for key, value in current.items():
                        if isinstance(value, list):
                            contacts.extend(value)
                            
            except:
                continue
                
        return contacts
    
    def _extract_call_logs(self, data: Dict) -> List[Dict]:
        """Extract call logs"""
        call_logs = []
        
        call_paths = [
            "report.calls",
            "calls",
            "callHistory",
            "call_logs"
        ]
        
        for path in call_paths:
            try:
                current = data
                for key in path.split('.'):
                    current = current.get(key, {})
                
                if isinstance(current, list):
                    call_logs.extend(current)
                elif isinstance(current, dict):
                    for key, value in current.items():
                        if isinstance(value, list):
                            call_logs.extend(value)
                            
            except:
                continue
                
        return call_logs
    
    def _extract_media_files(self, data: Dict) -> List[Dict]:
        """Extract media files information"""
        media_files = []
        
        media_paths = [
            "report.media",
            "media",
            "files",
            "multimedia"
        ]
        
        for path in media_paths:
            try:
                current = data
                for key in path.split('.'):
                    current = current.get(key, {})
                
                if isinstance(current, list):
                    media_files.extend(current)
                elif isinstance(current, dict):
                    for key, value in current.items():
                        if isinstance(value, list):
                            media_files.extend(value)
                            
            except:
                continue
                
        return media_files

class CSVParser(UFDRParser):
    """Parser for CSV UFDR files"""
    
    def parse(self, file_path: str) -> UFDRDocument:
        """Parse CSV UFDR file"""
        if not self.validate_file(file_path):
            raise ValueError(f"Invalid file: {file_path}")
            
        logger.info(f"Parsing CSV file: {file_path}")
        
        try:
            # Read CSV with pandas
            df = pd.read_csv(file_path, encoding='utf-8')
            
            # Convert to list of dictionaries
            data_records = df.to_dict('records')
            
            # Try to infer the type of CSV (calls, messages, contacts, etc.)
            csv_type = self._infer_csv_type(df.columns.tolist())
            
            content_data = {
                "type": csv_type,
                "records": data_records,
                "columns": df.columns.tolist(),
                "row_count": len(df)
            }
            
            # Extract basic device info if available
            device_info = self._extract_device_info_from_csv(data_records)
            
            metadata = self.extract_metadata(file_path)
            
            return UFDRDocument(
                file_path=file_path,
                file_type="csv",
                case_id=None,  # Usually not available in CSV
                device_info=device_info,
                content=content_data,
                metadata=metadata,
                extracted_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error parsing CSV file {file_path}: {str(e)}")
            raise
    
    def _infer_csv_type(self, columns: List[str]) -> str:
        """Infer the type of CSV based on column names"""
        columns_lower = [col.lower() for col in columns]
        
        # Check for common column patterns
        if any(col in columns_lower for col in ['message', 'text', 'content', 'body']):
            return "messages"
        elif any(col in columns_lower for col in ['call', 'duration', 'direction']):
            return "calls"
        elif any(col in columns_lower for col in ['contact', 'name', 'phone']):
            return "contacts"
        elif any(col in columns_lower for col in ['file', 'media', 'image', 'video']):
            return "media"
        else:
            return "unknown"
    
    def _extract_device_info_from_csv(self, records: List[Dict]) -> Dict[str, Any]:
        """Extract device info from CSV records if available"""
        device_info = {}
        
        # Look for device-related information in the first few records
        for record in records[:5]:  # Check first 5 records
            for key, value in record.items():
                if key.lower() in ['device', 'imei', 'serial', 'model', 'brand']:
                    device_info[key.lower()] = value
                    
        return device_info

class TXTParser(UFDRParser):
    """Parser for plain text UFDR files"""
    
    def parse(self, file_path: str) -> UFDRDocument:
        """Parse TXT UFDR file"""
        if not self.validate_file(file_path):
            raise ValueError(f"Invalid file: {file_path}")
            
        logger.info(f"Parsing TXT file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Try to parse structured text
            parsed_content = self._parse_structured_text(content)
            
            content_data = {
                "raw_text": content,
                "parsed_sections": parsed_content,
                "line_count": len(content.splitlines())
            }
            
            metadata = self.extract_metadata(file_path)
            
            return UFDRDocument(
                file_path=file_path,
                file_type="txt",
                case_id=None,
                device_info={},
                content=content_data,
                metadata=metadata,
                extracted_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error parsing TXT file {file_path}: {str(e)}")
            raise
    
    def _parse_structured_text(self, content: str) -> Dict[str, Any]:
        """Parse structured text content"""
        sections = {}
        current_section = "general"
        current_content = []
        
        lines = content.splitlines()
        
        for line in lines:
            line = line.strip()
            
            # Check if line is a section header
            if self._is_section_header(line):
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content)
                    current_content = []
                
                current_section = line.lower().replace(":", "").replace("-", "_")
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = "\n".join(current_content)
        
        return sections
    
    def _is_section_header(self, line: str) -> bool:
        """Check if a line is a section header"""
        # Common section headers in UFDR text files
        headers = [
            "device information", "call logs", "messages", "contacts",
            "media files", "applications", "web history", "location data"
        ]
        
        line_lower = line.lower()
        return any(header in line_lower for header in headers) or line.endswith(":")

class JSONParser(UFDRParser):
    """Parser for JSON UFDR files"""
    
    def parse(self, file_path: str) -> UFDRDocument:
        """Parse JSON UFDR file"""
        if not self.validate_file(file_path):
            raise ValueError(f"Invalid file: {file_path}")
            
        logger.info(f"Parsing JSON file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Extract structured information
            device_info = data.get("device", {})
            case_id = data.get("case_id") or data.get("caseId")
            
            content_data = {
                "raw_data": data,
                "messages": data.get("messages", []),
                "calls": data.get("calls", []),
                "contacts": data.get("contacts", []),
                "media": data.get("media", [])
            }
            
            metadata = self.extract_metadata(file_path)
            
            return UFDRDocument(
                file_path=file_path,
                file_type="json",
                case_id=case_id,
                device_info=device_info,
                content=content_data,
                metadata=metadata,
                extracted_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error parsing JSON file {file_path}: {str(e)}")
            raise

class UFDRDataIngestion:
    """Main data ingestion orchestrator"""
    
    def __init__(self):
        self.parsers = {
            'xml': XMLParser(),
            'csv': CSVParser(),
            'txt': TXTParser(),
            'json': JSONParser()
        }
    
    def process_file(self, file_path: str) -> UFDRDocument:
        """Process a single UFDR file"""
        path = Path(file_path)
        file_extension = path.suffix.lower().lstrip('.')
        
        if file_extension not in self.parsers:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        parser = self.parsers[file_extension]
        return parser.parse(file_path)
    
    def process_directory(self, directory_path: str) -> List[UFDRDocument]:
        """Process all UFDR files in a directory"""
        directory = Path(directory_path)
        documents = []
        
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Find all supported files
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                extension = file_path.suffix.lower().lstrip('.')
                if extension in self.parsers:
                    try:
                        document = self.process_file(str(file_path))
                        documents.append(document)
                        logger.info(f"Successfully processed: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {str(e)}")
        
        return documents
    
    def save_processed_document(self, document: UFDRDocument, output_dir: str = None) -> str:
        """Save processed document to JSON"""
        if output_dir is None:
            output_dir = settings.processed_data_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename based on original file
        original_name = Path(document.file_path).stem
        output_file = output_path / f"{original_name}_processed.json"
        
        # Convert document to dict for JSON serialization
        doc_dict = {
            "file_path": document.file_path,
            "file_type": document.file_type,
            "case_id": document.case_id,
            "device_info": document.device_info,
            "content": document.content,
            "metadata": document.metadata,
            "extracted_at": document.extracted_at.isoformat()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(doc_dict, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Saved processed document: {output_file}")
        return str(output_file)