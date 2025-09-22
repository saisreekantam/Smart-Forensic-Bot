"""
Simple Data Processor for Forensic Evidence
A straightforward, working data processing system
"""

import os
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import sqlite3
import uuid
import re
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("PyPDF2 not available - PDF processing disabled")

# Import enum for status values
from src.database.models import ProcessingStatus

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataProcessor:
    """
    Simple, reliable data processor that actually works
    """
    
    def __init__(self, db_path: str = "data/forensic_cases.db"):
        self.db_path = db_path
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(exist_ok=True)
        
    def process_case_evidence(self, case_id: str) -> Dict[str, Any]:
        """
        Process all evidence for a case and save to processed directory
        """
        logger.info(f"🚀 Starting simple processing for case {case_id}")
        
        results = {
            "case_id": case_id,
            "processing_time": datetime.now().isoformat(),
            "processed_files": [],
            "errors": [],
            "total_processed": 0
        }
        
        try:
            # Get evidence files from database
            evidence_files = self._get_evidence_files(case_id)
            
            if not evidence_files:
                logger.warning(f"No evidence files found for case {case_id}")
                return results
            
            for evidence in evidence_files:
                try:
                    logger.info(f"Processing: {evidence['original_filename']}")
                    
                    # Process the file based on type
                    processed_data = self._process_single_file(evidence, case_id)
                    
                    if processed_data:
                        # Save processed data
                        output_file = self._save_processed_data(evidence, processed_data, case_id)
                        
                        # Update database
                        self._update_evidence_status(evidence['id'], ProcessingStatus.COMPLETED.value, True)
                        
                        results["processed_files"].append({
                            "evidence_id": evidence['id'],
                            "filename": evidence['original_filename'],
                            "output_file": output_file,
                            "record_count": len(processed_data.get('records', [])),
                            "processing_status": "completed"
                        })
                        results["total_processed"] += 1
                        
                        logger.info(f"✅ Successfully processed {evidence['original_filename']}")
                    
                except Exception as e:
                    error_msg = f"Failed to process {evidence['original_filename']}: {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
                    self._update_evidence_status(evidence['id'], ProcessingStatus.FAILED.value, False)
            
            # Update case processing progress
            self._update_case_progress(case_id)
            
            # Save overall results
            results_file = self.processed_dir / f"case_{case_id}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"🎉 Processing complete! Processed {results['total_processed']} files")
            return results
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {str(e)}")
            results["errors"].append(f"Overall processing error: {str(e)}")
            return results
    
    def _get_evidence_files(self, case_id: str) -> List[Dict[str, Any]]:
        """Get evidence files from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, original_filename, file_path, evidence_type, processing_status
                FROM evidence 
                WHERE case_id = ? AND processing_status = ?
            """, (case_id, ProcessingStatus.PENDING.value))
            
            evidence_files = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return evidence_files
            
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return []
    
    def _process_single_file(self, evidence: Dict[str, Any], case_id: str) -> Optional[Dict[str, Any]]:
        """Process a single evidence file"""
        try:
            # The file_path already contains the full path from data/cases/...
            file_path = Path(evidence['file_path'])
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            # Determine file type and process accordingly
            if evidence['evidence_type'] == 'ufdr' or file_path.suffix.lower() == '.xml':
                return self._process_xml_file(file_path, evidence)
            elif file_path.suffix.lower() == '.csv':
                return self._process_csv_file(file_path, evidence)
            elif file_path.suffix.lower() == '.json':
                return self._process_json_file(file_path, evidence)
            elif file_path.suffix.lower() == '.txt':
                return self._process_text_file(file_path, evidence)
            elif file_path.suffix.lower() == '.pdf':
                return self._process_pdf_file(file_path, evidence)
            else:
                # Try to process as text
                return self._process_text_file(file_path, evidence)
                
        except Exception as e:
            logger.error(f"Error processing file {evidence['original_filename']}: {str(e)}")
            return None
    
    def _process_xml_file(self, file_path: Path, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Process XML/UFDR files"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            processed_data = {
                "file_type": "xml",
                "source_file": evidence['original_filename'],
                "evidence_id": evidence['id'],
                "records": [],
                "metadata": {
                    "root_tag": root.tag,
                    "processing_time": datetime.now().isoformat()
                }
            }
            
            # Extract data based on common UFDR patterns
            self._extract_xml_data(root, processed_data["records"])
            
            return processed_data
            
        except Exception as e:
            logger.error(f"XML processing error: {str(e)}")
            return None
    
    def _process_csv_file(self, file_path: Path, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Process CSV files"""
        try:
            import csv
            
            processed_data = {
                "file_type": "csv",
                "source_file": evidence['original_filename'],
                "evidence_id": evidence['id'],
                "records": [],
                "metadata": {
                    "processing_time": datetime.now().isoformat()
                }
            }
            
            with open(file_path, 'r', encoding='utf-8') as f:
                # Try to detect delimiter
                sample = f.read(1024)
                f.seek(0)
                
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(f, delimiter=delimiter)
                for row_num, row in enumerate(reader):
                    record = {
                        "id": str(uuid.uuid4()),
                        "row_number": row_num + 1,
                        "data": dict(row),
                        "searchable_text": " ".join(str(v) for v in row.values() if v)
                    }
                    processed_data["records"].append(record)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"CSV processing error: {str(e)}")
            return None
    
    def _process_json_file(self, file_path: Path, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Process JSON files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            processed_data = {
                "file_type": "json",
                "source_file": evidence['original_filename'],
                "evidence_id": evidence['id'],
                "records": [],
                "metadata": {
                    "processing_time": datetime.now().isoformat()
                }
            }
            
            # Convert JSON to searchable records
            if isinstance(data, list):
                for i, item in enumerate(data):
                    record = {
                        "id": str(uuid.uuid4()),
                        "index": i,
                        "data": item,
                        "searchable_text": json.dumps(item, ensure_ascii=False)
                    }
                    processed_data["records"].append(record)
            else:
                record = {
                    "id": str(uuid.uuid4()),
                    "index": 0,
                    "data": data,
                    "searchable_text": json.dumps(data, ensure_ascii=False)
                }
                processed_data["records"].append(record)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"JSON processing error: {str(e)}")
            return None
    
    def _process_text_file(self, file_path: Path, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Process text files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            processed_data = {
                "file_type": "text",
                "source_file": evidence['original_filename'],
                "evidence_id": evidence['id'],
                "records": [],
                "metadata": {
                    "processing_time": datetime.now().isoformat(),
                    "character_count": len(content)
                }
            }
            
            # Split into chunks for better searchability
            chunk_size = 1000  # characters per chunk
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    record = {
                        "id": str(uuid.uuid4()),
                        "chunk_number": i + 1,
                        "data": {"content": chunk.strip()},
                        "searchable_text": chunk.strip()
                    }
                    processed_data["records"].append(record)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Text processing error: {str(e)}")
            return None
    
    def _process_pdf_file(self, file_path: Path, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Process PDF files by extracting text content"""
        try:
            if not PDF_AVAILABLE:
                logger.error("PyPDF2 not available - cannot process PDF files")
                return None
            
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                full_text = ""
                
                # Extract text from all pages
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    except Exception as e:
                        logger.warning(f"Could not extract text from page {page_num + 1}: {str(e)}")
                        continue
            
            if not full_text.strip():
                logger.warning(f"No text content extracted from PDF: {file_path}")
                return None
            
            processed_data = {
                "file_type": "pdf",
                "source_file": evidence['original_filename'],
                "evidence_id": evidence['id'],
                "records": [],
                "metadata": {
                    "processing_time": datetime.now().isoformat(),
                    "page_count": len(reader.pages),
                    "character_count": len(full_text)
                }
            }
            
            # Enhanced text processing for forensic data
            text_sections = self._extract_forensic_sections_from_text(full_text)
            
            # If we found structured sections, use them
            if text_sections:
                for section in text_sections:
                    record = {
                        "id": str(uuid.uuid4()),
                        "section_type": section.get("type", "general"),
                        "data": section.get("data", {}),
                        "searchable_text": section.get("text", "")
                    }
                    processed_data["records"].append(record)
            else:
                # Fall back to chunking the text
                chunk_size = 1500  # Larger chunks for PDF content
                chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
                
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        record = {
                            "id": str(uuid.uuid4()),
                            "chunk_number": i + 1,
                            "data": {"content": chunk.strip()},
                            "searchable_text": chunk.strip()
                        }
                        processed_data["records"].append(record)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
            return None
    
    def _extract_forensic_sections_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract structured forensic data sections from text"""
        sections = []
        
        try:
            # Split text into lines for analysis
            lines = text.split('\n')
            current_section = None
            current_content = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Detect different types of forensic data
                if self._is_message_section(line):
                    if current_section:
                        sections.append(self._finalize_section(current_section, current_content))
                    current_section = {"type": "message", "data": {}}
                    current_content = [line]
                    
                elif self._is_call_section(line):
                    if current_section:
                        sections.append(self._finalize_section(current_section, current_content))
                    current_section = {"type": "call", "data": {}}
                    current_content = [line]
                    
                elif self._is_contact_section(line):
                    if current_section:
                        sections.append(self._finalize_section(current_section, current_content))
                    current_section = {"type": "contact", "data": {}}
                    current_content = [line]
                    
                else:
                    if current_section:
                        current_content.append(line)
                    else:
                        # Start a general section
                        current_section = {"type": "general", "data": {}}
                        current_content = [line]
            
            # Don't forget the last section
            if current_section:
                sections.append(self._finalize_section(current_section, current_content))
                
        except Exception as e:
            logger.error(f"Error extracting forensic sections: {str(e)}")
        
        return sections
    
    def _is_message_section(self, line: str) -> bool:
        """Check if line indicates start of message section"""
        message_indicators = [
            'message', 'sms', 'text', 'chat', 'whatsapp', 'telegram',
            'sender', 'recipient', 'from:', 'to:', 'msg:', 'conversation'
        ]
        line_lower = line.lower()
        return any(indicator in line_lower for indicator in message_indicators)
    
    def _is_call_section(self, line: str) -> bool:
        """Check if line indicates start of call section"""
        call_indicators = [
            'call', 'phone', 'dial', 'duration', 'incoming', 'outgoing',
            'missed', 'answered', 'caller', 'callee', 'number:'
        ]
        line_lower = line.lower()
        return any(indicator in line_lower for indicator in call_indicators)
    
    def _is_contact_section(self, line: str) -> bool:
        """Check if line indicates start of contact section"""
        contact_indicators = [
            'contact', 'address book', 'phonebook', 'name:', 'phone:',
            'email:', 'address:', 'contact info'
        ]
        line_lower = line.lower()
        return any(indicator in line_lower for indicator in contact_indicators)
    
    def _finalize_section(self, section: Dict[str, Any], content: List[str]) -> Dict[str, Any]:
        """Finalize a section by extracting relevant data"""
        full_text = '\n'.join(content)
        section["text"] = full_text
        
        # Extract specific data based on section type
        if section["type"] == "message":
            section["data"] = self._extract_message_data(full_text)
        elif section["type"] == "call":
            section["data"] = self._extract_call_data(full_text)
        elif section["type"] == "contact":
            section["data"] = self._extract_contact_data(full_text)
        else:
            section["data"] = {"content": full_text}
        
        return section
    
    def _extract_message_data(self, text: str) -> Dict[str, Any]:
        """Extract structured message data from text"""
        data = {}
        
        # Look for phone numbers
        phone_pattern = r'(\+?\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{0,4})'
        phones = re.findall(phone_pattern, text)
        if phones:
            data["phone_numbers"] = list(set(phones))
        
        # Look for timestamps
        time_patterns = [
            r'\d{4}-\d{2}-\d{2}[\s\T]\d{2}:\d{2}:\d{2}',
            r'\d{2}/\d{2}/\d{4}\s\d{1,2}:\d{2}',
            r'\d{1,2}:\d{2}\s?(AM|PM)'
        ]
        for pattern in time_patterns:
            times = re.findall(pattern, text, re.IGNORECASE)
            if times:
                data["timestamps"] = times
                break
        
        # Extract message content
        data["message_content"] = text
        
        return data
    
    def _extract_call_data(self, text: str) -> Dict[str, Any]:
        """Extract structured call data from text"""
        data = {}
        
        # Look for phone numbers
        phone_pattern = r'(\+?\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{0,4})'
        phones = re.findall(phone_pattern, text)
        if phones:
            data["phone_numbers"] = list(set(phones))
        
        # Look for duration
        duration_pattern = r'duration[:\s]*(\d+:\d+:\d+|\d+\s?(min|sec|hour))'
        duration = re.search(duration_pattern, text, re.IGNORECASE)
        if duration:
            data["duration"] = duration.group(1)
        
        # Look for call type
        if re.search(r'incoming|received', text, re.IGNORECASE):
            data["call_type"] = "incoming"
        elif re.search(r'outgoing|made|dialed', text, re.IGNORECASE):
            data["call_type"] = "outgoing"
        elif re.search(r'missed', text, re.IGNORECASE):
            data["call_type"] = "missed"
        
        data["call_content"] = text
        
        return data
    
    def _extract_contact_data(self, text: str) -> Dict[str, Any]:
        """Extract structured contact data from text"""
        data = {}
        
        # Look for phone numbers
        phone_pattern = r'(\+?\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{0,4})'
        phones = re.findall(phone_pattern, text)
        if phones:
            data["phone_numbers"] = list(set(phones))
        
        # Look for emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            data["emails"] = emails
        
        # Look for names (simple heuristic)
        name_pattern = r'name[:\s]*([A-Za-z\s]+?)(?:\n|phone|email|$)'
        name_match = re.search(name_pattern, text, re.IGNORECASE)
        if name_match:
            data["name"] = name_match.group(1).strip()
        
        data["contact_content"] = text
        
        return data
    
    def _extract_xml_data(self, element, records, parent_path=""):
        """Recursively extract data from XML elements"""
        try:
            current_path = f"{parent_path}/{element.tag}" if parent_path else element.tag
            
            # If element has text content, create a record
            if element.text and element.text.strip():
                record = {
                    "id": str(uuid.uuid4()),
                    "path": current_path,
                    "tag": element.tag,
                    "data": {
                        "text": element.text.strip(),
                        "attributes": element.attrib
                    },
                    "searchable_text": element.text.strip()
                }
                records.append(record)
            
            # Process child elements
            for child in element:
                self._extract_xml_data(child, records, current_path)
                
        except Exception as e:
            logger.error(f"XML extraction error: {str(e)}")
    
    def _save_processed_data(self, evidence: Dict[str, Any], processed_data: Dict[str, Any], case_id: str) -> str:
        """Save processed data to file"""
        try:
            # Create case-specific directory
            case_dir = self.processed_dir / case_id
            case_dir.mkdir(exist_ok=True)
            
            # Generate filename
            base_name = Path(evidence['original_filename']).stem
            output_file = case_dir / f"{base_name}_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Save data
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Save error: {str(e)}")
            return ""
    
    def _update_evidence_status(self, evidence_id: str, status: str, has_embeddings: bool):
        """Update evidence processing status in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE evidence 
                SET processing_status = ?, has_embeddings = ?, updated_at = ?
                WHERE id = ?
            """, (status, has_embeddings, datetime.now().isoformat(), evidence_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database update error: {str(e)}")
    
    def _update_case_progress(self, case_id: str):
        """Update case processing progress"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate progress
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN processing_status = ? THEN 1 ELSE 0 END) as completed
                FROM evidence 
                WHERE case_id = ?
            """, (ProcessingStatus.COMPLETED.value, case_id))
            
            result = cursor.fetchone()
            total, completed = result[0], result[1]
            
            progress = (completed / total * 100) if total > 0 else 0
            
            # Update case
            cursor.execute("""
                UPDATE cases 
                SET processed_evidence_count = ?, updated_at = ?
                WHERE id = ?
            """, (completed, datetime.now().isoformat(), case_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Case progress updated: {completed}/{total} files ({progress:.1f}%)")
            
        except Exception as e:
            logger.error(f"Progress update error: {str(e)}")

def main():
    """Test the processor"""
    processor = SimpleDataProcessor()
    
    # Get case ID for testing
    conn = sqlite3.connect("data/forensic_cases.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM cases WHERE case_number = 'ENHANCED-2024-001'")
    result = cursor.fetchone()
    conn.close()
    
    if result:
        case_id = result[0]
        print(f"Testing with case ID: {case_id}")
        results = processor.process_case_evidence(case_id)
        print(f"Results: {json.dumps(results, indent=2)}")
    else:
        print("No test case found")

if __name__ == "__main__":
    main()