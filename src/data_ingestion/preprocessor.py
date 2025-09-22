"""
Text Preprocessing Module for Project Sentinel
Handles text cleaning, normalization, and preparation for AI processing
"""

import re
import string
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import unicodedata
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import ForensicEntities

logger = logging.getLogger(__name__)

class TextCleaner:
    """Handles text cleaning and normalization"""
    
    def __init__(self):
        self.forensic_patterns = ForensicEntities.PATTERNS
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for processing"""
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove or replace special characters while preserving forensic entities
        text = self._preserve_forensic_entities(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _preserve_forensic_entities(self, text: str) -> str:
        """Preserve important forensic entities during cleaning"""
        # Store found entities temporarily
        entities = {}
        entity_counter = 0
        
        # Phone numbers
        phone_pattern = self.forensic_patterns['phone_number']
        for match in re.finditer(phone_pattern, text):
            placeholder = f"__PHONE_{entity_counter}__"
            entities[placeholder] = match.group()
            text = text.replace(match.group(), placeholder)
            entity_counter += 1
        
        # Email addresses
        email_pattern = self.forensic_patterns['email']
        for match in re.finditer(email_pattern, text):
            placeholder = f"__EMAIL_{entity_counter}__"
            entities[placeholder] = match.group()
            text = text.replace(match.group(), placeholder)
            entity_counter += 1
        
        # Crypto addresses
        for crypto_type, pattern in self.forensic_patterns['crypto_address'].items():
            for match in re.finditer(pattern, text):
                placeholder = f"__CRYPTO_{entity_counter}__"
                entities[placeholder] = match.group()
                text = text.replace(match.group(), placeholder)
                entity_counter += 1
        
        # Clean the text (remove excessive punctuation, normalize spaces)
        text = re.sub(r'[^\w\s\-_@.\+]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Restore entities
        for placeholder, original in entities.items():
            text = text.replace(placeholder, original)
        
        return text
    
    def extract_timestamps(self, text: str) -> List[Dict]:
        """Extract timestamp information from text"""
        timestamps = []
        
        # Common timestamp patterns
        patterns = [
            r'\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\b',  # 2023-12-25 14:30:00
            r'\b\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}\b',        # 25/12/2023 14:30
            r'\b\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}\b',        # 25-12-2023 14:30
            r'\b\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}\b',        # 2023/12/25 14:30
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                timestamps.append({
                    'timestamp': match.group(),
                    'position': match.span(),
                    'format': pattern
                })
        
        return timestamps

class ForensicEntityExtractor:
    """Extract forensic-specific entities from text"""
    
    def __init__(self):
        self.patterns = ForensicEntities.PATTERNS
        
    def extract_all_entities(self, text: str) -> Dict[str, List[Dict]]:
        """Extract all forensic entities from text"""
        entities = {
            'phone_numbers': self.extract_phone_numbers(text),
            'crypto_addresses': self.extract_crypto_addresses(text),
            'bank_accounts': self.extract_bank_accounts(text),
            'emails': self.extract_emails(text),
            'ip_addresses': self.extract_ip_addresses(text),
            'imei_numbers': self.extract_imei_numbers(text),
            'license_plates': self.extract_license_plates(text)
        }
        
        return entities
    
    def extract_phone_numbers(self, text: str) -> List[Dict]:
        """Extract phone numbers"""
        phone_numbers = []
        pattern = self.patterns['phone_number']
        
        for match in re.finditer(pattern, text):
            phone_numbers.append({
                'value': match.group(),
                'position': match.span(),
                'normalized': self._normalize_phone_number(match.group())
            })
        
        return phone_numbers
    
    def extract_crypto_addresses(self, text: str) -> List[Dict]:
        """Extract cryptocurrency addresses"""
        crypto_addresses = []
        
        for crypto_type, pattern in self.patterns['crypto_address'].items():
            for match in re.finditer(pattern, text):
                crypto_addresses.append({
                    'value': match.group(),
                    'type': crypto_type,
                    'position': match.span()
                })
        
        return crypto_addresses
    
    def extract_bank_accounts(self, text: str) -> List[Dict]:
        """Extract bank account numbers"""
        bank_accounts = []
        pattern = self.patterns['bank_account']
        
        for match in re.finditer(pattern, text):
            bank_accounts.append({
                'value': match.group(),
                'position': match.span()
            })
        
        return bank_accounts
    
    def extract_emails(self, text: str) -> List[Dict]:
        """Extract email addresses"""
        emails = []
        pattern = self.patterns['email']
        
        for match in re.finditer(pattern, text):
            emails.append({
                'value': match.group(),
                'position': match.span(),
                'domain': match.group().split('@')[1] if '@' in match.group() else None
            })
        
        return emails
    
    def extract_ip_addresses(self, text: str) -> List[Dict]:
        """Extract IP addresses"""
        ip_addresses = []
        pattern = self.patterns['ip_address']
        
        for match in re.finditer(pattern, text):
            ip_addresses.append({
                'value': match.group(),
                'position': match.span()
            })
        
        return ip_addresses
    
    def extract_imei_numbers(self, text: str) -> List[Dict]:
        """Extract IMEI numbers"""
        imei_numbers = []
        pattern = self.patterns['imei']
        
        for match in re.finditer(pattern, text):
            imei_numbers.append({
                'value': match.group(),
                'position': match.span()
            })
        
        return imei_numbers
    
    def extract_license_plates(self, text: str) -> List[Dict]:
        """Extract license plate numbers"""
        license_plates = []
        pattern = self.patterns['license_plate']
        
        for match in re.finditer(pattern, text):
            license_plates.append({
                'value': match.group(),
                'position': match.span()
            })
        
        return license_plates
    
    def _normalize_phone_number(self, phone: str) -> str:
        """Normalize phone number format"""
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)
        
        # Add country code if missing (assuming US/international format)
        if len(digits) == 10:
            digits = '1' + digits  # Add US country code
        
        return digits

class ConversationExtractor:
    """Extract and structure conversation data"""
    
    def __init__(self):
        self.message_patterns = [
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*-\s*([^:]+):\s*(.+)',  # WhatsApp format
            r'\[(\d{2}/\d{2}/\d{4},\s+\d{2}:\d{2}:\d{2})\]\s+([^:]+):\s*(.+)',  # Another WhatsApp format
            r'(\d{2}:\d{2})\s+([^:]+):\s*(.+)',  # Simple time format
        ]
    
    def extract_conversations(self, text: str) -> List[Dict]:
        """Extract conversation messages from text"""
        conversations = []
        
        for pattern in self.message_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                groups = match.groups()
                if len(groups) >= 3:
                    conversations.append({
                        'timestamp': groups[0],
                        'sender': groups[1].strip(),
                        'message': groups[2].strip(),
                        'position': match.span()
                    })
        
        return conversations
    
    def group_by_conversation(self, messages: List[Dict]) -> Dict[str, List[Dict]]:
        """Group messages by conversation participants"""
        conversations = {}
        
        for message in messages:
            sender = message['sender']
            if sender not in conversations:
                conversations[sender] = []
            conversations[sender].append(message)
        
        return conversations

class DataPreprocessor:
    """Main preprocessor orchestrating all text processing tasks"""
    
    def __init__(self):
        self.text_cleaner = TextCleaner()
        self.entity_extractor = ForensicEntityExtractor()
        self.conversation_extractor = ConversationExtractor()
    
    def preprocess_document(self, raw_content: Dict) -> Dict:
        """Preprocess a complete UFDR document"""
        preprocessed = {
            'cleaned_content': {},
            'extracted_entities': {},
            'conversations': {},
            'metadata': {
                'processed_at': datetime.now().isoformat(),
                'total_entities': 0
            }
        }
        
        # Process different content types
        for content_type, content_data in raw_content.items():
            if isinstance(content_data, str):
                # Clean text
                cleaned_text = self.text_cleaner.clean_text(content_data)
                preprocessed['cleaned_content'][content_type] = cleaned_text
                
                # Extract entities
                entities = self.entity_extractor.extract_all_entities(content_data)
                preprocessed['extracted_entities'][content_type] = entities
                
                # Count total entities
                for entity_type, entity_list in entities.items():
                    preprocessed['metadata']['total_entities'] += len(entity_list)
                
                # Extract conversations if applicable
                if 'message' in content_type.lower() or 'chat' in content_type.lower():
                    conversations = self.conversation_extractor.extract_conversations(content_data)
                    preprocessed['conversations'][content_type] = conversations
            
            elif isinstance(content_data, list):
                # Process list of items (e.g., messages, calls)
                processed_items = []
                for item in content_data:
                    if isinstance(item, dict):
                        processed_item = {}
                        for key, value in item.items():
                            if isinstance(value, str):
                                processed_item[key] = self.text_cleaner.clean_text(value)
                                
                                # Extract entities from string fields
                                entities = self.entity_extractor.extract_all_entities(value)
                                if any(entities.values()):
                                    processed_item[f"{key}_entities"] = entities
                            else:
                                processed_item[key] = value
                        processed_items.append(processed_item)
                    else:
                        processed_items.append(item)
                
                preprocessed['cleaned_content'][content_type] = processed_items
        
        return preprocessed
    
    def extract_searchable_text(self, preprocessed_content: Dict) -> List[str]:
        """Extract all searchable text chunks from preprocessed content"""
        text_chunks = []
        
        def extract_text_recursive(data, current_path=""):
            if isinstance(data, str) and data.strip():
                text_chunks.append(data.strip())
            elif isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{current_path}.{key}" if current_path else key
                    extract_text_recursive(value, new_path)
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    new_path = f"{current_path}[{i}]" if current_path else f"[{i}]"
                    extract_text_recursive(item, new_path)
        
        extract_text_recursive(preprocessed_content['cleaned_content'])
        
        return text_chunks