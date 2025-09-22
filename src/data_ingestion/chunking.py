"""
Text Chunking Module for Project Sentinel
Intelligent text chunking optimized for forensic data analysis and vector embeddings
"""

import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import hashlib
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    id: str
    content: str
    chunk_type: str
    source_file: str
    source_section: str
    start_position: int
    end_position: int
    token_count: int
    entities: Dict[str, List[Dict]]
    metadata: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict:
        """Convert chunk to dictionary for storage"""
        return {
            'id': self.id,
            'content': self.content,
            'chunk_type': self.chunk_type,
            'source_file': self.source_file,
            'source_section': self.source_section,
            'start_position': self.start_position,
            'end_position': self.end_position,
            'token_count': self.token_count,
            'entities': self.entities,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }

class ForensicTextChunker:
    """Intelligent text chunker for forensic documents"""
    
    def __init__(self, 
                 chunk_size: int = None,
                 chunk_overlap: int = None,
                 max_chunk_size: int = None):
        
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.max_chunk_size = max_chunk_size or settings.max_chunk_size
        
        # Conversation-specific settings
        self.conversation_chunk_size = 5  # Number of messages per chunk
        self.preserve_conversation_context = True
        
    def chunk_document(self, preprocessed_document: Dict, source_file: str) -> List[TextChunk]:
        """Chunk a complete preprocessed UFDR document"""
        all_chunks = []
        
        cleaned_content = preprocessed_document.get('cleaned_content', {})
        extracted_entities = preprocessed_document.get('extracted_entities', {})
        conversations = preprocessed_document.get('conversations', {})
        
        for section_name, content in cleaned_content.items():
            if isinstance(content, str):
                # Handle text content
                chunks = self._chunk_text_content(
                    content, section_name, source_file, 
                    extracted_entities.get(section_name, {})
                )
                all_chunks.extend(chunks)
                
            elif isinstance(content, list):
                # Handle structured data (messages, calls, etc.)
                if section_name in conversations:
                    # Special handling for conversations
                    chunks = self._chunk_conversations(
                        conversations[section_name], section_name, source_file
                    )
                else:
                    # Regular list chunking
                    chunks = self._chunk_list_content(
                        content, section_name, source_file
                    )
                all_chunks.extend(chunks)
        
        # Add cross-references and relationships
        self._add_chunk_relationships(all_chunks)
        
        return all_chunks
    
    def _chunk_text_content(self, 
                           text: str, 
                           section_name: str, 
                           source_file: str,
                           entities: Dict[str, List[Dict]]) -> List[TextChunk]:
        """Chunk plain text content"""
        chunks = []
        
        if not text or len(text.strip()) == 0:
            return chunks
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_position = 0
        chunk_start = 0
        
        for sentence in sentences:
            sentence_with_space = sentence + " "
            
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence_with_space) > self.chunk_size:
                if current_chunk.strip():
                    # Create chunk
                    chunk = self._create_text_chunk(
                        current_chunk.strip(),
                        "text",
                        source_file,
                        section_name,
                        chunk_start,
                        current_position,
                        entities
                    )
                    chunks.append(chunk)
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and chunks:
                    overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + sentence_with_space
                    chunk_start = current_position - len(overlap_text)
                else:
                    current_chunk = sentence_with_space
                    chunk_start = current_position
            else:
                current_chunk += sentence_with_space
            
            current_position += len(sentence_with_space)
        
        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_text_chunk(
                current_chunk.strip(),
                "text",
                source_file,
                section_name,
                chunk_start,
                current_position,
                entities
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_conversations(self, 
                           conversations: List[Dict], 
                           section_name: str, 
                           source_file: str) -> List[TextChunk]:
        """Chunk conversation data maintaining context"""
        chunks = []
        
        if not conversations:
            return chunks
        
        # Sort conversations by timestamp if available
        sorted_conversations = self._sort_conversations_by_time(conversations)
        
        # Group conversations by participants for better context
        participant_groups = self._group_conversations_by_participants(sorted_conversations)
        
        for participants, messages in participant_groups.items():
            # Chunk each conversation thread
            conversation_chunks = self._chunk_conversation_thread(
                messages, participants, section_name, source_file
            )
            chunks.extend(conversation_chunks)
        
        return chunks
    
    def _chunk_conversation_thread(self, 
                                 messages: List[Dict], 
                                 participants: str,
                                 section_name: str, 
                                 source_file: str) -> List[TextChunk]:
        """Chunk a single conversation thread"""
        chunks = []
        
        for i in range(0, len(messages), self.conversation_chunk_size):
            chunk_messages = messages[i:i + self.conversation_chunk_size]
            
            # Create conversation context
            conversation_text = self._format_conversation_chunk(chunk_messages)
            
            # Extract entities from conversation
            entities = self._extract_entities_from_conversation(chunk_messages)
            
            # Create metadata
            metadata = {
                'participants': participants,
                'message_count': len(chunk_messages),
                'time_range': self._get_conversation_time_range(chunk_messages),
                'conversation_position': i // self.conversation_chunk_size
            }
            
            chunk = self._create_text_chunk(
                conversation_text,
                "conversation",
                source_file,
                section_name,
                i,
                i + len(chunk_messages),
                entities,
                metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_list_content(self, 
                          content_list: List[Any], 
                          section_name: str, 
                          source_file: str) -> List[TextChunk]:
        """Chunk structured list content (calls, contacts, etc.)"""
        chunks = []
        
        if not content_list:
            return chunks
        
        # Determine optimal chunk size for this data type
        optimal_chunk_size = self._get_optimal_chunk_size_for_data_type(section_name)
        
        for i in range(0, len(content_list), optimal_chunk_size):
            chunk_items = content_list[i:i + optimal_chunk_size]
            
            # Convert items to text representation
            text_content = self._convert_items_to_text(chunk_items, section_name)
            
            # Extract entities
            entities = self._extract_entities_from_items(chunk_items)
            
            # Create metadata
            metadata = {
                'data_type': section_name,
                'item_count': len(chunk_items),
                'chunk_position': i // optimal_chunk_size
            }
            
            chunk = self._create_text_chunk(
                text_content,
                "structured_data",
                source_file,
                section_name,
                i,
                i + len(chunk_items),
                entities,
                metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_text_chunk(self, 
                          content: str,
                          chunk_type: str,
                          source_file: str,
                          source_section: str,
                          start_position: int,
                          end_position: int,
                          entities: Dict[str, List[Dict]] = None,
                          additional_metadata: Dict = None) -> TextChunk:
        """Create a TextChunk object"""
        
        # Generate unique ID
        chunk_id = self._generate_chunk_id(content, source_file, start_position)
        
        # Count tokens (approximate)
        token_count = len(content.split())
        
        # Prepare entities
        if entities is None:
            entities = {}
        
        # Prepare metadata
        metadata = {
            'content_length': len(content),
            'word_count': len(content.split()),
            'has_entities': bool(any(entities.values())),
            'language_detected': self._detect_language(content)
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return TextChunk(
            id=chunk_id,
            content=content,
            chunk_type=chunk_type,
            source_file=source_file,
            source_section=source_section,
            start_position=start_position,
            end_position=end_position,
            token_count=token_count,
            entities=entities,
            metadata=metadata,
            created_at=datetime.now()
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using multiple delimiters"""
        # Common sentence endings
        sentence_endings = r'[.!?]+\s+'
        
        # Split by sentence endings
        sentences = re.split(sentence_endings, text)
        
        # Filter out empty sentences and very short ones
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of current chunk"""
        words = text.split()
        if len(words) <= overlap_size:
            return text
        
        overlap_words = words[-overlap_size:]
        return " ".join(overlap_words) + " "
    
    def _sort_conversations_by_time(self, conversations: List[Dict]) -> List[Dict]:
        """Sort conversations by timestamp"""
        def get_timestamp(conv):
            timestamp = conv.get('timestamp', '')
            # Try to parse common timestamp formats
            # Return conversation as-is if parsing fails
            return timestamp
        
        try:
            return sorted(conversations, key=get_timestamp)
        except:
            # If sorting fails, return original order
            return conversations
    
    def _group_conversations_by_participants(self, conversations: List[Dict]) -> Dict[str, List[Dict]]:
        """Group conversations by participants"""
        groups = {}
        
        for conv in conversations:
            sender = conv.get('sender', 'unknown')
            # Create a simple key for now - could be enhanced to detect conversation threads
            if sender not in groups:
                groups[sender] = []
            groups[sender].append(conv)
        
        return groups
    
    def _format_conversation_chunk(self, messages: List[Dict]) -> str:
        """Format a chunk of conversation messages into readable text"""
        formatted_lines = []
        
        for msg in messages:
            timestamp = msg.get('timestamp', '')
            sender = msg.get('sender', 'Unknown')
            message = msg.get('message', '')
            
            if timestamp:
                line = f"[{timestamp}] {sender}: {message}"
            else:
                line = f"{sender}: {message}"
            
            formatted_lines.append(line)
        
        return "\n".join(formatted_lines)
    
    def _extract_entities_from_conversation(self, messages: List[Dict]) -> Dict[str, List[Dict]]:
        """Extract entities from conversation messages"""
        # This would integrate with the ForensicEntityExtractor
        # For now, return empty dict - will be enhanced when integrating with the entity extractor
        return {}
    
    def _get_conversation_time_range(self, messages: List[Dict]) -> Dict[str, str]:
        """Get time range for conversation chunk"""
        timestamps = [msg.get('timestamp', '') for msg in messages if msg.get('timestamp')]
        
        if timestamps:
            return {
                'start': timestamps[0],
                'end': timestamps[-1]
            }
        
        return {}
    
    def _get_optimal_chunk_size_for_data_type(self, data_type: str) -> int:
        """Get optimal chunk size based on data type"""
        chunk_sizes = {
            'calls': 10,     # 10 call records per chunk
            'contacts': 20,  # 20 contacts per chunk
            'media': 5,      # 5 media files per chunk
            'messages': 5,   # 5 messages per chunk (if not conversation format)
            'default': 10
        }
        
        return chunk_sizes.get(data_type.lower(), chunk_sizes['default'])
    
    def _convert_items_to_text(self, items: List[Any], data_type: str) -> str:
        """Convert structured items to text representation"""
        text_lines = []
        
        for item in items:
            if isinstance(item, dict):
                # Create readable representation of the item
                item_text = self._dict_to_text(item, data_type)
                text_lines.append(item_text)
            else:
                text_lines.append(str(item))
        
        return "\n".join(text_lines)
    
    def _dict_to_text(self, item_dict: Dict, data_type: str) -> str:
        """Convert dictionary item to readable text"""
        if data_type.lower() == 'calls':
            return self._call_record_to_text(item_dict)
        elif data_type.lower() == 'contacts':
            return self._contact_to_text(item_dict)
        elif data_type.lower() == 'messages':
            return self._message_to_text(item_dict)
        else:
            # Generic conversion
            return ", ".join(f"{k}: {v}" for k, v in item_dict.items() if v)
    
    def _call_record_to_text(self, call: Dict) -> str:
        """Convert call record to text"""
        timestamp = call.get('timestamp', call.get('time', ''))
        number = call.get('number', call.get('phone_number', ''))
        direction = call.get('direction', call.get('type', ''))
        duration = call.get('duration', '')
        
        text = f"Call {direction} {number}"
        if timestamp:
            text = f"[{timestamp}] {text}"
        if duration:
            text = f"{text}, duration: {duration}"
        
        return text
    
    def _contact_to_text(self, contact: Dict) -> str:
        """Convert contact to text"""
        name = contact.get('name', contact.get('display_name', ''))
        phone = contact.get('phone', contact.get('number', ''))
        email = contact.get('email', '')
        
        text = f"Contact: {name}"
        if phone:
            text = f"{text}, Phone: {phone}"
        if email:
            text = f"{text}, Email: {email}"
        
        return text
    
    def _message_to_text(self, message: Dict) -> str:
        """Convert message to text"""
        timestamp = message.get('timestamp', message.get('time', ''))
        sender = message.get('sender', message.get('from', ''))
        content = message.get('content', message.get('message', message.get('body', '')))
        
        text = f"{sender}: {content}"
        if timestamp:
            text = f"[{timestamp}] {text}"
        
        return text
    
    def _extract_entities_from_items(self, items: List[Any]) -> Dict[str, List[Dict]]:
        """Extract entities from structured items"""
        # This would integrate with the ForensicEntityExtractor
        # For now, return empty dict - will be enhanced when integrating
        return {}
    
    def _generate_chunk_id(self, content: str, source_file: str, position: int) -> str:
        """Generate unique ID for chunk"""
        # Create hash from content and metadata
        hash_input = f"{source_file}_{position}_{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text (simplified)"""
        # Simple language detection - could be enhanced with proper language detection
        if re.search(r'[a-zA-Z]', text):
            return 'en'  # Assume English for now
        return 'unknown'
    
    def _add_chunk_relationships(self, chunks: List[TextChunk]) -> None:
        """Add relationships between chunks"""
        for i, chunk in enumerate(chunks):
            # Add sequence information
            chunk.metadata['chunk_sequence'] = i
            chunk.metadata['total_chunks'] = len(chunks)
            
            # Add adjacent chunk references
            if i > 0:
                chunk.metadata['previous_chunk'] = chunks[i-1].id
            if i < len(chunks) - 1:
                chunk.metadata['next_chunk'] = chunks[i+1].id

class ChunkManager:
    """Manages text chunks for the application"""
    
    def __init__(self):
        self.chunker = ForensicTextChunker()
    
    def process_document_for_chunking(self, preprocessed_document: Dict, source_file: str) -> List[TextChunk]:
        """Process document and return chunks"""
        return self.chunker.chunk_document(preprocessed_document, source_file)
    
    def optimize_chunks_for_embedding(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Optimize chunks for vector embedding"""
        optimized_chunks = []
        
        for chunk in chunks:
            # Ensure chunk is within optimal size for embeddings
            if chunk.token_count > settings.max_chunk_size:
                # Split large chunks
                sub_chunks = self._split_large_chunk(chunk)
                optimized_chunks.extend(sub_chunks)
            elif chunk.token_count < 50:  # Too small chunks might not be meaningful
                # Could merge with adjacent chunks or mark for special handling
                chunk.metadata['small_chunk'] = True
                optimized_chunks.append(chunk)
            else:
                optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _split_large_chunk(self, chunk: TextChunk) -> List[TextChunk]:
        """Split a large chunk into smaller ones"""
        sub_chunks = []
        content = chunk.content
        
        # Split by sentences or paragraphs
        sentences = re.split(r'[.!?]+\s+', content)
        
        current_text = ""
        current_position = 0
        
        for sentence in sentences:
            if len(current_text.split()) + len(sentence.split()) > settings.max_chunk_size:
                if current_text.strip():
                    # Create sub-chunk
                    sub_chunk = TextChunk(
                        id=f"{chunk.id}_sub_{len(sub_chunks)}",
                        content=current_text.strip(),
                        chunk_type=chunk.chunk_type,
                        source_file=chunk.source_file,
                        source_section=chunk.source_section,
                        start_position=chunk.start_position + current_position,
                        end_position=chunk.start_position + current_position + len(current_text),
                        token_count=len(current_text.split()),
                        entities=chunk.entities,  # Inherit entities - could be refined
                        metadata={**chunk.metadata, 'is_sub_chunk': True, 'parent_chunk': chunk.id},
                        created_at=datetime.now()
                    )
                    sub_chunks.append(sub_chunk)
                
                current_text = sentence
                current_position += len(current_text)
            else:
                current_text += " " + sentence
        
        # Add final sub-chunk
        if current_text.strip():
            sub_chunk = TextChunk(
                id=f"{chunk.id}_sub_{len(sub_chunks)}",
                content=current_text.strip(),
                chunk_type=chunk.chunk_type,
                source_file=chunk.source_file,
                source_section=chunk.source_section,
                start_position=chunk.start_position + current_position,
                end_position=chunk.end_position,
                token_count=len(current_text.split()),
                entities=chunk.entities,
                metadata={**chunk.metadata, 'is_sub_chunk': True, 'parent_chunk': chunk.id},
                created_at=datetime.now()
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def get_chunk_statistics(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        if not chunks:
            return {}
        
        total_chunks = len(chunks)
        chunk_types = {}
        total_tokens = 0
        total_entities = 0
        
        for chunk in chunks:
            # Count chunk types
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
            
            # Sum tokens
            total_tokens += chunk.token_count
            
            # Count entities
            for entity_list in chunk.entities.values():
                total_entities += len(entity_list)
        
        avg_chunk_size = total_tokens / total_chunks if total_chunks > 0 else 0
        
        return {
            'total_chunks': total_chunks,
            'chunk_types': chunk_types,
            'total_tokens': total_tokens,
            'total_entities': total_entities,
            'average_chunk_size': avg_chunk_size,
            'chunks_with_entities': sum(1 for chunk in chunks if any(chunk.entities.values()))
        }