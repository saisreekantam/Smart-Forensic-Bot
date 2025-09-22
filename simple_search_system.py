"""
Simple Search System for Processed Evidence Data
Provides basic search functionality for the chatbot to retrieve relevant information
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class SimpleSearchSystem:
    """
    Simple search system that works with our processed JSON files
    """
    
    def __init__(self, processed_data_dir: str = "data/processed"):
        self.processed_data_dir = Path(processed_data_dir)
        self.cache = {}  # Simple cache for loaded data
        
    def search_case_data(self, case_id: str, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search through processed data for a specific case
        
        Args:
            case_id: The case ID to search in
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant records with source information
        """
        try:
            # Load all processed files for the case
            case_data = self._load_case_data(case_id)
            
            if not case_data:
                return []
            
            # Search through all records
            results = []
            query_terms = self._extract_search_terms(query)
            
            for file_data in case_data:
                source_file = file_data.get("source_file", "unknown")
                file_type = file_data.get("file_type", "unknown")
                
                for record in file_data.get("records", []):
                    # Calculate relevance score
                    score = self._calculate_relevance_score(record, query_terms)
                    
                    if score > 0:
                        result = {
                            "id": record.get("id"),
                            "source_file": source_file,
                            "file_type": file_type,
                            "data": record.get("data", {}),
                            "searchable_text": record.get("searchable_text", ""),
                            "relevance_score": score,
                            "record_type": self._determine_record_type(record, file_type)
                        }
                        results.append(result)
            
            # Sort by relevance score and return top results
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return results[:max_results]
            
        except Exception as e:
            print(f"Error calculating relevance: {e}")
            return 0.0
    
    def get_case_summary(self, case_id):
        """Get summary information about processed data for a case"""
        try:
            import glob
            import json
            import os
            
            # Find all processed files for this case
            processed_files = glob.glob(f"data/processed/*{case_id}*.json")
            
            if not processed_files:
                return {
                    "total_records": 0,
                    "data_sources": [],
                    "case_id": case_id
                }
            
            total_records = 0
            data_sources = []
            
            for file_path in processed_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Count records
                    if isinstance(data, list):
                        file_records = len(data)
                    elif isinstance(data, dict):
                        file_records = 1
                    else:
                        file_records = 0
                    
                    total_records += file_records
                    data_sources.append({
                        "file": os.path.basename(file_path),
                        "records": file_records
                    })
                    
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue
            
            return {
                "total_records": total_records,
                "data_sources": data_sources,
                "case_id": case_id
            }
            
        except Exception as e:
            print(f"Error getting case summary for {case_id}: {e}")
            return {
                "total_records": 0,
                "data_sources": [],
                "case_id": case_id
            }
    
    def get_case_summary(self, case_id: str) -> Dict[str, Any]:
        """
        Get a summary of all data available for a case
        
        Args:
            case_id: The case ID
            
        Returns:
            Summary information about the case data
        """
        try:
            case_data = self._load_case_data(case_id)
            
            if not case_data:
                return {
                    "total_files": 0,
                    "total_records": 0,
                    "file_types": [],
                    "data_sources": []
                }
            
            total_records = 0
            file_types = set()
            data_sources = []
            
            for file_data in case_data:
                source_file = file_data.get("source_file", "unknown")
                file_type = file_data.get("file_type", "unknown")
                record_count = len(file_data.get("records", []))
                
                total_records += record_count
                file_types.add(file_type)
                data_sources.append({
                    "file": source_file,
                    "type": file_type,
                    "records": record_count
                })
            
            return {
                "total_files": len(case_data),
                "total_records": total_records,
                "file_types": list(file_types),
                "data_sources": data_sources
            }
            
        except Exception as e:
            logger.error(f"Summary error for case {case_id}: {str(e)}")
            return {}
    
    def search_specific_data_type(self, case_id: str, data_type: str, query: str = "") -> List[Dict[str, Any]]:
        """
        Search for specific types of data (calls, messages, etc.)
        
        Args:
            case_id: The case ID
            data_type: Type of data to search for ('calls', 'messages', 'contacts', etc.)
            query: Optional additional search terms
            
        Returns:
            List of relevant records of the specified type
        """
        try:
            case_data = self._load_case_data(case_id)
            
            if not case_data:
                return []
            
            results = []
            query_terms = self._extract_search_terms(query) if query else []
            
            for file_data in case_data:
                source_file = file_data.get("source_file", "unknown")
                file_type = file_data.get("file_type", "unknown")
                
                # Check if this file contains the requested data type
                if not self._file_contains_data_type(source_file, file_type, data_type):
                    continue
                
                for record in file_data.get("records", []):
                    # If no query terms, include all records of this type
                    if not query_terms:
                        score = 1.0
                    else:
                        score = self._calculate_relevance_score(record, query_terms)
                    
                    if score > 0:
                        result = {
                            "id": record.get("id"),
                            "source_file": source_file,
                            "file_type": file_type,
                            "data": record.get("data", {}),
                            "searchable_text": record.get("searchable_text", ""),
                            "relevance_score": score,
                            "record_type": data_type
                        }
                        results.append(result)
            
            # Sort by relevance and timestamp if available
            results.sort(key=lambda x: (x["relevance_score"], self._extract_timestamp(x)), reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Specific search error for case {case_id}, type {data_type}: {str(e)}")
            return []
    
    def _load_case_data(self, case_id: str) -> List[Dict[str, Any]]:
        """Load all processed data files for a case"""
        if case_id in self.cache:
            return self.cache[case_id]
        
        case_dir = self.processed_data_dir / case_id
        
        if not case_dir.exists():
            logger.warning(f"No processed data found for case {case_id}")
            return []
        
        case_data = []
        
        # Load all JSON files in the case directory
        for json_file in case_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    case_data.append(data)
            except Exception as e:
                logger.error(f"Error loading {json_file}: {str(e)}")
                continue
        
        # Cache the data
        self.cache[case_id] = case_data
        return case_data
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract search terms from a query"""
        # Remove common words and split into terms
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'who', 'when', 'where', 'why', 'how', 'show', 'me', 'all', 'any', 'some'}
        
        # Extract words and numbers, including phone numbers
        terms = re.findall(r'\b(?:\+?\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}|\w+)\b', query.lower())
        
        # Filter out common words but keep important forensic terms
        filtered_terms = [term for term in terms if term not in common_words or len(term) > 3]
        
        return filtered_terms
    
    def _calculate_relevance_score(self, record: Dict[str, Any], query_terms: List[str]) -> float:
        """Calculate enhanced relevance score for a record with comprehensive search"""
        if not query_terms:
            return 0.0
        
        searchable_text = record.get("searchable_text", "").lower()
        data = record.get("data", {})
        
        # Create comprehensive searchable content
        searchable_content = []
        
        # Add searchable text
        if searchable_text:
            searchable_content.append(searchable_text)
        
        # Add all data values as searchable content
        def extract_text_from_data(obj, depth=0):
            """Recursively extract text from data structures"""
            if depth > 3:  # Prevent infinite recursion
                return []
            
            extracted = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Add key names as searchable
                    extracted.append(str(key).lower())
                    # Add values
                    if isinstance(value, (dict, list)):
                        extracted.extend(extract_text_from_data(value, depth + 1))
                    else:
                        extracted.append(str(value).lower())
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        extracted.extend(extract_text_from_data(item, depth + 1))
                    else:
                        extracted.append(str(item).lower())
            else:
                extracted.append(str(obj).lower())
            
            return extracted
        
        searchable_content.extend(extract_text_from_data(data))
        
        # Combine all searchable content
        combined_text = " ".join(searchable_content)
        
        score = 0.0
        total_matches = 0
        
        for term in query_terms:
            term_lower = term.lower()
            term_score = 0.0
            
            # Exact word match in searchable text (highest weight)
            if re.search(r'\b' + re.escape(term_lower) + r'\b', searchable_text):
                term_score += 5.0
            
            # Exact phrase match in searchable text
            elif term_lower in searchable_text:
                term_score += 3.0
            
            # Exact word match in combined content
            elif re.search(r'\b' + re.escape(term_lower) + r'\b', combined_text):
                term_score += 4.0
            
            # Partial match in combined content
            elif term_lower in combined_text:
                term_score += 2.0
            
            # Phone number matching (special handling)
            if self._is_phone_number(term):
                clean_term = re.sub(r'[-.\s()]', '', term)
                clean_text = re.sub(r'[-.\s()]', '', combined_text)
                if clean_term in clean_text:
                    term_score += 6.0  # High score for phone matches
            
            # Email matching
            elif '@' in term and '.' in term:
                if term_lower in combined_text:
                    term_score += 5.0
            
            # Number matching (for IDs, amounts, etc.)
            elif term.isdigit():
                if term in combined_text:
                    term_score += 3.0
            
            # Fuzzy matching for names and text content
            else:
                fuzzy_score = self._fuzzy_match_score(term_lower, combined_text)
                term_score += fuzzy_score
            
            if term_score > 0:
                total_matches += 1
                score += term_score
        
        # Normalize score by number of query terms
        if query_terms:
            score = score / len(query_terms)
            
        # Boost score if multiple terms match
        if total_matches > 1:
            score *= (1 + (total_matches - 1) * 0.2)  # 20% boost per additional match
        
        return min(score, 10.0)  # Cap at 10.0
    
    def _is_phone_number(self, text: str) -> bool:
        """Check if text looks like a phone number"""
        # Remove common separators and check if remaining chars are mostly digits
        cleaned = re.sub(r'[-.\s()+]', '', text)
        return len(cleaned) >= 7 and cleaned.replace('+', '').isdigit()
    
    def _fuzzy_match_score(self, term: str, text: str) -> float:
        """Calculate fuzzy matching score for partial word matches"""
        if len(term) < 3:
            return 0.0
        
        # Look for partial matches
        words = text.split()
        best_score = 0.0
        
        for word in words:
            if len(word) < 3:
                continue
                
            # Check if term is a substring of word
            if term in word:
                # Score based on how much of the word matches
                match_ratio = len(term) / len(word)
                best_score = max(best_score, match_ratio * 1.5)
            
            # Check if word is a substring of term
            elif word in term:
                match_ratio = len(word) / len(term)
                best_score = max(best_score, match_ratio * 1.0)
        
        return best_score
    
    def search_message_content(self, case_id: str, query: str, max_results: int = 50) -> Dict[str, Any]:
        """
        Enhanced search specifically for message content with comprehensive matching
        
        Args:
            case_id: The case ID
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Dict containing search results with detailed message information
        """
        try:
            case_data = self._load_case_data(case_id)
            if not case_data:
                return {"results": [], "total": 0, "query": query}
            
            query_terms = self._process_query_terms(query)
            if not query_terms:
                return {"results": [], "total": 0, "query": query}
            
            message_results = []
            
            for file_data in case_data:
                for record in file_data.get("records", []):
                    # Focus on message-like content
                    if self._is_message_record(record):
                        score = self._calculate_message_relevance(record, query_terms, query)
                        
                        if score > 0:
                            result = {
                                "id": record.get("id"),
                                "source_file": file_data.get("source_file", "unknown"),
                                "message_data": self._extract_message_details(record),
                                "searchable_text": record.get("searchable_text", ""),
                                "relevance_score": score,
                                "match_highlights": self._get_match_highlights(record, query_terms)
                            }
                            message_results.append(result)
            
            # Sort by relevance and return top results
            message_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return {
                "results": message_results[:max_results],
                "total": len(message_results),
                "query": query,
                "search_type": "message_content"
            }
            
        except Exception as e:
            logger.error(f"Message search error: {str(e)}")
            return {"results": [], "total": 0, "query": query, "error": str(e)}
    
    def _is_message_record(self, record: Dict[str, Any]) -> bool:
        """Check if a record contains message-like content"""
        data = record.get("data", {})
        text = record.get("searchable_text", "").lower()
        
        # Check for message-specific fields
        message_fields = ["message", "content", "text", "sender", "recipient", "chat", "conversation"]
        
        # Check data keys
        for field in message_fields:
            if field in data or field in text:
                return True
        
        # Check for message-like patterns
        if any(pattern in text for pattern in ["from:", "to:", "msg:", "said:", "replied:"]):
            return True
        
        return False
    
    def _calculate_message_relevance(self, record: Dict[str, Any], query_terms: List[str], original_query: str) -> float:
        """Calculate relevance score specifically for message content"""
        base_score = self._calculate_relevance_score(record, query_terms)
        
        # Boost for message-specific content
        data = record.get("data", {})
        text = record.get("searchable_text", "").lower()
        
        # Additional scoring for message context
        message_boost = 0.0
        
        # Check if query terms appear in message content specifically
        message_content = ""
        for key, value in data.items():
            if key.lower() in ["message", "content", "text", "body"]:
                message_content += str(value).lower() + " "
        
        if message_content:
            for term in query_terms:
                if term.lower() in message_content:
                    message_boost += 1.0
        
        # Boost for complete phrase matches
        if len(original_query.split()) > 1:
            if original_query.lower() in text or original_query.lower() in message_content:
                message_boost += 2.0
        
        return base_score + message_boost
    
    def _extract_message_details(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detailed message information from a record"""
        data = record.get("data", {})
        details = {}
        
        # Extract common message fields
        field_mapping = {
            "sender": ["sender", "from", "caller", "source"],
            "recipient": ["recipient", "to", "callee", "destination"],
            "message": ["message", "content", "text", "body"],
            "timestamp": ["timestamp", "time", "date", "created_at"],
            "phone": ["phone", "number", "phone_number"]
        }
        
        for detail_key, possible_keys in field_mapping.items():
            for key in possible_keys:
                if key in data:
                    details[detail_key] = data[key]
                    break
        
        # Add all original data for completeness
        details["raw_data"] = data
        
        return details
    
    def _get_match_highlights(self, record: Dict[str, Any], query_terms: List[str]) -> List[str]:
        """Get highlighted text snippets showing where query terms match"""
        highlights = []
        text = record.get("searchable_text", "")
        
        for term in query_terms:
            # Find contexts where the term appears
            term_lower = term.lower()
            text_lower = text.lower()
            
            start_pos = text_lower.find(term_lower)
            while start_pos != -1:
                # Extract context around the match
                context_start = max(0, start_pos - 50)
                context_end = min(len(text), start_pos + len(term) + 50)
                
                context = text[context_start:context_end]
                
                # Highlight the matching term
                highlighted = context.replace(
                    text[start_pos:start_pos + len(term)],
                    f"**{text[start_pos:start_pos + len(term)]}**"
                )
                
                highlights.append(highlighted)
                
                # Look for next occurrence
                start_pos = text_lower.find(term_lower, start_pos + 1)
                
                # Limit highlights per term
                if len(highlights) >= 3:
                    break
        
        return highlights[:5]  # Return max 5 highlights
    
    def _determine_record_type(self, record: Dict[str, Any], file_type: str) -> str:
        """Determine the type of record based on its data"""
        data = record.get("data", {})
        
        if "phone_number" in data and "duration" in data:
            return "call_log"
        elif "sender" in data and "message" in data:
            return "message"
        elif "contact_name" in data and "phone_number" in data:
            return "contact"
        elif "content" in data:
            return "text_content"
        else:
            return file_type
    
    def _file_contains_data_type(self, source_file: str, file_type: str, data_type: str) -> bool:
        """Check if a file contains the requested data type"""
        mapping = {
            "calls": ["call", "log"],
            "messages": ["message", "chat", "sms"],
            "contacts": ["contact", "address"],
            "text": ["text", "report", "document"]
        }
        
        search_terms = mapping.get(data_type, [data_type])
        source_lower = source_file.lower()
        
        return any(term in source_lower for term in search_terms)
    
    def _extract_timestamp(self, record: Dict[str, Any]) -> str:
        """Extract timestamp from record for sorting"""
        data = record.get("data", {})
        
        # Look for common timestamp fields
        for field in ["timestamp", "date", "created_at", "time"]:
            if field in data:
                return str(data[field])
        
        return "0000-00-00"
    
    def _process_query_terms(self, query: str) -> List[str]:
        """Process query into search terms"""
        if not query:
            return []
        
        # Use the existing _extract_search_terms method
        return self._extract_search_terms(query)

# Global instance
search_system = SimpleSearchSystem()

def search_case_evidence(case_id: str, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Global function to search case evidence
    """
    return search_system.search_case_data(case_id, query, max_results)

def get_case_data_summary(case_id: str) -> Dict[str, Any]:
    """
    Global function to get case data summary
    """
    return search_system.get_case_summary(case_id)

if __name__ == "__main__":
    # Test the search system
    search = SimpleSearchSystem()
    
    # Test case ID
    case_id = "c0c91912-9ba2-4ecc-9af6-22e3296d562c"
    
    # Test summary
    print("=== Case Summary ===")
    summary = search.get_case_summary(case_id)
    print(json.dumps(summary, indent=2))
    
    # Test search
    print("\n=== Search Results for 'Alex Rivera' ===")
    results = search.search_case_data(case_id, "Alex Rivera")
    for result in results:
        print(f"Score: {result['relevance_score']:.2f} | {result['source_file']} | {result['searchable_text'][:100]}")
    
    # Test phone number search
    print("\n=== Search Results for phone number ===")
    results = search.search_case_data(case_id, "+1-555-0987")
    for result in results:
        print(f"Score: {result['relevance_score']:.2f} | {result['source_file']} | {result['searchable_text'][:100]}")