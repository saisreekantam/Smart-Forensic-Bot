"""
Utility functions for Smart Forensic Bot
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

def calculate_file_hash(file_path: Path, algorithm: str = 'sha256') -> str:
    """
    Calculate hash of a file
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hex digest of the hash
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe filesystem storage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255 - len(ext)] + ext
    
    return filename

def create_case_report(case_data: Dict[str, Any], output_path: Path) -> bool:
    """
    Create a formatted case report
    
    Args:
        case_data: Case analysis results
        output_path: Path to save the report
        
    Returns:
        True if successful, False otherwise
    """
    try:
        report = {
            'case_information': {
                'case_id': case_data.get('case_id', 'N/A'),
                'analyst': case_data.get('analyst', 'N/A'),
                'timestamp': case_data.get('timestamp', datetime.now().isoformat()),
                'status': case_data.get('status', 'unknown')
            },
            'evidence_summary': {
                'total_files': len(case_data.get('evidence_files', [])),
                'files_analyzed': len([r for r in case_data.get('analysis_results', {}).values() 
                                    if 'error' not in r])
            },
            'key_findings': case_data.get('findings', []),
            'recommendations': case_data.get('recommendations', []),
            'detailed_analysis': case_data.get('analysis_results', {}),
            'knowledge_context': case_data.get('knowledge_context', [])
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error creating report: {e}")
        return False

def validate_evidence_integrity(file_path: Path, expected_hash: str, 
                               algorithm: str = 'sha256') -> bool:
    """
    Validate evidence file integrity using hash comparison
    
    Args:
        file_path: Path to the evidence file
        expected_hash: Expected hash value
        algorithm: Hash algorithm to use
        
    Returns:
        True if hashes match, False otherwise
    """
    try:
        actual_hash = calculate_file_hash(file_path, algorithm)
        return actual_hash.lower() == expected_hash.lower()
    except Exception:
        return False

def extract_metadata_summary(metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract summary statistics from metadata list
    
    Args:
        metadata_list: List of metadata dictionaries
        
    Returns:
        Summary statistics
    """
    summary = {
        'total_items': len(metadata_list),
        'types': {},
        'sources': {},
        'date_range': {'earliest': None, 'latest': None}
    }
    
    for metadata in metadata_list:
        # Count types
        item_type = metadata.get('type', 'unknown')
        summary['types'][item_type] = summary['types'].get(item_type, 0) + 1
        
        # Count sources
        source = metadata.get('source', 'unknown')
        summary['sources'][source] = summary['sources'].get(source, 0) + 1
        
        # Track date range
        timestamp = metadata.get('timestamp')
        if timestamp:
            if summary['date_range']['earliest'] is None or timestamp < summary['date_range']['earliest']:
                summary['date_range']['earliest'] = timestamp
            if summary['date_range']['latest'] is None or timestamp > summary['date_range']['latest']:
                summary['date_range']['latest'] = timestamp
    
    return summary

def generate_evidence_id(file_path: Path) -> str:
    """
    Generate a unique evidence ID for a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        Unique evidence ID
    """
    # Combine file path and modification time for uniqueness
    path_str = str(file_path.absolute())
    mtime = file_path.stat().st_mtime if file_path.exists() else 0
    
    unique_string = f"{path_str}_{mtime}"
    return hashlib.sha256(unique_string.encode()).hexdigest()[:16]

def is_suspicious_file_extension(extension: str) -> bool:
    """
    Check if file extension is potentially suspicious
    
    Args:
        extension: File extension (with or without dot)
        
    Returns:
        True if suspicious, False otherwise
    """
    if not extension.startswith('.'):
        extension = '.' + extension
    
    suspicious_extensions = {
        '.exe', '.scr', '.bat', '.cmd', '.com', '.pif', '.vbs', '.js',
        '.jar', '.app', '.deb', '.pkg', '.dmg', '.iso', '.msi'
    }
    
    return extension.lower() in suspicious_extensions

def parse_timeline_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parse and sort timeline events
    
    Args:
        events: List of event dictionaries
        
    Returns:
        Sorted list of events
    """
    # Sort events by timestamp
    sorted_events = sorted(events, key=lambda x: x.get('timestamp', ''))
    
    # Add sequence numbers
    for i, event in enumerate(sorted_events):
        event['sequence'] = i + 1
    
    return sorted_events