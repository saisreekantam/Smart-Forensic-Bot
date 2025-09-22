"""
File analysis module for digital forensics
"""

import hashlib
import magic
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

class FileAnalyzer:
    """Analyzes files for forensic investigation"""
    
    def __init__(self):
        self.supported_formats = [
            '.exe', '.dll', '.pdf', '.doc', '.docx', '.xls', '.xlsx',
            '.jpg', '.jpeg', '.png', '.gif', '.mp4', '.avi', '.zip',
            '.rar', '.tar', '.gz', '.log', '.txt', '.csv'
        ]
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Perform comprehensive file analysis
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        analysis = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'created_time': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'accessed_time': datetime.fromtimestamp(file_path.stat().st_atime).isoformat(),
            'file_extension': file_path.suffix.lower(),
            'hashes': self._calculate_hashes(file_path),
            'file_type': self._identify_file_type(file_path),
            'metadata': self._extract_metadata(file_path),
            'suspicious_indicators': self._check_suspicious_indicators(file_path)
        }
        
        return analysis
    
    def _calculate_hashes(self, file_path: Path) -> Dict[str, str]:
        """Calculate various hashes for the file"""
        hashes = {}
        
        with open(file_path, 'rb') as f:
            content = f.read()
            
            # MD5
            hashes['md5'] = hashlib.md5(content).hexdigest()
            
            # SHA1
            hashes['sha1'] = hashlib.sha1(content).hexdigest()
            
            # SHA256
            hashes['sha256'] = hashlib.sha256(content).hexdigest()
        
        return hashes
    
    def _identify_file_type(self, file_path: Path) -> Dict[str, str]:
        """Identify file type using magic numbers"""
        try:
            mime_type = magic.from_file(str(file_path), mime=True)
            description = magic.from_file(str(file_path))
            
            return {
                'mime_type': mime_type,
                'description': description
            }
        except Exception as e:
            return {
                'mime_type': 'unknown',
                'description': f'Error identifying file type: {str(e)}'
            }
    
    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from the file"""
        metadata = {}
        
        # Basic file system metadata
        stat = file_path.stat()
        metadata['permissions'] = oct(stat.st_mode)[-3:]
        metadata['owner_uid'] = stat.st_uid
        metadata['group_gid'] = stat.st_gid
        
        # TODO: Add specific metadata extraction for different file types
        # (EXIF for images, PE headers for executables, etc.)
        
        return metadata
    
    def _check_suspicious_indicators(self, file_path: Path) -> List[str]:
        """Check for suspicious indicators in the file"""
        indicators = []
        
        # Check for suspicious file extensions
        suspicious_extensions = ['.exe', '.scr', '.bat', '.cmd', '.pif', '.com']
        if file_path.suffix.lower() in suspicious_extensions:
            indicators.append(f"Suspicious file extension: {file_path.suffix}")
        
        # Check for double extensions
        name_parts = file_path.name.split('.')
        if len(name_parts) > 2:
            indicators.append("Multiple file extensions detected")
        
        # Check file size anomalies
        if file_path.stat().st_size == 0:
            indicators.append("Zero-byte file")
        elif file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB
            indicators.append("Unusually large file size")
        
        # Check for hidden files
        if file_path.name.startswith('.'):
            indicators.append("Hidden file")
        
        return indicators
    
    def analyze_directory(self, directory_path: Path, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Analyze all files in a directory
        
        Args:
            directory_path: Path to the directory to analyze
            recursive: Whether to analyze subdirectories recursively
            
        Returns:
            List of file analysis results
        """
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory path: {directory_path}")
        
        results = []
        
        if recursive:
            for file_path in directory_path.rglob('*'):
                if file_path.is_file():
                    try:
                        analysis = self.analyze_file(file_path)
                        results.append(analysis)
                    except Exception as e:
                        results.append({
                            'file_path': str(file_path),
                            'error': str(e)
                        })
        else:
            for file_path in directory_path.iterdir():
                if file_path.is_file():
                    try:
                        analysis = self.analyze_file(file_path)
                        results.append(analysis)
                    except Exception as e:
                        results.append({
                            'file_path': str(file_path),
                            'error': str(e)
                        })
        
        return results