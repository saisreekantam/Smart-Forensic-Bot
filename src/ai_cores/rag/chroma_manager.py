"""
ChromaDB Client Manager

This module provides a singleton pattern for managing ChromaDB clients
to avoid conflicts when multiple components try to access the same database.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ChromaDBManager:
    """Singleton manager for ChromaDB clients to prevent conflicts"""
    
    _instances: Dict[str, chromadb.PersistentClient] = {}
    _settings_cache: Dict[str, ChromaSettings] = {}
    
    @classmethod
    def get_client(
        cls, 
        persist_directory: str = "data/vector_db",
        settings: Optional[ChromaSettings] = None
    ) -> chromadb.PersistentClient:
        """
        Get or create a ChromaDB client for the specified directory
        
        Args:
            persist_directory: Directory for persistent storage
            settings: ChromaDB settings (optional, will use defaults if not provided)
            
        Returns:
            ChromaDB PersistentClient instance
        """
        # Normalize the path
        persist_path = str(Path(persist_directory).absolute())
        
        # Check if we already have a client for this path
        if persist_path in cls._instances:
            logger.debug(f"Reusing existing ChromaDB client for {persist_path}")
            return cls._instances[persist_path]
        
        # Create default settings if not provided
        if settings is None:
            settings = ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        
        # Check if there's a settings mismatch (be more permissive)
        if persist_path in cls._settings_cache:
            existing_settings = cls._settings_cache[persist_path]
            # Only warn about major differences, don't fail
            if (hasattr(existing_settings, 'anonymized_telemetry') and 
                hasattr(settings, 'anonymized_telemetry') and
                existing_settings.anonymized_telemetry != settings.anonymized_telemetry):
                logger.warning(f"Telemetry setting mismatch for {persist_path}, using existing")
            settings = existing_settings
        
        try:
            # Ensure directory exists
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Create new client
            logger.info(f"Creating new ChromaDB client for {persist_path}")
            client = chromadb.PersistentClient(
                path=persist_path,
                settings=settings
            )
            
            # Cache the client and settings
            cls._instances[persist_path] = client
            cls._settings_cache[persist_path] = settings
            
            return client
            
        except Exception as e:
            logger.error(f"Failed to create ChromaDB client for {persist_path}: {e}")
            raise
    
    @classmethod
    def reset_client(cls, persist_directory: str = "data/vector_db"):
        """
        Reset (remove) a ChromaDB client for the specified directory
        
        Args:
            persist_directory: Directory for persistent storage
        """
        persist_path = str(Path(persist_directory).absolute())
        
        if persist_path in cls._instances:
            logger.info(f"Resetting ChromaDB client for {persist_path}")
            del cls._instances[persist_path]
            del cls._settings_cache[persist_path]
    
    @classmethod
    def reset_all_clients(cls):
        """Reset all ChromaDB clients"""
        logger.info("Resetting all ChromaDB clients")
        cls._instances.clear()
        cls._settings_cache.clear()
    
    @classmethod
    def list_active_clients(cls) -> Dict[str, chromadb.PersistentClient]:
        """Get a list of all active ChromaDB clients"""
        return cls._instances.copy()


# Convenience function for getting the default client
def get_chroma_client(persist_directory: str = "data/vector_db") -> chromadb.PersistentClient:
    """
    Get the default ChromaDB client
    
    Args:
        persist_directory: Directory for persistent storage
        
    Returns:
        ChromaDB PersistentClient instance
    """
    return ChromaDBManager.get_client(persist_directory)