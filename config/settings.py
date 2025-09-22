"""
Project Sentinel Configuration Module (Simplified)
Handles all configuration settings for the AI-powered UFDR analysis platform
"""

import os
from pathlib import Path
from typing import List, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use system environment variables

class Settings:
    """Application settings"""
    
    def __init__(self):
        # Application
        self.app_name = os.getenv("APP_NAME", "Project Sentinel")
        self.app_version = os.getenv("APP_VERSION", "1.0.0")
        self.debug = os.getenv("DEBUG", "True").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # API Configuration
        self.api_host = os.getenv("API_HOST", "127.0.0.1")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        
        # AI/LLM API Keys
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        
        # Database Configuration
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        # ChromaDB Configuration
        self.chroma_db_path = os.getenv("CHROMA_DB_PATH", "./data/vector_db")
        self.chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME", "ufdr_embeddings")
        
        # Model Configuration
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.llm_model = os.getenv("LLM_MODEL", "gemini-pro")
        self.ner_model = os.getenv("NER_MODEL", "en_core_web_sm")
        
        # Text Processing Configuration
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        self.max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
        
        # Data Processing
        self.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
        supported_formats_str = os.getenv("SUPPORTED_FORMATS", "xml,csv,txt,json")
        self.supported_formats = [fmt.strip() for fmt in supported_formats_str.split(",")]
        
        # Security
        self.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-this")
        self.algorithm = os.getenv("ALGORITHM", "HS256")
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        
        # Paths
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.sample_data_dir = self.data_dir / "sample"
        self.config_dir = self.project_root / "config"
        
        # Ensure data directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.raw_data_dir.mkdir(exist_ok=True)
        self.processed_data_dir.mkdir(exist_ok=True)
        self.sample_data_dir.mkdir(exist_ok=True)

class DatabaseConfig:
    """Database specific configurations"""
    
    @staticmethod
    def get_neo4j_config(settings: Settings) -> dict:
        """Get Neo4j configuration"""
        return {
            "uri": settings.neo4j_uri,
            "username": settings.neo4j_username,
            "password": settings.neo4j_password,
            "max_connection_lifetime": 3600,
            "max_connection_pool_size": 50,
            "connection_timeout": 30
        }
    
    @staticmethod
    def get_chroma_config(settings: Settings) -> dict:
        """Get ChromaDB configuration"""
        return {
            "persist_directory": settings.chroma_db_path,
            "collection_name": settings.chroma_collection_name
        }

class ForensicEntities:
    """Forensic-specific entity patterns and configurations"""
    
    # Regex patterns for forensic entities
    PATTERNS = {
        "phone_number": r"(\+?\d{1,4}[\s-]?)?\(?\d{1,4}\)?[\s-]?\d{1,4}[\s-]?\d{1,9}",
        "crypto_address": {
            "bitcoin": r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b",
            "ethereum": r"\b0x[a-fA-F0-9]{40}\b",
            "monero": r"\b4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}\b"
        },
        "bank_account": r"\b\d{8,18}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
        "imei": r"\b\d{15}\b",
        "license_plate": r"\b[A-Z]{1,3}[\s-]?\d{1,4}[\s-]?[A-Z]{0,3}\b"
    }
    
    # Entity types for NER
    ENTITY_TYPES = [
        "PERSON", "ORG", "GPE", "DATE", "TIME", "MONEY", 
        "PHONE", "CRYPTO", "BANK_ACCOUNT", "EMAIL", "IP_ADDRESS", "IMEI", "LICENSE_PLATE"
    ]

# Global settings instance
settings = Settings()