"""
Project Sentinel Configuration Module
Handles all configuration settings for the AI-powered UFDR analysis platform
"""

import os
from pathlib import Path
from typing import List, Optional
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    try:
        from pydantic import BaseSettings, Field
    except ImportError:
        # Fallback for very old pydantic versions or missing pydantic
        class BaseSettings:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        def Field(default=None, env=None):
            return default
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    app_name: str = Field(default="Project Sentinel", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API Configuration
    api_host: str = Field(default="127.0.0.1", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # AI/LLM API Keys
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    
    # Database Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", env="NEO4J_USERNAME")
    neo4j_password: Optional[str] = Field(default=None, env="NEO4J_PASSWORD")
    
    # ChromaDB Configuration
    chroma_db_path: str = Field(default="./data/vector_db", env="CHROMA_DB_PATH")
    chroma_collection_name: str = Field(default="ufdr_embeddings", env="CHROMA_COLLECTION_NAME")
    
    # Model Configuration
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    llm_model: str = Field(default="gemini-pro", env="LLM_MODEL")
    ner_model: str = Field(default="en_core_web_sm", env="NER_MODEL")
    
    # Text Processing Configuration
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    max_chunk_size: int = Field(default=1000, env="MAX_CHUNK_SIZE")
    
    # Data Processing
    max_file_size_mb: int = Field(default=100, env="MAX_FILE_SIZE_MB")
    supported_formats: List[str] = Field(default=["xml", "csv", "txt", "json"], env="SUPPORTED_FORMATS")
    
    # Security
    secret_key: str = Field(default="your-secret-key-change-this", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    sample_data_dir: Path = data_dir / "sample"
    config_dir: Path = project_root / "config"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

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

# Ensure data directories exist
settings.data_dir.mkdir(exist_ok=True)
settings.raw_data_dir.mkdir(exist_ok=True)
settings.processed_data_dir.mkdir(exist_ok=True)
settings.sample_data_dir.mkdir(exist_ok=True)