"""
Configuration management for Smart Forensic Bot
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config(BaseModel):
    """Main configuration class for Smart Forensic Bot"""
    
    # API Keys
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    
    # Neo4j Configuration
    neo4j_uri: str = Field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = Field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    
    # Paths
    vectordb_path: Path = Field(default_factory=lambda: Path(os.getenv("VECTORDB_PATH", "./data/vectordb")))
    evidence_path: Path = Field(default_factory=lambda: Path(os.getenv("EVIDENCE_PATH", "./evidence")))
    temp_path: Path = Field(default_factory=lambda: Path(os.getenv("TEMP_PATH", "./temp")))
    reports_path: Path = Field(default_factory=lambda: Path(os.getenv("REPORTS_PATH", "./reports")))
    langgraph_checkpoint_path: Path = Field(default_factory=lambda: Path(os.getenv("LANGGRAPH_CHECKPOINT_PATH", "./checkpoints")))
    
    # Model Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 2000
    
    class Config:
        env_file = ".env"
        
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        for path in [self.vectordb_path, self.evidence_path, self.temp_path, 
                    self.reports_path, self.langgraph_checkpoint_path]:
            path.mkdir(parents=True, exist_ok=True)

# Global configuration instance
config = Config()
config.create_directories()