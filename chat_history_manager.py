"""
Chat History Manager
Stores and retrieves chat conversations for each case
"""

import json
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ChatHistoryManager:
    """
    Manages chat history for forensic cases
    """
    
    def __init__(self, db_path: str = "data/forensic_cases.db"):
        self.db_path = db_path
        self.init_chat_tables()
    
    def init_chat_tables(self):
        """Initialize chat history tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create chat_sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    session_name TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (case_id) REFERENCES cases (id)
                )
            """)
            
            # Create chat_messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,  -- 'user' or 'assistant'
                    content TEXT NOT NULL,
                    sources TEXT,  -- JSON array of sources
                    confidence REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing chat tables: {str(e)}")
    
    def create_chat_session(self, case_id: str, session_name: Optional[str] = None) -> str:
        """Create a new chat session"""
        try:
            import uuid
            session_id = str(uuid.uuid4())
            
            if not session_name:
                session_name = f"Chat Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO chat_sessions (id, case_id, session_name, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, case_id, session_name, datetime.now().isoformat(), datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating chat session: {str(e)}")
            return ""
    
    def save_message(self, session_id: str, role: str, content: str, sources: List[Dict] = None, confidence: float = None) -> str:
        """Save a chat message"""
        try:
            import uuid
            message_id = str(uuid.uuid4())
            
            sources_json = json.dumps(sources) if sources else None
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO chat_messages (id, session_id, role, content, sources, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (message_id, session_id, role, content, sources_json, confidence, datetime.now().isoformat()))
            
            # Update session timestamp
            cursor.execute("""
                UPDATE chat_sessions SET updated_at = ? WHERE id = ?
            """, (datetime.now().isoformat(), session_id))
            
            conn.commit()
            conn.close()
            
            return message_id
            
        except Exception as e:
            logger.error(f"Error saving message: {str(e)}")
            return ""
    
    def get_chat_sessions(self, case_id: str) -> List[Dict[str, Any]]:
        """Get all chat sessions for a case"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    s.id,
                    s.session_name,
                    s.created_at,
                    s.updated_at,
                    COUNT(m.id) as message_count
                FROM chat_sessions s
                LEFT JOIN chat_messages m ON s.id = m.session_id
                WHERE s.case_id = ?
                GROUP BY s.id, s.session_name, s.created_at, s.updated_at
                ORDER BY s.updated_at DESC
            """, (case_id,))
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    "id": row["id"],
                    "session_name": row["session_name"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "message_count": row["message_count"]
                })
            
            conn.close()
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting chat sessions: {str(e)}")
            return []
    
    def get_chat_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a chat session"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, role, content, sources, confidence, timestamp
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))
            
            messages = []
            for row in cursor.fetchall():
                message = {
                    "id": row["id"],
                    "role": row["role"],
                    "content": row["content"],
                    "confidence": row["confidence"],
                    "timestamp": row["timestamp"]
                }
                
                if row["sources"]:
                    try:
                        message["sources"] = json.loads(row["sources"])
                    except:
                        message["sources"] = []
                else:
                    message["sources"] = []
                
                messages.append(message)
            
            conn.close()
            return messages
            
        except Exception as e:
            logger.error(f"Error getting chat messages: {str(e)}")
            return []
    
    def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session and all its messages"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete messages first
            cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            
            # Delete session
            cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting chat session: {str(e)}")
            return False

# Global instance
chat_history_manager = ChatHistoryManager()

if __name__ == "__main__":
    # Test the chat history manager
    manager = ChatHistoryManager()
    
    # Create a test session
    case_id = "test-case-123"
    session_id = manager.create_chat_session(case_id, "Test Session")
    print(f"Created session: {session_id}")
    
    # Save some test messages
    manager.save_message(session_id, "user", "Who is Alex Rivera?")
    manager.save_message(session_id, "assistant", "Alex Rivera is a contact found in the call logs...", [], 0.9)
    
    # Get sessions
    sessions = manager.get_chat_sessions(case_id)
    print(f"Sessions: {sessions}")
    
    # Get messages
    messages = manager.get_chat_messages(session_id)
    print(f"Messages: {messages}")