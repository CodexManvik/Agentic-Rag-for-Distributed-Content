"""
Session manager for handling session CRUD operations.

Provides persistent storage and management of user sessions using SQLite
for embedded deployment simplicity.
"""

import sqlite3
import uuid
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from .session_state import SessionState, ConversationMessage, ModelConfig

logger = logging.getLogger(__name__)


def _safe_user_id_for_log(user_id: Optional[str]) -> str:
    if not user_id:
        return "anonymous"
    return f"{user_id[:4]}..." if len(user_id) > 4 else user_id


class SessionManager:
    """
    Manages session persistence using SQLite.
    
    Handles creation, retrieval, updating, and deletion of user sessions
    with conversation history and state management.
    """
    
    def __init__(self, db_path: str = "data/sessions.db"):
        """
        Initialize session manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_dir()
        self._init_db()
    
    def close(self) -> None:
        """Close any open database connections."""
        # For now, we use context managers, so no explicit close needed
        # This method exists for compatibility and future connection pooling
        pass
    
    def _ensure_db_dir(self) -> None:
        """Ensure database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL,
                    conversation_history TEXT DEFAULT '[]',
                    active_agents TEXT DEFAULT '[]',
                    knowledge_base_id TEXT,
                    model_config TEXT,
                    session_metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_sessions 
                ON sessions(user_id, last_active DESC)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_active
                ON sessions(last_active DESC)
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    def create_session(
        self,
        user_id: Optional[str] = None,
        knowledge_base_id: Optional[str] = None,
        model_config: Optional[ModelConfig] = None
    ) -> SessionState:
        """
        Create a new session.
        
        Args:
            user_id: Optional user identifier
            knowledge_base_id: Optional knowledge base to use
            model_config: Optional model configuration
            
        Returns:
            Created session state
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session = SessionState(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_active=now,
            knowledge_base_id=knowledge_base_id,
            model_configuration=model_config
        )
        
        self._save_session(session)
        
        logger.info(f"Created new session: {session_id} for user: {_safe_user_id_for_log(user_id)}")
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        Retrieve session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session state if found, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM sessions WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return self._row_to_session(row)
        finally:
            conn.close()
    
    def update_session(self, session: SessionState) -> bool:
        """
        Update existing session.
        
        Args:
            session: Session state to update
            
        Returns:
            True if updated successfully, False otherwise
        """
        session.update_last_active()
        return self._save_session(session)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM sessions WHERE session_id = ?
                """, (session_id,))
                
                deleted = cursor.rowcount > 0
                conn.commit()
                
                if deleted:
                    logger.info(f"Deleted session: {session_id}")
                
                return deleted
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def list_sessions(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[SessionState]:
        """
        List sessions with optional filtering.
        
        Args:
            user_id: Optional user filter
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip
            
        Returns:
            List of session states
        """
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute("""
                    SELECT * FROM sessions 
                    WHERE user_id = ?
                    ORDER BY last_active DESC
                    LIMIT ? OFFSET ?
                """, (user_id, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM sessions 
                    ORDER BY last_active DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            
            return [self._row_to_session(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_session_count(self, user_id: Optional[str] = None) -> int:
        """
        Get total count of sessions with optional filtering.
        
        Args:
            user_id: Optional user filter
            
        Returns:
            Total number of sessions
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute("""
                    SELECT COUNT(*) FROM sessions 
                    WHERE user_id = ?
                """, (user_id,))
            else:
                cursor.execute("""
                    SELECT COUNT(*) FROM sessions
                """)
            
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get session statistics matching SessionStatsResponse schema.
        
        Returns:
            Dictionary containing session statistics
        """
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Total sessions
            cursor.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0]
            
            # Active sessions (sessions accessed in last 24 hours)
            yesterday = datetime.now().timestamp() - (24 * 3600)
            yesterday_str = datetime.fromtimestamp(yesterday).isoformat()
            cursor.execute("SELECT COUNT(*) FROM sessions WHERE last_active > ?", (yesterday_str,))
            active_sessions = cursor.fetchone()[0]
            
            # Messages today (count conversation entries from today's sessions)
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_str = today_start.isoformat()
            cursor.execute(
                """
                SELECT COALESCE(SUM(
                    CASE
                        WHEN json_valid(conversation_history) THEN json_array_length(conversation_history)
                        ELSE 0
                    END
                ), 0)
                FROM sessions
                WHERE last_active > ?
                """,
                (today_str,),
            )
            messages_today = int(cursor.fetchone()[0] or 0)

            # Average session length (in conversation turns)
            cursor.execute(
                """
                SELECT COALESCE(AVG(
                    CASE
                        WHEN json_valid(conversation_history) THEN json_array_length(conversation_history)
                        ELSE 0
                    END
                ), 0)
                FROM sessions
                """
            )
            avg_session_length = float(cursor.fetchone()[0] or 0)
            
            # Top knowledge bases (group by knowledge_base_id)
            cursor.execute("""
                SELECT knowledge_base_id, COUNT(*) as usage_count 
                FROM sessions 
                WHERE knowledge_base_id IS NOT NULL 
                GROUP BY knowledge_base_id 
                ORDER BY usage_count DESC 
                LIMIT 5
            """)
            top_knowledge_bases = []
            for row in cursor.fetchall():
                top_knowledge_bases.append({
                    "knowledge_base_id": row['knowledge_base_id'],
                    "session_count": row['usage_count']
                })
            
            return {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "messages_today": messages_today,
                "avg_session_length": round(avg_session_length, 2),
                "top_knowledge_bases": top_knowledge_bases
            }
        finally:
            conn.close()
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """
        Clean up sessions older than specified days.
        
        Args:
            days_old: Number of days to consider as old
            
        Returns:
            Number of sessions deleted
        """
        cutoff = datetime.now().timestamp() - (days_old * 24 * 3600)
        cutoff_str = datetime.fromtimestamp(cutoff).isoformat()
        
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM sessions 
                    WHERE last_active < ?
                """, (cutoff_str,))
                
                deleted = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted} old sessions (older than {days_old} days)")
                return deleted
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0
    
    def _save_session(self, session: SessionState) -> bool:
        """Save session to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                
                # Serialize complex fields
                conversation_history = json.dumps([
                    msg.to_dict() for msg in session.conversation_history
                ])
                active_agents = json.dumps(session.active_agents)
                model_config = json.dumps(
                    session.model_configuration.to_dict() if session.model_configuration else None
                )
                session_metadata = json.dumps(session.session_metadata)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO sessions (
                        session_id, user_id, created_at, last_active,
                        conversation_history, active_agents, knowledge_base_id,
                        model_config, session_metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.user_id,
                    session.created_at.isoformat(),
                    session.last_active.isoformat(),
                    conversation_history,
                    active_agents,
                    session.knowledge_base_id,
                    model_config,
                    session_metadata
                ))
                
                conn.commit()
                return True
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            return False
    
    def _row_to_session(self, row: sqlite3.Row) -> SessionState:
        """Convert database row to SessionState."""
        # Deserialize complex fields
        conversation_history = [
            ConversationMessage.from_dict(msg_data)
            for msg_data in json.loads(row["conversation_history"] or "[]")
        ]
        
        active_agents = json.loads(row["active_agents"] or "[]")
        
        model_config = None
        if row["model_config"]:
            model_config_data = json.loads(row["model_config"])
            if model_config_data:
                model_config = ModelConfig.from_dict(model_config_data)
        
        session_metadata = json.loads(row["session_metadata"] or "{}")
        
        return SessionState(
            session_id=row["session_id"],
            user_id=row["user_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_active=datetime.fromisoformat(row["last_active"]),
            conversation_history=conversation_history,
            active_agents=active_agents,
            knowledge_base_id=row["knowledge_base_id"],
            model_configuration=model_config,
            session_metadata=session_metadata
        )