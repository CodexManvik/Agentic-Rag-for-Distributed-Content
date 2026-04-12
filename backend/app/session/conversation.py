"""
Conversation history management.

Handles conversation message storage, retrieval, and context window management
for maintaining chat context within model limits.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from .session_state import ConversationMessage, MessageRole

logger = logging.getLogger(__name__)


class ConversationHistory:
    """
    Manages conversation message history with context window awareness.
    
    Provides methods for adding messages, retrieving history, and truncating
    to fit within model context windows while preserving important context.
    """
    
    def __init__(self, messages: Optional[List[ConversationMessage]] = None):
        """
        Initialize conversation history.
        
        Args:
            messages: Optional initial messages
        """
        self.messages: List[ConversationMessage] = messages or []
        self._max_history_length = 1000  # Safety limit
    
    def add_message(
        self,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationMessage:
        """
        Add a message to the conversation.
        
        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            metadata: Optional message metadata
            
        Returns:
            Created message
        """
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.messages.append(message)
        
        # Prevent unlimited growth
        if len(self.messages) > self._max_history_length:
            self._trim_history()
        
        logger.debug(f"Added {role.value} message ({len(content)} chars)")
        return message
    
    def get_messages(
        self,
        limit: Optional[int] = None,
        role_filter: Optional[MessageRole] = None
    ) -> List[ConversationMessage]:
        """
        Get conversation messages with optional filtering.
        
        Args:
            limit: Maximum number of messages to return (most recent)
            role_filter: Optional filter by message role
            
        Returns:
            List of messages
        """
        messages = self.messages
        
        if role_filter:
            messages = [msg for msg in messages if msg.role == role_filter]
        
        if limit is None:
            pass
        elif limit == 0:
            messages = []
        elif limit > 0:
            messages = messages[-limit:]
        
        return messages
    
    def get_context_messages(self, max_tokens: int, tokens_per_char: float = 0.25) -> List[ConversationMessage]:
        """
        Get messages that fit within a token budget.
        
        Uses a simple heuristic for token estimation. For production,
        consider using tiktoken or the model's tokenizer.
        
        Args:
            max_tokens: Maximum tokens to include
            tokens_per_char: Estimated tokens per character
            
        Returns:
            Messages that fit within token budget
        """
        if not self.messages:
            return []
        
        # Always include system messages
        system_messages = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]
        other_messages = [msg for msg in self.messages if msg.role != MessageRole.SYSTEM]
        
        # Calculate tokens for system messages
        system_tokens = sum(len(msg.content) * tokens_per_char for msg in system_messages)
        available_tokens = max_tokens - system_tokens
        
        if available_tokens <= 0:
            logger.warning("System messages exceed token budget")
            return system_messages
        
        # Add other messages from most recent, staying within budget
        selected_messages = []
        current_tokens = 0
        
        for message in reversed(other_messages):
            message_tokens = len(message.content) * tokens_per_char
            
            if current_tokens + message_tokens > available_tokens:
                break
            
            selected_messages.insert(0, message)
            current_tokens += message_tokens
        
        # Combine system messages (first) with selected messages
        result = system_messages + selected_messages
        
        logger.debug(f"Selected {len(result)} messages (~{system_tokens + current_tokens:.0f} tokens)")
        return result
    
    def truncate_to_context_length(
        self,
        context_length: int,
        preserve_system: bool = True,
        preserve_recent: int = 2
    ) -> List[ConversationMessage]:
        """
        Truncate history to fit context length while preserving important messages.
        
        Args:
            context_length: Maximum context length in tokens (estimated)
            preserve_system: Whether to always preserve system messages
            preserve_recent: Number of recent messages to always preserve
            
        Returns:
            Truncated message list
        """
        if not self.messages:
            return []
        
        # Use conservative token estimation (4 chars per token)
        max_chars = context_length * 4
        
        # Identify messages to preserve
        preserved_messages = []
        
        if preserve_system:
            preserved_messages.extend([
                msg for msg in self.messages if msg.role == MessageRole.SYSTEM
            ])
        
        if preserve_recent > 0:
            recent_non_system = [
                msg for msg in self.messages[-preserve_recent:]
                if msg.role != MessageRole.SYSTEM
            ]
            preserved_messages.extend(recent_non_system)
        
        # Calculate space taken by preserved messages
        preserved_chars = sum(len(msg.content) for msg in preserved_messages)
        
        if preserved_chars >= max_chars:
            logger.warning("Preserved messages exceed context length")
            recent_part = preserved_messages[-preserve_recent:] if preserve_recent else []
            if preserve_system:
                system_part = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]
                merged = []
                seen_ids = set()
                for msg in system_part + recent_part:
                    msg_key = id(msg)
                    if msg_key in seen_ids:
                        continue
                    seen_ids.add(msg_key)
                    merged.append(msg)
                merged.sort(key=lambda msg: self.messages.index(msg))
                return merged
            return recent_part
        
        # Fill remaining space with other messages
        available_chars = max_chars - preserved_chars
        other_messages = [
            msg for msg in self.messages
            if msg not in preserved_messages
        ]
        
        selected_messages = []
        current_chars = 0
        
        # Add from most recent to oldest
        for message in reversed(other_messages):
            message_chars = len(message.content)
            
            if current_chars + message_chars > available_chars:
                break
            
            selected_messages.insert(0, message)
            current_chars += message_chars
        
        # Combine and sort by original order
        all_selected = preserved_messages + selected_messages
        all_selected.sort(key=lambda msg: self.messages.index(msg))
        
        logger.info(f"Truncated to {len(all_selected)} messages (~{preserved_chars + current_chars} chars)")
        return all_selected
    
    def clear_history(self) -> None:
        """Clear all conversation history."""
        message_count = len(self.messages)
        self.messages.clear()
        logger.info(f"Cleared {message_count} messages from history")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about conversation history."""
        if not self.messages:
            return {
                "total_messages": 0,
                "total_characters": 0,
                "by_role": {},
                "first_message": None,
                "last_message": None
            }
        
        by_role = {}
        total_chars = 0
        
        for message in self.messages:
            role = message.role.value
            by_role[role] = by_role.get(role, 0) + 1
            total_chars += len(message.content)
        
        return {
            "total_messages": len(self.messages),
            "total_characters": total_chars,
            "by_role": by_role,
            "first_message": self.messages[0].timestamp.isoformat(),
            "last_message": self.messages[-1].timestamp.isoformat()
        }
    
    def serialize(self) -> str:
        """Serialize conversation to JSON string."""
        return json.dumps([msg.to_dict() for msg in self.messages])
    
    @classmethod
    def deserialize(cls, data: str) -> "ConversationHistory":
        """Deserialize conversation from JSON string."""
        try:
            messages_data = json.loads(data)
            messages = [ConversationMessage.from_dict(msg_data) for msg_data in messages_data]
            return cls(messages)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to deserialize conversation: {e}")
            return cls()  # Return empty history on error
    
    def _trim_history(self) -> None:
        """Trim history to reasonable size when it gets too long."""
        if len(self.messages) > self._max_history_length:
            other_messages = [msg for msg in self.messages if msg.role != MessageRole.SYSTEM]

            # Keep last 800 non-system messages.
            keep_count = 800
            keep_non_system = set(id(msg) for msg in other_messages[-keep_count:])
            self.messages = [
                msg
                for msg in self.messages
                if msg.role == MessageRole.SYSTEM or id(msg) in keep_non_system
            ]
            logger.info(f"Trimmed conversation history to {len(self.messages)} messages")
    
    def __len__(self) -> int:
        """Return number of messages in history."""
        return len(self.messages)
    
    def __iter__(self):
        """Iterate over messages."""
        return iter(self.messages)