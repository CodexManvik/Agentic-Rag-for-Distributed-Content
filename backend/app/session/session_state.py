"""
Session state dataclass for session management.

Defines the structure for session data including conversation history,
active agents, model configuration, and metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class MessageRole(str, Enum):
    """Message roles for conversation history."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ConversationMessage:
    """Individual message in conversation history."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMessage":
        """Create from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class ModelConfig:
    """Model configuration for session."""
    backend: str  # "llama.cpp", "ollama", "local"
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 4096
    context_length: int = 8192
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "context_length": self.context_length,
            "additional_params": self.additional_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create from dictionary."""
        return cls(
            backend=data["backend"],
            model_name=data["model_name"],
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 4096),
            context_length=data.get("context_length", 8192),
            additional_params=data.get("additional_params", {})
        )


@dataclass
class SessionState:
    """
    Complete session state for managing user interactions.
    
    Stores all necessary information for maintaining context
    across multi-turn conversations including history, agent state,
    and configuration.
    """
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    conversation_history: List[ConversationMessage] = field(default_factory=list)
    active_agents: List[str] = field(default_factory=list)
    knowledge_base_id: Optional[str] = None
    model_configuration: Optional[ModelConfig] = None
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_last_active(self) -> None:
        """Update the last active timestamp."""
        self.last_active = datetime.now()
    
    def add_message(self, message: ConversationMessage) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append(message)
        self.update_last_active()
    
    def get_recent_messages(self, count: int = 10) -> List[ConversationMessage]:
        """Get the most recent messages."""
        return self.conversation_history[-count:]
    
    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        self.update_last_active()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "conversation_history": [msg.to_dict() for msg in self.conversation_history],
            "active_agents": self.active_agents,
            "knowledge_base_id": self.knowledge_base_id,
            "model_configuration": self.model_configuration.to_dict() if self.model_configuration else None,
            "session_metadata": self.session_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Create from dictionary."""
        conversation_history = [
            ConversationMessage.from_dict(msg_data)
            for msg_data in data.get("conversation_history", [])
        ]
        
        model_configuration = None
        if data.get("model_configuration"):
            model_configuration = ModelConfig.from_dict(data["model_configuration"])
        
        return cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
            conversation_history=conversation_history,
            active_agents=data.get("active_agents", []),
            knowledge_base_id=data.get("knowledge_base_id"),
            model_configuration=model_configuration,
            session_metadata=data.get("session_metadata", {})
        )