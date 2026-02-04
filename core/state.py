"""
Moltbot State Management
========================
Handles conversation persistence, loading, and context pruning.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass, field, asdict

if TYPE_CHECKING:
    from .inference import Message

logger = logging.getLogger(__name__)


@dataclass
class MessageData:
    """Serializable message data."""
    role: str
    content: str
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationData:
    """Serializable conversation data."""
    user_id: str
    messages: list[MessageData] = field(default_factory=list)
    current_model: str = "forecaster"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class ConversationState:
    """Manages a single conversation's state."""

    def __init__(
        self,
        user_id: str,
        data_dir: Path,
        max_history_tokens: int = 2048
    ):
        self.user_id = user_id
        self.data_dir = data_dir
        self.max_history_tokens = max_history_tokens
        self._data: Optional[ConversationData] = None
        self._file_path = self.data_dir / f"{user_id}.json"

        # Load existing or create new
        self._load_or_create()

    def _load_or_create(self) -> None:
        """Load existing conversation or create new one."""
        if self._file_path.exists():
            try:
                with open(self._file_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                    messages = [MessageData(**m) for m in raw.get('messages', [])]
                    self._data = ConversationData(
                        user_id=raw['user_id'],
                        messages=messages,
                        current_model=raw.get('current_model', 'forecaster'),
                        created_at=raw.get('created_at', time.time()),
                        updated_at=raw.get('updated_at', time.time())
                    )
                logger.info(f"Loaded conversation for user {self.user_id} ({len(self._data.messages)} messages)")
            except Exception as e:
                logger.error(f"Failed to load conversation: {e}")
                self._data = ConversationData(user_id=self.user_id)
        else:
            self._data = ConversationData(user_id=self.user_id)
            logger.info(f"Created new conversation for user {self.user_id}")

    def save(self) -> None:
        """Save conversation to disk."""
        self._data.updated_at = time.time()
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self._file_path, 'w', encoding='utf-8') as f:
                # Convert to dict for JSON serialization
                data_dict = {
                    'user_id': self._data.user_id,
                    'messages': [asdict(m) for m in self._data.messages],
                    'current_model': self._data.current_model,
                    'created_at': self._data.created_at,
                    'updated_at': self._data.updated_at
                }
                json.dump(data_dict, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved conversation for user {self.user_id}")
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")

    def add_message(self, message: "Message") -> None:
        """Add a message to the conversation."""
        msg_data = MessageData(
            role=message.role,
            content=message.content,
            tool_calls=message.tool_calls,
            tool_call_id=message.tool_call_id,
            name=message.name
        )
        self._data.messages.append(msg_data)
        self._prune_if_needed()
        self.save()

    def get_messages(self) -> list["Message"]:
        """Get messages in Message format for inference."""
        from .inference import Message
        return [
            Message(
                role=m.role,
                content=m.content,
                tool_calls=m.tool_calls,
                tool_call_id=m.tool_call_id,
                name=m.name
            )
            for m in self._data.messages
        ]

    def _prune_if_needed(self) -> None:
        """Prune old messages if context exceeds limit."""
        # Rough token estimation: ~4 chars per token
        total_chars = sum(len(m.content or '') for m in self._data.messages)
        estimated_tokens = total_chars // 4

        if estimated_tokens > self.max_history_tokens:
            # Keep removing oldest messages (except system) until under limit
            while estimated_tokens > self.max_history_tokens and len(self._data.messages) > 2:
                removed = self._data.messages.pop(0)
                total_chars -= len(removed.content or '')
                estimated_tokens = total_chars // 4
                logger.debug(f"Pruned old message, {len(self._data.messages)} remaining")

    def clear(self) -> None:
        """Clear conversation history."""
        self._data.messages = []
        self.save()
        logger.info(f"Cleared conversation for user {self.user_id}")

    @property
    def current_model(self) -> str:
        """Get current model for this conversation."""
        return self._data.current_model

    @current_model.setter
    def current_model(self, model: str) -> None:
        """Set current model for this conversation."""
        self._data.current_model = model
        self.save()

    @property
    def message_count(self) -> int:
        """Get number of messages in conversation."""
        return len(self._data.messages)


class StateManager:
    """Manages all conversation states."""

    def __init__(self, data_dir: Path, max_history_tokens: int = 2048):
        self.data_dir = data_dir / "conversations"
        self.max_history_tokens = max_history_tokens
        self._conversations: dict[str, ConversationState] = {}

    def get_conversation(self, user_id: str) -> ConversationState:
        """Get or create a conversation state for a user."""
        user_id = str(user_id)
        if user_id not in self._conversations:
            self._conversations[user_id] = ConversationState(
                user_id=user_id,
                data_dir=self.data_dir,
                max_history_tokens=self.max_history_tokens
            )
        return self._conversations[user_id]

    def clear_conversation(self, user_id: str) -> None:
        """Clear a user's conversation history."""
        conv = self.get_conversation(user_id)
        conv.clear()

    def save_all(self) -> None:
        """Save all active conversations."""
        for conv in self._conversations.values():
            conv.save()
