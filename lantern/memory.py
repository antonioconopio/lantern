"""Per-session chat history"""

import threading

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class ChatHistoryStore:
    def __init__(self, max_messages: int = 10):
        # max_messages caps what's sent to the model each turn. Old turns are
        # dropped so prompts (and cost) don't grow without limit.
        self.max_messages = max_messages
        self._sessions: dict[str, list[BaseMessage]] = {}
        self._lock = threading.Lock()  # requests can run in parallel threads

    def get(self, session_id: str) -> list[BaseMessage]:
        """Return the most recent messages for a session (empty if new)."""
        with self._lock:
            return list(self._sessions.get(session_id, [])[-self.max_messages :])

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.extend([HumanMessage(question), AIMessage(answer)])
            del history[: -self.max_messages]  # keep only what we'd ever send

    def clear(self, session_id: str) -> bool:
        """Delete a session. Returns False if it didn't exist."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None