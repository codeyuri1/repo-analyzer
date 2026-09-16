from typing import Any


class ConversationMemory:
    """Mantém somente mensagens válidas e limita o contexto recente."""

    def __init__(self, max_messages: int) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages deve ser maior que zero.")
        self._max_messages = max_messages
        self._messages: list[dict[str, str]] = []

    def clear(self) -> None:
        self._messages.clear()

    def replace(self, messages: list[dict[str, object]]) -> None:
        normalized: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            normalized.append({"role": role, "content": content})
        self._messages = normalized[-self._max_messages :]

    def context(self) -> list[dict[str, Any]]:
        return [dict(message) for message in self._messages]

    def save_turn(self, user_message: dict[str, str], answer: str) -> None:
        self._messages.extend(
            [user_message, {"role": "assistant", "content": answer}]
        )
        self._messages = self._messages[-self._max_messages :]
