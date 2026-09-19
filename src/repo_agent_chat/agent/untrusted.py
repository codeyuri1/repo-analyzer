"""Tratamento de conteúdo não confiável vindo do repositório analisado."""

import json
import re
from typing import Any

# Target imperative prompt-like prose, not ordinary source code. Line count is
# preserved, so source references remain meaningful after redaction.
PROMPT_INJECTION_PATTERN = re.compile(
    r"(?ix)\b(?:"
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|rules?)|"
    r"disregard\s+(?:the\s+)?(?:system|previous|prior)\b|"
    r"system\s+(?:message|prompt|instructions?)\b|"
    r"you\s+are\s+(?:chatgpt|an\s+assistant)\b|"
    r"reveal\s+(?:the\s+)?(?:prompt|secret|token|password)\b|"
    r"(?:call|use)\s+(?:this\s+)?tool\b|"
    r"ignore\s+.*\b(?:safety|guardrail)"
    r")"
)
REDACTED_INJECTION_LINE = "[linha omitida: possível instrução maliciosa no repositório]"


def sanitize_tool_result_for_model(tool_result: object) -> str:
    """Redacts likely prompt injections before a tool result reaches the model."""

    if not isinstance(tool_result, str):
        return json.dumps({"ok": False, "error": "Resultado de tool inválido."})
    try:
        payload = json.loads(tool_result)
    except json.JSONDecodeError:
        return json.dumps({"ok": False, "error": "Resultado de tool inválido."})
    if not isinstance(payload, dict):
        return json.dumps({"ok": False, "error": "Resultado de tool inválido."})

    sanitized, redacted = _sanitize_value(payload, untrusted=False)
    if not redacted:
        return tool_result
    if isinstance(sanitized, dict):
        sanitized["untrusted_content_redacted"] = True
    return json.dumps(sanitized, ensure_ascii=False)


def _sanitize_value(value: Any, *, untrusted: bool) -> tuple[Any, bool]:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        redacted = False
        for key, item in value.items():
            clean_item, item_redacted = _sanitize_value(
                item, untrusted=untrusted or key == "result"
            )
            sanitized[key] = clean_item
            redacted = redacted or item_redacted
        return sanitized, redacted
    if isinstance(value, list):
        values = [_sanitize_value(item, untrusted=untrusted) for item in value]
        return [item for item, _ in values], any(redacted for _, redacted in values)
    if isinstance(value, str) and untrusted:
        lines = value.splitlines(keepends=True)
        sanitized_lines = [
            REDACTED_INJECTION_LINE + ("\n" if line.endswith("\n") else "")
            if PROMPT_INJECTION_PATTERN.search(line)
            else line
            for line in lines
        ]
        return "".join(sanitized_lines), sanitized_lines != lines
    return value, False
