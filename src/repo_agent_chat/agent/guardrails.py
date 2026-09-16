import re

from repo_agent_chat.tools.registry import TOOL_DEFINITIONS

TOOL_NAMES = {definition["function"]["name"] for definition in TOOL_DEFINITIONS}
CITATION_PATTERN = re.compile(
    r"[\w./-]+\.[a-zA-Z0-9]+:(?P<start>\d+)(?:-(?P<end>\d+))?"
)
WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÿ_][\w]*")
EVIDENCE_STOP_WORDS = {
    "como",
    "para",
    "uma",
    "das",
    "dos",
    "que",
    "com",
    "por",
    "the",
    "and",
    "from",
    "return",
    "self",
}
