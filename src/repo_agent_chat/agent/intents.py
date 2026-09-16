from enum import StrEnum
from unicodedata import combining, normalize


class UserIntent(StrEnum):
    """Intenções que precisam de workflows próprios e previsíveis."""

    GENERAL = "general"
    MERMAID_DIAGRAM = "mermaid_diagram"
    PROJECT_OVERVIEW = "project_overview"
    SECURITY_ANALYSIS = "security_analysis"


OVERVIEW_PHRASES = (
    "explique o projeto",
    "explica o projeto",
    "visão geral do projeto",
    "visao geral do projeto",
    "como funciona o projeto",
    "explique este repositório",
    "explique esse repositório",
    "explique este repositorio",
    "explique esse repositorio",
    "fluxo principal da aplicação",
    "fluxo principal da aplicacao",
)

SECURITY_PHRASES = (
    "analise de seguranca",
    "analisar a seguranca",
    "analise a seguranca",
    "auditoria de seguranca",
    "audite a seguranca",
    "falhas de seguranca",
    "problemas de seguranca",
    "riscos de seguranca",
    "verifique a seguranca",
    "verificar a seguranca",
)


def _normalize_question(question: str) -> str:
    folded = normalize("NFKD", question.casefold())
    without_accents = "".join(character for character in folded if not combining(character))
    return " ".join(without_accents.split())


def classify_intent(question: str) -> UserIntent:
    """Classifica somente intenções amplas com alta confiança."""

    normalized = _normalize_question(question)
    if "mermaid" in normalized or (
        "diagrama" in normalized and "depend" in normalized
    ):
        return UserIntent.MERMAID_DIAGRAM
    if "vulnerabil" in normalized or any(
        phrase in normalized for phrase in SECURITY_PHRASES
    ):
        return UserIntent.SECURITY_ANALYSIS
    if any(phrase in normalized for phrase in OVERVIEW_PHRASES):
        return UserIntent.PROJECT_OVERVIEW
    return UserIntent.GENERAL
