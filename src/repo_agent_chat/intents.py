from enum import StrEnum


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


def classify_intent(question: str) -> UserIntent:
    """Classifica somente intenções amplas com alta confiança."""

    normalized = " ".join(question.casefold().split())
    if "mermaid" in normalized or (
        "diagrama" in normalized and "depend" in normalized
    ):
        return UserIntent.MERMAID_DIAGRAM
    if "vulnerabil" in normalized or "análise de segurança" in normalized:
        return UserIntent.SECURITY_ANALYSIS
    if any(phrase in normalized for phrase in OVERVIEW_PHRASES):
        return UserIntent.PROJECT_OVERVIEW
    return UserIntent.GENERAL
