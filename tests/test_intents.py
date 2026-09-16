import pytest

from repo_agent_chat.agent.intents import UserIntent, classify_intent


@pytest.mark.parametrize(
    "question",
    [
        "Explique o projeto",
        "Quero uma visão geral do projeto",
        "Como funciona o projeto?",
        "Explique o fluxo principal da aplicação",
    ],
)
def test_classifica_visao_geral(question: str) -> None:
    assert classify_intent(question) is UserIntent.PROJECT_OVERVIEW


def test_pergunta_especifica_permanece_no_fluxo_geral() -> None:
    assert classify_intent("Como funciona o chunk_source_file?") is UserIntent.GENERAL


@pytest.mark.parametrize(
    "question",
    [
        "Analise possíveis vulnerabilidades em src/",
        "Faça uma análise de segurança do projeto",
        "Verifique a segurança deste repositório",
        "Faça uma auditoria de segurança",
        "Procure falhas de segurança no código",
    ],
)
def test_classifica_analise_de_seguranca(question: str) -> None:
    assert classify_intent(question) is UserIntent.SECURITY_ANALYSIS


def test_classifica_diagrama() -> None:
    assert (
        classify_intent("Gere um diagrama Mermaid das dependências")
        is UserIntent.MERMAID_DIAGRAM
    )


def test_leitura_segura_nao_e_confundida_com_analise_de_vulnerabilidade() -> None:
    assert classify_intent("Onde ocorre a leitura segura?") is UserIntent.GENERAL
