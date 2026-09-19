"""Estratégias declarativas das investigações expostas pela interface."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    require_sources: bool = True
    require_citations: bool = True
    require_confirmed_read: bool = False
    require_mermaid: bool = False


@dataclass(frozen=True, slots=True)
class QuestionDefinition:
    """Contrato usado pela UI, pelo agente e pelos avaliadores."""

    id: str
    label: str
    prompt_template: str
    description: str
    allowed_tools: tuple[str, ...]
    expected_tool_groups: tuple[tuple[str, tuple[str, ...]], ...]
    deterministic_evals: tuple[str, ...]
    semantic_evals: tuple[str, ...]
    evidence_policy: EvidencePolicy = EvidencePolicy()

    @property
    def prompt(self) -> str:
        return self.prompt_template


_EVIDENCE = ("semantic_search", "read_file")
_BROWSE = ("semantic_search", "list_files", "read_file")
_STANDARD = ("answer_non_empty", "allowed_tools", "sources_present", "citations_valid", "symbols_exist")

QUESTION_CATALOG = (
    QuestionDefinition("overview", "Visão geral do projeto", "Explique este projeto para alguém que está conhecendo o código.", "Propósito, estrutura e componentes principais", _BROWSE, (("retrieval", _EVIDENCE), ("structure", ("list_files",))), _STANDARD + ("scope",), ("groundedness", "completeness", "no_invented_components")),
    QuestionDefinition("architecture", "Arquitetura", "Qual é a arquitetura deste projeto?", "Componentes, responsabilidades e relações", _BROWSE, (("retrieval", _EVIDENCE),), _STANDARD + ("components_exist", "relations_have_evidence"), ("groundedness", "fact_vs_inference", "relevance"), EvidencePolicy(require_confirmed_read=True)),
    QuestionDefinition("main_flow", "Fluxo principal", "Rastreie o fluxo principal da aplicação, a partir de um entrypoint plausível, citando cada transição observada.", "Execução e caminhos mais importantes", _BROWSE, (("retrieval", _EVIDENCE),), _STANDARD + ("flow_entrypoint",), ("flow_coherence", "groundedness"), EvidencePolicy(require_confirmed_read=True)),
    QuestionDefinition("rag_pipeline", "Pipeline de RAG", "Como o pipeline de RAG é implementado? Descreva somente etapas comprovadas e diferencie retrieval de reranking.", "Retrieval, contexto e reranking", _BROWSE, (("retrieval", _EVIDENCE),), _STANDARD + ("only_present_stages",), ("groundedness", "retrieval_vs_reranking"), EvidencePolicy(require_confirmed_read=True)),
    QuestionDefinition("engineering_decisions", "Decisões de engenharia", "Quais decisões de engenharia se destacam neste projeto? Separe evidência observada de interpretação e trade-offs.", "Escolhas técnicas e seus trade-offs", _BROWSE, (("retrieval", _EVIDENCE),), _STANDARD, ("evidence_vs_interpretation", "groundedness"), EvidencePolicy(require_confirmed_read=True)),
    QuestionDefinition("testing", "Estratégia de testes", "Como a estratégia de testes protege a qualidade? Identifique apenas frameworks, tipos de teste e cobertura comprovados.", "Testes automatizados e cobertura de riscos", _BROWSE, (("test_discovery", ("list_files", "semantic_search")), ("evidence", _EVIDENCE)), _STANDARD + ("test_files_exist", "no_unproven_coverage"), ("testing_completeness", "groundedness"), EvidencePolicy(require_confirmed_read=True)),
    QuestionDefinition("security", "Segurança", "Encontre possíveis riscos de segurança no repositório. Diferencie achados confirmados de riscos potenciais e mencione controles observados.", "Riscos encontrados e controles existentes", ("semantic_search", "read_file", "analyze_vulnerabilities"), (("static_analysis", ("analyze_vulnerabilities",)),), _STANDARD + ("security_findings_have_evidence",), ("severity_proportionality", "groundedness")),
    QuestionDefinition("diagram", "Gerar diagrama", "Gere um diagrama Mermaid das dependências entre módulos, baseado somente nos arquivos analisados.", "Visão arquitetural baseada no código", ("semantic_search", "read_file", "list_files", "generate_mermaid_diagram"), (("diagram", ("generate_mermaid_diagram",)), ("evidence", _EVIDENCE)), _STANDARD + ("mermaid_valid", "diagram_paths_exist"), ("diagram_groundedness",), EvidencePolicy(require_mermaid=True)),
)
QUESTION_BY_ID = {question.id: question for question in QUESTION_CATALOG}


def resolve_question(question_id: str) -> QuestionDefinition:
    try:
        return QUESTION_BY_ID[question_id]
    except (KeyError, TypeError) as error:
        raise ValueError("Ação de investigação inválida.") from error
