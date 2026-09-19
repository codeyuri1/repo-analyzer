import re
from dataclasses import dataclass
from pathlib import Path

from repo_agent_chat.agent import ToolAgent
from repo_agent_chat.evaluation.citations import CitationEvaluator
from repo_agent_chat.evaluation.faithfulness import (
    extract_referenced_symbols,
    symbol_exists,
)
from repo_agent_chat.repository import discover_source_files

CITATION_PATTERN = re.compile(
    r"(?P<path>[\w./-]+\.[a-zA-Z0-9]+):(?P<start>\d+)(?:-(?P<end>\d+))?"
)
QUOTED_CITATION_PATTERN = re.compile(r"`[\w./-]+\.[a-zA-Z0-9]+:\d+(?:-\d+)?`")


@dataclass(frozen=True, slots=True)
class EvalCase:
    """Define uma pergunta e critérios objetivos para sua resposta."""

    name: str
    question: str
    required_terms: tuple[str, ...] = ()
    required_concepts: tuple[tuple[str, tuple[str, ...]], ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    required_tool_groups: tuple[tuple[str, tuple[str, ...]], ...] = ()
    required_evidence_paths: tuple[str, ...] = ()
    minimum_citations: int = 0
    maximum_citations: int | None = None
    maximum_citation_span: int | None = None
    forbid_code_blocks: bool = False
    require_reference_only_citations: bool = False
    require_faithful_symbols: bool = False
    require_mermaid_block: bool = False


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Resultado detalhado de um caso de avaliação."""

    case_name: str
    passed: bool
    checks: dict[str, bool]
    answer: str
    tool_calls: tuple[str, ...]
    evidence_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Citation:
    """Referência a um intervalo de linhas encontrada em uma resposta."""

    path: str
    start_line: int
    end_line: int


REPOSITORY_SAFETY_CASE = EvalCase(
    name="repository_safe_read",
    question=(
        "Liste os módulos do projeto e explique onde ocorre a leitura segura "
        "de arquivos."
    ),
    required_terms=(
        "agent/orchestrator.py",
        "repository/reader.py",
        "tools/registry.py",
        "read_source_file",
    ),
    required_concepts=(
        ("path_inside_root", ("is_relative_to", "dentro da raiz")),
        (
            "file_size_limit",
            ("MAX_FILE_SIZE_BYTES", "tamanho máximo", "limite permitido"),
        ),
        ("utf8_read", ("read_text", "utf-8")),
    ),
    required_tools=("list_files", "read_file"),
    forbidden_terms=(
        "arquivo de código-fonte suportado",
        "injeção de código",
    ),
    minimum_citations=2,
    maximum_citations=5,
    maximum_citation_span=20,
    forbid_code_blocks=True,
    require_reference_only_citations=True,
    require_faithful_symbols=True,
)

CHUNKING_CASE = EvalCase(
    name="chunking_strategy",
    question=(
        "Explique como o projeto divide arquivos em chunks, incluindo tamanho, "
        "sobreposição e metadados de localização."
    ),
    required_terms=("chunk_source_file",),
    required_concepts=(
        ("chunk_representation", ("CodeChunk", "cada chunk")),
        ("chunk_size", ("chunk_size", "tamanho do chunk")),
        ("overlap", ("overlap", "sobreposição")),
        (
            "line_metadata",
            ("start_line", "linha inicial", "linhas de início e fim"),
        ),
    ),
    required_tool_groups=(("evidence", ("read_file", "semantic_search")),),
    required_evidence_paths=("src/repo_agent_chat/retrieval/chunking.py",),
    minimum_citations=2,
    maximum_citations=5,
    maximum_citation_span=20,
    forbid_code_blocks=True,
    require_reference_only_citations=True,
    require_faithful_symbols=True,
)

VECTOR_SEARCH_CASE = EvalCase(
    name="embedding_vector_search",
    question=(
        "Explique o fluxo desde a geração dos embeddings dos chunks até a busca "
        "pelos trechos mais similares."
    ),
    required_concepts=(
        ("embedding_component", ("OllamaEmbeddings", "embed_chunks")),
        ("retriever_component", ("RepositoryRetriever", "retriever.py")),
        ("vector_store", ("InMemoryVectorStore", "vector_store.py")),
        ("batching", ("lote", "batch_size")),
        ("cosine_similarity", ("similaridade de cosseno", "cosine_similarity")),
        ("top_k", ("top_k", "mais similares")),
    ),
    required_tool_groups=(("evidence", ("read_file", "semantic_search")),),
    required_evidence_paths=(
        "src/repo_agent_chat/retrieval/embeddings.py",
        "src/repo_agent_chat/retrieval/retriever.py",
        "src/repo_agent_chat/retrieval/vector_store.py",
    ),
    minimum_citations=2,
    maximum_citations=5,
    maximum_citation_span=20,
    forbid_code_blocks=True,
    require_reference_only_citations=True,
    require_faithful_symbols=True,
)

RAG_FLOW_CASE = EvalCase(
    name="rag_flow",
    question=(
        "Como o RagAssistant recupera evidências, monta o contexto e gera a "
        "resposta final?"
    ),
    required_terms=("RagAssistant", "format_retrieved_context"),
    required_concepts=(
        ("retrieval", ("retriever.search", "recupera")),
        ("context", ("context", "contexto")),
        ("streaming", ("yield from", "stream")),
    ),
    required_tool_groups=(("evidence", ("read_file", "semantic_search")),),
    required_evidence_paths=("src/repo_agent_chat/retrieval/rag.py",),
    forbidden_terms=("semantic_search", "read_file"),
    minimum_citations=2,
    maximum_citations=5,
    maximum_citation_span=50,
    forbid_code_blocks=True,
    require_reference_only_citations=True,
    require_faithful_symbols=True,
)

VULNERABILITY_CASE = EvalCase(
    name="vulnerability_analysis",
    question=(
        "Faça uma análise estática de possíveis vulnerabilidades em src/ e explique "
        "as limitações dos achados."
    ),
    required_concepts=(
        ("uncertainty", ("possível", "potencial", "candidato")),
        ("static_analysis", ("análise estática", "heurística")),
        (
            "limitations",
            ("falso positivo", "revisão manual", "não garante", "não pode confirmar"),
        ),
    ),
    forbidden_terms=("completamente seguro", "garante que não existem"),
    required_tools=("analyze_vulnerabilities",),
)

MERMAID_CASE = EvalCase(
    name="mermaid_dependency_diagram",
    question="Gere um diagrama Mermaid das dependências entre arquivos dentro de src/.",
    required_terms=("flowchart LR",),
    required_tools=("generate_mermaid_diagram",),
    require_mermaid_block=True,
)

# Kept for callers that import the symbol.  Fixed cases above document examples
# for this repository, but must never be used to grade an arbitrary repository.
DEFAULT_EVAL_CASES: tuple[EvalCase, ...] = ()
EVAL_EXCLUDED_PATHS = frozenset()


def default_evaluation_cases(repository_root: Path) -> tuple[EvalCase, ...]:
    """Create repository-neutral evals from files that actually exist.

    The former suite asserted implementation names from this project, which made
    ``--eval`` misleading for every other repository.  These checks verify the
    agent's investigation behaviour instead of a particular architecture.
    """

    paths = [
        path.relative_to(repository_root).as_posix()
        for path in discover_source_files(repository_root)
    ]
    if not paths:
        return ()
    target = next(
        (path for path in paths if Path(path).suffix.lower() not in {".md", ".json"}),
        paths[0],
    )
    return (
        EvalCase(
            name="grounded_file_explanation",
            question=(
                f"Explique a responsabilidade observável de `{target}`. "
                "Diferencie fatos do código de inferências."
            ),
            required_tools=("read_file",),
            required_evidence_paths=(target,),
            minimum_citations=1,
            maximum_citations=5,
            maximum_citation_span=20,
            forbid_code_blocks=True,
            require_reference_only_citations=True,
            require_faithful_symbols=True,
        ),
        EvalCase(
            name="repository_vulnerability_analysis",
            question=(
                "Faça uma análise estática de possíveis vulnerabilidades no "
                "repositório inteiro e explique as limitações dos achados."
            ),
            required_concepts=VULNERABILITY_CASE.required_concepts,
            forbidden_terms=VULNERABILITY_CASE.forbidden_terms,
            required_tools=("analyze_vulnerabilities",),
        ),
        EvalCase(
            name="repository_dependency_diagram",
            question="Gere um diagrama Mermaid das dependências entre os arquivos do repositório.",
            required_terms=("flowchart",),
            required_tools=("generate_mermaid_diagram",),
            require_mermaid_block=True,
        ),
    )


def evaluate_answer(
    case: EvalCase,
    answer: str,
    tool_calls: list[str],
    repository_root: Path,
    evidence_paths: set[str] | None = None,
    evidence_excerpts: list[str] | None = None,
    evidence_ranges: set[str] | None = None,
) -> EvalResult:
    """Aplica verificações determinísticas à resposta de um agente."""

    normalized_answer = answer.casefold()
    checks = {
        f"term:{term}": term.casefold() in normalized_answer
        for term in case.required_terms
    }
    checks.update(
        {
            f"concept:{name}": any(
                alternative.casefold() in normalized_answer
                for alternative in alternatives
            )
            for name, alternatives in case.required_concepts
        }
    )
    checks.update({f"tool:{tool}": tool in tool_calls for tool in case.required_tools})
    checks.update(
        {
            f"tool_group:{name}": any(tool in tool_calls for tool in alternatives)
            for name, alternatives in case.required_tool_groups
        }
    )
    checks.update(
        {
            f"forbidden:{term}": term.casefold() not in normalized_answer
            for term in case.forbidden_terms
        }
    )
    citations = extract_citations(answer)
    normalized_evidence = evidence_paths or set()
    checks["citations:count"] = len(citations) >= case.minimum_citations
    if case.maximum_citations is not None:
        checks["citations:not_excessive"] = len(citations) <= case.maximum_citations
    citation_evaluation = CitationEvaluator().evaluate(
        citations, repository_root, evidence_ranges, normalized_evidence
    )
    checks["citations:valid"] = (
        all(check.exists and check.lines_exist and check.inside_repository for check in citation_evaluation.citations)
        if citations
        else case.minimum_citations == 0
    )
    checks.update(
        {
            f"evidence:{path}": path in normalized_evidence
            for path in case.required_evidence_paths
        }
    )
    checks["citations:grounded"] = (
        citation_evaluation.passed
        if citations
        else case.minimum_citations == 0
    )
    if case.maximum_citation_span is not None:
        checks["citations:precise"] = bool(citations) and all(
            citation.end_line - citation.start_line + 1 <= case.maximum_citation_span
            for citation in citations
        )
    if case.forbid_code_blocks:
        checks["format:no_code_blocks"] = "```" not in answer
    if case.require_reference_only_citations:
        checks["citations:reference_only"] = len(
            QUOTED_CITATION_PATTERN.findall(answer)
        ) == len(citations)
    if case.require_faithful_symbols:
        evidence = "\n".join(evidence_excerpts or [])
        symbols = extract_referenced_symbols(answer)
        checks["symbols:faithful"] = all(
            symbol_exists(symbol, evidence) for symbol in symbols
        )
    if case.require_mermaid_block:
        checks["format:mermaid_block"] = re.search(
            r"```mermaid\s+flowchart\s+(?:LR|RL|TD|TB|BT)\b.*?```",
            answer,
            re.DOTALL | re.IGNORECASE,
        ) is not None
    return EvalResult(
        case_name=case.name,
        passed=all(checks.values()),
        checks=checks,
        answer=answer,
        tool_calls=tuple(tool_calls),
        evidence_paths=tuple(sorted(normalized_evidence)),
    )


def run_evaluation_suite(
    agent: ToolAgent,
    repository_root: Path,
    cases: tuple[EvalCase, ...] | None = None,
) -> list[EvalResult]:
    """Executa casos reais contra o agente local."""

    results = []
    selected_cases = cases if cases is not None else default_evaluation_cases(repository_root)
    for case in selected_cases:
        agent.reset_conversation()
        answer = agent.ask(case.question)
        results.append(
            evaluate_answer(
                case,
                answer,
                agent.last_tool_calls,
                repository_root,
                agent.last_evidence_paths,
                agent.last_evidence_excerpts,
                agent.last_evidence_ranges
                if isinstance(agent.last_evidence_ranges, set)
                else None,
            )
        )
    return results


def extract_citations(answer: str) -> list[Citation]:
    """Extrai citações estruturadas de uma resposta."""

    citations = []
    for match in CITATION_PATTERN.finditer(answer):
        start_line = int(match.group("start"))
        end_line = int(match.group("end") or start_line)
        citations.append(Citation(match.group("path"), start_line, end_line))
    return citations


def citation_is_valid(citation: Citation, repository_root: Path) -> bool:
    """Confirma que caminho e linhas citados existem dentro do repositório."""

    check = CitationEvaluator().evaluate([citation], repository_root).citations[0]
    return check.exists and check.lines_exist and check.inside_repository


def print_evaluation_report(results: list[EvalResult]) -> None:
    """Exibe um relatório compacto e útil para diagnosticar regressões."""

    for result in results:
        status = "PASSOU" if result.passed else "FALHOU"
        print(f"\n[{status}] {result.case_name}")
        for check, passed in result.checks.items():
            marker = "✓" if passed else "✗"
            print(f"  {marker} {check}")
        print(f"  tools: {', '.join(result.tool_calls) or 'nenhuma'}")
        print(f"  evidências: {', '.join(result.evidence_paths) or 'nenhuma'}")
        print(f"  resposta: {result.answer}")

    passed = sum(result.passed for result in results)
    print(f"\nResultado: {passed}/{len(results)} casos aprovados.")
