from pathlib import Path
from unittest.mock import Mock

from repo_agent_chat.evals import EvalCase, evaluate_answer, run_evaluation_suite


def test_evaluate_answer_aprova_todos_os_criterios(tmp_path: Path) -> None:
    (tmp_path / "repository.py").write_text("\n" * 70, encoding="utf-8")
    case = EvalCase(
        name="safe_read",
        question="Onde ocorre a leitura segura?",
        required_terms=("read_source_file", "is_relative_to"),
        required_concepts=(("size_limit", ("MAX_FILE_SIZE_BYTES", "tamanho máximo")),),
        required_tools=("read_file",),
        minimum_citations=2,
    )
    answer = (
        "`read_source_file` usa `is_relative_to` em `repository.py:57-64` "
        "e aplica um tamanho máximo em `repository.py:69`."
    )

    result = evaluate_answer(
        case,
        answer,
        ["list_files", "read_file"],
        tmp_path,
        {"repository.py"},
    )

    assert result.passed is True
    assert all(result.checks.values())


def test_evaluate_answer_mostra_criterios_ausentes(tmp_path: Path) -> None:
    case = EvalCase(
        name="safe_read",
        question="Onde ocorre a leitura segura?",
        required_terms=("read_source_file",),
        required_tools=("read_file",),
        minimum_citations=1,
    )

    result = evaluate_answer(case, "Não encontrei.", ["list_files"], tmp_path)

    assert result.passed is False
    assert result.checks == {
        "term:read_source_file": False,
        "tool:read_file": False,
        "citations:count": False,
        "citations:valid": False,
        "citations:grounded": False,
    }


def test_evaluate_answer_reprova_caminho_ou_linha_inexistente(
    tmp_path: Path,
) -> None:
    (tmp_path / "repository.py").write_text("linha 1\nlinha 2", encoding="utf-8")
    case = EvalCase(name="citations", question="Onde?", minimum_citations=2)
    answer = "Veja `repository.py:20` e `../segredo.py:1`."

    result = evaluate_answer(case, answer, [], tmp_path)

    assert result.checks["citations:count"] is True
    assert result.checks["citations:valid"] is False
    assert result.checks["citations:grounded"] is False
    assert result.passed is False


def test_evaluate_answer_reprova_afirmacao_proibida(tmp_path: Path) -> None:
    case = EvalCase(
        name="groundedness",
        question="Como funciona?",
        forbidden_terms=("injeção de código",),
    )

    result = evaluate_answer(
        case,
        "O método impede injeção de código.",
        [],
        tmp_path,
    )

    assert result.checks["forbidden:injeção de código"] is False
    assert result.passed is False


def test_evaluate_answer_aceita_alternativa_para_um_conceito(
    tmp_path: Path,
) -> None:
    case = EvalCase(
        name="concept",
        question="Como funciona?",
        required_concepts=(("path_inside_root", ("is_relative_to", "dentro da raiz")),),
    )

    result = evaluate_answer(
        case,
        "A função confirma que o arquivo está dentro da raiz permitida.",
        [],
        tmp_path,
    )

    assert result.checks["concept:path_inside_root"] is True
    assert result.passed is True


def test_evaluate_answer_aceita_limite_permitido_como_conceito(
    tmp_path: Path,
) -> None:
    case = EvalCase(
        name="size",
        question="Como funciona?",
        required_concepts=(
            (
                "file_size_limit",
                ("MAX_FILE_SIZE_BYTES", "tamanho máximo", "limite permitido"),
            ),
        ),
    )

    result = evaluate_answer(
        case,
        "Verifica se o tamanho não excede o limite permitido.",
        [],
        tmp_path,
    )

    assert result.checks["concept:file_size_limit"] is True


def test_evaluate_answer_reprova_codigo_copiado_e_citacao_ampla(
    tmp_path: Path,
) -> None:
    (tmp_path / "repository.py").write_text("\n" * 100, encoding="utf-8")
    case = EvalCase(
        name="format",
        question="Explique.",
        minimum_citations=1,
        maximum_citation_span=20,
        forbid_code_blocks=True,
    )
    answer = "Veja `repository.py:1-50`.\n```python\ndef read(): ...\n```"

    result = evaluate_answer(
        case,
        answer,
        [],
        tmp_path,
        {"repository.py"},
    )

    assert result.checks["citations:precise"] is False
    assert result.checks["format:no_code_blocks"] is False
    assert result.passed is False


def test_evaluate_answer_reprova_excesso_de_citacoes(tmp_path: Path) -> None:
    (tmp_path / "repository.py").write_text("\n" * 10, encoding="utf-8")
    case = EvalCase(
        name="concise_evidence",
        question="Explique.",
        minimum_citations=1,
        maximum_citations=2,
    )
    answer = "`repository.py:1`, `repository.py:2` e `repository.py:3`"

    result = evaluate_answer(
        case,
        answer,
        [],
        tmp_path,
        {"repository.py"},
    )

    assert result.checks["citations:count"] is True
    assert result.checks["citations:not_excessive"] is False
    assert result.passed is False


def test_evaluate_answer_reprova_codigo_dentro_da_citacao(
    tmp_path: Path,
) -> None:
    (tmp_path / "repository.py").write_text("linha", encoding="utf-8")
    case = EvalCase(
        name="citation_format",
        question="Explique.",
        minimum_citations=1,
        require_reference_only_citations=True,
    )

    result = evaluate_answer(
        case,
        "`repository.py:1: conteúdo copiado`",
        [],
        tmp_path,
        {"repository.py"},
    )

    assert result.checks["citations:reference_only"] is False
    assert result.passed is False


def test_evaluate_answer_reprova_simbolo_ausente_das_evidencias(
    tmp_path: Path,
) -> None:
    case = EvalCase(
        name="faithfulness",
        question="Como funciona?",
        require_faithful_symbols=True,
    )

    result = evaluate_answer(
        case,
        "A função `RagAssistant().generate_response(query)` gera a resposta.",
        [],
        tmp_path,
        evidence_excerpts=["rag.py:20: class RagAssistant:"],
    )

    assert result.checks["symbols:faithful"] is False
    assert result.passed is False


def test_evaluate_answer_valida_bloco_mermaid(tmp_path: Path) -> None:
    case = EvalCase(
        name="diagram",
        question="Gere o diagrama.",
        require_mermaid_block=True,
    )

    valid = evaluate_answer(
        case,
        "```mermaid\nflowchart LR\n  a --> b\n```",
        [],
        tmp_path,
    )
    invalid = evaluate_answer(case, "flowchart LR\na --> b", [], tmp_path)

    assert valid.passed is True
    assert invalid.checks["format:mermaid_block"] is False


def test_run_evaluation_suite_usa_resposta_e_tracing_do_agent(
    tmp_path: Path,
) -> None:
    (tmp_path / "repository.py").write_text("\n" * 10, encoding="utf-8")
    agent = Mock()
    agent.ask.return_value = "Implementado em `repository.py:10`."
    agent.last_tool_calls = ["read_file"]
    agent.last_evidence_paths = {"repository.py"}
    case = EvalCase(
        name="case",
        question="Onde?",
        required_tools=("read_file",),
        minimum_citations=1,
    )

    results = run_evaluation_suite(agent, tmp_path, (case,))

    assert results[0].passed is True
    agent.reset_conversation.assert_called_once_with()
    agent.ask.assert_called_once_with("Onde?")
