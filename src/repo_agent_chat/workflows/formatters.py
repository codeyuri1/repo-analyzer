import json


def format_security_analysis(tool_result: str) -> str:
    """Formata achados estáticos sem delegar conclusões de segurança à LLM."""

    result = _result_object(tool_result)
    findings = result.get("findings", [])
    if not isinstance(findings, list):
        findings = []

    lines = ["## Possíveis achados da análise estática", ""]
    if not findings:
        lines.append("Nenhum possível achado foi identificado pelas regras disponíveis.")
    else:
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            rule = finding.get("rule", "regra-desconhecida")
            severity = finding.get("severity", "unknown")
            path = finding.get("path")
            line = finding.get("line")
            message = finding.get("message", "Possível achado a revisar.")
            reference = f"`{path}:{line}`" if isinstance(path, str) and isinstance(line, int) else ""
            lines.append(f"- **{severity} · {rule}** {reference}: {message}")

    lines.extend(
        [
            "",
            "## Limitações",
            "",
            (
                "Esta é uma análise estática heurística: pode produzir falso positivo ou "
                "falso negativo, não confirma exploração e não garante ausência de "
                "vulnerabilidades. Os achados exigem revisão manual e testes adicionais."
            ),
        ]
    )
    return "\n".join(lines)


def format_mermaid_diagram(tool_result: str) -> str:
    """Reproduz exatamente o diagrama devolvido pela tool."""

    result = _result_object(tool_result)
    diagram = result.get("diagram")
    if not isinstance(diagram, str) or not diagram.startswith("flowchart "):
        return "Não foi possível gerar um diagrama Mermaid válido."
    nodes = result.get("nodes", 0)
    edges = result.get("edges", 0)
    truncated = result.get("truncated") is True
    paths = result.get("paths", [])
    edge_evidence = result.get("edge_evidence", [])
    note = " O grafo foi limitado ao máximo configurado de nós." if truncated else ""
    sources = _format_node_sources(paths) + _format_edge_evidence(edge_evidence)
    return (
        f"Diagrama de dependências com {nodes} nós e {edges} arestas.{note}\n\n"
        f"```mermaid\n{diagram}\n```{sources}"
    )


def _format_edge_evidence(edge_evidence: object) -> str:
    if not isinstance(edge_evidence, list) or not edge_evidence:
        return ""
    references = []
    for item in edge_evidence:
        if not isinstance(item, dict):
            continue
        source, target, line = item.get("source"), item.get("target"), item.get("line")
        if isinstance(source, str) and isinstance(target, str) and isinstance(line, int):
            references.append(f"- `{source}` → `{target}`: `{source}:{line}`")
    return "\n\nEvidências das relações:\n" + "\n".join(references) if references else ""


def _format_node_sources(paths: object) -> str:
    if not isinstance(paths, list) or not paths:
        return ""
    references = [f"`{path}:1`" for path in paths if isinstance(path, str)]
    return "\n\nFontes dos nós: " + ", ".join(references) if references else ""


def _result_object(tool_result: str) -> dict[str, object]:
    try:
        payload = json.loads(tool_result)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return {}
    result = payload.get("result")
    return result if isinstance(result, dict) else {}
