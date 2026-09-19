import re
from dataclasses import asdict, dataclass

from repo_agent_chat.repository import SourceFile


@dataclass(frozen=True, slots=True)
class VulnerabilityFinding:
    """Possível vulnerabilidade sem incluir conteúdo potencialmente secreto."""

    rule: str
    severity: str
    path: str
    line: int
    message: str


SECURITY_RULES = (
    (
        "hardcoded-secret",
        "high",
        re.compile(
            r"(?i)\b(?:password|passwd|api[_-]?key|secret|access[_-]?token)\b"
            r"\s*[:=]\s*['\"][^'\"]{6,}['\"]"
        ),
        "Possível segredo definido diretamente no código.",
    ),
    (
        "dynamic-code-execution",
        "high",
        re.compile(r"\b(?:eval|exec)\s*\("),
        "Execução dinâmica de código exige validação rigorosa da entrada.",
    ),
    (
        "subprocess-shell",
        "high",
        re.compile(r"\b(?:subprocess\.)?(?:run|call|Popen)\s*\([^\n]*shell\s*=\s*True"),
        "Comando de sistema executado com shell habilitado.",
    ),
    (
        "unsafe-deserialization",
        "high",
        re.compile(r"\b(?:pickle\.loads?|yaml\.unsafe_load)\s*\("),
        "Desserialização insegura pode executar conteúdo não confiável.",
    ),
    (
        "weak-hash",
        "medium",
        re.compile(r"\b(?:hashlib\.)?(?:md5|sha1)\s*\("),
        "Hash criptográfico fraco; confirme se não é usado para segurança.",
    ),
    (
        "sql-query-concatenation",
        "medium",
        re.compile(
            r"(?i)\b(?:select|insert|update|delete)\b[^\n]*(?:\+|f[\"']|\.format\s*\()"
        ),
        "Consulta SQL construída dinamicamente; use parâmetros preparados.",
    ),
    (
        "insecure-tls-verification",
        "high",
        re.compile(r"(?i)\b(?:verify\s*=\s*False|ssl[_-]?verify\s*[:=]\s*false)\b"),
        "Verificação de certificado TLS desabilitada.",
    ),
    (
        "prompt-injection-content",
        "medium",
        re.compile(
            r"(?i)\b(?:ignore\s+(?:previous|prior)\s+instructions?|"
            r"disregard\s+(?:the\s+)?system|reveal\s+(?:the\s+)?prompt)\b"
        ),
        "Texto com padrão de prompt injection; trate-o como dado não confiável.",
    ),
)


def analyze_source_files(
    source_files: list[SourceFile],
    max_findings: int = 50,
) -> list[dict[str, object]]:
    """Executa regras estáticas conservadoras e retorna achados serializáveis."""

    findings: list[VulnerabilityFinding] = []
    for source_file in source_files:
        for line_number, line in enumerate(source_file.content.splitlines(), start=1):
            for rule, severity, pattern, message in SECURITY_RULES:
                if pattern.search(line):
                    findings.append(
                        VulnerabilityFinding(
                            rule=rule,
                            severity=severity,
                            path=source_file.path,
                            line=line_number,
                            message=message,
                        )
                    )
                    if len(findings) == max_findings:
                        return [asdict(finding) for finding in findings]
    return [asdict(finding) for finding in findings]
