"""Validação determinística de citações produzidas pelo agente."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CitationCheck:
    path: str
    start_line: int
    end_line: int
    exists: bool
    lines_exist: bool
    retrieved: bool
    inside_repository: bool

    @property
    def valid(self) -> bool:
        return self.exists and self.lines_exist and self.retrieved and self.inside_repository


@dataclass(frozen=True, slots=True)
class CitationEvaluation:
    citations: tuple[CitationCheck, ...]

    @property
    def passed(self) -> bool:
        return all(citation.valid for citation in self.citations)


class CitationEvaluator:
    """Confirma arquivo, faixa e recuperação no repositório analisado."""

    def evaluate(
        self,
        citations: list[object],
        repository_root: Path,
        evidence_ranges: set[str] | None = None,
        evidence_paths: set[str] | None = None,
    ) -> CitationEvaluation:
        root = repository_root.resolve()
        ranges = _parse_ranges(evidence_ranges or set())
        paths = evidence_paths or set()
        checks = []
        for citation in citations:
            path = str(citation.path)
            start = int(citation.start_line)
            end = int(citation.end_line)
            target = (root / path).resolve()
            inside = target.is_relative_to(root)
            exists = inside and target.is_file()
            lines_exist = False
            if exists and start >= 1 and end >= start:
                try:
                    lines_exist = end <= len(target.read_text(encoding="utf-8").splitlines())
                except (OSError, UnicodeDecodeError):
                    lines_exist = False
            retrieved = path in paths or any(
                candidate_path == path and candidate_start <= start and end <= candidate_end
                for candidate_path, candidate_start, candidate_end in ranges
            )
            checks.append(CitationCheck(path, start, end, exists, lines_exist, retrieved, inside))
        return CitationEvaluation(tuple(checks))


def _parse_ranges(ranges: set[str]) -> list[tuple[str, int, int]]:
    parsed = []
    for value in ranges:
        try:
            path, span = value.rsplit(":", 1)
            start, end = span.split("-", 1)
            parsed.append((path, int(start), int(end)))
        except (ValueError, TypeError):
            continue
    return parsed
