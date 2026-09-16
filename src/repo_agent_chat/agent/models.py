from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequestedTool:
    """Representação interna uniforme de uma chamada de tool."""

    id: str
    name: str
    arguments: str
