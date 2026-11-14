"""Serve only captured PDF sources inside the project's corpus directories."""
from pathlib import Path


def resolve_source(project_root, source_file):
    root = Path(project_root).resolve()
    allowed = [root / "data", root / "artifacts"]
    source = Path(source_file)
    if source.suffix.lower() != ".pdf":
        return None
    candidate = (source if source.is_absolute() else root / source).resolve()
    if candidate.is_file() and any(candidate.is_relative_to(p.resolve()) for p in allowed):
        return candidate
    # Some registries retain only the basename. Ambiguous names must not open
    # an unrelated source; captured text remains available in the debugger.
    if source.name != source_file:
        return None
    matches = [p.resolve() for base in allowed if base.exists() for p in base.rglob(source.name)
               if p.is_file() and p.resolve().is_relative_to(base.resolve())]
    return matches[0] if len(matches) == 1 else None
