"""Read-only research evidence. Never runs models or pools different runs."""
import json
from pathlib import Path
import re


def experiment_directory(project_root, source):
    if source not in ("active", "legacy"):
        raise ValueError("source must be active or legacy")
    base = Path(project_root) / "artifacts" / "experiments"
    pointer = base / "active_run.json"
    if source == "legacy" or not pointer.is_file():
        return base
    try:
        run_id = json.loads(pointer.read_text(encoding="utf-8-sig"))["run_id"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Active run pointer could not be read") from exc
    if not isinstance(run_id, str) or not re.fullmatch(r"run[A-Za-z0-9_-]{1,127}", run_id):
        raise RuntimeError("Invalid active run identifier")
    directory = base / "runs" / run_id
    if not directory.resolve().is_relative_to((base / "runs").resolve()):
        raise RuntimeError("Active run escapes the results directory")
    if not directory.is_dir():
        raise RuntimeError("Active run results are unavailable")
    return directory


def research_summary(project_root, source="active"):
    directory = experiment_directory(project_root, source)
    warnings, studies = [], []
    for path in sorted(directory.glob("exp*/summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=lambda _: None)
            if not isinstance(data, dict) or not isinstance(data.get("headline"), dict):
                raise ValueError("Missing headline")
            n = data.get("n_examples")
            studies.append({"key": path.parent.name, "title": str(data.get("title", path.parent.name)),
                "mode": str(data.get("mode", "unknown")),
                "n_examples": n if type(n) is int and n >= 0 else None,
                "headline": data["headline"],
                "caveats": [str(v) for key in ("caveats", "interpretation_notes") for v in (data.get(key) if isinstance(data.get(key), list) else [])],
                "arm_counts": {str(k): v.get("n") for k, v in (data.get("by_arm") or {}).items() if isinstance(v, dict) and type(v.get("n")) is int} if isinstance(data.get("by_arm"), dict) else {},
                "source_path": path.relative_to(project_root).as_posix()})
        except (OSError, ValueError, TypeError):
            warnings.append(f"{path.parent.name}: summary unavailable or invalid; excluded from charts.")
    run = None
    manifest = directory / "run_manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8-sig"))
            run = {key: data.get(key) for key in ("run_id", "created_at", "mode", "seed", "limit")} if isinstance(data, dict) else None
        except (OSError, ValueError):
            warnings.append("Run metadata unavailable.")
    resolved_source = "legacy" if directory.name == "experiments" else "active"
    return {"source": resolved_source, "directory": directory.relative_to(project_root).as_posix(),
            "run": run, "studies": studies, "warnings": warnings,
            "note": "Saved results, not a fresh evaluation. Counts are study records, not unique independent questions. Mode labels describe the saved runner configuration; injected and replayed studies are not live human validation."}
