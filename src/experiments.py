"""Persistencia explícita: un registro JSONL por ejecución evita un CSV compartido."""
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
import json
import platform
import shutil
import subprocess
import uuid
from .config import under
from .preparation import sha256
from .repository import repository_state


def write_json_new(path, value):
    # Serializar antes de crear el archivo: un fallo no deja un JSON truncado.
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")


def start_run(paths, repo, manifest, config, data_path):
    state = repository_state(repo)
    if state["dirty"]:
        raise ValueError("El repositorio tiene cambios sin commit. Confírmelos para registrar código reproducible.")
    splits = {}
    for name, relative in manifest["dataset"].get("splits", {}).items():
        path = under(data_path, relative)
        splits[name] = {"path": relative, "sha256": sha256(path)}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
    output = paths["results"] / run_id
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                "git": state, "dataset": manifest, "splits": splits, "config": config,
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]},
                "data_path": str(data_path), "weights_path": str(paths["weights"] / run_id)}
    write_json_new(output / "metadata.json", metadata)
    freeze = subprocess.run([__import__("sys").executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True)
    # La instalación editable propia incluye rutas locales; el commit ya la identifica.
    lines = [line for line in freeze.stdout.splitlines() if not line.startswith(("-e ", "#")) and " @ file:" not in line]
    (output / "requirements-resolved.txt").write_text("\n".join(lines) + "\n")
    return output


def save_artifact(source, destination_dir, name=None):
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Artefacto no encontrado: {source}")
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = under(destination_dir, name or source.name)
    if target.parent != destination_dir.resolve():
        raise ValueError("El nombre del artefacto no debe incluir subdirectorios.")
    # Apertura exclusiva: un nombre repetido se rechaza, sin sobrescribir.
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    if sha256(source) != sha256(target):
        raise OSError("La verificación del artefacto falló. Conserve el original y revise Drive.")
    return {"path": str(target), "sha256": sha256(target)}


def finish_run(output, metrics, artifacts=None):
    output = Path(output)
    metadata = json.loads((output / "metadata.json").read_text())
    record = {"run_id": metadata["run_id"], "git": metadata["git"],
              "dataset_version": metadata["dataset"]["dataset"]["version"],
              "config": metadata["config"], "metrics": metrics, "artifacts": artifacts or [],
              "metadata": str(output / "metadata.json")}
    text = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
    # Una línea por ejecución; agregar offline con collect_runs evita escrituras concurrentes.
    with (output / "record.jsonl").open("x", encoding="utf-8") as stream:
        stream.write(text)
    return record


def collect_runs(results_root):
    return [json.loads(p.read_text()) for p in sorted(Path(results_root).glob("*/record.jsonl"))]
