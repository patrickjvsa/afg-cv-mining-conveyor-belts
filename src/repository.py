"""Operaciones Git conservadoras, también utilizables desde Colab."""
from pathlib import Path
import os
import subprocess


def git(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True,
                            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if result.returncode:
        # Evitar imprimir remotos o diagnósticos que pudieran contener credenciales.
        raise RuntimeError("Git no pudo completar la operación. Revise acceso, rama y estado local; consulte docs/guia_equipo.md.")
    return result.stdout.strip()


def repository_state(repo):
    return {"commit": git(repo, "rev-parse", "HEAD"),
            "branch": git(repo, "branch", "--show-current"),
            "dirty": bool(git(repo, "status", "--porcelain"))}


def ensure_repository(url, branch, destination, update=False):
    destination = Path(destination).expanduser().resolve()
    if not branch or branch.startswith("-"):
        raise ValueError("Indique una rama válida.")
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        git(destination.parent, "clone", "--branch", branch, "--", url, str(destination))
    else:
        if not (destination / ".git").exists():
            raise ValueError("El directorio existe pero no es un repositorio. Elija otro.")
        if git(destination, "remote", "get-url", "origin").rstrip("/") != url.rstrip("/"):
            raise ValueError("La copia local corresponde a otro remoto. No se modificó.")
        state = repository_state(destination)
        if state["branch"] != branch:
            raise ValueError("La rama local difiere de la solicitada. Guarde sus cambios y cambie de rama manualmente.")
        if update:
            if state["dirty"]:
                raise ValueError("Hay cambios locales. Guárdelos mediante un commit antes de actualizar.")
            git(destination, "fetch", "origin", branch)
            git(destination, "merge", "--ff-only", "FETCH_HEAD")
    return repository_state(destination)
