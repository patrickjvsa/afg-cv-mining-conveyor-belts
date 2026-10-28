"""Configuración centralizada y resolución segura de rutas."""
from pathlib import Path


def under(root, relative):
    root = Path(root).expanduser().resolve()
    if not isinstance(relative, str) or not relative.strip() or Path(relative).is_absolute():
        raise ValueError("La ruta debe ser relativa y no estar vacía.")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError("La ruta debe permanecer dentro de su directorio raíz.")
    return candidate


def resolve_paths(drive_root, manifest):
    """Resolver una raíz de almacenamiento: data/ local o carpeta montada de Drive."""
    root = Path(drive_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Raíz de almacenamiento no encontrada: {root}. Revise la ruta local o el montaje y permisos de Drive.")
    dataset = manifest["dataset"]
    if not isinstance(dataset.get("version"), str) or not dataset["version"].strip():
        raise ValueError("Complete dataset.version con un identificador estable.")
    paths = {key: under(root, manifest[key + "_dir"]) for key in ("raw", "processed", "results", "weights")}
    roots = list(paths.values())
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(roots) for b in roots[i + 1:]):
        raise ValueError("Las carpetas de datos, resultados y pesos deben estar separadas.")
    paths["dataset"] = under(paths["raw"], dataset["relative_path"])
    paths["root"] = root
    return paths
