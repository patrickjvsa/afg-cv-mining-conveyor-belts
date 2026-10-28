"""Copias verificadas y extracción ZIP sin alterar los originales."""
import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")
    if path.is_symlink() or (path.is_dir() and any(p.is_symlink() for p in path.rglob("*"))):
        raise ValueError("No se admiten enlaces simbólicos en las copias.")
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    return {p.name if path.is_file() else p.relative_to(path).as_posix():
            {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files}


def prepare_dataset(source, destination, extract_zip=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Origen y destino deben ser independientes.")
    source_inventory = inventory(source)
    identity = {"source": str(source), "inventory": source_inventory, "extract_zip": extract_zip}
    marker = destination / ".preparation.json"
    if destination.exists():
        if not marker.is_file():
            raise FileExistsError("Destino existente sin manifiesto; elija otro directorio.")
        saved = json.loads(marker.read_text())
        actual = inventory(destination / "data")
        if saved["identity"] != identity or saved["prepared"] != actual:
            raise FileExistsError("Origen o copia cambiaron; elija un destino nuevo. No se sobrescribió nada.")
        return destination / "data"
    required = sum(item["bytes"] for item in source_inventory.values())
    if extract_zip:
        if not source.is_file() or not zipfile.is_zipfile(source):
            raise ValueError("Se solicitó extracción pero el archivo no es un ZIP válido.")
        with zipfile.ZipFile(source) as archive:
            seen = set()
            required = 0
            for item in archive.infolist():
                relative = PurePosixPath(item.filename)
                mode = item.external_attr >> 16
                if (relative.is_absolute() or ".." in relative.parts or "\\" in item.filename
                        or ":" in item.filename or not relative.parts or stat.S_ISLNK(mode)):
                    raise ValueError(f"Ruta ZIP no segura: {item.filename}")
                normalized = relative.as_posix().casefold()
                if normalized in seen:
                    raise ValueError("ZIP con rutas duplicadas o ambiguas.")
                seen.add(normalized)
                required += item.file_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < required + max(required // 10, 16 * 1024**2):
        raise OSError(f"Espacio insuficiente: se requieren {required} bytes más margen.")
    # Solo se elimina nuestra carpeta temporal en caso de fallo.
    staging = Path(tempfile.mkdtemp(prefix=".preparing-", dir=destination.parent))
    try:
        target = staging / "data"
        if extract_zip:
            target.mkdir()
            with zipfile.ZipFile(source) as archive:
                archive.extractall(target)
        elif source.is_dir():
            shutil.copytree(source, target)
        else:
            target.mkdir()
            shutil.copy2(source, target / source.name)
        if inventory(source) != source_inventory:
            raise RuntimeError("El origen cambió durante la copia. Repita cuando esté estable.")
        prepared = inventory(target)
        if not extract_zip and prepared != source_inventory:
            raise RuntimeError("La copia no coincide con el origen.")
        (staging / ".preparation.json").write_text(json.dumps({"identity": identity, "prepared": prepared}, indent=2))
        if destination.exists():
            raise FileExistsError("El destino apareció durante la copia; no se reemplazará.")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination / "data"
