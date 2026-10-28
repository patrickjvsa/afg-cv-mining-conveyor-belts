"""Inspección general, de solo lectura, sin asumir formato de etiquetas."""
from collections import Counter
from pathlib import Path
import importlib
import os
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


def validate_dataset(path, sample_size=20, validator=None):
    if not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError("sample_size debe ser un entero positivo.")
    root = Path(path)
    report = {"path": str(root), "files": 0, "bytes": 0, "extensions": {},
              "top_level": [], "images_checked": [], "errors": [], "warnings": []}
    if not root.exists():
        report["errors"].append("Dataset inexistente. Revise relative_path y permisos de Drive.")
        report["ok"] = False
        return report
    files = []
    if root.is_file():
        files = [root]
    else:
        try:
            report["top_level"] = sorted(p.name for p in root.iterdir())[:50]
        except OSError as exc:
            report["errors"].append(f"No se puede listar el directorio: {exc}")
        for folder, dirs, names in os.walk(root, onerror=lambda e: report["errors"].append(str(e))):
            for name in dirs[:]:
                if (Path(folder) / name).is_symlink():
                    report["errors"].append(f"Enlace simbólico no admitido: {name}")
                    dirs.remove(name)
            files.extend(Path(folder) / name for name in sorted(names))
    extensions = Counter()
    for file in sorted(files):
        report["files"] += 1
        extensions[file.suffix.lower() or "(sin extensión)"] += 1
        try:
            if file.is_symlink():
                raise ValueError("Enlace simbólico no admitido")
            size = file.stat().st_size
            report["bytes"] += size
            if size == 0:
                raise ValueError("Archivo vacío")
            with file.open("rb") as stream:
                stream.read(1)
            if file.suffix.lower() in IMAGE_EXTENSIONS and len(report["images_checked"]) < sample_size:
                with Image.open(file) as img:
                    item = {"file": str(file), "size": list(img.size), "format": img.format}
                    report["images_checked"].append(item)
                    img.verify()
                with Image.open(file) as img:
                    img.load()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            report["errors"].append(f"{file}: {exc}")
    report["extensions"] = dict(extensions)
    if not files:
        report["errors"].append("No se encontraron archivos. Compruebe la carpeta elegida.")
    if not report["images_checked"]:
        report["warnings"].append("No se inspeccionaron imágenes: puede tratarse de un archivo comprimido u otro formato.")
    report["warnings"].append("La muestra no verifica todas las imágenes ni la correspondencia de etiquetas o particiones.")
    if validator:
        module, function = validator.split(":", 1)
        # Contrato: función(path) devuelve una lista de errores; no modifica datos.
        errors = getattr(importlib.import_module(module), function)(root)
        if not isinstance(errors, list) or any(not isinstance(e, str) for e in errors):
            raise ValueError("El validador específico debe devolver list[str].")
        report["errors"].extend(errors)
    report["ok"] = not report["errors"]
    return report
