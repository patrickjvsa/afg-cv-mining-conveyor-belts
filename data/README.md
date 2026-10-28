# Datos locales

Coloque sus datasets en `data/raw/`, en una carpeta por dataset o versión. Solo este README se versiona: los datos y resultados locales quedan excluidos de Git. No modifique los originales durante un experimento.

En Colab, complete la sección 1 de `00_setup.ipynb`. El notebook obtiene los datos de Drive y entrega `DATA_PATH`. Si activa copia local, esta queda en `data/prepared/data/` del clon; los resultados importantes siguen guardándose en Drive.

Para usar las utilidades en su computador, instale `requirements.txt` y ejecute desde la raíz del repositorio, completando los dos valores vacíos:

```python
from pathlib import Path
import sys
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from src.config import resolve_paths
from src.validation import validate_dataset

MANIFEST = {
    "raw_dir": "raw", "processed_dir": "processed",
    "results_dir": "results", "weights_dir": "weights",
    "dataset": {
        "relative_path": "",  # Nombre o ruta real dentro de data/raw/.
        "version": "",  # Versión acordada por el equipo.
        "splits": {},
    },
}
PATHS = resolve_paths(ROOT / "data", MANIFEST)
DATA_PATH = PATHS["dataset"]
report = validate_dataset(DATA_PATH)
print(report)
if not report["ok"]:
    raise ValueError("Revise los errores antes de cargar datos.")
```

El futuro cargador recibe `DATA_PATH` tanto localmente como en Colab. Las rutas locales de procesados, resultados y pesos quedan bajo `data/`; las utilidades crean las salidas cuando se necesitan. No hay archivos de configuración adicionales.

Vea la [guía del equipo](../docs/guia_equipo.md) para el recorrido completo.
