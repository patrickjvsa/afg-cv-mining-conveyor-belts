# Entrega de Patrick: exploración y primeros modelos

El objetivo es clasificar imágenes de `coal` (normal, bloque grande, perno), explorar `iron` y documentar iteraciones reales. Clasificación no requiere cajas. La evaluación usa un benchmark agrupado nuevo; no es una evaluación en otra planta.

## Abrir resultados

Abra `entrega_modelado.ipynb` y ejecute sus celdas desde un kernel nuevo dentro del repositorio. La vista por defecto reconstruye resultados guardados y verifica consistencia; no reentrena. El notebook se genera una vez cerrada la selección y evaluado el test.

Se requieren los artefactos locales en:

- `data/processed/patrick_v1`: inventario, calidad, decisiones, manifiesto y splits.
- `data/results/patrick_v1`: figuras, métricas, predicciones, diario, selección y verificación.
- `data/weights/patrick_v1`: checkpoints de las CNN.

Están excluidos de Git. El notebook ejecutado incorpora salidas para lectura local; una copia nueva del código no incluye automáticamente datos ni pesos. No se han publicado ni subido archivos externos.

## Entorno local

Desde la raíz, use el entorno del proyecto o cree uno aislado:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r notebooks/patrick/requirements.txt
```

`requirements-resolved.txt` registra el entorno local probado. Sus ruedas se eligieron para macOS ARM; en otra plataforma se deben resolver versiones compatibles. En Colab conserve la pareja compatible de torch/torchvision del runtime y añada el resto de dependencias necesarias. No reinstale CUDA automáticamente. El montaje de Drive y la ejecución Colab de este notebook no han sido probados.

Los scripts resuelven la raíz desde su ubicación. El notebook busca la raíz en el directorio actual y sus padres. Para Colab, coloque código/datos en la estructura esperada o adapte las rutas explícitamente; no presuponga que otro notebook comparte variables o archivos temporales.

## Reconstruir análisis y experimentos

Ejecute desde la raíz, en este orden:

```bash
.venv/bin/python notebooks/patrick/study_data.py
.venv/bin/python notebooks/patrick/modeling.py baselines
.venv/bin/python notebooks/patrick/modeling.py head --name M2_raw --variant raw
.venv/bin/python notebooks/patrick/modeling.py head --name M2_masked --variant masked
.venv/bin/python notebooks/patrick/modeling.py finetune --name M3_masked_ft --parent M2_masked --epochs 8
.venv/bin/python notebooks/patrick/modeling.py finetune --name M4_masked_jitter --parent M2_masked --epochs 8 --augment
.venv/bin/python notebooks/patrick/modeling.py head --name M5_masked_320 --variant masked --size 320
.venv/bin/python notebooks/patrick/modeling.py table
```

Estos comandos reconstruyen la secuencia efectivamente investigada. Las decisiones originales se tomaron después de revisar resultados, no mediante búsqueda automática de métricas. El diario explica cada paso. Cambiar de plataforma puede producir diferencias numéricas y decisiones de early stopping distintas.

Las ejecuciones existentes no se sobrescriben: conserve sus artefactos y utilice otra versión/ubicación o nombres nuevos para un estudio nuevo. `study_data.py` reutiliza el inventario tras verificar hashes; si cambia el dataset, requiere una versión nueva. Las cachés de características están ligadas al hash del manifiesto.

Revise las comparaciones, curvas y errores de **validación** antes de cerrar los experimentos. Solo entonces:

```bash
.venv/bin/python notebooks/patrick/modeling.py final
.venv/bin/python notebooks/patrick/build_notebook.py
```

`final` fija la selección por macro-F1 de validación y el umbral de alarma por F2 de validación, evalúa el modelo elegido y M0 en test y escribe un sello. Rechaza volver a ejecutarse sobre ese sello. No borre el sello para continuar ajustando sobre los mismos resultados de test.

`build_notebook.py` reconstruye tablas/figuras y el notebook, sin entrenar ni modificar predicciones o selección. También exporta incidencias descriptivas y 288 candidatos de iron para anotación futura, todos explícitamente sin revisar y sin etiqueta verificada. Sobrescribe el notebook generado: si lo edita manualmente, guarde antes su versión o adapte el generador.

## Verificación

```bash
.venv/bin/python notebooks/patrick/verify_study.py --raw --inference
```

Verifica rutas y hashes de originales, ausencia de cruces, exclusiones, soporte, fuentes de cada ejecución, probabilidades, métricas reconstruidas, criterio de selección/umbral, checkpoint y notebook ejecutado. `--without-notebook` permite comprobar los artefactos antes de ejecutar el notebook. La revisión de figuras y contenido visual complementa estas comprobaciones: ninguna certifica semántica de etiquetas ni independencia real de los eventos.

## Archivos y decisiones

- `study_data.py`: lectura de originales, métricas de calidad, agrupación conservadora y EDA.
- `modeling.py`: referencias, extracción de características, cabezas, fine-tuning y test final.
- `build_notebook.py`: exposición en español a partir de evidencia guardada.
- `verify_study.py`: auditoría de invariantes y reconstrucción.
- `PROGRESO.md`: continuidad del trabajo.

Por ejecución, `metadata.json` identifica manifiesto, parámetros, semillas, versiones y código. `source_snapshot/` preserva los scripts efectivos, aunque el worktree siga cambiando. Para las primeras ejecuciones se aclaró el dispositivo efectivo mediante una nota de procedencia: sklearn y la optimización de las cabezas usan CPU; la extracción convolucional y el fine-tuning usan MPS cuando está disponible. Los metadatos distinguen acelerador disponible y dispositivos efectivos.

La máscara de texto tiene tres regiones fijas, independientes de la clase. Oculta también parte del fondo y podría eliminar extremos de objetos: la comparación mide ese tratamiento completo, no el efecto causal puro de leer el texto.

Se excluyeron conflictos exactos y copias consistentes. Los grupos horarios/visuales son proxies: faltan fechas/videos para certificar eventos. Las etiquetas de carpetas restantes no están adjudicadas por un especialista; ejemplos discutibles no se relabelan a partir de la predicción del modelo.

Iron, el entrenamiento conjunto, los sintéticos y las cajas quedan como extensiones condicionadas. Primero se necesitan etiquetas verificadas por fotograma y una definición compartida de riesgo. No se reportarán resultados supervisados de iron que no se hayan ejecutado.

## Ejecutar el notebook en un kernel nuevo desde terminal

Después de construirlo:

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python - <<'PYCODE'
from pathlib import Path
import nbformat
from nbclient import NotebookClient
p = Path("notebooks/patrick/entrega_modelado.ipynb")
nb = nbformat.read(p, as_version=4)
NotebookClient(nb, timeout=300, kernel_name="python3",
               resources={"metadata": {"path": str(Path.cwd())}}).execute()
nbformat.write(nb, p)
PYCODE
```

Esta ejecución reproduce la exposición desde artefactos y un checkpoint, sin entrenar ni volver a seleccionar por test. Tras ejecutarla, use `verify_study.py --raw --inference` para auditar el conjunto.
