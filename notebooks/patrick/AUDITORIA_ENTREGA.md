# Auditoría de la entrega temprana

Este registro contrasta el objetivo con evidencia local. La validación semántica por especialista, la independencia real de eventos y la ejecución en Colab no se presentan como verificadas.

| Requisito del objetivo | Evidencia | Resultado |
|---|---|---|
| Leer plan, guía y reglas del repositorio | PROGRESO.md; documentación revisada antes de implementar | Realizado; no se encontraron AGENTS.md aplicables |
| Explorar ambos datasets y muestras visuales | inventory.csv, quality.csv, figuras coal/iron y notebook | 24.011 archivos; calidad de coal completo e iron 288 muestras |
| Descripción, visualizaciones, resúmenes y relaciones | dataset_counts.csv, descriptive_summary.csv, quality_correlations.csv, overlay_development.csv | Conteos, dimensiones, distribución, IQR, brillo/nitidez y asociación de overlay |
| Limpiar contradicciones y duplicados sin inventar etiquetas | annotation_decisions.csv y quality_issues.csv | Excluidos 135 archivos en 65 contenidos conflictivos y 165 copias consistentes; 4.500 candidatos únicos |
| Preservar originales | inventory.csv y verification.json | Coincidencia completa de rutas, bytes y SHA-256 de los 24.011 originales |
| Particiones antes de entrenar y sin cruces conocidos | protocol.json, coal_manifest.csv, splits/*.csv, near_duplicate_edges.csv | 3.211/644/645 imágenes; 205/36/34 grupos proxy; cero cruces de hashes/grupos/pares confirmados |
| Declarar límites de eventos y etiquetas | Notebook, README.md y protocol.json | Horas sin fechas y similitud como proxies; etiquetas restantes heredadas, no adjudicadas |
| Referencias y modelo preentrenado ejecutados | runs/*/metrics.json, config/metadata, histories, checkpoints | Siete ejecuciones terminadas: M0, M1, dos M2, M3, M4 y M5 |
| Iterar por resultados, no solo producir código | experiment_journal.json, curvas y galerías de errores | Comparación de texto, fine-tuning, sobreajuste, jitter y resolución; resultados negativos conservados |
| Tratamiento coherente del texto | mask_overlay, control visual, M2_raw/M2_masked | Regiones fijas por imagen, sin usar etiqueta; pérdida de escena y límites documentados |
| Seleccionar con validación y reservar test | selection.json, final_test.json y predicciones | M3 época 3; test únicamente del elegido y M0; sello y orden temporal comprobados |
| Métricas pertinentes y alarmas | Reportes por clase, matrices, curva PR, umbral y análisis de prevalencia | Macro-F1 test 0,93185; TP432/FN1/FP24/TN188 al umbral fijado con validación |
| Analizar errores concretos | Galerías de validación/test, alarma y diario | Pernos finos/oscuros, otra vista y etiquetas discutibles; sin relabelar por predicción |
| Pesos, trazabilidad y reconstrucción | metadata.json, source_snapshot, requirements-resolved.txt, predicciones, checkpoint | Hashes de código/datos/pesos; dispositivos efectivos y cambios locales documentados |
| Notebook comprensible y ejecución limpia | entrega_modelado.ipynb y verify_study.py | 44 celdas, 17 de código ejecutadas sin errores en kernel nuevo; reconstrucción e inferencia verificadas |
| Decidir siguiente paso sin fingir resultados de iron | iron_annotation_candidates.csv, guía y cierre del notebook | 288 candidatos sin revisar y sin etiqueta; combinación/síntesis/cajas aplazadas justificadamente |
| Autonomía y límites | Trabajo en notebooks/patrick y artefactos bajo data | Instalación/entrenamiento locales; sin publicación, push, servicios pagos ni cambios en raw |

Las rutas de artefactos se resuelven desde `data/processed/patrick_v1`, `data/results/patrick_v1` y `data/weights/patrick_v1`. El notebook y el README explican cómo abrirlos y reconstruirlos.

El resultado es una **evaluación interna temprana**. Se usó una semilla por configuración. El intervalo bootstrap por grupos no mide variabilidad entre entrenamientos ni generalización a una planta distinta. El umbral F2 es exploratorio y la tasa de falsas alarmas por imagen normal no es una tasa por hora.

## Cierre verificado

`verify_study.py --raw --inference` terminó con `status: passed`, incluyendo el notebook final. Se inspeccionaron visualmente galerías de ambos datasets, comparaciones, curvas, máscaras/augmentation, matriz de test y errores de alarma. La selección precede temporalmente a las predicciones de test y a su sello. `pip check`, compilación de los scripts y `git diff --check` no detectaron problemas. No se crearon commits; el commit base sigue siendo 116e6f5.
