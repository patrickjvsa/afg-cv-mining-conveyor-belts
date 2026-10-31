# Progreso · etapa temprana terminada y verificada

## Entrega
- Notebook principal: entrega_modelado.ipynb, 44 celdas, 17 de código ejecutadas en kernel nuevo local.
- README.md: dependencias, comandos de reproducción, artefactos y límites de Colab.
- AUDITORIA_ENTREGA.md: contraste del objetivo con evidencia y verificaciones.
- Scripts: study_data.py, modeling.py, build_notebook.py, verify_study.py.

## Evidencia ejecutada
- Inventario SHA-256 de 24.011 archivos; raw revalidado completo al cierre sin cambios.
- Calidad: coal 4.800 imágenes, iron muestra temporal 288. Inspección visual por muestreo.
- Limpieza: 135 archivos en 65 contenidos con etiquetas contradictorias y 165 copias consistentes excluidos; 4.500 candidatos únicos.
- Benchmark nuevo, agrupación por similitud confirmada y hora como proxies: train 3.211/205 grupos, val 644/36, test 645/34. Cero cruces de hashes/grupos/pares confirmados.
- Siete ejecuciones reales. Val macro-F1: M0 0,16744; M1 0,76799; M2_raw 0,90418; M2_masked 0,88859; M3_masked_ft 0,92052; M4_masked_jitter 0,91737; M5_masked_320 0,90092.
- Diario de hipótesis y decisiones en data/results/patrick_v1/experiment_journal.json; fuentes de cada entrenamiento preservadas en source_snapshot.
- Selección por validación: M3_masked_ft, época 3. Test sellado; no reabrir selección para ajustar por test.
- Test macro-F1 0,93185; balanced accuracy 0,93171. Umbral de alarma fijado en validación 0,019967: TP432/FN1/FP24/TN188.
- Intervalo exploratorio bootstrap de grupos para macro-F1 [0,86797; 0,95806].
- Se inspeccionaron errores de validación/test y la única omisión de alarma; no se corrigieron etiquetas a partir de predicciones.

## Verificaciones finales
- verify_study.py --raw --inference: passed; métricas y probabilidades reconstruidas, checkpoint consistente, fuentes, invariantes y notebook verificados.
- Notebook ejecutado desde sesión limpia local. Figuras revisadas visualmente.
- pip check, compilación de scripts y git diff --check aprobados.
- No se verificó Colab ni autorización de Drive. No se publicaron datos/código ni se hicieron commits o push.

## Límites y siguiente paso
Grupos temporales/visuales son proxies; faltan fechas/videos para certificar eventos. Etiquetas restantes heredadas sin adjudicación experta; una semilla por configuración. Máscara de texto también oculta parte de escena. Métricas internas no certifican una planta distinta ni costos operacionales.

Iron no tiene etiquetas verificadas: se prepararon 288 candidatos, todos unreviewed y sin etiqueta. Próximo hito propuesto: adjudicar etiquetas dudosas y anotar piloto de iron, antes de comparar especialistas y modelo conjunto. Sintéticos y bounding boxes son extensiones posteriores justificadas, no tareas obligatorias pendientes de esta entrega.
