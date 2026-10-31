# Revisión de datos y plan de notebooks: exploración y primeros modelos

Fecha: 31 de octubre de 2026. Fuente de la revisión: copia local de `data/raw/`.

**Recomendación:** comenzar con clasificación de imágenes en `coal`, después de auditar etiquetas y particiones. En paralelo, convertir una muestra temporal de `iron` en un conjunto revisado que permita estudiar transferencia entre dominios. Combinar ambos y generar datos sintéticos son hipótesis experimentales posteriores, condicionadas a etiquetas compatibles y evaluación real independiente.

Para una primera experiencia con computer vision, leer la [guía práctica de decisiones](guia_practica_computer_vision.md): explica clasificación frente a detección, bounding boxes, anotación asistida, elección de modelo, tratamiento del texto y alcance recomendado de la entrega. **La entrega próxima se centra en coal; anotar iron, combinar datos y generar sintéticos son ampliaciones, no condiciones para presentar los primeros modelos.**

Este documento planifica el desarrollo de las semanas 3 y 4. No se entrenaron modelos, no se corrigieron etiquetas y no se modificaron los originales. Los experimentos descritos son propuestas, no resultados.

## 1. Alcance y evidencia de la revisión

- Se inventariaron los **24.011 archivos**: rutas, tamaños, extensiones, dimensiones, modo de color y SHA-256 del contenido completo.
- Se leyeron cabeceras de todas las imágenes. No se detectaron errores de apertura en esa lectura.
- Se decodificaron **216 imágenes por muestreo**, con semilla 42: 12 por cada combinación clase/partición de `coal` y 12 por carpeta de `iron`. No falló ninguna de esas decodificaciones. Esto no certifica la integridad completa de los demás archivos.
- Se inspeccionaron visualmente dos galerías de **72 posiciones** en total: cuatro muestras por cada una de las seis carpetas de `coal` y de las doce de `iron`. En `iron`, la selección cubre inicio, dos puntos intermedios y final según el índice numérico del nombre. En `coal`, cubre posiciones del orden de nombres; ese orden no se presume cronológico.
- También se observaron ejemplos de duplicados con clases contradictorias. No se realizó inspección visual exhaustiva.
- Se revisaron `README.md`, `data/README.md`, `docs/guia_equipo.md`, el setup y las utilidades existentes. `notebooks/patrick/entrega_modelado.ipynb` es un archivo vacío de 0 bytes; no contiene un experimento previo.

Evidencia local, excluida de Git por estar bajo `data/`:

- [Galería de coal](../data/processed/auditoria_inicial/coal_contact.jpg).
- [Galería de iron](../data/processed/auditoria_inicial/iron_contact.jpg).
- [Ejemplos de conflictos de etiquetas](../data/processed/auditoria_inicial/conflictos_etiquetas.jpg).
- [Resumen de auditoría](../data/processed/auditoria_inicial/summary.json), [inventario por archivo](../data/processed/auditoria_inicial/inventory.json) y [duplicados con rutas y hashes](../data/processed/auditoria_inicial/duplicados.csv).

Estas rutas corresponden a esta revisión local; otra copia del repositorio necesitará regenerar los artefactos. El plan conserva aquí los hallazgos esenciales.

## 2. Qué datos tenemos realmente

| Característica | `coal` | `iron` |
|---|---|---|
| Archivos | 4.800 JPG | 19.211 JPEG |
| Tamaño de archivos, sumado | 346,0 MiB | 1.389,3 MiB |
| Dimensiones | Todos 416 × 416 | Todos 1.280 × 720 |
| Modo almacenado | Todos RGB; apariencia principalmente monocromática | Todos RGB; imágenes en color |
| Organización | `train/test` → tres clases | `image-files` → doce carpetas de secuencias |
| Supervisión local | Clase de imagen inferida de la carpeta | Sin etiquetas por imagen verificadas |
| Particiones | 3.900 train y 900 test; sin validación | Sin train/validación/test |
| Cajas, máscaras u otras anotaciones | No hay archivos de anotación | No hay archivos de anotación |
| Archivos auxiliares dentro de raw | Ninguno | Ninguno |

En `raw` solo hay imágenes: no se encontraron CSV, JSON, XML, TXT, videos ni documentos de procedencia. La autoría, licencia, FPS, cámara y condiciones de captura deben registrarse desde la fuente original cuando se recupere. La apariencia visual no basta para certificar que un conjunto es de planta o de laboratorio.

### 2.1. Coal: clasificación disponible, pero etiquetas y evaluación requieren limpieza

| Clase original | Train | Test | Total |
|---|---:|---:|---:|
| `Normal Coal Flow` | 1.300 | 300 | 1.600 |
| `Oversized Coal` | 1.300 | 300 | 1.600 |
| `Rock Bolts` | 1.300 | 300 | 1.600 |
| **Total** | **3.900** | **900** | **4.800** |

El balance es exacto en los archivos originales; debe recalcularse después de limpiar. No representa necesariamente la frecuencia de incidentes en operación.

Las muestras muestran una cinta en perspectiva, mineral, bloques grandes y objetos alargados. Algunas tienen fecha, hora y texto de alarma sobreimpreso; otras no. Cambian brillo, encuadre y visibilidad del objeto. Los pernos pueden ser delgados y poco contrastados. El texto, los bordes o la iluminación podrían funcionar como atajos para un clasificador; es una hipótesis que hay que medir.

**Problemas comprobados mediante identidad SHA-256:**

| Hallazgo | Cantidad |
|---|---:|
| Contenidos únicos | 4.565 |
| Grupos con archivos idénticos | 208 |
| Copias adicionales sobre los contenidos únicos | 235 |
| Archivos participantes en esos grupos | 443 |
| Grupos idénticos con clases contradictorias | 65 |
| Archivos en grupos de clases contradictorias | 135 |
| Grupos idénticos que cruzan train/test | 1 |

De los 65 conflictos, 64 enfrentan `Normal Coal Flow` con `Oversized Coal` y uno enfrenta `Normal Coal Flow` con `Rock Bolts`. Hay 64 grupos conflictivos íntegramente en train y uno entre train y test.

Ejemplo del cruce entre particiones: `coal/test/Oversized Coal/大块测试集_051541_0641.jpg` es idéntico a `coal/train/Normal Coal Flow/正常煤流训练集_051541_0641.jpg`. Aquí coinciden fuga de información y contradicción de clase.

Consecuencias: no aceptar el test original como limpio; no resolver conflictos por mayoría ni conservar la etiqueta de una carpeta automáticamente. Cuarentenar los grupos contradictorios hasta revisión. Si se excluyen los 65 contenidos conflictivos y se conserva una copia de cada contenido restante, quedan **4.500 imágenes únicas candidatas**, todavía sujetas a revisión semántica y de similitud visual.

### 2.2. Iron: más imágenes, pero principalmente secuencias correlacionadas

Prefijo común de carpetas: `iron/image-files/boardcamera_iron_ore`.

| Carpeta, sufijo del prefijo | Imágenes | Índices numéricos mínimos–máximos | Observación |
|---|---:|---|---|
| Sin sufijo | 5.740 | 5.726–27.663 | Dos tramos separados por un salto de índice |
| `_2` | 1.512 | 1.806–3.317 | Índices consecutivos |
| `_3` | 1.848 | 3.640–5.487 | Índices consecutivos |
| `_4` | 1.134 | 2.702–3.835 | Índices consecutivos |
| `_5` | 1.008 | 12.152–13.159 | Archivos incluyen `fastandfurious`; desenfoque en muestras |
| `_6` | 2.265 | 4.309–6.573 | Índices consecutivos |
| `_7_wood` | 1.211 | 1.964–3.265 | Un salto; madera visible en algunas muestras |
| `_7_wood_2` | 1.153 | 2.106–3.258 | Madera no visible en las cuatro muestras iniciales |
| `_7_wood_3` | 1.299 | 1.868–3.166 | Madera visible en una muestra intermedia |
| `_canga` | 790 | 1.486–2.275 | Roca rojiza y elemento de manipulación visibles en muestras |
| `_canga2` | 1.091 | 2.601–3.691 | No asumir una clase por su nombre |
| `_canga3` | 160 | 1.666–1.875 | Un salto; roca y elemento de manipulación visibles |
| **Total** | **19.211** | | |

Los índices se obtienen del número antes del primer guion, no del orden lexicográfico. No conocemos su equivalencia en segundos. En las carpetas sin sufijo, `_7_wood` y `_canga3`, los saltos entre índices son de 16.199, 92 y 51 respectivamente; son discontinuidades, no prueba por sí mismas de archivos perdidos.

Las muestras muestran una cinta vista desde arriba, mineral y también tramos de cinta vacía. Los bordes, una unión de la cinta y elementos que parecen intervenir en la colocación de objetos pueden generar atajos visuales. Hay diferencias marcadas respecto de `coal`: orientación, perspectiva, color, iluminación y escala aparente.

**No se encontraron duplicados exactos por bytes en iron.** Esto no descarta imágenes casi iguales: los nombres consecutivos y las muestras sugieren extracción de video. El número de imágenes no equivale al número de eventos, objetos físicos ni escenarios independientes.

`wood` describe una secuencia donde puede aparecer madera, no una etiqueta positiva de todos sus fotogramas. Tampoco `canga` establece por sí solo si una roca es mineral normal, sobretamaño u objeto de riesgo. Hace falta un criterio visual y operacional explícito.

## 3. Definición inicial del problema

Proponer dos niveles, conservando siempre las etiquetas originales:

1. **Tarea inmediata:** clasificación supervisada en `coal` con tres clases: flujo normal, bloque sobredimensionado y perno. Responde qué condición se observa en una imagen, sin localizar el objeto.
2. **Salida operacional derivada:** normal frente a condición de riesgo, agregando bloque/perno como alarma. Esto permite comenzar a estudiar sensibilidad y falsas alarmas, pero no presupone que ambas anomalías tengan la misma causa o costo.

Si el objetivo final incluye localizar objetos sobre la cinta, un detector supervisado convencional necesita cajas de referencia; una segmentación supervisada necesita máscaras. Las carpetas de clase no sustituyen esas anotaciones. La clasificación inicial permite estudiar presencia de riesgo, pero no prueba capacidad de localización. El alcance final debe distinguir explícitamente alerta por imagen y localización de objetos.

Para `iron`, definir una tabla de anotación separada: `normal_mineral`, `cinta_vacia`, `madera`, `otro_objeto`, `sobretamano_confirmado` y `incierto`, según lo que realmente sea identificable. Son categorías de revisión propuestas, no clases cuya existencia ya esté demostrada. Permitir coexistencia de objetos mediante indicadores por objeto; la clasificación exclusiva en `coal` debe examinarse por separado. Un bloque grande aparente no demuestra sobretamaño físico sin escala o umbral acordado.

No asignar etiquetas de perno a madera ni asumir que toda roca rojiza es un cuerpo extraño. Decidir explícitamente si la cinta vacía pertenece al estado operativo normal o queda fuera del alcance; conservar su indicador aunque después se agregue a normal.

## 4. Secuencia de desarrollo mediante notebooks

Los nombres siguientes son propuestos. En esta fase se entrega el plan; no se crean notebooks vacíos ni se reemplaza el notebook personal existente.

Cada notebook deberá ejecutarse desde una sesión nueva, tener una pregunta al inicio y cerrar con hallazgos, decisiones y pendientes. Reutilizará el setup existente para acceso, sin presuponer variables de otra sesión. Para comparar ambos conjuntos se resolverán explícitamente `COAL_PATH` e `IRON_PATH` desde la raíz de datos; hoy `DATA_PATH` apunta a un dataset elegido. Las funciones repetidas podrán pasar a `src/` cuando exista una necesidad concreta.

### 01 · `01_inventario_calidad.ipynb` — qué existe y qué es confiable

**Entradas:** ambos datasets originales. **Cómputo:** CPU.

- Crear un manifiesto por imagen con ID estable, ruta relativa, origen, clase original si existe, partición original, carpeta/secuencia candidata, índice candidato, dimensiones, bytes, modo de color, hash y estado de lectura.
- Separar campos observados de campos inferidos; una secuencia candidata no equivale todavía a un video independiente confirmado.
- Reproducir conteos, duplicados, conflictos y cruces. Guardar los grupos, no solo una cifra agregada.
- Añadir similitud perceptual para proponer vecinos casi idénticos y revisarlos en galerías pequeñas. Verificar candidatos; ni un hash perceptual ni un embedding prueban por sí solos que sean duplicados.
- Revisar las 65 contradicciones por contenido único, con estados `excluir`, `revisar` o `corregir_con_evidencia`; registrar razón, etiqueta anterior, propuesta y responsable. Priorizar casos ambiguos para segunda revisión.
- Reutilizar duplicados consistentes como una única observación en el manifiesto limpio. Mantener raw intacto.
- Recuperar y documentar fuente, versión y condiciones de captura si están disponibles. No bloquear el inventario por esa ausencia, pero declarar sus límites.

**Salidas:** `inventory`, tabla de incidencias, registro de decisiones y manifiesto candidato limpio bajo `data/processed/`. Tabla antes/después por clase y partición. Gráficos de barras de conteos y de incidencias.

**Criterio de cierre:** toda imagen candidata a entrenamiento tiene trazabilidad; ningún conflicto se acepta silenciosamente. Los pendientes quedan explícitamente excluidos o señalados.

### 02 · `02_exploracion_visual_descriptiva.ipynb` — patrones, calidad y diferencias de dominio

**Entradas:** inventario y manifiesto candidato, sin entrenar aún. **Cómputo:** CPU; embeddings opcionales según recursos.

- Galerías estratificadas por clase, split, carpeta y condiciones visuales; añadir casos extremos de brillo/desenfoque y vecinos casi iguales. Usar muestras reproducibles y tamaños manejables.
- En `iron`, recorrer una cuadrícula temporal por las doce carpetas y refinar únicamente intervalos con cambio o posible objeto. No revisar manualmente los 19.211 fotogramas. No fijar FPS ni llamar segundos a los índices.
- Medir brillo, contraste, saturación, fracción oscura/clara y una medida consistente de nitidez. Empezar con una muestra estratificada; ampliar automáticamente si aporta a una decisión. Las métricas de calidad son proxies y dependen de resolución, textura y recorte.
- Describir dimensiones, peso y calidad con media, mediana, desviación estándar, cuartiles, IQR y rango; para clases/cámaras/texto, frecuencias y proporciones. Señalar cuántas observaciones sustentan cada resumen.
- Crear histogramas por origen y clase, boxplots de calidad, scatterplot brillo–nitidez y tablas de contingencia clase × texto visible y clase × encuadre candidato. Explorar correlaciones entre métricas numéricas; no correlacionar códigos arbitrarios de clases como si fueran cantidades.
- Comparar train/test de `coal` y secuencias de `iron`. Revisar si texto, fondo o condiciones de captura se asocian con etiquetas. La asociación no demuestra causalidad.
- Proponer regiones de interés de cinta y zonas con overlays. Verificar que un recorte o máscara no borre pernos, bloques ni objetos en los bordes. Registrar cobertura y limitaciones.
- Si se usan embeddings/PCA para visualizar, ajustar la proyección con datos de desarrollo. Usar los gráficos como diagnóstico, sin interpretar separación visual como desempeño probado.

**Salidas:** ficha de cada dataset, galerías, tablas resumen, gráficos y lista priorizada de hipótesis. Guardar también IDs de las muestras.

**Criterio de cierre:** justificar con evidencia qué variaciones preservar, qué problemas limpiar y qué transformaciones podrían distorsionar el objeto. No eliminar automáticamente imágenes oscuras o borrosas: son condiciones relevantes del problema.

### 03 · `03_etiquetas_particiones_preparacion.ipynb` — pasar de imágenes a un experimento válido

**Entradas:** revisión anterior y decisiones de etiquetas. **Cómputo:** CPU y revisión humana acotada.

Para la próxima entrega, la parte obligatoria de este notebook es preparar coal. El piloto y las particiones de iron se realizan solo como ampliación si queda tiempo; su ausencia no bloquea los baselines de coal.

- Consolidar la taxonomía de `coal` y el criterio de alarma. Confirmar reglas para coexistencia de objetos y casos dudosos.
- Crear un piloto revisado de `iron`: **objetivo inicial de 240–360 fotogramas**, distribuidos por las doce carpetas, cubriendo fondo, presencia/ausencia de objetos, tramos vacíos y transiciones. Anotar imágenes concretas, no carpetas enteras; ampliar solo si faltan eventos o clases.
- Distinguir la muestra dirigida que busca anomalías de una muestra temporal sistemática. No usar la primera para estimar prevalencia natural. Registrar `selection_reason`, visibilidad y confianza; dejar desconocido lo no verificable.
- Identificar grupos por evento/toma/objeto físico cuando sea posible. Revisar similitud también entre carpetas: una misma pieza podría reaparecer en distintas tomas.
- **Coal:** conservar `original_split`, pero construir una evaluación limpia. Si el test original puede sanearse y separarse por eventos, mantenerlo y retirar de desarrollo todo grupo que lo contamine. Si no, definir un benchmark nuevo agrupado y declarar que no es comparable directamente con el split original.
- Usar como referencia una división de grupos cercana a 70/15/15 para train/validación/test, ajustada a grupos y soporte de clases. Nunca forzar porcentajes repartiendo fotogramas del mismo evento. Si se conserva el test original saneado, obtener validación solo del train saneado.
- **Iron:** separar por toma/evento, inicialmente manteniendo cada carpeta completa en un split salvo evidencia que obligue a agrupar varias. No mezclar fotogramas vecinos entre particiones. Con solo doce carpetas, verificar que haya soporte real por clase; si falta, limitar la tarea o recopilar más eventos. No prometer una evaluación robusta de madera con tres carpetas.
- Si no se pueden reconstruir eventos de `coal`, usar agrupación conservadora por evidencia visual/temporal y declarar que la independencia no está garantizada. Una hora en el nombre sin fecha no identifica un evento global.
- Congelar los manifiestos de evaluación antes de modelar. Aplicar muestreo temporal y augmentations únicamente al train; no construir un conjunto común y dividir después.
- Mantener RGB para aprovechar pesos preentrenados, normalización documentada y escala coherente. Comparar resolución 224 con una mayor solo si los objetos delgados pierden visibilidad. En `iron`, conservar relación de aspecto con padding o ROI verificada; no aplastar 16:9 a cuadrado.
- Ensayar transformaciones moderadas de iluminación, contraste y desenfoque en train. Revisar primero una galería antes/después. Evitar giros grandes y recortes aleatorios que saquen el objeto del campo de visión manteniendo su etiqueta positiva.

**Salidas:** diccionario de etiquetas, piloto anotado de `iron`, log de exclusiones, `train/val/test` con hashes, informe de soporte por grupos/clases y especificación de preprocesamiento. Los datos inciertos no entran como negativos.

**Criterio de cierre:** cero hashes compartidos entre splits, cero grupos conocidos compartidos, clases evaluables y test fijado. Sin ello, el entrenamiento solo puede etiquetarse como prueba técnica, no como evidencia de generalización. La separación por grupos está respaldada por la [documentación de StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html); su estratificación no garantiza proporciones perfectas.

### 04 · `04_baselines_clasificacion.ipynb` — primeros modelos y aprendizaje documentado

**Entrada principal:** `coal` limpio con splits congelados. **Cómputo:** CPU para referencias; GPU de Colab para CNN.

| Experimento | Modelo | Pregunta y justificación |
|---|---|---|
| M0 | `DummyClassifier` | ¿Cuál es el piso de referencia con el balance que quedó tras limpiar? |
| M1 | Regresión logística sobre brillo, contraste, bordes/textura simples | ¿Cuánto se explica por señales visuales simples o sesgos? Escalado ajustado solo en train |
| M2 | ResNet-18 preentrenada, backbone congelado y cabeza de tres clases | ¿Las características transferidas permiten separar las condiciones con poco entrenamiento? |
| M3 | Misma ResNet-18, ajuste del último bloque y cabeza | ¿La adaptación al dominio mejora M2 sin aumentar el sobreajuste? |

Usar una misma arquitectura para aislar congelación frente a ajuste. La [guía oficial de transferencia de PyTorch](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) describe ambas estrategias. Los [pesos de ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) aportan normalización y una referencia de tamaño de entrada; revisar su recorte central antes de usarlo en cintas con objetos periféricos. No entrenar desde cero ni comparar un catálogo amplio de modelos en esta etapa. **M0 y M2 son el mínimo recomendado; M3 es la ampliación prioritaria y M1 es un diagnóstico opcional.**

Protocolo inicial propuesto, ajustable solo con validación:

- Prueba técnica con un batch e imágenes transformadas; comprobar correspondencia imagen–etiqueta y salida de tres clases.
- Semilla inicial 42. M2: hasta 5 épocas de cabeza. M3: hasta 10 épocas, learning rate menor para backbone, early stopping con paciencia 3 sobre macro-F1 de validación. Registrar optimizador y tasas exactas al implementar.
- Batch inicial 32, reducible a 16 según memoria. Sin búsqueda extensa de hiperparámetros. Recalcular balance antes de decidir pesos de clase; no usar SMOTE sobre píxeles.
- Comparar los experimentos ejecutados con el mismo conjunto, preprocesamiento y validación. Después, una ablación focalizada del mejor candidato con/sin texto sobreimpreso mediante una máscara o ROI previamente verificada. Cada variante se entrena y valida con su propio tratamiento consistente; no inferir dependencia del texto solo por taparlo después de entrenar. Si ninguna opción preserva los objetos, documentar la limitación antes de recortar.
- Ajustar criterio de alarma con validación. Reportar precision, recall y F1 por clase, macro-F1, balanced accuracy y matriz de confusión. Para alarma agregada, reportar curva precisión–recall y recall frente a tasa de falsas alarmas por imagen normal. No fijar un umbral operacional sin conocer costos.
- Accuracy queda como métrica secundaria. Un conjunto equilibrado artificialmente no permite inferir directamente precisión de alarmas con prevalencia real.
- Guardar curvas train/validación, predicciones por imagen y una galería de falsos positivos y falsos negativos con confianza. Mapas de activación pueden apoyar el diagnóstico, no demostrar una localización correcta.
- Seleccionar modelo y configuración solo con validación. Repetir la configuración final con tres semillas si el tiempo lo permite; evaluar el protocolo ya fijado en test y reportar variabilidad sin seleccionar por su resultado. Si hay suficientes eventos, intervalos de confianza por remuestreo de grupos, no por fotogramas como si fueran independientes.
- Registrar tiempo, hardware y latencia con tamaño de entrada y batch explícitos. No declarar FPS operacional ni alarmas por hora sin video, FPS y medición de extremo a extremo.

**Salidas:** tabla de experimentos ejecutados, configuración, curvas, métricas, predicciones, pesos y análisis de errores. Dejar `no ejecutado` donde corresponda; nunca completar resultados esperados como si fueran medidos.

**Criterio de cierre:** M0 y al menos una CNN ejecutados, comparación honesta, dificultades observadas y siguiente decisión sustentada. Un modelo que no supera M0 o que parece usar overlays como atajo también produce un aprendizaje válido si queda documentado. El alcance de la entrega no depende de completar M1, M3 ni el trabajo de iron.

### 05 · `05_transferencia_y_sinteticos.ipynb` — experimento posterior, condicionado

Este notebook es una extensión: **no es requisito para cerrar los primeros pasos de modelado**. Depende del piloto de `iron`, etiquetas compatibles y baselines estables.

Su primer objetivo es comparar especialistas por dominio frente a un modelo conjunto real; la síntesis se agrega después como una variante separada. La [sección 7 de la guía práctica](guia_practica_computer_vision.md#7-tiene-sentido-un-modelo-que-sirva-para-ambos-datasets) define qué significa servir para ambos, cómo comprobarlo y qué afirmaciones no permiten los datos actuales.

1. Evaluar transferencia de coal a iron solo sobre categorías homologables. La alarma binaria podría ser compatible si ambos dominios tienen normales y riesgos confirmados. Madera sería una anomalía no vista en coal: evaluar esa condición por separado, sin presentarla como reconocimiento aprendido de madera.
2. Comparar, sobre el mismo test real de iron: entrenamiento solo con coal, solo con train etiquetado de iron y entrenamiento conjunto real. Igualar el presupuesto de etiquetas del dominio destino y reportar resultados por origen y clase. No permitir que los 19.211 fotogramas dominen por cantidad: muestrear por evento y equilibrar contribuciones de dominio si corresponde.
3. Antes de sintetizar, construir una pequeña biblioteca de objetos con máscaras revisadas y fondos de train verificados sin objetos de riesgo. Los JPG completos no son recortes de objetos. Priorizar copy-paste dentro de iron; pasar un perno de coal a una vista cenital de iron puede ser geométricamente inverosímil y se ensayaría después.
4. Pilotear 50–100 composiciones de train, con objeto de procedencia conocida, escala plausible, perspectiva, iluminación y oclusión controladas. Incluir controles de fondos sin inserción para detectar aprendizaje de artefactos. Revisar visualmente el piloto antes de escalar.
5. Cada composición hereda el split y la trazabilidad de **todos** sus padres: `source_object_id`, `source_background_id`, grupo, máscara, transformación y semilla. Ningún objeto, fondo ni fotograma vecino reservado para evaluación se usa como padre de train.
6. Comparar real, real + augmentations convencionales y real + augmentations + copy-paste, con una proporción sintética inicial acotada, por ejemplo 25% del batch. Mantener presupuesto de entrenamiento comparable y el mismo test exclusivamente real. Seleccionar proporción usando validación real.
7. Escalar solo si mejora la validación real sin deteriorar clases o dominios críticos; confirmar luego en test reservado. Si solo mejora sobre sintéticos, descartar esa variante.

El [estudio original de Copy-Paste](https://arxiv.org/abs/2012.07177) respalda investigar la técnica en segmentación; no demuestra que funcionará con estas cintas ni elimina la necesidad de anotaciones. En esta etapa no se necesita generación con modelos de difusión: primero comprobar la utilidad de composiciones trazables.

**Alternativa si hay pocas etiquetas:** estudiar anomalías en embeddings usando exclusivamente imágenes normales confirmadas para entrenar y datos etiquetados aparte para validar el umbral. No asumir que todo iron es normal. Dejar esta vía como alternativa acotada, sin abrir simultáneamente varios pipelines complejos.

### 06 · `06_resultados_iniciales_y_decisiones.ipynb` — evidencia para la pauta

Notebook breve que carga artefactos existentes y no reentrena:

- Problema, alcance de clasificación y relación con futura detección.
- Descripción real de datos, muestra de cada origen y conteos antes/después de limpiar.
- Visualizaciones y medidas resumen seleccionadas por lo que enseñan.
- Decisiones de etiquetas, exclusiones y particiones con justificación.
- Tabla de modelos considerados, ejecutados y aplazados; razones y resultados disponibles.
- Dificultades comprobadas: etiquetas contradictorias, duplicados, incertidumbre de eventos, objetos delgados, overlays y diferencias de dominio. Distinguirlas de dificultades hipotéticas o de entrenamiento que aún no ocurren.
- Aprendizajes, errores representativos y una próxima decisión concreta. Explicar qué puede concluirse y qué requiere más datos o anotación.

## 5. Cómo decidir si combinar, sintetizar o continuar separados

| Ruta | Viabilidad actual | Condición para avanzar |
|---|---|---|
| Clasificación en coal | Alta tras limpieza | Resolver/excluir conflictos y fijar evaluación |
| Clasificación supervisada en iron | Parcial | Anotar presencia real por imagen y garantizar eventos evaluables |
| Unir las carpetas como clases comunes | No justificada | Antes hay que definir semántica y comprobar etiquetas |
| Modelo conjunto con alarma compartida | Hipótesis razonable | Homologación explícita y métricas por dominio |
| Copy-paste dentro de iron | Piloto posible después | Máscaras, fondos y separación de padres |
| Copy-paste entre coal e iron | Más incierta | Demostrar plausibilidad geométrica y ganancia real |
| Detección con cajas | Requiere nueva supervisión | Cajas revisadas de objetos y negativos reales |

Para pasar a detección, seleccionar un piloto de unos 200–400 fotogramas diversos y poco redundantes, según eventos disponibles; anotar todos los objetos de las categorías elegidas y revisar una fracción por otra persona. La cifra es un presupuesto piloto, no garantía de suficiencia. Como siguiente ruta concreta se propone un [YOLO11n preentrenado](https://docs.ultralytics.com/models/yolo11/) ajustado a nuestras clases y anotaciones, por su variante pequeña y flujo de entrenamiento documentado. Faster R-CNN, siguiendo el [tutorial de TorchVision](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html), queda como alternativa si se prefiere continuar íntegramente en ese ecosistema; no es necesario entrenar ambos. Usar métricas de detección solo cuando existan cajas de referencia.

## 6. Orden de trabajo y entregables verificables

| Orden | Trabajo | Evidencia terminada |
|---|---|---|
| Semana 3 · A | Notebook 01 | Inventario, incidencias, duplicados y decisiones de limpieza |
| Semana 3 · B | Notebook 02 | Fichas, galerías, resúmenes y relaciones con hipótesis priorizadas |
| Puente a semana 4 | Notebook 03 | Taxonomía y manifiestos congelados de coal; piloto de iron opcional |
| Semana 4 · A | Notebook 04 | Baselines, primer entrenamiento, métricas y análisis de errores |
| Semana 4 · B | Notebook 06 | Relato reproducible de decisiones, dificultades y aprendizajes |
| Después, si se cumplen condiciones | Notebook 05 / piloto de detección | Evidencia de transferencia, síntesis o localización |

El mínimo para responder bien a la pauta es **01–04 y 06, con preparación/modelado centrados en coal y M0 + M2 ejecutados**. La integración de datasets no debe retrasar los primeros resultados válidos de coal. El piloto de iron puede avanzar junto con la exploración sin convertir todos sus fotogramas en anotaciones manuales. Para priorizar bajo presión de tiempo: limpieza/splits → CNN inicial y errores → tratamiento del texto → fine-tuning → piloto de iron. Mezcla, síntesis y detector quedan después.

Estructura propuesta de artefactos bajo las carpetas excluidas de Git:

```text
data/processed/<version>/
  inventory.csv
  quality_issues.csv
  annotation_decisions.csv
  iron_pilot_annotations.csv
  splits/{train,val,test}.csv
data/results/<run_id>/
  config.json
  metrics.json
  predictions.csv
  figures/
data/weights/<run_id>/
```

Conservar rutas relativas, versión de datos/splits, semilla, clases, versiones de paquetes y commit del código. Reutilizar `src/experiments.py` si su contrato encaja; hoy exige commit limpio y configuración de rutas que el setup no genera por completo. Resolver esa integración al implementar, sin confundir infraestructura disponible con flujo ya conectado.

El entorno actual declara únicamente Pillow como dependencia del proyecto. Para desarrollar se añadirán las dependencias realmente utilizadas de análisis y modelado, documentando las versiones probadas, sin reinstalar automáticamente PyTorch/CUDA de Colab. Los resultados persistentes deberán guardarse de manera explícita: el setup actual no los guarda ni comparte estado entre sesiones.

## 7. Decisiones recomendadas al iniciar el desarrollo

1. Priorizar calidad de etiquetas y evaluación sobre la elección entre arquitecturas.
2. Mantener clasificación de coal como ruta principal para la semana 4 y anotar iron por muestreo temporal como ruta de extensión.
3. Mantener separados origen, semántica de clase y condición de riesgo; solo agregarlos cuando exista una equivalencia defendible.
4. Reservar eventos reales para evaluación antes de augmentations, selección de modelos y generación sintética.
5. Usar el análisis de errores para decidir el siguiente esfuerzo: más etiquetas, recorte/iluminación, resolución, combinación de dominios o detección con cajas.
