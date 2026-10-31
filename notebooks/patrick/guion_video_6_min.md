# Guion de video · seis minutos

Leer únicamente estos seis bloques. Las pausas y la velocidad de lectura determinan la duración final.

## 00:00–01:00 · Qué estamos prediciendo

Nuestro proyecto busca reconocer condiciones de riesgo en imágenes de una cinta transportadora. En esta etapa entrenamos un solo modelo que clasifica cada foto en tres categorías: flujo normal, bloque grande o perno. Normal no significa que la cinta esté vacía: puede tener mineral circulando. Tampoco cualquier roca es un riesgo; el bloque grande es una categoría específica del dataset.

Esto es clasificación: damos una etiqueta a la imagen completa. Ubicar el objeto con un rectángulo sería detección, y señalar qué píxeles le pertenecen sería segmentación. Esas tareas quedan para después. Además de la etiqueta, usamos las probabilidades del mismo modelo para producir una alerta de riesgo. Por lo tanto, mostramos dos lecturas del mismo resultado, no dos modelos distintos.

## 01:00–02:00 · Qué datos usamos y qué sacamos

Revisamos dos conjuntos: uno de carbón, con cuatro mil ochocientas imágenes, y otro de hierro, con diecinueve mil doscientas once. Para modelar usamos carbón, porque ya tenía etiquetas por imagen. En hierro, el nombre de una carpeta no asegura que un objeto esté presente en cada foto.

Antes de entrenar retiramos ciento treinta y cinco archivos que repetían imágenes con etiquetas contradictorias. No elegimos una etiqueta al azar. También retiramos ciento sesenta y cinco copias repetidas que tenían etiquetas consistentes. Quedaron cuatro mil quinientas imágenes únicas. Conservamos los originales y no eliminamos automáticamente las fotos oscuras o borrosas, porque también representan dificultades reales del problema. Las imágenes parecidas se agruparon para reducir contaminación entre entrenamiento y evaluación.

## 02:00–03:00 · El flujo que elegimos

El flujo elegido es simple. Primero limpiamos los datos y separamos los grupos. Después ocultamos las zonas del texto sobreimpreso, como fechas y mensajes de alarma, y ajustamos las imágenes a doscientos veinticuatro píxeles por lado. Buscamos que el modelo se apoye en la escena y no en un aviso de otro sistema.

Usamos ResNet dieciocho, una red preentrenada que ya aprendió características visuales de otras imágenes. Primero entrenamos su salida para nuestras tres categorías. Luego ajustamos su último bloque a estas cintas. Los rectángulos negros solo ocultan texto: no son detecciones ni segmentación. También borran parte de la escena, por lo que revisamos ejemplos y dejamos esa limitación documentada. Este es el camino del modelo que finalmente seleccionamos.

## 03:00–04:00 · Cómo elegimos el modelo

Separamos tres grupos de datos con funciones distintas. El entrenamiento permite aprender. La validación permite comparar opciones. La prueba final se reserva para evaluar el modelo ya elegido. Usamos tres mil doscientas once imágenes para entrenar, seiscientas cuarenta y cuatro para validar y seiscientas cuarenta y cinco para la prueba.

La mejor opción de validación fue la red con texto ocultado y ajuste del último bloque: obtuvo un macro F uno de cero coma novecientos veintiuno. Este indicador da igual importancia a las tres clases. Conservamos la tercera época, porque seguir entrenando mejoraba el entrenamiento pero no la validación. Las variaciones adicionales de iluminación y tamaño no superaron esa opción. La elección se hizo antes de consultar la prueba final.

## 04:00–05:00 · Resultados: etiqueta y alerta

En la prueba final, el modelo asignó correctamente la etiqueta a seiscientas una de seiscientas cuarenta y cinco imágenes: aproximadamente noventa y tres por ciento. El macro F uno fue cero coma novecientos treinta y dos.

Para la alerta sumamos las probabilidades de bloque grande y perno, y las comparamos con un umbral elegido en validación para priorizar la detección de riesgos. No entrenamos otra red. Esta regla detectó cuatrocientas treinta y dos de cuatrocientas treinta y tres imágenes etiquetadas como riesgo, pero generó veinticuatro falsas alarmas entre doscientas doce imágenes normales. Eso equivale a un once coma tres por ciento de falsas alarmas. Confundir un bloque con un perno puede ser un error de etiqueta y aun activar correctamente la alerta: son dos formas distintas de evaluar el mismo modelo.

## 05:00–06:00 · Qué aprendimos y qué sigue

El principal aprendizaje es que preparar bien los datos importa tanto como elegir la red. Encontramos duplicados, etiquetas contradictorias y texto que podía servir como atajo. Ajustar parte de una red preentrenada mejoró la clasificación, pero seguir entrenando produjo sobreajuste.

Todavía hay dificultades con pernos delgados, escenas oscuras y algunas etiquetas discutibles. Además, una alta detección de riesgos no elimina el problema de las falsas alarmas. Estos resultados corresponden a una evaluación interna, con agrupaciones aproximadas de eventos; no prueban funcionamiento en otra planta. El siguiente paso es revisar las etiquetas dudosas y etiquetar una muestra de hierro. Después podremos evaluar si conviene combinar los datasets. Las cajas, la segmentación y los datos sintéticos quedan para experimentos posteriores, según lo que necesitemos resolver.
