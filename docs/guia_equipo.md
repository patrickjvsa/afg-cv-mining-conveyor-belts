# Guía del equipo

El recorrido habitual es **00: preparar → 01: explorar → 02: consultar resultados**. Por ahora no hay entrenamiento implementado. Para usar Colab no necesita instalar Python ni Git en su computador ni editar `src/`.

## 1. Preparación del equipo

El responsable publica el repositorio en GitHub e invita a sus compañeros. Cada integrante acepta la invitación. Por separado, el responsable comparte una carpeta de Google Drive con las cuentas Google del equipo: GitHub no concede acceso a Drive.

Dentro de la raíz compartida se proponen `raw` (originales), `processed` (procesados), `results` (métricas y visualizaciones) y `weights` (pesos). Los nombres son configurables en el notebook. Suba los datos a una subcarpeta de `raw`, acuerde un identificador de versión y comparta con el equipo la URL del repositorio, rama y ubicación de los datos. No modifique una versión ya usada en experimentos; publique otra al corregir o agregar archivos.

Si la carpeta aparece en «Compartido conmigo», añada un acceso directo a «Mi unidad» desde el menú Organizar de Drive. Después del montaje en Colab, copie la ruta desde su explorador de archivos. Cada integrante debe verificar sus propios permisos de lectura de datos y escritura de resultados.

## 2. Paso 00: preparar Colab

1. Entre en [Colab](https://colab.research.google.com/), seleccione **Archivo → Abrir notebook → GitHub** y busque la URL real del repositorio. Abra `notebooks/00_setup.ipynb`. También puede descargar el archivo y usar **Subir**.
2. Conecte una sesión. CPU es suficiente para preparar y explorar datos; GPU es opcional y su disponibilidad puede variar.
3. Complete la sección 1: URL, rama, raíz de Drive, ruta relativa del dataset y versión. La ruta del dataset es relativa a `raw`, no a la raíz compartida. Los demás valores pueden conservarse inicialmente.
4. Ejecute las celdas en orden con ▶ y espere a que cada una termine. No continúe si falla una celda: lea el mensaje y corrija su causa.
5. Autorice Drive mediante el diálogo oficial de Google. Nunca escriba contraseñas en celdas.
6. Revise los archivos encontrados y el reporte de imágenes. `ok: true` significa que pasó la inspección general, no que se hayan validado las etiquetas ni todas las imágenes.
7. Si necesita copiar a disco local, active `copy_to_local` en la sección 1. Para un ZIP active también `extract_zip`. La copia se guarda bajo `data/prepared/data/` del clon; las ejecuciones repetidas reutilizan una copia idéntica y rechazan cambios inesperados.
8. Compruebe el resumen final: datos, commit, CPU/GPU y carpetas de salida. `DATA_PATH` es la ruta que debe recibir el futuro cargador del modelo.

Toda la configuración está en la sección 1 del notebook, incluidos semilla, parámetros y versión de datos. No hay archivos YAML que editar. Conserve una copia personal del notebook configurado; no publique rutas personales innecesarias ni salidas privadas.

### Repositorio privado

La autorización para abrir un notebook desde GitHub no necesariamente autentica su clon. Para clonar un repositorio privado, cree un token de lectura limitado a ese repositorio y guárdelo en el panel **Secretos** de Colab con el nombre `GITHUB_TOKEN`. Autorice el acceso desde el notebook y active `private_repo` en la sección 1. Nunca pegue el token en código, URLs, archivos ni mensajes. El helper temporal no guarda el token en su contenido.

## 3. Pasos 01 y 02: continuar sin perder la sesión

En Colab, abrir otro notebook puede asignarle otra máquina. El camino más sencillo es terminar `00_setup.ipynb` y añadir una celda al final con:

```python
run_step("01_exploracion_datos.ipynb")
```

Esto ejecuta el notebook de exploración en la misma sesión y muestra sus instrucciones y ejemplos de imágenes. El archivo 01 sigue siendo un notebook independiente que se puede abrir para leer o editar sus pasos.

Cuando haya resultados registrados, ejecute en otra celda:

```python
run_step("02_evaluacion.ipynb")
```

El paso 02 muestra métricas persistidas; si no hay experimentos finalizados, mostrará cero. No calcula todavía métricas de un detector. Si reinicia o pierde la máquina, repita el paso 00 antes de continuar. Un `session.json` de una máquina anterior no recupera sus archivos temporales.

## 4. Datos y experimentos

Mantenga los originales sin cambios. Registre la procedencia real o simulada y la versión en la sección 1. Las clases, anotaciones y particiones están pendientes de definición por el equipo. No genere particiones aleatorias nuevas en cada ejecución si quiere comparar resultados.

Si ya existen listas de train/validation/test, indique en `splits` sus rutas relativas al dataset preparado. Al registrar un experimento se guardan sus hashes. La infraestructura no comprueba aún la correspondencia entre imágenes y etiquetas: cuando se conozca el formato, el responsable puede añadir una función de solo lectura `modulo:funcion` en `validator`, que devuelva una lista de errores. La muestra general no reemplaza esa validación.

Cuando exista un modelo, complete `hyperparameters` y aplique la semilla desde el código de entrenamiento. Guardar `seed` no la aplica automáticamente. Antes de registrar una ejecución haga commit y push del código que la genera.

Ejemplo de persistencia, para usar cuando existan resultados reales:

```python
RUN = start_run(PATHS, REPO, MANIFEST, EXPERIMENT_CONFIG, DATA_PATH)
# Ejecutar aquí el entrenamiento que defina el equipo.
artifacts = [save_artifact(ruta_peso_final, PATHS["weights"] / RUN.name)]
finish_run(RUN, metrics=metricas_reales, artifacts=artifacts)
```

Las variables `ruta_peso_final` y `metricas_reales` deben provenir del entrenamiento. No ejecute este ejemplo esperando un modelo ya implementado.

Cada ejecución guarda configuración, versión de datos, commit y versiones del entorno en `results/<id>`. Sus métricas se guardan en `record.jsonl`; los pesos seleccionados en `weights/<id>`. La carpeta `experiments/` no es necesaria. Las llamadas rechazan sobrescrituras y código sin commit. Los checkpoints útiles y visualizaciones también pueden copiarse explícitamente con `save_artifact`, usando nombres distintos.

Antes de desconectar, abra Drive y compruebe los archivos guardados. El disco de Colab es temporal; no se sincroniza automáticamente. Una exportación interrumpida puede dejar archivos parciales: conserve el original y revise el error/hash. Reanudar un entrenamiento requerirá soporte del futuro modelo.

Para comparar resultados revise versión de datos, particiones, configuración, métricas y commit. Para repetirlos necesita recuperar esos mismos datos y código; no basta con copiar el número de la métrica.

## 5. Git sin complicaciones

Git guarda versiones del código. Un **commit** guarda un cambio local; una **rama** permite trabajar separado; **push** envía cambios a GitHub; **pull** trae cambios; un **Pull Request (PR)** pide revisión antes de integrar la rama mediante **merge**.

Para desarrollar localmente, instale [Git](https://git-scm.com/downloads). Ejecute una vez, reemplazando los marcadores:

```bash
git config --global user.name "Su nombre"
git config --global user.email "Su correo"
git clone <URL_REAL>
cd <CARPETA_DEL_REPOSITORIO>
```

Flujo por tarea, empezando sin cambios pendientes:

```bash
git status
git switch <RAMA_PRINCIPAL>
git pull --ff-only
git switch -c tarea/descripcion
# Edite los archivos y compruebe su funcionamiento.
git add <ARCHIVOS_MODIFICADOS>
git diff --cached
git commit -m "Describir el cambio"
git push -u origin tarea/descripcion
```

`git add` selecciona archivos y `git diff --cached` permite revisarlos antes del commit. En GitHub abra un PR y pida revisión a un compañero. Explique el cambio, cómo lo verificó y si afecta datos o resultados. Acuerden quién edita cada notebook y borren sus salidas antes de enviarlo.

Si dos personas editan las mismas líneas puede haber un conflicto. Guarde primero su trabajo en su rama; use `git fetch origin` y `git merge origin/<RAMA_PRINCIPAL>`. Si Git pide resolver conflictos, revise `git status`, acuerde el contenido correcto, quite los marcadores de conflicto y haga `git add` y `git commit`. `git merge --abort` cancela esa integración si necesita ayuda. No use resets forzados ni borre trabajo para resolverlo.

No suba datasets, pesos ni credenciales. `.gitignore` excluye los datos locales y artefactos habituales, pero siempre revise lo que va a enviar. Para autenticación local use un gestor de credenciales; no ponga tokens en la URL.

## 6. Trabajo local opcional

Colab no requiere instalación local. Si quiere usar las utilidades en su computador, necesita Python 3.10+ y, desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
# En Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

No se instala el proyecto como paquete. Los notebooks agregan `src/` a la ruta de importación. Para sus scripts locales siga el ejemplo de [data/README.md](../data/README.md). PyTorch/CUDA se elegirán cuando exista un modelo; el setup no los reinstala automáticamente.

## 7. Problemas frecuentes

| Problema | Qué hacer |
|---|---|
| GitHub no permite clonar | Compruebe invitación, URL, rama y permisos del secreto GITHUB_TOKEN. |
| La copia tiene cambios o pertenece a otra rama | Guarde el trabajo; use otro work_dir o resuelva la rama manualmente. No se sobrescribe automáticamente. |
| Drive no monta | Repita la autorización oficial con la cuenta correcta. |
| No aparece la carpeta compartida | Compruebe acceso en el navegador, añada acceso directo y copie la ruta desde Colab. La falta de permisos puede parecer una ruta inexistente. |
| No existe el dataset | Revise su ruta dentro de raw y la versión subida. |
| ZIP corrupto o con rutas inseguras | Solicite una copia válida al responsable; no desactive la protección de extracción. |
| No queda espacio o cambió la copia | Use lectura directa cuando sea posible o un work_dir nuevo; conserve antes los resultados importantes. |
| No hay GPU | Continúe la preparación en CPU. Solicite GPU cuando exista un entrenamiento que la necesite. |
| Falla pip o PyTorch | Revise el error; pruebe una sesión limpia. No reinstale CUDA a ciegas. |
| No se encuentra session.json | Vuelva al paso 00 y ejecute los siguientes mediante run_step en esa sesión. |
| No aparecen resultados en Drive | Revise las rutas, permisos/cuota y la exportación explícita; guardar el notebook no guarda sus archivos locales. |
| El notebook sigue mostrando código antiguo | Guarde su trabajo, vuelva a abrirlo desde la rama correcta y reinicie explícitamente la sesión si cambió módulos ya importados. |

## 8. Comprobación breve antes de trabajar

Se retiraron la suite de tests y la CI para mantener este repositorio pequeño. Al cambiar las utilidades, el responsable debe comprobar manualmente:

- [ ] Setup completo en una sesión nueva de Colab; lectura y escritura con la cuenta de un integrante.
- [ ] Una ruta inexistente produce un error comprensible.
- [ ] Repetir la preparación no duplica ni sobrescribe datos.
- [ ] La exploración muestra imágenes y la evaluación lee registros existentes.
- [ ] Un artefacto pequeño persiste en Drive después de iniciar otra sesión; su nombre repetido se rechaza.
- [ ] El código y los notebooks no contienen secretos ni salidas privadas.

Las comprobaciones locales de sintaxis y lógica no prueban autorización de Google, permisos reales de Drive ni disponibilidad de GPU. Esas integraciones deben verificarse en Colab.
