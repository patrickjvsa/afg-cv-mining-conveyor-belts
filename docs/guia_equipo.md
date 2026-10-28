# Guía del equipo

Por ahora usamos **00_setup.ipynb** para preparar el entorno y acceder a los datos. Todavía no definimos un flujo común de entrenamiento o evaluación. Para usar Colab no necesita instalar Python ni Git en su computador ni editar `src/`.

## 1. Preparación del equipo

El responsable publica el repositorio en GitHub e invita a sus compañeros. Cada integrante acepta la invitación. Por separado, el responsable comparte una carpeta de Google Drive con las cuentas Google del equipo: GitHub no concede acceso a Drive.

Los datos compartidos siguen la estructura de `data/`: datasets dentro de `raw/`, actualmente `iron` y `coal`. Elija cuál quiere inspeccionar en la sección 1 del setup. No necesita crear carpetas de resultados o pesos para comenzar. Mantenga intactos los originales.

Si la carpeta aparece en «Compartido conmigo», añada un acceso directo a «Mi unidad» desde el menú Organizar de Drive. Después del montaje en Colab, copie la ruta desde su explorador de archivos. Cada integrante debe verificar sus propios permisos de lectura de datos.

## 2. Paso 00: preparar Colab

1. Entre en [Colab](https://colab.research.google.com/), seleccione **Archivo → Abrir notebook → GitHub** y busque la URL real del repositorio. Abra `notebooks/00_setup.ipynb`. También puede descargar el archivo y usar **Subir**.
2. Conecte una sesión. CPU es suficiente para preparar y explorar datos; GPU es opcional y su disponibilidad puede variar.
3. Revise la sección 1: URL pública y `main` ya están configuradas. Para hacer cambios, cree primero su rama en GitHub desde `main` y escriba su nombre en `branch`. Complete la raíz montada de Drive y la ruta relativa del dataset. La ruta del dataset es relativa a `raw`, no a la raíz compartida. Los demás valores pueden conservarse inicialmente.
4. Ejecute las celdas en orden con ▶ y espere a que cada una termine. No continúe si falla una celda: lea el mensaje y corrija su causa.
5. Autorice Drive mediante el diálogo oficial de Google. Nunca escriba contraseñas en celdas.
6. Revise los archivos encontrados y el reporte de imágenes. `ok: true` significa que pasó la inspección general, no que se hayan validado las etiquetas ni todas las imágenes.
7. Si necesita copiar a disco local, active `copy_to_local` en la sección 1. Para un ZIP active también `extract_zip`. La copia se guarda bajo `data/prepared/data/` del clon; las ejecuciones repetidas reutilizan una copia idéntica y rechazan cambios inesperados.
8. Al finalizar la sección 7, compruebe el mensaje «Setup completado». `DATA_PATH` contiene la ruta que puede usar para explorar los datos en esta sesión.

Toda la configuración de repositorio, rutas, validación y copia está en la sección 1 del notebook. No hay archivos YAML que editar. Conserve una copia personal del notebook configurado; no publique rutas personales innecesarias ni salidas privadas.

El repositorio actual es público: deje `private_repo=False`. No necesita un token para clonar, pero sí permisos de colaborador para publicar cambios. Tener lectura pública de Drive no concede escritura de resultados.

### Si en el futuro usa un repositorio privado

La autorización para abrir un notebook desde GitHub no necesariamente autentica su clon. Para clonar un repositorio privado, cree un token de lectura limitado a ese repositorio y guárdelo en el panel **Secretos** de Colab con el nombre `GITHUB_TOKEN`. Autorice el acceso desde el notebook y active `private_repo` en la sección 1. Nunca pegue el token en código, URLs, archivos ni mensajes. El helper temporal no guarda el token en su contenido.

## 3. Comenzar a explorar

Después del setup puede agregar celdas en esa sesión y usar `DATA_PATH` para leer los archivos. Cada integrante puede desarrollar sus propios notebooks en su rama; aún no se prescribe una secuencia de modelado o evaluación.

Abrir otro notebook en Colab puede asignarle otra máquina: las variables y archivos temporales no se comparten automáticamente. Si pierde la sesión, repita el setup. Este notebook no genera `session.json`, no crea carpetas de salida y no escribe en Drive.

La validación general cuenta archivos y revisa una muestra de imágenes. No comprueba todavía etiquetas, clases ni particiones. Si el equipo define un formato, el responsable puede incorporar una función `modulo:funcion` en `validator`, que devuelva una lista de errores sin modificar los datos. Mantenga los originales intactos mientras explora.

## 4. Git sin complicaciones

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

## 5. Trabajo local opcional

Colab no requiere instalación local. Si quiere usar las utilidades en su computador, necesita Python 3.10+ y, desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
# En Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

No se instala el proyecto como paquete. El setup agrega la raíz del repositorio a la ruta de importación para cargar las utilidades de `src/`. Para sus scripts locales siga el ejemplo de [data/README.md](../data/README.md). PyTorch/CUDA se elegirán cuando exista un modelo; el setup no los reinstala automáticamente.

## 6. Problemas frecuentes

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
| El notebook sigue mostrando código antiguo | Guarde su trabajo, vuelva a abrirlo desde la rama correcta y reinicie explícitamente la sesión si cambió módulos ya importados. |

## 7. Comprobación breve antes de trabajar

Se retiraron la suite de tests y la CI para mantener este repositorio pequeño. Al cambiar las utilidades, el responsable debe comprobar manualmente:

- [ ] Setup completo en una sesión nueva de Colab; lectura con la cuenta de un integrante.
- [ ] Una ruta inexistente produce un error comprensible.
- [ ] Repetir la preparación no duplica ni sobrescribe datos.
- [ ] `DATA_PATH` apunta al dataset elegido y el reporte muestra los archivos esperados.
- [ ] El setup no modifica los originales ni crea carpetas de resultados en Drive.
- [ ] El código y los notebooks no contienen secretos ni salidas privadas.

Las comprobaciones locales de sintaxis y lógica no prueban autorización de Google, permisos reales de Drive ni disponibilidad de GPU. Esas integraciones deben verificarse en Colab.
