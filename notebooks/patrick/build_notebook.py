"""Construir un notebook de exposición desde resultados ejecutados y sellados."""
from pathlib import Path
import json
import sys
import nbformat as nbf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_data import ROOT, RAW, PROC, RESULTS, FIG, CLASSES, SPANISH, gallery, digest


def make_figures():
    from modeling import mask_overlay, diagnostics
    df = pd.read_csv(PROC/'coal_manifest.csv')
    inv = pd.read_csv(PROC/'inventory.csv')
    q = pd.read_csv(PROC/'quality.csv')
    decisions = pd.read_csv(PROC/'annotation_decisions.csv')
    issues = []
    for r in decisions[decisions.decision!='candidate'].itertuples():
        issues.append(dict(path=r.path,issue=r.decision,evidence=r.sha256,action='exclude_from_manifest'))
    for r in q[(q.dataset=='coal') & (q.gray_mean<35)].itertuples():
        issues.append(dict(path=r.path,issue='low_brightness_proxy',evidence=f'mean={r.gray_mean:.3f}<35',action='retain_if_otherwise_eligible'))
    near = pd.read_csv(PROC/'near_duplicate_edges.csv')
    for r in near[~near.same_label].itertuples():
        for path in [r.path_a,r.path_b]:
            issues.append(dict(path=path,issue='near_pair_different_labels',evidence=f'MAE={r.roi_mae:.3f}; RMSE={r.roi_rmse:.3f}',action='same_group_no_relabel'))
    pd.DataFrame(issues).to_csv(PROC/'quality_issues.csv',index=False)
    candidates=q[q.dataset=='iron'][['path','group']].copy()
    candidates['frame_index']=candidates.path.str.extract(r'/(\d+)-',expand=False).astype(int)
    candidates['selection_reason']='24_equally_spaced_positions_in_numeric_frame_order'
    candidates['review_status']='unreviewed'
    candidates['verified_label']=''
    candidates['notes']=''
    candidates.to_csv(PROC/'iron_annotation_candidates.csv',index=False)
    # Muestras ya inspeccionadas, sin convertirlas en etiquetas supervisadas.
    names = [
      ('boardcamera_iron_ore','5726-boardcamera_iron_ore.jpeg','Mineral y vista cenital'),
      ('boardcamera_iron_ore_7_wood','1964-boardcamera_iron_ore_7_wood.jpeg','Carpeta wood: madera no visible aquí'),
      ('boardcamera_iron_ore_7_wood','2458-boardcamera_iron_ore_7_wood.jpeg','Misma carpeta: madera visible'),
      ('boardcamera_iron_ore_canga','2012-boardcamera_iron_ore_canga.jpeg','Roca rojiza: semántica por definir'),
      ('boardcamera_iron_ore_5','12488-boardcamera_iron_ore_5_fastandfurious.jpeg','Desenfoque aparente'),
      ('boardcamera_iron_ore_6','6573-boardcamera_iron_ore_6.jpeg','Tramo de cinta casi vacío')]
    rows=[dict(path=f'iron/image-files/{g}/{f}') for g,f,_ in names]
    assert all((RAW/r['path']).exists() for r in rows)
    gallery(rows,FIG/'iron_compact.jpg','Iron: ejemplos visuales, sin etiquetas por fotograma verificadas',cols=3,labels=[t for _,_,t in names])
    # Una muestra por clase/presencia de texto; región constante, sin consultar test.
    dev=df[df.split!='test']
    sample=pd.concat([g.sample(1,random_state=42) for _,g in dev.groupby(['target','overlay_proxy'])])
    from PIL import Image
    fig,axes=plt.subplots(len(sample),2,figsize=(7,len(sample)*2.9))
    for i,r in enumerate(sample.itertuples()):
        im=Image.open(RAW/r.path).convert('RGB')
        for j,view in enumerate([im,mask_overlay(im)]):
            axes[i,j].imshow(view);axes[i,j].axis('off')
            axes[i,j].set_title(SPANISH[int(r.target)]+(' · original' if j==0 else ' · entrada enmascarada'),fontsize=9)
    fig.suptitle('Control visual del preprocesamiento: muestra de desarrollo',y=.999)
    fig.tight_layout(rect=(0,0,1,.985));fig.savefig(FIG/'mask_audit.jpg',bbox_inches='tight');plt.close(fig)
    from torchvision.transforms import ColorJitter
    import torch
    torch.manual_seed(42)
    train_sample=df[df.split=='train'].groupby('target',group_keys=False).sample(1,random_state=42)
    fig,axes=plt.subplots(3,2,figsize=(7,9))
    for i,r in enumerate(train_sample.itertuples()):
        im=mask_overlay(Image.open(RAW/r.path).convert('RGB'))
        changed=ColorJitter(brightness=.15,contrast=.15)(im)
        for j,view in enumerate([im,changed]):
            axes[i,j].imshow(view);axes[i,j].axis('off')
            axes[i,j].set_title(SPANISH[int(r.target)]+(' · entrada' if j==0 else ' · jitter de train'),fontsize=9)
    fig.suptitle('Transformación moderada: cambia iluminación, conserva geometría')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(FIG/'augmentation_audit.jpg',bbox_inches='tight');plt.close(fig)
    # Curvas reales por época; no rellenar las de modelos sin entrenamiento iterativo.
    runs=[]
    for p in sorted((RESULTS/'runs').glob('*/history.csv')):
        runs.append((p.parent.name,pd.read_csv(p)))
    fig,axes=plt.subplots(len(runs),2,figsize=(11,len(runs)*2.6),squeeze=False)
    for row,(name,h) in enumerate(runs):
        axes[row,0].plot(h.epoch,h.train_loss,label='train');axes[row,0].plot(h.epoch,h.val_loss,label='validación')
        axes[row,0].set(title=name+' · entropía cruzada',xlabel='Época',ylabel='Pérdida');axes[row,0].legend(fontsize=8)
        axes[row,1].plot(h.epoch,h.train_f1,label='train');axes[row,1].plot(h.epoch,h.val_f1,label='validación')
        axes[row,1].set(title=name+' · desempeño',xlabel='Época',ylabel='Macro-F1',ylim=(.6,1.01));axes[row,1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(FIG/'learning_curves.png',bbox_inches='tight');plt.close(fig)
    comp=pd.read_csv(RESULTS/'validation_comparison.csv')
    fig,ax=plt.subplots(figsize=(10,4))
    x=np.arange(len(comp));ax.bar(x-.18,comp.train_f1,width=.36,label='train');ax.bar(x+.18,comp.val_f1,width=.36,label='validación')
    ax.set(xticks=x,xticklabels=comp.run,ylabel='Macro-F1',ylim=(0,1.06),title='Comparación antes de abrir test')
    ax.tick_params(axis='x',rotation=25);ax.legend();fig.tight_layout();fig.savefig(FIG/'model_comparison.png');plt.close(fig)
    seal=json.loads((RESULTS/'final_test.json').read_text());selected=seal['selection']['run']
    selected_cfg=json.loads((RESULTS/'runs'/selected/'metadata.json').read_text())['config']
    test_predictions=pd.read_csv(RESULTS/'runs'/selected/'test_predictions.csv')
    test_predictions['risk_probability']=test_predictions[['p1','p2']].sum(axis=1)
    threshold=seal['selection']['risk_threshold']
    false_negatives=test_predictions[(test_predictions.target!=0)&(test_predictions.risk_probability<threshold)]
    false_positives=test_predictions[(test_predictions.target==0)&(test_predictions.risk_probability>=threshold)].sort_values('risk_probability',ascending=False).head(3)
    alarm_examples=pd.concat([false_negatives,false_positives])
    titles=[f"Etiqueta: {SPANISH[int(r.target)]}\n{'Sin alarma' if r.risk_probability<threshold else 'Alarma'} · P(riesgo)={r.risk_probability:.4f}" for r in alarm_examples.itertuples()]
    gallery(alarm_examples.to_dict('records'),FIG/'alarm_errors.jpg','Errores de alarma según carpetas · umbral fijado con validación',cols=2,labels=titles,transform=mask_overlay if selected_cfg.get('variant')=='masked' else None)
    vp=pd.read_csv(RESULTS/'runs'/selected/'val_predictions.csv')
    from sklearn.metrics import precision_recall_curve
    p,r,thresholds=precision_recall_curve(vp.target!=0,vp[['p1','p2']].sum(axis=1))
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    axes[0].plot(r,p);axes[0].set(xlabel='Sensibilidad a riesgo',ylabel='Precisión de alarmas',title='Validación: curva precisión–recall',xlim=(0,1),ylim=(0,1.02))
    thresholds=np.unique(np.r_[np.linspace(0,1,101),seal['selection']['risk_threshold']])
    truth=(vp.target.to_numpy()!=0);scores=vp[['p1','p2']].sum(axis=1).to_numpy()
    recall=[];far=[]
    for t in thresholds:
        pred=scores>=t;recall.append(float(pred[truth].mean()));far.append(float(pred[~truth].mean()))
    axes[1].plot(thresholds,recall,label='Sensibilidad');axes[1].plot(thresholds,far,label='Falsas alarmas / normal')
    axes[1].axvline(seal['selection']['risk_threshold'],color='black',ls='--',label='Umbral elegido con val')
    axes[1].set(xlabel='Umbral de probabilidad agregada',ylabel='Proporción',title='Tradeoff de alarma, solo validación',ylim=(0,1.02));axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(FIG/'alarm_tradeoff.png');plt.close(fig)
    # Refrescar únicamente presentación; no cambiar predicciones, métricas ni selección.
    for run in (RESULTS/'runs').iterdir():
        if not (run/'metadata.json').exists():continue
        for split in ['val','test']:
            predfile=run/f'{split}_predictions.csv'
            if predfile.exists():
                pred=pd.read_csv(predfile);frame=df.set_index('path').loc[pred.path].reset_index()
                diagnostics(run,frame,pred[['p0','p1','p2']].to_numpy(),split)


def build():
    seal=json.loads((RESULTS/'final_test.json').read_text())
    sel=seal['selection'];test=seal['test'];name=sel['run']
    comp=pd.read_csv(RESULTS/'validation_comparison.csv')
    metadata=json.loads((RESULTS/'runs'/name/'metadata.json').read_text())
    journal=json.loads((RESULTS/'experiment_journal.json').read_text())
    protocol=json.loads((PROC/'protocol.json').read_text())
    cells=[]
    md=lambda s:cells.append(nbf.v4.new_markdown_cell(s.strip()))
    code=lambda s:cells.append(nbf.v4.new_code_cell(s.strip()))
    md('''# Exploración y primeros modelos en cintas transportadoras
**Patrick · entrega de modelado temprano · ejecución local.**

Este notebook presenta un trabajo **ejecutado**: datos revisados, limpieza, evaluación agrupada, referencias, transferencia de aprendizaje e iteraciones guiadas por validación. Los scripts auxiliares implementan los experimentos; aquí se reconstruyen sus resultados guardados sin volver a entrenar.

**Pregunta inicial:** ¿podemos clasificar una imagen de coal como flujo normal, bloque grande o perno? La salida normal/riesgo se deriva de esas tres clases. No entrenamos un detector espacial: las etiquetas actuales no contienen bounding boxes. Iron participa de la exploración; no se inventan etiquetas para entrenar con sus carpetas.

Objetivos de la pauta: **(1)** exploración y descripción que orientan decisiones y **(2)** primeros modelos, dificultades, aprendizajes y ajustes respaldados por resultados.''')
    md('''## 1. Cómo ejecutar y qué se necesita
Desde la raíz del repositorio: `python -m pip install -r notebooks/patrick/requirements.txt`. Para reproducir exactamente el entorno local, consultar `requirements-resolved.txt` y el README de esta carpeta. No se reinstala CUDA ni se requiere GPU para leer los resultados.

Se necesitan los scripts de esta carpeta y los artefactos locales `data/processed/patrick_v1`, `data/results/patrick_v1` y `data/weights/patrick_v1`. Los datos y pesos están excluidos de Git: clonar el repositorio no los descarga. El README incluye comandos para reconstruirlos desde raw.

Ejecutar todas las celdas **lee resultados y verifica consistencia**; el entrenamiento se solicita explícitamente por CLI. El notebook se ha probado con kernel nuevo local; el montaje de Drive y la ejecución en Colab no se han verificado. En Colab deben resolverse rutas y dependencias en esa sesión, sin presuponer estado del setup anterior.''')
    code('''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Image as DisplayImage

# Resolver desde la raíz o desde notebooks/patrick, sin rutas personales fijas.
PROJECT_ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                     if (p / 'notebooks/patrick/modeling.py').exists()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError('Abra el notebook dentro del clon del repositorio.')
sys.path.insert(0, str(PROJECT_ROOT / 'notebooks/patrick'))
from study_data import PROC, RESULTS, FIG, CLASSES, SPANISH, digest
from modeling import summarize

required = [PROC/'inventory.csv', PROC/'coal_manifest.csv', RESULTS/'final_test.json']
for p in required:
    if not p.exists():
        raise FileNotFoundError(f'Falta {p}. Vea los comandos de reproducción en README.md.')
inventory = pd.read_csv(PROC/'inventory.csv')
quality = pd.read_csv(PROC/'quality.csv')
coal = pd.read_csv(PROC/'coal_manifest.csv')
protocol = json.loads((PROC/'protocol.json').read_text())
final = json.loads((RESULTS/'final_test.json').read_text())
selection = final['selection']
selected = selection['run']
run_dir = RESULTS/'runs'/selected
comparison = pd.read_csv(RESULTS/'validation_comparison.csv')

def show(path, width=1000):
    display(DisplayImage(filename=str(path), width=width))

pd.set_option('display.max_columns', 20)
print('Python:', sys.version.split()[0])
print('Intérprete:', sys.executable)
print('Proyecto:', PROJECT_ROOT.name)
print('Versión de datos:', PROC.name)
print('Lectura de resultados, sin entrenamiento automático.')''')
    md('''## 2. Qué existe y qué falta
Las carpetas de coal proporcionan etiquetas de **imagen completa**; no cajas. Iron contiene imágenes ordenables mediante el índice numérico del nombre, sin anotaciones locales por fotograma. Un nombre como `wood` no garantiza presencia de madera en cada imagen.

Inventariamos los originales y los contrastamos por SHA-256 con la auditoría previa. Se decodificó todo coal y una muestra temporal de 24 imágenes por cada carpeta de iron. El resto de iron pasó lectura de cabecera y hash; no se certifica su decodificación completa. No se revisaron visualmente todas las imágenes.''')
    code('''dataset_summary = inventory.groupby('dataset').agg(
    imagenes=('path','size'), bytes=('bytes','sum'),
    ancho_min=('width','min'), ancho_max=('width','max'),
    alto_min=('height','min'), alto_max=('height','max'), carpetas=('group','nunique'))
dataset_summary['MiB'] = dataset_summary['bytes'] / 2**20
display(dataset_summary.drop(columns='bytes').round(2))
display(pd.crosstab(inventory.loc[inventory.dataset=='coal','label'],
                    inventory.loc[inventory.dataset=='coal','original_split']))
print('Formatos:', inventory.groupby('dataset')['format'].unique().to_dict())
print('Campos sin etiqueta semántica en iron:', int((inventory.label=='unlabeled').sum()))
print('Archivos vacíos:', int((inventory.bytes==0).sum()))''')
    code("show(FIG/'coal_gallery.jpg', 1050)\nshow(FIG/'iron_compact.jpg', 1050)")
    md('''**Lectura visual.** Coal presenta vistas en perspectiva, imágenes oscuras y texto sobreimpreso. Iron presenta vista cenital, color, cinta vacía, madera visible en algunos tramos y desenfoque. Una roca grande aparente no demuestra un umbral físico de sobretamaño: falta escala y criterio operacional. La procedencia planta/laboratorio y los FPS no se deducen solo de la apariencia.

Dentro de iron los índices consecutivos sugieren fotogramas relacionados; no representan observaciones independientes. Las discontinuidades se documentan como saltos de índices, no como segundos ni prueba automática de archivos perdidos.''')
    code('''iron_sequences = []
for group, g in inventory[inventory.dataset=='iron'].groupby('group'):
    frames = np.sort(g.path.str.extract(r'/(\\d+)-', expand=False).astype(int).to_numpy())
    gaps = np.diff(frames)
    iron_sequences.append({'secuencia':Path(group).name.replace('boardcamera_iron_ore','iron'),
        'imagenes':len(g),'indice_min':int(frames.min()),'indice_max':int(frames.max()),
        'saltos_mayores_a_1':int((gaps>1).sum())})
display(pd.DataFrame(iron_sequences))''')
    md('''## 3. Calidad y relaciones descriptivas
El brillo es el promedio de intensidad en escala de grises [0,255]; el contraste es su desviación estándar. La nitidez es la varianza del Laplaciano después de llevar a 160×160: mide cambios locales, pero también responde a textura y overlays. No equivale a una medida física de enfoque.

Comparamos media, mediana, cuartiles e IQR y exploramos asociaciones. Las imágenes oscuras o borrosas **no se eliminan automáticamente**: representan parte del problema. En iron la muestra temporal sirve para explorar, no para inferir prevalencia de anomalías ni independencia estadística.''')
    code('''variables = ['gray_mean','gray_std','sharpness','dark_fraction','bright_fraction']
summary = quality.groupby('dataset')[variables].agg(['count','mean','std','median','min','max'])
display(summary.round(3))
quartiles = quality.groupby('dataset')[variables].quantile([.25,.75]).unstack()
iqr = pd.DataFrame({v:quartiles[(v,.75)]-quartiles[(v,.25)] for v in variables})
display(iqr.rename_axis('IQR por dataset').round(3))
show(FIG/'quality.png', 1150)
display(quality[variables].corr().round(2))''')
    md('''**Interpretación:** las condiciones visuales difieren entre datasets y entre clases. Una correlación de métricas de calidad no identifica una causa de riesgo. El número de imágenes y la textura del mineral afectan estas descripciones; no realizamos pruebas de significancia suponiendo independencia entre fotogramas.''')
    md('''## 4. Limpieza y particiones: evitar resultados engañosos
La misma imagen no puede ofrecer dos clases contradictorias a un clasificador exclusivo. Se excluyeron **todos los miembros de los grupos con conflictos exactos**, sin decidir por mayoría. Después se conservó una copia de cada contenido consistente. Raw permanece intacto; el registro de exclusiones conserva cada ruta y motivo.

No se resolvieron etiquetas nuevas basándose en las predicciones del modelo. Los casos visualmente dudosos permanecen como limitación de la supervisión heredada.''')
    code('''decisions = pd.read_csv(PROC/'annotation_decisions.csv')
display(decisions.groupby('decision').size().rename('archivos').to_frame())
print('Contenidos conflictivos excluidos:', protocol['conflicting_contents_excluded'])
print('Candidatos únicos conservados:', len(coal))
display(pd.read_csv(PROC/'quality_issues.csv').groupby(['issue','action']).size().rename('registros').to_frame())
display(pd.crosstab(coal.split, coal.label).reindex(['train','val','test']))
display(coal.groupby('split').event_group.nunique().rename('grupos proxy').to_frame())
show(FIG/'split_counts.png', 950)''')
    md('''**La dificultad de los vecinos visuales.** pHash y dHash solos propusieron 310.648 pares: algunos compartían fondo pero mostraban objetos distintos. Antes de modelar se añadió una confirmación por diferencia de píxeles en una ROI central de cinta a 64×64 (MAE≤2,5 y RMSE≤6, además de ambos hashes≤4). Así se conservaron 5.163 pares para agrupación, sin relabelar.

Se unieron esos grupos con ventanas de cinco minutos extraídas de HHMMSS en los nombres, sin distinguir clase ni partición original. Falta la fecha: se pueden unir capturas de días distintos y todavía pueden quedar eventos relacionados a ambos lados de una ventana. Son **275 grupos proxy, no eventos certificados**.

StratifiedGroupKFold de siete folds, semilla 42: fold 0 para test, fold 1 para validación y el resto para train. La elección se fijó antes del entrenamiento. Este es un benchmark nuevo, no el test original. Train aprende parámetros; validación permite decidir; test se consulta después de cerrar las iteraciones.''')
    code('''for field in ['sha256','visual_group','event_group','time_bin_5min']:
    assert coal.groupby(field).split.nunique().max()==1, field
assert coal.sha256.nunique()==len(coal)
assert digest(PROC/'coal_manifest.csv')==protocol['split_sha256']==selection['manifest_sha256']
edges = pd.read_csv(PROC/'near_duplicate_edges.csv')
print('Pares confirmados por umbrales:', len(edges))
print('Pares con carpetas/clases diferentes:', int((~edges.same_label).sum()))
print('Máximo tamaño de grupo proxy:', protocol['largest_proxy_group'])
print('Verificados: ningún hash ni grupo proxy aparece en más de un split.')
show(FIG/'near_cross_class.jpg', 750)''')
    md('''## 5. Texto sobreimpreso: una hipótesis que se prueba
El proxy detecta ≥100 píxeles amarillos en bandas superior/inferior. No transcribe el texto, no determina la clase y puede fallar. La tabla usa únicamente train y validación.

El texto puede estar asociado al origen de captura, a la fecha o a un sistema previo de alarma. No queremos presentar reconocimiento de letras como reconocimiento del objeto. Comparamos una entrada original con otra enmascarada mediante **tres rectángulos fijos normalizados**, aplicados a todas las imágenes independientemente de clase y presencia de texto. No se usa reconstrucción generativa.''')
    code('''overlay = pd.read_csv(RESULTS/'overlay_development.csv').set_index('label')
counts = overlay.copy()
counts['proporcion_con_proxy'] = counts['1'] / (counts['0']+counts['1'])
display(counts.round(3))
show(FIG/'overlay_association.png', 900)
show(FIG/'mask_audit.jpg', 700)''')
    md('''**Límite de la comparación:** revisamos la máscara en una muestra de desarrollo: conserva los objetos centrales mostrados, pero puede quitar extremos de objetos que entren por esas regiones. Además de texto oculta fondo. Por ello una caída de desempeño no prueba causalmente cuánto dependía el modelo de leer el texto. Las galerías muestran la imagen fuente con la máscara aplicada. La red recibe además el redimensionado y la normalización de su configuración; por eso un detalle visible en la fuente puede perderse a 224 píxeles.''')
    md('''## 6. Modelos y decisiones de entrenamiento
- **M0:** prior de clases del train. No usa imágenes; entrega un piso de referencia.
- **M1:** regresión logística sobre calidad y proxy de overlay, con escalado ajustado solo en train. Es un diagnóstico de señales simples.
- **M2:** ResNet-18 preentrenada, extractor congelado y cabeza lineal de tres clases entrenada con entropía cruzada. Se almacenan características para no repetir el backbone en cada época.
- **M3 y siguientes:** ajustes decididos tras ver errores y validación; no un catálogo de arquitecturas.

“Preentrenada” significa que se reutilizan características de ImageNet, no que el modelo ya conozca nuestros pernos. Las imágenes de coal se redimensionan completas a 224×224 (o al tamaño explícito del experimento), sin recorte central, y se normalizan con los valores de esos pesos. Esto conserva el campo completo, pero puede perder detalle de objetos finos.

Los scripts eligen CUDA, MPS o CPU. En esta ejecución se utilizó MPS en Apple M1. En fine-tuning se ajustan layer4 y la cabeza, manteniendo las estadísticas BatchNorm fijas. Los resultados no se prometen idénticos bit a bit entre dispositivos. Se ejecutó una semilla por configuración: diferencias pequeñas requieren confirmación con otras semillas; el bootstrap de grupos no captura variabilidad del entrenamiento. La cabeza usa hasta 80 épocas baratas sobre embeddings cacheados y paciencia 15; el fine-tuning usa hasta 8 épocas y paciencia 3.''')
    code('''configs = []
for run in comparison.run:
    meta = json.loads((RESULTS/'runs'/run/'metadata.json').read_text())
    cfg = meta['config']
    configs.append({k:cfg.get(k) for k in ['kind','variant','size','seed','epochs','best_epoch','lr','augment','batch_size']}
                   | {'run':run,'device':str(meta.get('compute_devices',meta['device']))})
display(pd.DataFrame(configs).set_index('run'))
display(comparison.round(4))
show(FIG/'model_comparison.png', 1100)
show(FIG/'learning_curves.png', 1050)''')
    md('''## 7. Iteraciones: qué observamos y por qué cambiamos el experimento
El registro siguiente contiene hipótesis, evidencia observada y decisiones tomadas durante el trabajo. Los resultados negativos y los intentos descartados también se conservan. Todas las iteraciones de modelado se eligieron con validación; no a partir del test.''')
    for entry in journal:
        md(f"### {entry['stage']}\n**Hipótesis:** {entry['hypothesis']}\n\n**Evidencia:** {entry['evidence']}\n\n**Decisión:** {entry['decision']}")
        if entry['stage']=='M4_masked_jitter':
            code("show(FIG/'augmentation_audit.jpg', 700)")
    md('''## 8. Selección final y evaluación reservada
La selección se guarda antes de abrir el test. Se utiliza macro-F1 de validación según el criterio registrado. La salida de alarma suma las probabilidades de bloque y perno; su umbral se elige maximizando F2 en validación, ponderando más la sensibilidad. Es una elección exploratoria, no una función de costos acordada con una planta.

Macro-F1 promedia el F1 de las tres clases; balanced accuracy promedia sus recalls. Para riesgo, falsa alarma significa una imagen etiquetada normal que genera alarma. No podemos convertir esa tasa en alarmas por hora sin frecuencia de captura y lógica temporal.''')
    code('''display(pd.Series(selection, name='selección fijada antes del test').to_frame())
test_metrics = final['test']
dummy_metrics = final['dummy_test']
display(pd.DataFrame([
    {'modelo':'M0_dummy','macro_f1':dummy_metrics['macro_f1'],'balanced_accuracy':dummy_metrics['balanced_accuracy'],'accuracy':dummy_metrics['accuracy']},
    {'modelo':selected,'macro_f1':test_metrics['macro_f1'],'balanced_accuracy':test_metrics['balanced_accuracy'],'accuracy':test_metrics['accuracy']}
]).set_index('modelo').round(4))
report = {k:v for k,v in test_metrics['report'].items() if isinstance(v,dict)}
display(pd.DataFrame(report).T.round(4))
display(pd.Series(test_metrics['risk'], name='alarma en test al umbral fijado').to_frame())
display(pd.Series(final['inference'], name='inferencia local por lotes, no latencia de planta').to_frame())
print('IC exploratorio 95% macro-F1, bootstrap por grupos proxy:', test_metrics['macro_f1_group_bootstrap_95'])
print('Grupos proxy de test:', test_metrics['proxy_groups'])
show(run_dir/'test_confusion.png', 650)
show(FIG/'alarm_tradeoff.png', 1050)''')
    code('''test_predictions = pd.read_csv(run_dir/'test_predictions.csv')
recomputed = summarize(test_predictions.target, test_predictions[['p0','p1','p2']].to_numpy(), selection['risk_threshold'])
assert abs(recomputed['macro_f1']-test_metrics['macro_f1']) < 1e-12
assert recomputed['confusion']==test_metrics['confusion']
assert recomputed['risk']['tp']==test_metrics['risk']['tp']
assert len(test_predictions)==len(coal[coal.split=='test'])
weight_path = PROJECT_ROOT/'data/weights/patrick_v1'/f'{selected}.pt'
assert digest(weight_path)==selection['weights_sha256']
print('Métricas reconstruidas desde predicciones; manifest y checkpoint coinciden con la selección.')''')
    md(f'''**Resultado observado:** {name} obtuvo macro-F1 **{test['macro_f1']:.3f}** y balanced accuracy **{test['balanced_accuracy']:.3f}** en las {test['n']} imágenes de test. Al umbral de alarma fijado con validación, detectó {test['risk']['tp']} de {test['risk']['tp']+test['risk']['fn']} imágenes de riesgo y activó {test['risk']['fp']} falsas alarmas entre {test['risk']['fp']+test['risk']['tn']} imágenes normales.

El intervalo por bootstrap usa grupos proxy: describe variabilidad interna bajo esa agrupación, no incertidumbre de desempeño en otra planta. Las etiquetas heredadas y el número limitado de grupos restringen la conclusión. El test no se empleó para volver a ajustar después de conocer este resultado.''')
    md('''**El balance y la prevalencia importan.** En este test cerca de dos tercios de las imágenes son de riesgo, por construcción de las clases. Por eso su precisión de alarmas no se puede trasladar directamente a una planta donde los incidentes podrían ser mucho menos frecuentes.

Como ejemplo matemático, no como estimación de una planta: si solo 1% de las imágenes tuviera riesgo y se mantuvieran exactamente las mismas sensibilidad y tasa de falsas alarmas, la precisión esperada de las alarmas se calcula con la fórmula siguiente. Esa hipótesis de tasas constantes tampoco está demostrada fuera de estos datos.''')
    code('''hypothetical_prevalence = .01  # Escenario ilustrativo, no prevalencia medida.
sensitivity = test_metrics['risk']['recall']
false_alarm_rate = test_metrics['risk']['false_alarm_rate']
illustrative_precision = (sensitivity*hypothetical_prevalence /
                         (sensitivity*hypothetical_prevalence + false_alarm_rate*(1-hypothetical_prevalence)))
print(f'Escenario hipotético con 1% de riesgo: precisión de alarmas ≈ {illustrative_precision:.1%}.')
print('Esto explica por qué una alta sensibilidad no basta para declarar un sistema operacional.')''')
    md('''## 9. Errores, confianza y dificultades que quedan
Las galerías priorizan errores de alta confianza, con representación de clases. Son una muestra dirigida para diagnóstico, no una muestra para estimar frecuencias.

Entre los errores de desarrollo observamos pernos finos sobre mineral oscuro, cambios de vista y ejemplos donde la etiqueta de carpeta parece discutible. Una predicción plausible no autoriza a corregir la etiqueta automáticamente. Para una siguiente versión hace falta adjudicación con reglas explícitas y revisión independiente, manteniendo congelada la evaluación de esta versión.''')
    code('''if (run_dir/'val_errors.jpg').exists():
    show(run_dir/'val_errors.jpg', 1100)
if (run_dir/'test_errors.jpg').exists():
    show(run_dir/'test_errors.jpg', 1100)
show(FIG/'alarm_errors.jpg', 750)
display(pd.read_csv(run_dir/'val_overlay_slices.csv').round(4))
error_counts = pd.crosstab(test_predictions.target, test_predictions.prediction).reindex(index=[0,1,2],columns=[0,1,2],fill_value=0)
error_counts.index = SPANISH; error_counts.columns = SPANISH
display(error_counts)''')
    md('''La galería de alarma separa el objetivo binario de la clasificación de tres clases. Confundir bloque y perno puede contar como error de clase y aun activar correctamente una alarma. La única omisión de riesgo muestra una línea delgada y curva sobre mineral oscuro; en los ejemplos de falsa alarma también hay etiquetas heredadas que conviene adjudicar. Esta es inspección posterior del test para explicar límites, **sin modificar el modelo ni el umbral**.''')
    md('''## 10. Cargar el modelo y reproducir una predicción
El checkpoint conserva pesos, orden de clases y transformación. Usar el mismo preprocesamiento importa tanto como cargar los pesos. Este ejemplo reconstruye una predicción ya guardada como comprobación; no abre otro experimento ni selecciona por test.''')
    code('''import torch
from modeling import build_model, CoalImages, DEVICE
checkpoint = torch.load(weight_path, map_location='cpu', weights_only=True)
model = build_model(pretrained=False)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(DEVICE).eval()
cfg = checkpoint['config']
example = coal.set_index('path').loc[[test_predictions.path.iloc[0]]].reset_index()
x, _ = CoalImages(example, cfg['variant'], cfg['size'])[0]
with torch.inference_mode():
    probabilities = model(x.unsqueeze(0).to(DEVICE)).softmax(1).cpu().numpy()[0]
assert np.allclose(probabilities, test_predictions.iloc[0][['p0','p1','p2']].to_numpy(dtype=float), atol=5e-4)
display(pd.Series(probabilities, index=CLASSES, name='probabilidad').to_frame())
print('Etiqueta heredada:', example.label.iloc[0])
print('Predicción:', CLASSES[int(probabilities.argmax())])
print('Entrada:', cfg['variant'], cfg['size'], '×', cfg['size'])''')
    md('''## 11. Qué presentar y qué desarrollar después
**La etapa temprana está centrada en coal.** Podemos exponer la auditoría, el problema de los fondos para agrupar imágenes, la referencia de calidad, el aporte de la CNN, el experimento con texto, los ajustes ejecutados, la selección por validación y los errores finales.

**Iron:** el siguiente paso concreto es etiquetar una muestra temporal por presencia real, manteniendo casos inciertos y cinta vacía separados. Solo después compararíamos coal, iron y ambos, evaluando por dominio. No hay evidencia aquí de que el modelo sirva para iron, otra cámara o cualquier objeto extraño.

**Sintéticos:** una composición de objetos recortados y fondos de train sería un experimento posterior para cubrir una carencia concreta. No arregla etiquetas ni prueba generalización. Compararíamos real frente a real+sintético sobre la misma evaluación real.

**Bounding boxes:** se requieren cuando definamos una tarea supervisada de localización. El primer piloto usaría imágenes diversas por evento con todos los objetos objetivo anotados; las imágenes normales verificadas no llevan una caja ficticia. Esta entrega clasifica imágenes y no estima coordenadas.

Las extensiones están **aplazadas con justificación**, no ejecutadas. No se generan tablas de resultados ficticios de iron, sintéticos o detectores.''')
    code('''display(pd.DataFrame([
    {'línea':'Exploración coal/iron','estado':'Ejecutada','evidencia':'Inventario, calidad, galerías y resúmenes'},
    {'línea':'Limpieza y particiones coal','estado':'Ejecutada con límites','evidencia':'Decisiones, hashes y grupos proxy'},
    {'línea':'Referencias y CNN','estado':'Ejecutada','evidencia':'Comparación, curvas, checkpoints y predicciones'},
    {'línea':'Tratamiento del texto y ajustes','estado':'Ejecutada','evidencia':'Diario de iteraciones y métricas de validación'},
    {'línea':'Supervisión y modelo conjunto iron','estado':'Aplazada','evidencia':'Faltan etiquetas verificadas y eventos de evaluación'},
    {'línea':'Sintéticos y detector con cajas','estado':'Aplazada','evidencia':'Requieren objetivo/anotaciones y evaluación real'}
]))''')
    code('''iron_candidates = pd.read_csv(PROC/'iron_annotation_candidates.csv', keep_default_na=False)
assert (iron_candidates.review_status=='unreviewed').all()
assert (iron_candidates.verified_label=='').all()
print('Próximo paso preparado:', len(iron_candidates), 'candidatos temporales de iron, todos sin revisar ni etiquetar.')
display(iron_candidates.head(3))''')
    md('''## 12. Reproducibilidad y fuentes
Los comandos y dependencias están en `README.md`. `study_data.py` produce inventario/EDA; `modeling.py` ejecuta referencias, cabezas y fine-tuning y sella el test; `build_notebook.py` construye esta exposición; `verify_study.py` audita resultados y entradas. `PROGRESO.md` registra el estado del trabajo y `experiment_journal.json` conserva las hipótesis.

Cada ejecución tiene configuración, versiones, hash del manifiesto y fuentes preservadas. Los artefactos son locales; no se publicaron datos ni se hicieron commits/push. Las instrucciones de Colab son orientación y no una integración verificada.

Fuentes técnicas primarias:
- [Transferencia de aprendizaje en PyTorch](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
- [Pesos y normalización de ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- [Particiones con StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

La evidencia del proyecto procede de los archivos locales y de los experimentos guardados; las fuentes explican las técnicas, no certifican nuestros resultados.''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':sys.version.split()[0]}})
    destination=Path(__file__).parent/'entrega_modelado.ipynb'
    nbf.write(nb,destination)
    print('Notebook construido:',destination,'celdas:',len(cells))


if __name__=='__main__':
    make_figures()
    build()
