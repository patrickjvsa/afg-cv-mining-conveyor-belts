"""Auditoría científica y de reconstrucción, sin seleccionar ni reentrenar."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from study_data import ROOT, RAW, PROC, RESULTS, CLASSES, digest, write_json
from modeling import summarize


def verify(check_raw=False, check_notebook=True, inference=False):
    checks=[]
    inv=pd.read_csv(PROC/'inventory.csv');m=pd.read_csv(PROC/'coal_manifest.csv')
    protocol=json.loads((PROC/'protocol.json').read_text())
    assert len(inv)==24011 and inv.path.nunique()==len(inv)
    assert len(m)==4500 and m.sha256.nunique()==len(m)
    assert len(set(m.path)-set(inv.path))==0
    assert m.groupby('split').target.nunique().min()==3
    for key in ['sha256','visual_group','event_group','time_bin_5min']:
        assert m.groupby(key).split.nunique().max()==1,key
    assert digest(PROC/'coal_manifest.csv')==protocol['split_sha256']
    decisions=pd.read_csv(PROC/'annotation_decisions.csv')
    bad=set(decisions.loc[decisions.decision=='exclude_conflicting_label','sha256'])
    assert len(bad)==65 and not set(m.sha256)&bad
    assert len(decisions[decisions.decision=='exclude_conflicting_label'])==135
    assert len(decisions)==4800
    issues=pd.read_csv(PROC/'quality_issues.csv')
    assert set(issues.path)<=set(inv.path)
    candidates=pd.read_csv(PROC/'iron_annotation_candidates.csv',keep_default_na=False)
    assert len(candidates)==288 and candidates.path.nunique()==288
    assert set(candidates.path)<=set(inv.loc[inv.dataset=='iron','path'])
    assert (candidates.review_status=='unreviewed').all() and (candidates.verified_label=='').all()
    lookup=m.set_index('path')
    edges=pd.read_csv(PROC/'near_duplicate_edges.csv')
    assert (lookup.loc[edges.path_a,'split'].to_numpy()==lookup.loc[edges.path_b,'split'].to_numpy()).all()
    checks.append('Inventario, incidencias, exclusiones, soporte y cero cruces de hashes/grupos/pares confirmados; candidatos iron sin etiquetas inventadas')
    if check_raw:
        actual=sorted(p.relative_to(RAW).as_posix() for p in RAW.rglob('*') if p.is_file())
        assert actual==sorted(inv.path.tolist())
        for r in inv.itertuples():
            p=RAW/r.path
            assert p.stat().st_size==r.bytes and digest(p)==r.sha256,r.path
        checks.append('24.011 originales actuales coinciden en rutas, tamaño y SHA-256 con el inventario')
    runs=[]
    for run in sorted((RESULTS/'runs').iterdir()):
        if not (run/'metadata.json').exists():continue
        meta=json.loads((run/'metadata.json').read_text())
        assert (run/'metrics.json').exists(),f'Ejecución incompleta: {run.name}'
        assert meta['manifest_sha256']==digest(PROC/'coal_manifest.csv')
        for name,expected in meta['source_hashes'].items():
            assert digest(run/'source_snapshot'/name)==expected,(run.name,name)
        metrics=json.loads((run/'metrics.json').read_text())
        for split in ['train','val']:
            p=pd.read_csv(run/f'{split}_predictions.csv');expected=m[m.split==split]
            assert p.path.tolist()==expected.path.tolist()
            assert p.target.tolist()==expected.target.tolist()
            probs=p[['p0','p1','p2']].to_numpy()
            assert np.isfinite(probs).all() and (probs>=0).all() and (probs<=1).all()
            assert np.allclose(probs.sum(axis=1),1,atol=1e-6)
            rec=summarize(p.target,probs)
            for metric in ['accuracy','balanced_accuracy','macro_f1']:
                assert abs(rec[metric]-metrics[split][metric])<1e-12,(run.name,split,metric)
            assert rec['confusion']==metrics[split]['confusion']
        if (run/'history.csv').exists():
            h=pd.read_csv(run/'history.csv')
            assert abs(h.val_f1.max()-metrics['val']['macro_f1'])<1e-6,run.name
        cfg=meta['config']
        if 'weights_relative' in cfg:
            assert digest(ROOT/cfg['weights_relative'])==cfg['weights_sha256']
        runs.append(run.name)
    checks.append('Todas las ejecuciones: fuentes preservadas, métricas reconstruidas, probabilidades, historia y pesos')
    final=json.loads((RESULTS/'final_test.json').read_text());sel=final['selection'];run=RESULTS/'runs'/sel['run']
    comp=pd.read_csv(RESULTS/'validation_comparison.csv')
    candidates=comp[comp.kind.isin(['frozen_head','finetune_layer4'])]
    expected=candidates.sort_values(['val_f1','run'],ascending=[False,True]).iloc[0].run
    assert sel['run']==expected
    assert set(sel['candidate_runs'])==set(comp.run)==set(runs)
    vp=pd.read_csv(run/'val_predictions.csv')
    from sklearn.metrics import precision_recall_curve
    precision,recall,th=precision_recall_curve(vp.target!=0,vp[['p1','p2']].sum(axis=1))
    f2=5*precision[:-1]*recall[:-1]/np.maximum(4*precision[:-1]+recall[:-1],1e-12)
    assert abs(float(th[int(np.argmax(f2))])-sel['risk_threshold'])<1e-12
    p=pd.read_csv(run/'test_predictions.csv');expected_rows=m[m.split=='test']
    assert p.path.tolist()==expected_rows.path.tolist()
    rec=summarize(p.target,p[['p0','p1','p2']].to_numpy(),sel['risk_threshold'])
    assert abs(rec['macro_f1']-final['test']['macro_f1'])<1e-12
    assert rec['confusion']==final['test']['confusion']
    for key in ['tp','fp','tn','fn']:
        assert rec['risk'][key]==final['test']['risk'][key]
    assert sel['manifest_sha256']==digest(PROC/'coal_manifest.csv')
    weight=ROOT/'data/weights/patrick_v1'/f"{sel['run']}.pt"
    assert digest(weight)==sel['weights_sha256']
    evaluated={p.parent.name for p in (RESULTS/'runs').glob('*/test_predictions.csv')}
    assert evaluated=={'M0_dummy',sel['run']},evaluated
    assert len(final['test']['macro_f1_group_bootstrap_95'])==2
    assert final['test']['proxy_groups']==expected_rows.event_group.nunique()
    checks.append('Selección y umbral reproducidos únicamente desde validación; test de M0 y modelo elegido reconstruido')
    if inference:
        import torch
        from modeling import build_model,predict_images,DEVICE
        checkpoint=torch.load(weight,map_location='cpu',weights_only=True)
        model=build_model(False);model.load_state_dict(checkpoint['state_dict']);model.to(DEVICE)
        # Lote que incluye ejemplos de las tres clases, sin tomar decisiones por desempeño.
        subset=pd.concat([g.head(4) for _,g in expected_rows.groupby('target')]).sort_index()
        probs=predict_images(model,subset,checkpoint['config']['variant'],checkpoint['config']['size'])
        stored=p.set_index('path').loc[subset.path,['p0','p1','p2']].to_numpy()
        assert np.allclose(probs,stored,atol=5e-4)
        checks.append('Checkpoint y transformación reconstruyen probabilidades guardadas en una muestra de las tres clases')
    if check_notebook:
        import nbformat
        path=Path(__file__).parent/'entrega_modelado.ipynb'
        nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
        code=[c for c in nb.cells if c.cell_type=='code']
        assert code and all(c.execution_count is not None for c in code)
        assert not any(o.output_type=='error' for c in code for o in c.outputs)
        assert sum(any(o.output_type=='display_data' and 'image/png' in o.get('data',{}) for o in c.outputs) for c in code)>=5
        checks.append(f'Notebook válido, {len(code)} celdas de código ejecutadas sin errores y figuras embebidas')
    report={'status':'passed','checks':checks,'runs_verified':runs,'selected':sel['run'],
            'manifest_sha256':digest(PROC/'coal_manifest.csv'),'notebook_checked':check_notebook,
            'raw_sha256_rechecked':check_raw,'inference_rechecked':inference}
    write_json(RESULTS/'verification.json',report)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',action='store_true');p.add_argument('--inference',action='store_true');p.add_argument('--without-notebook',action='store_true')
    args=p.parse_args();verify(args.raw,not args.without_notebook,args.inference)
