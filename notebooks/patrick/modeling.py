"""Experimentos ejecutables. Selección con validación; test sellado por separado."""
import argparse
import copy
import importlib.metadata
import json
import platform
import random
import subprocess
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, ImageDraw
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, classification_report,
                             confusion_matrix, average_precision_score, precision_recall_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from study_data import ROOT, RAW, PROC, RESULTS, FIG, CLASSES, SPANISH, SEED, digest, write_json, gallery

WEIGHTS = ROOT / "data/weights/patrick_v1"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MEAN, STD = [0.485, .456, .406], [.229, .224, .225]


def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mask_overlay(im):
    """Máscara fija independiente de etiqueta; comparar pérdida de escena explícitamente."""
    im = im.copy()
    w, h = im.size
    d = ImageDraw.Draw(im)
    # Coordenadas normalizadas desde la revisión de desarrollo; incluye margen de texto.
    for box in [(0, 0, .86, .11), (0, .11, .46, .215), (0, .89, .35, 1)]:
        d.rectangle(tuple(round(v*(w if j % 2 == 0 else h)) for j, v in enumerate(box)), fill=(0, 0, 0))
    return im


class CoalImages(Dataset):
    def __init__(self, frame, variant="raw", size=224, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.variant = variant
        ops = [transforms.Resize((size, size))]
        if augment:
            ops += [transforms.ColorJitter(brightness=.15, contrast=.15)]
        ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        r = self.frame.iloc[index]
        with Image.open(RAW / r.path) as im:
            im = im.convert("RGB")
            if self.variant == "masked":
                im = mask_overlay(im)
            x = self.transform(im)
        return x, int(r.target)


def manifest():
    df = pd.read_csv(PROC / "coal_manifest.csv")
    assert df.sha256.nunique() == len(df)
    for key in ["event_group", "visual_group", "time_bin_5min", "sha256"]:
        assert df.groupby(key).split.nunique().max() == 1
    return df


def metadata(config):
    git = lambda *args: subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout
    sources = {p.name: digest(p) for p in [Path(__file__), Path(__file__).parent / "study_data.py"]}
    sources["requirements.txt"] = digest(Path(__file__).parent / "requirements.txt")
    kind=config.get("kind", "")
    devices = {"optimizer":"cpu", "features":DEVICE} if kind=="frozen_head" else {"model":"cpu" if kind in ["dummy","quality_logistic"] else DEVICE}
    return dict(config=config, device="cpu" if kind in ["dummy","quality_logistic"] else DEVICE,
                available_accelerator=DEVICE, compute_devices=devices, recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                platform=platform.platform(), python=platform.python_version(),
                packages={p:importlib.metadata.version(p) for p in ["torch", "torchvision", "numpy", "pandas", "scikit-learn", "Pillow"]},
                code_commit=git("rev-parse", "HEAD").strip(), code_dirty=bool(git("status", "--porcelain").strip()),
                source_hashes=sources, manifest_sha256=digest(PROC / "coal_manifest.csv"))


def summarize(y, probs, threshold=.5):
    y = np.asarray(y, dtype=int)
    probs = np.asarray(probs)
    pred = probs.argmax(1)
    risk_true = y != 0
    risk_pred = probs[:, 1:].sum(1) >= threshold
    tp = int((risk_true & risk_pred).sum()); fn = int((risk_true & ~risk_pred).sum())
    fp = int((~risk_true & risk_pred).sum()); tn = int((~risk_true & ~risk_pred).sum())
    return dict(n=len(y), accuracy=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, labels=[0,1,2], average="macro", zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                report=classification_report(y, pred, labels=[0,1,2], target_names=CLASSES, zero_division=0, output_dict=True),
                confusion=confusion_matrix(y, pred, labels=[0,1,2]).tolist(),
                risk=dict(threshold=float(threshold), recall=tp/max(tp+fn,1), precision=tp/max(tp+fp,1),
                          false_alarm_rate=fp/max(fp+tn,1), average_precision=float(average_precision_score(risk_true, probs[:,1:].sum(1))),
                          tp=tp,fn=fn,fp=fp,tn=tn))


def save_predictions(frame, probs, path):
    cols = ["path", "sha256", "target", "label", "event_group", "split", "overlay_proxy", "gray_mean", "sharpness"]
    out = frame[cols].copy()
    for j in range(3):
        out[f"p{j}"] = probs[:, j]
    out["prediction"] = probs.argmax(1)
    out["confidence"] = probs.max(1)
    out["correct"] = out.target == out.prediction
    out.to_csv(path, index=False)
    return out


def run_directory(name):
    path = RESULTS / "runs" / name
    if path.exists():
        raise FileExistsError(f"La ejecución {name} existe. Use otro nombre para preservar resultados.")
    path.mkdir(parents=True)
    snapshot=path/"source_snapshot";snapshot.mkdir()
    for name in ["modeling.py","study_data.py","requirements.txt"]:
        shutil.copy2(Path(__file__).parent/name,snapshot/name)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    return path


def finish(path, config, frame, train_probs, val_probs, history=None, elapsed=0):
    train = frame[frame.split == "train"]
    val = frame[frame.split == "val"]
    metrics = {"train":summarize(train.target, train_probs), "val":summarize(val.target, val_probs), "elapsed_seconds":elapsed}
    write_json(path / "metadata.json", metadata(config))
    write_json(path / "metrics.json", metrics)
    for split, group, probs in [("train", train, train_probs), ("val", val, val_probs)]:
        save_predictions(group, probs, path / f"{split}_predictions.csv")
    if history:
        pd.DataFrame(history).to_csv(path / "history.csv", index=False)
    diagnostics(path, val, val_probs, "val")
    print(json.dumps(dict(name=path.name, train_f1=metrics["train"]["macro_f1"], val_f1=metrics["val"]["macro_f1"],
                          val_balanced_accuracy=metrics["val"]["balanced_accuracy"], seconds=elapsed), indent=2), flush=True)


def diagnostics(path, frame, probs, split):
    predictions = pd.read_csv(path / f"{split}_predictions.csv")
    cm = confusion_matrix(frame.target, probs.argmax(1), labels=[0,1,2])
    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()*.55 else "black")
    ax.set(xticks=range(3),yticks=range(3),xticklabels=SPANISH,yticklabels=SPANISH,
           xlabel="Predicción",ylabel="Etiqueta heredada",title=f"{path.name} · {split}")
    fig.tight_layout();fig.savefig(path / f"{split}_confusion.png");plt.close(fig)
    errors = predictions[~predictions.correct].sort_values("confidence", ascending=False)
    # Balancear clases en la galería sin perder los errores de alta confianza.
    examples = errors.groupby("target", group_keys=False).head(4).sort_values("confidence", ascending=False).head(12)
    captions = [f"Etiqueta: {SPANISH[int(r.target)]}\nPred.: {SPANISH[int(r.prediction)]} · p={r.confidence:.2f}" for r in examples.itertuples()]
    cfg=json.loads((path/"metadata.json").read_text())["config"]
    variant=cfg.get("variant","raw")
    gallery(examples.to_dict("records"), path / f"{split}_errors.jpg", f"{path.name}: errores por clase · entrada {variant}",
            labels=captions,transform=mask_overlay if variant=="masked" else None)
    if not len(examples):
        (path / f"{split}_no_errors.txt").write_text("No hay errores en esta partición; esto no prueba generalización externa.\n")
    slices = []
    for flag, g in predictions.groupby("overlay_proxy"):
        slices.append(dict(overlay_proxy=int(flag), n=len(g), accuracy=float(g.correct.mean()),
                           macro_f1=float(f1_score(g.target,g.prediction,labels=[0,1,2],average="macro",zero_division=0))))
    pd.DataFrame(slices).to_csv(path / f"{split}_overlay_slices.csv", index=False)


def baselines():
    seed_all(); df=manifest(); train=df[df.split=="train"]; val=df[df.split=="val"]
    path=run_directory("M0_dummy")
    clf=DummyClassifier(strategy="prior",random_state=SEED).fit(np.zeros((len(train),1)),train.target)
    finish(path,dict(kind="dummy",strategy="prior",seed=SEED),df,
           clf.predict_proba(np.zeros((len(train),1))),clf.predict_proba(np.zeros((len(val),1))))
    path=run_directory("M1_quality")
    features=["gray_mean","gray_std","sharpness","dark_fraction","bright_fraction","overlay_proxy"]
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=SEED))
    t=time.perf_counter();clf.fit(train[features],train.target)
    finish(path,dict(kind="quality_logistic",features=features,C=1,seed=SEED),df,
           clf.predict_proba(train[features]),clf.predict_proba(val[features]),elapsed=time.perf_counter()-t)


def build_model(pretrained=True):
    model=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc=nn.Linear(model.fc.in_features,3)
    return model


def features(frame, variant, size, splitset="development"):
    path=PROC / f"features_{variant}_{size}_{splitset}.npz"
    if path.exists():
        z=np.load(path)
        assert list(z["paths"])==frame.path.tolist()
        assert str(z["manifest_sha256"])==digest(PROC / "coal_manifest.csv")
        return z["features"]
    model=build_model();model.fc=nn.Identity();model.to(DEVICE).eval()
    loader=DataLoader(CoalImages(frame,variant,size),batch_size=48,shuffle=False,num_workers=0)
    collected=[];t=time.perf_counter()
    with torch.inference_mode():
        for i,(x,_) in enumerate(loader):
            collected.append(model(x.to(DEVICE)).cpu().numpy())
            if i % 20==0:
                print(f"Features {variant} {size}: {i}/{len(loader)} · {time.perf_counter()-t:.1f}s",flush=True)
    a=np.concatenate(collected)
    np.savez_compressed(path,features=a,paths=frame.path.to_numpy(dtype=str),manifest_sha256=digest(PROC / "coal_manifest.csv"))
    del model
    if DEVICE=="mps":torch.mps.empty_cache()
    return a


def predict_head(head,x):
    with torch.inference_mode():
        return head(torch.from_numpy(x)).softmax(1).numpy()


def head_experiment(name,variant="raw",size=224,epochs=80,lr=.003,decay=.001,seed=SEED):
    seed_all(seed); path=run_directory(name); df=manifest();dev=df[df.split!="test"]
    config=dict(kind="frozen_head",variant=variant,size=size,epochs=epochs,lr=lr,weight_decay=decay,seed=seed,
                batch_size=256,pretrained="ResNet18_Weights.IMAGENET1K_V1",normalization=dict(mean=MEAN,std=STD),
                early_stopping_patience=15,selection="maximum val macro-F1, earliest tie")
    write_json(path / "metadata.json",metadata(config))
    t=time.perf_counter();a=features(dev,variant,size)
    tr=dev.split.to_numpy()=="train";va=dev.split.to_numpy()=="val"
    tx=torch.from_numpy(a[tr]);ty=torch.tensor(dev.target.to_numpy()[tr],dtype=torch.long)
    head=nn.Linear(512,3)
    loader=DataLoader(TensorDataset(tx,ty),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed))
    optim=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=decay)
    criterion=nn.CrossEntropyLoss();history=[];best=-1;stale=0;best_state=None
    for epoch in range(1,epochs+1):
        head.train();loss_sum=0
        for x,y in loader:
            optim.zero_grad();loss=criterion(head(x),y);loss.backward();optim.step();loss_sum+=loss.item()*len(y)
        head.eval();tp=predict_head(head,a[tr]);vp=predict_head(head,a[va])
        tm=summarize(dev.target.to_numpy()[tr],tp);vm=summarize(dev.target.to_numpy()[va],vp)
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(ty),train_f1=tm["macro_f1"],val_f1=vm["macro_f1"],
                            val_loss=float(criterion(head(torch.from_numpy(a[va])),torch.tensor(dev.target.to_numpy()[va],dtype=torch.long)).detach())))
        if vm["macro_f1"]>best+1e-6:
            best=vm["macro_f1"];best_state=copy.deepcopy(head.state_dict());stale=0;config["best_epoch"]=epoch
        else:stale+=1
        if epoch % 10==0:print(name,epoch,"train",round(tm["macro_f1"],4),"val",round(vm["macro_f1"],4),flush=True)
        if stale>=15:break
    head.load_state_dict(best_state);model=build_model();model.fc.load_state_dict(best_state)
    weight=WEIGHTS / f"{name}.pt"
    torch.save(dict(state_dict=model.state_dict(),config=config,classes=CLASSES,manifest_sha256=digest(PROC / "coal_manifest.csv")),weight)
    config["weights_relative"]=str(weight.relative_to(ROOT));config["weights_sha256"]=digest(weight)
    finish(path,config,df,predict_head(head,a[tr]),predict_head(head,a[va]),history,time.perf_counter()-t)


def predict_images(model,frame,variant,size):
    loader=DataLoader(CoalImages(frame,variant,size),batch_size=48,shuffle=False,num_workers=0)
    model.eval();out=[]
    with torch.inference_mode():
        for x,_ in loader:out.append(model(x.to(DEVICE)).softmax(1).cpu().numpy())
    return np.concatenate(out)


def finetune(name,parent,epochs=8,lr=.0001,augment=False,seed=SEED):
    seed_all(seed);path=run_directory(name);df=manifest();tr=df[df.split=="train"];va=df[df.split=="val"]
    checkpoint=torch.load(WEIGHTS/f"{parent}.pt",map_location="cpu",weights_only=True)
    config=dict(checkpoint["config"],kind="finetune_layer4",parent=parent,epochs=epochs,lr=lr,augment=augment,
                seed=seed,early_stopping_patience=3,batch_size=24)
    variant=config["variant"];size=config["size"]
    model=build_model(False);model.load_state_dict(checkpoint["state_dict"])
    for p in model.parameters():p.requires_grad=False
    for module in [model.layer4,model.fc]:
        for p in module.parameters():p.requires_grad=True
    model.to(DEVICE)
    optim=torch.optim.AdamW([{"params":model.layer4.parameters(),"lr":lr},{"params":model.fc.parameters(),"lr":lr*5}],weight_decay=.001)
    loader=DataLoader(CoalImages(tr,variant,size,augment),batch_size=24,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed))
    criterion=nn.CrossEntropyLoss();history=[];best=-1;stale=0;t=time.perf_counter()
    write_json(path/"metadata.json",metadata(config))
    for epoch in range(1,epochs+1):
        # BN permanece congelado (incluido layer4) para batches pequeños y comparabilidad.
        model.eval();loss_sum=0
        for i,(x,y) in enumerate(loader):
            x=x.to(DEVICE);y=y.to(DEVICE);optim.zero_grad()
            loss=criterion(model(x),y);loss.backward();optim.step();loss_sum+=loss.item()*len(y)
            if i%40==0:print(name,"epoch",epoch,"batch",i,"/",len(loader),flush=True)
        tp=predict_images(model,tr,variant,size);vp=predict_images(model,va,variant,size)
        tm=summarize(tr.target,tp);vm=summarize(va.target,vp)
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(tr),train_f1=tm["macro_f1"],val_f1=vm["macro_f1"],
                            val_loss=float(-np.log(np.maximum(vp[np.arange(len(va)),va.target.to_numpy()],1e-9)).mean())))
        pd.DataFrame(history).to_csv(path/"history.csv",index=False)
        print(name,"epoch",epoch,"train",tm["macro_f1"],"val",vm["macro_f1"],"seconds",time.perf_counter()-t,flush=True)
        if vm["macro_f1"]>best+1e-6:
            best=vm["macro_f1"];best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0;config["best_epoch"]=epoch
            torch.save(dict(state_dict=best_state,config=config,classes=CLASSES,manifest_sha256=digest(PROC / "coal_manifest.csv")),WEIGHTS/f"{name}.pt")
        else:stale+=1
        if stale>=3:break
    model.load_state_dict(best_state);weight=WEIGHTS/f"{name}.pt"
    config["weights_relative"]=str(weight.relative_to(ROOT));config["weights_sha256"]=digest(weight)
    finish(path,config,df,predict_images(model,tr,variant,size),predict_images(model,va,variant,size),history,time.perf_counter()-t)


def validation_table():
    rows=[]
    for p in sorted((RESULTS/"runs").glob("*/metrics.json")):
        m=json.loads(p.read_text());cfg=json.loads((p.parent/"metadata.json").read_text())["config"]
        rows.append(dict(run=p.parent.name,kind=cfg["kind"],variant=cfg.get("variant","quality/prior"),
                         train_f1=m["train"]["macro_f1"],val_f1=m["val"]["macro_f1"],
                         val_balanced_accuracy=m["val"]["balanced_accuracy"],val_risk_recall=m["val"]["risk"]["recall"],
                         val_false_alarm_rate=m["val"]["risk"]["false_alarm_rate"],seconds=m["elapsed_seconds"]))
    table=pd.DataFrame(rows)
    table.to_csv(RESULTS/"validation_comparison.csv",index=False)
    return table


def select_and_test():
    """Ejecutar solo al cerrar iteraciones; un sello impide selección retroactiva."""
    seed_all();seal=RESULTS/"final_test.json"
    if seal.exists():
        raise FileExistsError("Test ya evaluado. No reoptimizar ni sobrescribir resultados.")
    table=validation_table();candidates=table[table.kind.isin(["frozen_head","finetune_layer4"])]
    best=candidates.sort_values(["val_f1","run"],ascending=[False,True]).iloc[0]
    name=best.run;path=RESULTS/"runs"/name
    vp=pd.read_csv(path/"val_predictions.csv");scores=vp[["p1","p2"]].sum(1).to_numpy();truth=vp.target.to_numpy()!=0
    prec,rec,th=precision_recall_curve(truth,scores)
    f2=5*prec[:-1]*rec[:-1]/np.maximum(4*prec[:-1]+rec[:-1],1e-12)
    threshold=float(th[int(np.argmax(f2))])
    selection=dict(run=name,criterion="highest validation macro-F1; deterministic lexical tie",val_macro_f1=float(best.val_f1),
                   risk_threshold=threshold,threshold_criterion="maximum F2 on validation; exploratory, no operational cost model",
                   validation_f2=float(f2.max()),manifest_sha256=digest(PROC/"coal_manifest.csv"),
                   weights_sha256=digest(WEIGHTS/f"{name}.pt"),candidate_runs=table.run.tolist())
    write_json(RESULTS/"selection.json",selection)
    df=manifest();test=df[df.split=="test"];checkpoint=torch.load(WEIGHTS/f"{name}.pt",map_location="cpu",weights_only=True)
    cfg=checkpoint["config"];model=build_model(False);model.load_state_dict(checkpoint["state_dict"]);model.to(DEVICE)
    start=time.perf_counter();probs=predict_images(model,test,cfg["variant"],cfg["size"]);elapsed=time.perf_counter()-start
    save_predictions(test,probs,path/"test_predictions.csv")
    diagnostics(path,test,probs,"test")
    metrics=summarize(test.target,probs,threshold)
    # Incertidumbre por grupos proxy: remuestrear bloques, no imágenes independientes.
    rng=np.random.default_rng(SEED);groups=test.event_group.unique();ix={g:np.flatnonzero(test.event_group.to_numpy()==g) for g in groups}
    boot=[]
    for _ in range(500):
        indexes=np.concatenate([ix[g] for g in rng.choice(groups,len(groups),replace=True)])
        if len(np.unique(test.target.to_numpy()[indexes]))==3:
            boot.append(f1_score(test.target.to_numpy()[indexes],probs[indexes].argmax(1),labels=[0,1,2],average="macro",zero_division=0))
    metrics["macro_f1_group_bootstrap_95"]=np.quantile(boot,[.025,.975]).tolist()
    metrics["bootstrap_valid_replicates"]=len(boot);metrics["proxy_groups"]=len(groups)
    train=df[df.split=="train"];prior=train.target.value_counts(normalize=True).reindex(range(3),fill_value=0).to_numpy()
    dummy_probs=np.tile(prior,(len(test),1))
    save_predictions(test,dummy_probs,RESULTS/"runs/M0_dummy/test_predictions.csv")
    write_json(seal,dict(selection=selection,test=metrics,dummy_test=summarize(test.target,dummy_probs),
                         inference=dict(seconds=elapsed,n=len(test),batch_size=48,device=DEVICE,includes_io=True),
                         interpretation="Evaluación interna agrupada mediante proxies, no generalización a otra planta."))
    print(json.dumps(json.loads(seal.read_text()),indent=2,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=["baselines","head","finetune","table","final"])
    parser.add_argument("--name");parser.add_argument("--parent");parser.add_argument("--variant",default="raw",choices=["raw","masked"])
    parser.add_argument("--size",type=int,default=224);parser.add_argument("--epochs",type=int);parser.add_argument("--lr",type=float)
    parser.add_argument("--augment",action="store_true");parser.add_argument("--seed",type=int,default=SEED)
    args=parser.parse_args()
    if args.action=="baselines":baselines()
    elif args.action=="head":head_experiment(args.name,args.variant,args.size,args.epochs or 80,args.lr or .003,seed=args.seed)
    elif args.action=="finetune":finetune(args.name,args.parent,args.epochs or 8,args.lr or .0001,args.augment,args.seed)
    elif args.action=="table":print(validation_table().to_string(index=False))
    else:select_and_test()


if __name__=="__main__":
    main()
