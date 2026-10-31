"""Inventario, EDA y particiones de la entrega. Solo lee data/raw."""
from pathlib import Path
import hashlib
import json
import re
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import imagehash
from scipy.ndimage import laplace
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
VERSION = "patrick_v1"
PROC = ROOT / "data/processed" / VERSION
RESULTS = ROOT / "data/results" / VERSION
FIG = RESULTS / "figures"
CLASSES = ["Normal Coal Flow", "Oversized Coal", "Rock Bolts"]
SPANISH = ["Normal", "Bloque grande", "Perno"]
SEED = 42


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def setup():
    for p in [PROC, RESULTS, FIG]:
        p.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10})


def inspect_image(path):
    with Image.open(path) as im:
        im.load()
        rgb = im.convert("RGB")
        a = np.asarray(rgb.resize((160, 160)), dtype=np.float32)
        g = np.asarray(rgb.convert("L").resize((160, 160)), dtype=np.float32)
        color = np.asarray(rgb, dtype=np.float32)
        yellow = ((color[:, :, 0] > 100) & (color[:, :, 1] > 90) &
                  (color[:, :, 2] < .65 * np.minimum(color[:, :, 0], color[:, :, 1])))
        # Proxy, no OCR: contar píxeles amarillos en las bandas de overlays.
        h = rgb.height
        yellow[round(.24*h):round(.78*h)] = False
        return {"gray_mean": float(g.mean()), "gray_std": float(g.std()),
                "sharpness": float(laplace(g).var()), "dark_fraction": float((g < 25).mean()),
                "bright_fraction": float((g > 230).mean()),
                "saturation_proxy": float((a.max(2)-a.min(2)).mean()),
                "yellow_pixels": int(yellow.sum()), "overlay_proxy": int(yellow.sum() >= 100),
                "phash": str(imagehash.phash(rgb)), "dhash": str(imagehash.dhash(rgb))}


def inventory():
    """Reutiliza metadatos previos solo tras revalidar rutas, bytes y SHA-256."""
    setup()
    target = PROC / "inventory.csv"
    if target.exists():
        old = pd.read_csv(target)
        paths = sorted(p for p in RAW.rglob("*") if p.is_file())
        assert len(paths) == len(old), "Cambió el inventario; usar otra versión."
        for r in old.itertuples():
            p = RAW / r.path
            assert p.stat().st_size == r.bytes and digest(p) == r.sha256, r.path
        print("Inventario existente revalidado por SHA-256.", flush=True)
        return old
    prior = RAW.parent / "processed/auditoria_inicial/inventory.json"
    previous = {r["path"]: r for r in json.loads(prior.read_text())} if prior.exists() else {}
    files = sorted(p for p in RAW.rglob("*") if p.is_file())
    rows = []
    for i, p in enumerate(files):
        rel = p.relative_to(RAW).as_posix()
        sha = digest(p)
        old = previous.get(rel, {})
        if old:
            assert old["sha256"] == sha and old["bytes"] == p.stat().st_size, rel
        with Image.open(p) as im:
            row = dict(path=rel, dataset=rel.split("/")[0], group=str(p.parent.relative_to(RAW)),
                       bytes=p.stat().st_size, sha256=sha, width=im.width, height=im.height,
                       mode=im.mode, format=im.format)
        if row["dataset"] == "coal":
            row.update(original_split=rel.split("/")[1], label=p.parent.name,
                       target=CLASSES.index(p.parent.name))
        else:
            row.update(original_split="unassigned", label="unlabeled", target=-1)
        rows.append(row)
        if i % 4000 == 0:
            print(f"Inventario/hash: {i}/{len(files)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(target, index=False)
    return df


def quality(df):
    target = PROC / "quality.csv"
    if target.exists():
        return pd.read_csv(target, dtype={"phash": str, "dhash": str})
    coal = df[df.dataset == "coal"]
    samples = [coal]
    for _, group in df[df.dataset == "iron"].groupby("group"):
        g = group.copy()
        g["frame"] = g.path.str.extract(r"/(\d+)-", expand=False).astype(int)
        g = g.sort_values("frame")
        samples.append(g.iloc[np.linspace(0, len(g)-1, min(24, len(g))).astype(int)])
    sample = pd.concat(samples, ignore_index=True)
    rows = []
    for i, r in enumerate(sample.to_dict("records")):
        r.update(inspect_image(RAW / r["path"]))
        rows.append(r)
        if i % 500 == 0:
            print(f"Calidad/decodificación: {i}/{len(sample)}", flush=True)
    q = pd.DataFrame(rows)
    q.to_csv(target, index=False)
    return q


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def prepare_coal(df, q):
    path = PROC / "coal_manifest.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"phash": str, "dhash": str})
    c = q[q.dataset == "coal"].copy()
    nlabels = c.groupby("sha256").target.nunique()
    conflicting = set(nlabels[nlabels > 1].index)
    c["decision"] = np.where(c.sha256.isin(conflicting), "exclude_conflicting_label", "candidate")
    c.loc[(c.decision == "candidate") & c.duplicated("sha256"), "decision"] = "exclude_exact_copy"
    c[["path", "sha256", "label", "original_split", "decision"]].to_csv(PROC / "annotation_decisions.csv", index=False)
    clean = c[c.decision == "candidate"].sort_values("path").reset_index(drop=True)
    assert len(clean) == clean.sha256.nunique()
    # Los hashes globales confunden fondos estáticos con objetos iguales. Confirmar
    # candidatos con distancia de píxeles dentro de la cinta, no solo dos hashes.
    bits = np.array([[int(bit) for bit in f"{int(s, 16):064b}"] for s in clean.phash], dtype=np.uint8)
    neighbors = NearestNeighbors(radius=4/64 + 1e-8, metric="hamming", algorithm="brute", n_jobs=2).fit(bits)
    candidates = neighbors.radius_neighbors(bits, return_distance=False)
    uf = UnionFind(len(clean))
    edges = []
    phashes = [int(s,16) for s in clean.phash]
    dhashes = [int(s,16) for s in clean.dhash]
    paths = clean.path.tolist(); targets = clean.target.tolist()
    crops = []
    for p in paths:
        with Image.open(RAW / p) as im:
            small = np.asarray(im.convert("L").resize((64,64)),dtype=np.float32)
        crops.append(small[12:60,15:53])
    crops = np.stack(crops)
    for i, js in enumerate(candidates):
        for j in js:
            j = int(j)
            if j <= i:
                continue
            pdist = (phashes[i] ^ phashes[j]).bit_count()
            ddist = (dhashes[i] ^ dhashes[j]).bit_count()
            if ddist <= 4:
                diff = crops[i]-crops[j]
                mae = float(np.abs(diff).mean()); rmse = float(np.sqrt((diff**2).mean()))
                if mae > 2.5 or rmse > 6:
                    continue
                uf.union(i, j)
                edges.append(dict(path_a=paths[i], path_b=paths[j], phash_distance=pdist,
                                  dhash_distance=ddist, roi_mae=mae, roi_rmse=rmse, same_label=bool(targets[i] == targets[j])))
    pd.DataFrame(edges, columns=["path_a", "path_b", "phash_distance", "dhash_distance", "roi_mae", "roi_rmse", "same_label"]).to_csv(PROC / "near_duplicate_edges.csv", index=False)
    clean["visual_group"] = [f"visual_{uf.find(i):05}" for i in range(len(clean))]
    # La hora es un proxy conservador: faltan fechas. Agrupar sin clase ni split.
    # Algunos nombres terminan en ' - 副本.jpg'; extraer sin exigir sufijo exacto.
    times = clean.path.str.extract(r"_(\d{6})_\d+", expand=False)
    assert times.notna().all(), "Hay nombres sin hora candidata; revisar antes de dividir."
    seconds = times.map(lambda t: int(t[:2])*3600+int(t[2:4])*60+int(t[4:]))
    clean["time_bin_5min"] = seconds // 300
    for _, indexes in clean.groupby("time_bin_5min").groups.items():
        indexes = list(indexes)
        for i in indexes[1:]:
            uf.union(indexes[0], i)
    clean["event_group"] = [f"proxy_{uf.find(i):05}" for i in range(len(clean))]
    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    clean["fold"] = -1
    for fold, (_, ix) in enumerate(splitter.split(clean, clean.target, clean.event_group)):
        clean.loc[ix, "fold"] = fold
    clean["split"] = np.where(clean.fold == 0, "test", np.where(clean.fold == 1, "val", "train"))
    print("Soporte preliminar:",pd.crosstab(clean.split,clean.label).to_string(),flush=True)
    print("Grupos:",clean.event_group.nunique(),"mayores:",clean.event_group.value_counts().head().to_dict(),flush=True)
    for field in ["sha256", "visual_group", "event_group", "time_bin_5min"]:
        assert clean.groupby(field).split.nunique().max() == 1, field
    assert clean.groupby("split").target.nunique().min() == 3
    clean.to_csv(path, index=False)
    (PROC / "splits").mkdir(exist_ok=True)
    for split, g in clean.groupby("split"):
        g.to_csv(PROC / "splits" / f"{split}.csv", index=False)
    protocol = {"seed": SEED, "classes": CLASSES, "clean_unique": len(clean),
                "conflicting_contents_excluded": len(conflicting), "near_edges": len(edges),
                "visual_groups": clean.visual_group.nunique(), "proxy_groups": clean.event_group.nunique(),
                "largest_proxy_group": int(clean.event_group.value_counts().max()),
                "split_sha256": digest(path), "split_counts": clean.groupby("split").size().to_dict(),
                "split_class_counts": pd.crosstab(clean.split, clean.label).to_dict(),
                "split_group_counts": clean.groupby("split").event_group.nunique().to_dict(),
                "method": "SGKF7 seed42; fold0 test, fold1 val, remaining train; union of pHash<=4 AND dHash<=4 AND central ROI MAE<=2.5/RMSE<=6 (64x64 gray), plus HHMMSS five-minute bins across all classes/splits",
                "limitations": ["Sin fechas/videos originales; grupos horarios y visuales son proxies, no eventos confirmados.",
                                "Agrupación visual puede unir escenas distintas y no detecta toda similitud semántica.",
                                "Nuevo benchmark; no comparable directamente con el test original.",
                                "Etiquetas heredadas no certificadas por especialista; conflictos exactos excluidos."]}
    write_json(PROC / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2, ensure_ascii=False), flush=True)
    return clean


def gallery(rows, destination, title, cols=4, labels=None, transform=None):
    rows = list(rows)
    if not rows:
        return
    nrows = (len(rows)+cols-1)//cols
    fig, axes = plt.subplots(nrows, cols, figsize=(cols*3.1, nrows*3.2), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= len(rows):
            continue
        r = rows[i]
        with Image.open(RAW / r["path"]) as im:
            im = im.convert("RGB")
            if transform:
                im = transform(im)
            ax.imshow(im)
        identifier = re.sub(r"[^a-zA-Z0-9_\-]", "", Path(r["path"]).stem)[-24:]
        text = labels[i] if labels is not None else f"{r.get('label', '')}\n{identifier}"
        ax.set_title(text, fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def eda(df, q, clean):
    setup()
    counts = df.groupby(["dataset", "group"]).size().rename("images").reset_index()
    counts.to_csv(RESULTS / "dataset_counts.csv", index=False)
    metrics = ["gray_mean", "gray_std", "sharpness", "dark_fraction", "bright_fraction", "bytes"]
    q.groupby(["dataset", "label"])[metrics].describe().to_csv(RESULTS / "descriptive_summary.csv")
    q[metrics].corr().to_csv(RESULTS / "quality_correlations.csv")
    plt.figure(figsize=(9, 4))
    pd.crosstab(clean.split, clean.label).reindex(["train", "val", "test"]).rename(columns=dict(zip(CLASSES, SPANISH))).plot.bar(ax=plt.gca())
    plt.ylabel("Imágenes únicas"); plt.xlabel("Partición agrupada"); plt.xticks(rotation=0)
    plt.title("Coal después de sanear: soporte por clase y partición")
    plt.tight_layout(); plt.savefig(FIG / "split_counts.png"); plt.close()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ds, group in q.groupby("dataset"):
        axes[0].hist(group.gray_mean, bins=35, alpha=.55, density=True, label=f"{ds} (n={len(group)})")
        axes[1].scatter(group.gray_mean, np.log1p(group.sharpness), alpha=.18, s=5, label=ds)
    axes[0].set(xlabel="Brillo medio [0,255]", ylabel="Densidad"); axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Brillo medio", ylabel="log(1 + varianza Laplaciano)")
    data = [q.loc[(q.dataset == "coal") & (q.label == c), "gray_mean"] for c in CLASSES]
    axes[2].boxplot(data, tick_labels=SPANISH)
    axes[2].set(ylabel="Brillo medio (coal)")
    fig.suptitle("Calidad: coal completo; iron muestra sistemática de 24 por carpeta")
    fig.tight_layout(); fig.savefig(FIG / "quality.png"); plt.close(fig)
    dev = clean[clean.split != "test"]
    overlay = pd.crosstab(dev.label, dev.overlay_proxy).reindex(columns=[0, 1], fill_value=0)
    overlay.to_csv(RESULTS / "overlay_development.csv")
    overlay.div(overlay.sum(axis=1), axis=0).rename(index=dict(zip(CLASSES, SPANISH)), columns={0:"Sin proxy amarillo",1:"Con proxy amarillo"}).plot.barh(stacked=True, figsize=(8, 3.4))
    plt.xlabel("Proporción en desarrollo (train + val)"); plt.xlim(0,1)
    plt.title("Asociación de clase con texto amarillo: proxy, no OCR")
    plt.tight_layout(); plt.savefig(FIG / "overlay_association.png"); plt.close()
    coal_sample = pd.concat([g.sample(min(2,len(g)),random_state=SEED) for _,g in dev.groupby(["label", "overlay_proxy"])])
    gallery(coal_sample.to_dict("records"), FIG / "coal_gallery.jpg", "Coal: variedad de clase y presencia de overlay", cols=4)
    iron = []
    for _, g in q[q.dataset == "iron"].groupby("group"):
        g = g.copy(); g["frame"] = g.path.str.extract(r"/(\d+)-", expand=False).astype(int)
        g = g.sort_values("frame")
        iron.extend(g.iloc[[0,len(g)//2,len(g)-1]].to_dict("records"))
    gallery(iron, FIG / "iron_gallery.jpg", "Iron: inicio, centro y final por carpeta; no son etiquetas", cols=3,
            labels=[Path(r["group"]).name.replace("boardcamera_iron_ore", "iron")+"\n"+Path(r["path"]).name.split("-")[0] for r in iron])
    # Muestra de candidatos de similitud, no una revisión exhaustiva.
    edges = pd.read_csv(PROC / "near_duplicate_edges.csv")
    if len(edges):
        subset = edges.sample(min(8, len(edges)), random_state=SEED)
        lookup = clean.set_index("path").to_dict("index")
        items, titles = [], []
        for e in subset.itertuples():
            for p in [e.path_a, e.path_b]:
                items.append(dict(lookup[p], path=p))
                titles.append(f"pHash={e.phash_distance}, dHash={e.dhash_distance}\n{lookup[p]['label']}")
        gallery(items, FIG / "near_duplicates.jpg", "Pares similares agrupados: revisar como evidencia, no relabelar", cols=4, labels=titles)
    write_json(RESULTS / "eda_summary.json", {"inventory_count":len(df), "decoded_coal":int((q.dataset=="coal").sum()),
                "sampled_iron":int((q.dataset=="iron").sum()), "class_counts_raw":df[df.dataset=="coal"].label.value_counts().to_dict(),
                "descriptive_sample": "Coal completo; iron 24 índices equiespaciados por carpeta (no prevalencia independiente)."})


def main():
    df = inventory()
    q = quality(df)
    clean = prepare_coal(df, q)
    eda(df, q, clean)
    print("Preparación y EDA completas:", PROC, RESULTS, flush=True)


if __name__ == "__main__":
    main()
