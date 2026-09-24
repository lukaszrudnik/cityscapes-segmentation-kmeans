import os
import csv
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: batch run, no display
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- paths
DATA_DIR = "data"
RESULTS_DIR = "results"
DEMO_DIR = os.path.join(RESULTS_DIR, "demo")


def ensure_dirs():
    for d in (RESULTS_DIR, DEMO_DIR):
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------- default config
DEFAULT_CONFIG = {
    "split": "train",
    "city": "cologne",
    "image_names": [],            # empty -> first num_images frames of the city
    "num_images": 50,
    "resize": (512, 256),         # (W, H); None = full resolution
    "demo_k_values": [8, 12, 16],
    "demo_lambda_values": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    "color_space": "rgb",         # "rgb" or "lab"
    "minibatch": True,
    "n_init": 5,
    "random_state": 42,
    "ignore_ids": (0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30),
}


# ---------------------------------------------------------------- Cityscapes classes
CITYSCAPES_NAMES = {
    7: "road", 8: "sidewalk", 11: "building", 12: "wall", 13: "fence", 17: "pole",
    19: "traffic light", 20: "traffic sign", 21: "vegetation", 22: "terrain", 23: "sky",
    24: "person", 25: "rider", 26: "car", 27: "truck", 28: "bus", 31: "train",
    32: "motorcycle", 33: "bicycle",
}
CITYSCAPES_COLORS = {
    7: (128, 64, 128), 8: (244, 35, 232), 11: (70, 70, 70), 12: (102, 102, 156),
    13: (190, 153, 153), 17: (153, 153, 153), 19: (250, 170, 30), 20: (220, 220, 0),
    21: (107, 142, 35), 22: (152, 251, 152), 23: (70, 130, 180), 24: (220, 20, 60),
    25: (255, 0, 0), 26: (0, 0, 142), 27: (0, 0, 70), 28: (0, 60, 100),
    31: (0, 80, 100), 32: (0, 0, 230), 33: (119, 11, 32),
}


# ---------------------------------------------------------------- data loading
def find_pairs(split, city, num_images, image_names=None):
    img_dir = os.path.join(DATA_DIR, "leftImg8bit", split, city)
    gt_dir = os.path.join(DATA_DIR, "gtFine", split, city)
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*_leftImg8bit.png")))

    pairs = []
    if image_names:
        by_base = {os.path.basename(p).replace("_leftImg8bit.png", ""): p for p in img_paths}
        for name in image_names:
            base = name.replace("_leftImg8bit.png", "").replace("_leftImg8bit", "")
            ip = by_base.get(base)
            if ip is None:
                print(f"WARNING: {base} not found, skipping")
                continue
            lp = os.path.join(gt_dir, base + "_gtFine_labelIds.png")
            if os.path.exists(lp):
                pairs.append((ip, lp, base))
        return pairs

    for ip in img_paths:
        base = os.path.basename(ip).replace("_leftImg8bit.png", "")
        lp = os.path.join(gt_dir, base + "_gtFine_labelIds.png")
        if os.path.exists(lp):
            pairs.append((ip, lp, base))
        if len(pairs) >= num_images:
            break
    return pairs


def load_image(path, resize):
    img = Image.open(path).convert("RGB")
    if resize:
        img = img.resize(resize, Image.BILINEAR)
    return np.asarray(img)


def load_label(path, resize):
    lbl = Image.open(path)
    if resize:
        lbl = lbl.resize(resize, Image.NEAREST)  # never interpolate label IDs
    return np.asarray(lbl)


def load_data(cfg):
    pairs = find_pairs(cfg["split"], cfg["city"], cfg["num_images"], cfg.get("image_names"))
    assert pairs, "No image/label pairs found - check DATA_DIR, split, city."
    data = []
    for ip, lp, base in pairs:
        data.append({"image": load_image(ip, cfg["resize"]),
                     "gt": load_label(lp, cfg["resize"]),
                     "base": base})
    return data


# ---------------------------------------------------------------- features + k-means
def image_features(img, color_space):
    """Return (color, xs, ys) flattened to (N,3) and (N,1); color computed once."""
    H, W = img.shape[:2]
    if color_space == "lab":
        from skimage.color import rgb2lab
        color = rgb2lab(img / 255.0).reshape(-1, 3) / np.array([100.0, 128.0, 128.0])
    else:
        color = img.reshape(-1, 3).astype(np.float64) / 255.0
    ys, xs = np.mgrid[0:H, 0:W]
    xs = xs.reshape(-1, 1) / max(W - 1, 1)
    ys = ys.reshape(-1, 1) / max(H - 1, 1)
    return color, xs, ys


def run_kmeans(features, k, random_state=42, minibatch=False, n_init=5):
    if minibatch:
        km = MiniBatchKMeans(n_clusters=k, n_init=n_init, batch_size=4096, random_state=random_state)
    else:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    return km.fit_predict(features)


# ---------------------------------------------------------------- IoU
def map_clusters_to_classes(cluster_map, gt_map):
    lut = np.zeros(int(cluster_map.max()) + 1, dtype=np.int64)
    for c in np.unique(cluster_map):
        vals, counts = np.unique(gt_map[cluster_map == c], return_counts=True)
        lut[c] = int(vals[np.argmax(counts)])  # dominant GT class
    return lut[cluster_map]


def mean_iou(pred_map, gt_map, ignore_ids=(0,)):
    classes = [c for c in np.unique(gt_map) if c not in ignore_ids]
    per_class = {}
    for c in classes:
        p = pred_map == c
        g = gt_map == c
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        if union > 0:
            per_class[int(c)] = inter / union
    miou = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return miou, per_class


# ---------------------------------------------------------------- coloring
def colorize_classes(class_map):
    """Paint a class-id map with the official Cityscapes palette (void -> black)."""
    out = np.zeros(class_map.shape + (3,), dtype=np.uint8)
    for cid, col in CITYSCAPES_COLORS.items():
        out[class_map == cid] = col
    return out


def colorize_hybrid(cluster_map, gt_map):
    """Color each cluster by its dominant GT class, varying the shade per cluster."""
    from collections import defaultdict
    out = np.zeros(cluster_map.shape + (3,), dtype=np.uint8)
    by_class = defaultdict(list)
    for c in np.unique(cluster_map):
        vals, cnt = np.unique(gt_map[cluster_map == c], return_counts=True)
        by_class[int(vals[np.argmax(cnt)])].append(int(c))
    for cls, clusters in by_class.items():
        base = np.array(CITYSCAPES_COLORS.get(cls, (110, 110, 110)), dtype=np.float64)
        n = len(clusters)
        for i, c in enumerate(sorted(clusters)):
            if n == 1:
                col = base
            else:
                t = -0.4 + 0.8 * (i / (n - 1))   # -0.4 darker .. +0.4 lighter
                col = base + (255.0 - base) * t if t >= 0 else base * (1.0 + t)
            out[cluster_map == c] = np.clip(col, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------- standalone GT legend
def save_legend(path=None, ncol=2):
    """Plain ground-truth legend (one flat color per class, no shading) as a PNG."""
    from matplotlib.patches import Patch
    if path is None:
        path = os.path.join(DEMO_DIR, "legend.png")
    handles = [Patch(facecolor=np.array(CITYSCAPES_COLORS[c]) / 255.0,
                     edgecolor="0.4", label=CITYSCAPES_NAMES[c])
               for c in sorted(CITYSCAPES_NAMES)]
    fig = plt.figure(figsize=(5.5, 4.5))
    fig.legend(handles=handles, loc="center", ncol=ncol, frameon=False,
               title="Cityscapes ground-truth classes", fontsize=10, title_fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- per-image processing
def process_image(d, ks, lams, cfg):
    """Cluster one image for every k (color-only) and every (k, lambda) (color+pos)."""
    H, W = d["image"].shape[:2]
    color, xs, ys = image_features(d["image"], cfg["color_space"])
    rs, mb, ni, ig = cfg["random_state"], cfg["minibatch"], cfg["n_init"], cfg["ignore_ids"]
    segs, ious = {}, {}
    for k in ks:
        lab = run_kmeans(color, k, rs, mb, ni).reshape(H, W)
        segs[("color", k)] = lab
        ious[("color", k)] = mean_iou(map_clusters_to_classes(lab, d["gt"]), d["gt"], ig)[0]
        for lam in lams:
            feats = np.hstack([color, lam * xs, lam * ys])
            lab = run_kmeans(feats, k, rs, mb, ni).reshape(H, W)
            segs[("pos", k, lam)] = lab
            ious[("pos", k, lam)] = mean_iou(map_clusters_to_classes(lab, d["gt"]), d["gt"], ig)[0]
    return segs, ious


def _annotate(a, miou):
    a.text(0.03, 0.97, f"{miou:.3f}", transform=a.transAxes, va="top", ha="left",
           fontsize=8, color="white",
           bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45, lw=0))


def render_demo_figure(d, segs, ious, ks, lams, out_path):
    """One PNG: rows = k, cols = original | GT | RGB only | position(lambda)."""
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple

    ncols = 3 + len(lams)
    headers = ["original", "ground truth", "RGB only"] + [f"\u03bb={lam}" for lam in lams]
    fig, ax = plt.subplots(len(ks), ncols, figsize=(2.0 * ncols, 2.4 * len(ks) + 1.4), squeeze=False)
    gt_rgb = colorize_classes(d["gt"])
    for ri, k in enumerate(ks):
        ax[ri][0].imshow(d["image"])
        ax[ri][1].imshow(gt_rgb)
        ax[ri][2].imshow(colorize_hybrid(segs[("color", k)], d["gt"])); _annotate(ax[ri][2], ious[("color", k)])
        for ci, lam in enumerate(lams):
            a = ax[ri][3 + ci]
            a.imshow(colorize_hybrid(segs[("pos", k, lam)], d["gt"])); _annotate(a, ious[("pos", k, lam)])
        for ci in range(ncols):
            ax[ri][ci].set_xticks([]); ax[ri][ci].set_yticks([])
            if ri == 0:
                ax[ri][ci].set_title(headers[ci], fontsize=10)
        ax[ri][0].set_ylabel(f"k={k}", fontsize=12, rotation=0, ha="right", va="center", labelpad=22)

    # embedded legend with shade variants
    handles, labels = [], []
    for cid in sorted(CITYSCAPES_NAMES):
        b = np.array(CITYSCAPES_COLORS[cid], float)
        dk = np.clip(b * 0.6, 0, 255) / 255
        md = b / 255
        lt = np.clip(b + (255 - b) * 0.4, 0, 255) / 255
        handles.append((Patch(facecolor=dk), Patch(facecolor=md), Patch(facecolor=lt)))
        labels.append(CITYSCAPES_NAMES[cid])
    bf = 1.4 / (2.4 * len(ks) + 1.4)
    fig.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)},
               ncol=5, loc="lower center", fontsize=8, frameon=False,
               handlelength=2.4, columnspacing=1.2, bbox_to_anchor=(0.5, 0.005),
               title="Cityscapes classes  (dark -> light = different clusters of one class)",
               title_fontsize=9)
    fig.suptitle(f"{d['base']}  -  city demo | rows = k | cols = RGB only + position(\u03bb) | number = mIoU", fontsize=12)
    plt.tight_layout(rect=[0, bf, 1, 0.97])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- gallery HTML
_GALLERY_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>k-means demo - __CITY__</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #0f1115; color: #e6e6e6; }
  header { position: sticky; top: 0; background: #161922; padding: 14px 20px; border-bottom: 1px solid #2a2f3a; z-index: 5; }
  header h1 { margin: 0 0 8px; font-size: 18px; }
  .controls { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; font-size: 14px; }
  .controls input, .controls select { background: #0f1115; color: #e6e6e6; border: 1px solid #2a2f3a; border-radius: 6px; padding: 6px 8px; }
  .count { opacity: .7; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 16px; padding: 18px; }
  .card { background: #161922; border: 1px solid #2a2f3a; border-radius: 10px; overflow: hidden; }
  .card .meta { padding: 8px 12px; }
  .card .name { font-weight: 600; font-size: 13px; word-break: break-all; }
  .card .stats { font-size: 12px; opacity: .9; margin-top: 5px; }
  .badge { display: inline-block; background: #1f3b2a; color: #9fe8b8; border-radius: 5px; padding: 1px 6px; margin-right: 6px; }
  .badge.rgb { background: #1f2f3b; color: #8cc8ff; }
  .card a { display: block; }
  .card img { width: 100%; display: block; background: #fff; }
  footer { padding: 16px 20px; opacity: .6; font-size: 12px; }
  .legend { display: flex; flex-wrap: wrap; gap: 10px 14px; margin-top: 10px; font-size: 12px; opacity: .9; }
  .lg { display: inline-flex; align-items: center; }
  .lg i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 5px; border: 1px solid rgba(255,255,255,.25); }
</style></head>
<body>
<header>
  <h1>k-means demo - city __CITY__ - __COUNT__ images</h1>
  <div class="controls">
    <input id="q" type="search" placeholder="filter by name...">
    <label>sort:
      <select id="sort">
        <option value="name">name</option>
        <option value="pos_desc">best mIoU (pos) high-low</option>
        <option value="pos_asc">best mIoU (pos) low-high</option>
      </select>
    </label>
    <span class="count" id="count"></span>
  </div>
  <div class="legend">__LEGEND__</div>
</header>
<div class="grid" id="grid"></div>
<footer>Each tile: rows = k, cols = original / ground truth / RGB only / position(lambda). Number on a panel = mIoU. Colors follow Cityscapes classes; shades of one color = different clusters of that class. Click a tile to open it full size.</footer>
<script>
const DATA = __DATA__;
const grid = document.getElementById('grid');
const q = document.getElementById('q');
const sortSel = document.getElementById('sort');
const countEl = document.getElementById('count');
function render() {
  const term = q.value.trim().toLowerCase();
  let rows = DATA.filter(d => d.base.toLowerCase().includes(term));
  const mode = sortSel.value;
  if (mode === 'name') rows.sort((a,b) => a.base.localeCompare(b.base));
  if (mode === 'pos_desc') rows.sort((a,b) => b.best_pos - a.best_pos);
  if (mode === 'pos_asc') rows.sort((a,b) => a.best_pos - b.best_pos);
  countEl.textContent = rows.length + ' shown';
  grid.innerHTML = rows.map(d => `
    <div class="card">
      <a href="${d.img}" target="_blank" rel="noopener">
        <img loading="lazy" src="${d.img}" alt="${d.base}">
      </a>
      <div class="meta">
        <div class="name">${d.base}</div>
        <div class="stats">
          <span class="badge">pos ${d.best_pos.toFixed(3)} - ${d.set}</span>
          <span class="badge rgb">rgb ${d.best_rgb.toFixed(3)}</span>
        </div>
      </div>
    </div>`).join('');
}
q.addEventListener('input', render);
sortSel.addEventListener('change', render);
render();
</script>
</body></html>"""


def build_gallery(items, city, out_path=None):
    if out_path is None:
        out_path = os.path.join(DEMO_DIR, "index.html")
    legend = "".join(
        "<span class='lg'><i style='background:rgb({},{},{})'></i>{}</span>".format(
            CITYSCAPES_COLORS[c][0], CITYSCAPES_COLORS[c][1], CITYSCAPES_COLORS[c][2],
            CITYSCAPES_NAMES[c]) for c in sorted(CITYSCAPES_NAMES))
    html = (_GALLERY_TEMPLATE
            .replace("__DATA__", json.dumps(items))
            .replace("__COUNT__", str(len(items)))
            .replace("__LEGEND__", legend)
            .replace("__CITY__", str(city)))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


# ---------------------------------------------------------------- batch runner
def run_demo(cfg):
    """Full browsing demo: per-image PNGs, gallery, legend, and metric CSVs."""
    ensure_dirs()
    data = load_data(cfg)
    ks, lams = cfg["demo_k_values"], cfg["demo_lambda_values"]
    print(f"loaded {len(data)} images from city '{cfg['city']}'")

    records = []
    agg_color = {k: [] for k in ks}
    agg_pos = {(k, lam): [] for k in ks for lam in lams}
    summary = {}

    for i, d in enumerate(data, 1):
        segs, ious = process_image(d, ks, lams, cfg)
        s = summary.setdefault(d["base"], {"best_pos": -1.0, "best_set": "", "best_rgb": -1.0})
        for k in ks:
            m = ious[("color", k)]
            records.append((d["base"], k, "rgb", "", m)); agg_color[k].append(m)
            s["best_rgb"] = max(s["best_rgb"], m)
            for lam in lams:
                m = ious[("pos", k, lam)]
                records.append((d["base"], k, "pos", lam, m)); agg_pos[(k, lam)].append(m)
                if m > s["best_pos"]:
                    s["best_pos"], s["best_set"] = m, f"k={k}, \u03bb={lam}"
        render_demo_figure(d, segs, ious, ks, lams, os.path.join(DEMO_DIR, d["base"] + ".png"))
        if i % 10 == 0 or i == len(data):
            print(f"processed {i}/{len(data)} images")

    # per-image / per-setting IoU
    with open(os.path.join(DEMO_DIR, "iou_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "k", "method", "lambda", "miou"])
        for base, k, method, lam, m in records:
            w.writerow([base, k, method, lam, f"{m:.4f}"])

    # aggregate grid (mean over images): baseline color-only + color+pos per (k, lambda)
    with open(os.path.join(RESULTS_DIR, "metrics_grid.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "k", "lambda", "mean_miou"])
        for k in ks:
            w.writerow(["color_only", k, "", f"{np.mean(agg_color[k]):.4f}"])
        for lam in lams:
            for k in ks:
                w.writerow(["color_pos", k, lam, f"{np.mean(agg_pos[(k, lam)]):.4f}"])

    # gallery + standalone legend
    items = [{"base": b, "img": b + ".png",
              "best_pos": round(summary[b]["best_pos"], 3),
              "best_rgb": round(summary[b]["best_rgb"], 3),
              "set": summary[b]["best_set"]} for b in sorted(summary)]
    build_gallery(items, cfg["city"])
    save_legend()

    best = max(((k, lam) for k in ks for lam in lams), key=lambda kl: np.mean(agg_pos[kl]))
    print(f"done. best color+pos setting: k={best[0]}, lambda={best[1]} "
          f"(mean mIoU={np.mean(agg_pos[best]):.3f})")
    print("artifacts in:", DEMO_DIR, "and", RESULTS_DIR)
    return records


# ---------------------------------------------------------------- run
CONFIG = {
    **DEFAULT_CONFIG,
    "city": "cologne",
    "image_names": [],          # empty -> first num_images frames of the city
    "num_images": 150,           # frames to process (Cologne has 154)
    "resize": (512, 256),       # (384,192) / (256,128) for a faster big run
    "demo_k_values": [8, 12, 16],
    "demo_lambda_values": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    "minibatch": True,
    "n_init": 5,
}

if __name__ == "__main__":
    run_demo(CONFIG)
