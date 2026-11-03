"""Step 3: measure how well the model works, including on data it has never seen.

It produces three kinds of evidence (all saved in reports/ for your project report):
  1. Test-set accuracy, per-class precision/recall/F1 and a confusion matrix.
  2. Robustness test: the test photos are made darker, blurrier, rotated, partly covered,
     low quality ... and accuracy is measured again under each condition.
  3. Your own photos (my_photos/): notes photographed by you, with a different phone,
     place and lighting from the dataset. This is the most honest "unseen data" score.

Usage:  python evaluate.py            (add --no-tta to make it ~4x faster)
"""
import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from tqdm import tqdm
import io

import config
from currency.data import read_splits
from currency.labels import find_images
from currency.predictor import Predictor


def nice(label):
    return "no note" if label == config.BACKGROUND else f"Rs {label}"  # "Rs" prints on every console


# ---------- Real-world conditions applied to test photos ----------
def _jpeg_low(img):
    small = img.resize((max(1, img.width // 4), max(1, img.height // 4)))
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=15)
    return Image.open(buf).convert("RGB").resize(img.size)


def _covered(img, rng):
    img = img.copy()
    w, h = img.size
    cw, ch = int(w * 0.45), int(h * 0.65)          # ~30% of the photo hidden
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    ImageDraw.Draw(img).rectangle([x, y, x + cw, y + ch], fill=(198, 134, 103))  # skin-like colour
    return img


def _crop(img, frac=0.7):
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    return img.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))


CONDITIONS = {
    "Original": lambda im, r: im,
    "Dim light": lambda im, r: ImageEnhance.Brightness(im).enhance(0.4),
    "Very bright": lambda im, r: ImageEnhance.Brightness(im).enhance(1.7),
    "Blurry": lambda im, r: im.filter(ImageFilter.GaussianBlur(3)),
    "Rotated 90°": lambda im, r: im.transpose(Image.Transpose.ROTATE_90),
    "Upside down": lambda im, r: im.transpose(Image.Transpose.ROTATE_180),
    "Tilted 20°": lambda im, r: im.rotate(20, expand=True, fillcolor=(120, 120, 120)),
    "Part covered (~30%)": _covered,
    "Zoomed in (70% visible)": lambda im, r: _crop(im),
    "Low quality camera": lambda im, r: _jpeg_low(im),
}


def accuracy(pred, true):
    return float(np.mean(np.array(pred) == np.array(true))) if len(true) else float("nan")


def gallery(images, titles, colors, path, cols=6):
    if not images:
        return
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, im, t, c in zip(axes.flat, images, titles, colors):
        ax.imshow(im)
        ax.set_title(t, fontsize=8, color=c)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tta", action="store_true", help="faster, slightly less robust")
    args = ap.parse_args()

    pred = Predictor(tta=not args.no_tta)
    classes = pred.classes
    out = config.REPORTS_DIR
    out.mkdir(exist_ok=True)
    md = ["# Evaluation results\n", f"Model: `{config.ARCH}`, classes: {', '.join(nice(c) for c in classes)}\n"]

    # ---------------- 1. Test set ----------------
    test_rows = [r for r in read_splits() if r["split"] == "test"]
    images = [Image.open(config.ROOT / r["path"]).convert("RGB") for r in test_rows]
    true = [r["label"] for r in test_rows]
    print(f"Test set: {len(images)} photos the model never saw during training")
    probs = pred.probabilities(images)
    results = [pred.decide(p) for p in probs]
    predicted = [res.label for res in results]
    acc = accuracy(predicted, true)
    confident = [i for i, res in enumerate(results) if res.status != "uncertain"]
    conf_acc = accuracy([predicted[i] for i in confident], [true[i] for i in confident])
    coverage = len(confident) / len(results)

    report = classification_report(true, predicted, labels=classes,
                                   target_names=[nice(c) for c in classes], digits=4, zero_division=0)
    print(f"\nTest accuracy: {acc:.4f}")
    print(f"With the confidence check ({config.CONFIDENCE_THRESHOLD:.0%}): the app answers {coverage:.1%} "
          f"of photos, and is right on {conf_acc:.4f} of those (it asks to retake the rest).\n")
    print(report)
    md += ["## 1. Test set (unseen photos from the dataset)\n",
           f"- Photos: **{len(images)}**",
           f"- Accuracy: **{acc:.2%}**",
           f"- With the confidence check: answers **{coverage:.1%}** of photos, correct on **{conf_acc:.2%}** of those\n",
           "```", report, "```\n", "![Confusion matrix](confusion_matrix.png)\n"]

    cm = confusion_matrix(true, predicted, labels=classes)
    fig, ax = plt.subplots(figsize=(1.1 * len(classes) + 2, 1.1 * len(classes) + 1.5))
    ConfusionMatrixDisplay(cm, display_labels=[nice(c) for c in classes]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Test set confusion matrix (accuracy {acc:.2%})")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150); plt.close(fig)

    wrong = [i for i in range(len(true)) if predicted[i] != true[i]][:24]
    gallery([images[i] for i in wrong],
            [f"true {nice(true[i])}\npred {nice(predicted[i])} ({results[i].confidence:.0%})" for i in wrong],
            ["red"] * len(wrong), out / "mistakes.png")
    if wrong:
        md.append("Mistakes the model made: ![Mistakes](mistakes.png)\n")

    # ---------------- 2. Robustness ----------------
    print("Robustness test: the same test photos under difficult real-world conditions")
    md += ["## 2. Robustness to real-world conditions\n",
           "Each test photo was changed to imitate a difficult situation, then classified again.\n",
           "| Condition | Accuracy |", "|---|---|"]
    rob = {}
    for name, fn in tqdm(CONDITIONS.items(), desc="Conditions"):
        rng = random.Random(config.SEED)
        changed = [fn(im, rng) for im in images]
        p = pred.probabilities(changed)
        rob[name] = accuracy([classes[int(i)] for i in p.argmax(1)], true)
        md.append(f"| {name} | {rob[name]:.2%} |")
    for name, a in rob.items():
        print(f"   {name:<26} {a:.4f}")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(rob)[::-1]
    ax.barh(names, [rob[n] * 100 for n in names], color="#3b7dd8")
    for i, n in enumerate(names):
        ax.text(rob[n] * 100 + 0.5, i, f"{rob[n]:.1%}", va="center", fontsize=9)
    ax.set_xlim(0, 105); ax.set_xlabel("accuracy (%)"); ax.set_title("Accuracy under real-world conditions")
    fig.tight_layout(); fig.savefig(out / "robustness.png", dpi=150); plt.close(fig)
    md.append("\n![Robustness](robustness.png)\n")

    # ---------------- 3. Your own photos ----------------
    md.append("## 3. My own photos (completely new data)\n")
    own = [(p, l) for p, l in find_images(config.MY_PHOTOS_DIR)] if config.MY_PHOTOS_DIR.exists() else []
    own = [(p, l) for p, l in own if l in classes]
    if own:
        own_imgs = [Image.open(p) for p, _ in own]
        own_true = [l for _, l in own]
        own_res = [pred.decide(p) for p in pred.probabilities(own_imgs)]
        own_pred = [r.label for r in own_res]
        own_acc = accuracy(own_pred, own_true)
        unsure = sum(r.status == "uncertain" for r in own_res)
        print(f"\nYour own photos: {len(own)} photos, accuracy {own_acc:.4f} ({unsure} marked 'not sure')")
        md += [f"- Photos: **{len(own)}**, accuracy: **{own_acc:.2%}**, "
               f"marked 'not sure' by the app: {unsure}\n", "![My photos](my_photos_results.png)\n"]
        colors = ["green" if p == t else ("orange" if r.status == "uncertain" else "red")
                  for p, t, r in zip(own_pred, own_true, own_res)]
        gallery([Predictor._prepare(im) for im in own_imgs[:36]],
                [f"true {nice(t)}\npred {nice(p)} ({r.confidence:.0%})" for t, p, r in zip(own_true, own_pred, own_res)][:36],
                colors[:36], out / "my_photos_results.png")
    else:
        print("\nNo photos in my_photos/ yet. Add your own (see README) to test on truly new data.")
        md.append("_No photos added yet. Put your own photos in `my_photos/<denomination>/` and run again._\n")

    (out / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nAll charts and a summary (results.md) are in {out}")


if __name__ == "__main__":
    main()
