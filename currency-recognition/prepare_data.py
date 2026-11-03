"""Step 1: read the raw dataset, shrink the images, and split them into train / val / test.

Why the split is "group-aware": many datasets contain the same photo twice, or burst shots
that are almost identical. If one copy lands in train and the other in test, the test score
is inflated and says nothing about unseen data. We detect near-duplicates with a perceptual
hash and keep each group of look-alikes inside the same split.

Usage:  python prepare_data.py
"""
import csv
import hashlib
import random
import shutil
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

import config
from currency.labels import find_images, sort_classes

HASH_DISTANCE = 4  # images whose hashes differ in <= this many bits count as duplicates


def dhash(img: Image.Image) -> int:
    """64-bit difference hash: robust to resizing and JPEG compression."""
    g = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def load_small(path):
    img = Image.open(path)
    img.draft("RGB", (config.STORE_MAX_SIDE * 2, config.STORE_MAX_SIDE * 2))  # fast JPEG downscale
    img = ImageOps.exif_transpose(img)  # phone photos store rotation in EXIF
    img = img.convert("RGB")
    img.thumbnail((config.STORE_MAX_SIDE, config.STORE_MAX_SIDE), Image.Resampling.LANCZOS)
    return img


def group_duplicates(hashes):
    """Union-find over images whose hashes are within HASH_DISTANCE bits.
    Group size is capped so a chain of look-alikes can't swallow a whole class."""
    n = len(hashes)
    parent = list(range(n))
    size = [1] * n
    max_group = max(5, int(0.02 * n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    h = np.array(hashes, dtype=np.uint64)
    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    for i in range(n - 1):
        x = np.bitwise_xor(h[i + 1:], h[i])
        dist = popcount[x.view(np.uint8)].reshape(-1, 8).sum(1)
        for j in np.nonzero(dist <= HASH_DISTANCE)[0]:
            a, b = find(i), find(i + 1 + int(j))
            if a != b and size[a] + size[b] <= max_group:
                parent[b] = a
                size[a] += size[b]
    return [find(i) for i in range(n)]


def main():
    if not config.RAW_DIR.exists() or not any(config.RAW_DIR.iterdir()):
        raise SystemExit(f"No data found. Put the dataset folders inside {config.RAW_DIR} (see README).")

    items = list(find_images(config.RAW_DIR))
    unknown = Counter(str(p.parent.relative_to(config.RAW_DIR)) for p, l in items if l is None)
    items = [(p, l) for p, l in items if l is not None and l not in config.EXCLUDE]
    if unknown:
        print("Skipped folders with no recognisable denomination (rename them if they are notes):")
        for folder, n in unknown.most_common():
            print(f"   {folder}  ({n} images)")
    if not items:
        raise SystemExit("No labelled images found. Check the folder names (see README).")

    if config.PROCESSED_DIR.exists():
        shutil.rmtree(config.PROCESSED_DIR)
    config.PROCESSED_DIR.mkdir(parents=True)

    rows, hashes, seen = [], [], set()
    exact_dupes = 0
    for path, label in tqdm(items, desc="Resizing images"):
        raw = path.read_bytes()
        md5 = hashlib.md5(raw).hexdigest()
        if md5 in seen:
            exact_dupes += 1
            continue
        seen.add(md5)
        try:
            img = load_small(path)
        except Exception as e:  # corrupt file
            print(f"   could not read {path.name}: {e}")
            continue
        out = config.PROCESSED_DIR / label / f"{md5[:16]}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, quality=92)
        rows.append({"path": str(out.relative_to(config.ROOT)).replace("\\", "/"), "label": label})
        hashes.append(dhash(img))

    groups = group_duplicates(hashes)
    for r, g in zip(rows, groups):
        r["group"] = g

    # Stratified, group-aware split: a whole group always goes to one split,
    # and groups are balanced per class using the group's most common label.
    rng = random.Random(config.SEED)
    members = defaultdict(list)
    for i, r in enumerate(rows):
        members[r["group"]].append(i)
    by_class = defaultdict(list)
    for g, idx in sorted(members.items()):
        majority = Counter(rows[i]["label"] for i in idx).most_common(1)[0][0]
        by_class[majority].append(idx)
    for label in sorted(by_class):
        glist = by_class[label]
        rng.shuffle(glist)
        n = sum(len(g) for g in glist)
        counts = {"test": 0, "val": 0}
        for g in glist:
            if counts["test"] < config.TEST_FRAC * n:
                split = "test"
            elif counts["val"] < config.VAL_FRAC * n:
                split = "val"
            else:
                split = "train"
            if split in counts:
                counts[split] += len(g)
            for i in g:
                rows[i]["split"] = split

    with open(config.SPLITS_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "split", "group"])
        w.writeheader()
        w.writerows(rows)

    classes = sort_classes({r["label"] for r in rows})
    n_groups = len(set(groups))
    print(f"\nKept {len(rows)} images ({exact_dupes} exact duplicates removed, "
          f"{len(rows) - n_groups} near-duplicates kept together in the same split).")
    print(f"{'class':>12} {'train':>7} {'val':>6} {'test':>6}")
    table = Counter((r["label"], r["split"]) for r in rows)
    for c in classes:
        print(f"{c:>12} {table[(c, 'train')]:>7} {table[(c, 'val')]:>6} {table[(c, 'test')]:>6}")
    if config.BACKGROUND not in classes:
        print("\nTip: add a data/raw/background folder with 100+ photos that contain no note "
              "(tables, hands, paper, bags). The app then says 'no note found' instead of guessing.")
    print(f"\nSaved split list to {config.SPLITS_FILE}. Next: python train.py")


if __name__ == "__main__":
    main()
