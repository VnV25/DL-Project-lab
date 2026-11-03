"""Turn folder names from any dataset into a class label like "500" or "background".

Datasets name their folders differently ("500", "Rs_500_new", "5Hundrednote", "Tennote",
"2000 old" ...). Instead of forcing one layout, we read the label from the folder names.
"""
import re
from pathlib import Path

import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_BG_WORDS = ("background", "negative", "nonote", "no_note", "not_note", "notnote", "random", "other")
_SMALL = {"one": 1, "two": 2, "five": 5}


def label_from_name(name: str):
    """Return "10", "500", "background" ... for a folder name, or None if it is not a label."""
    s = name.lower()
    flat = re.sub(r"[^a-z0-9]+", "", s)
    if any(w.replace("_", "") in flat for w in _BG_WORDS):
        return config.BACKGROUND

    # 1) plain numbers: "500", "Rs_500_new", "10 old"
    for num in re.findall(r"\d+", s):
        if int(num) in config.DENOMINATIONS:
            return str(int(num))

    # 2) word forms: "5Hundrednote", "two thousand", "Fiftynote", "Tennote"
    value = None
    m = re.search(r"(\d|one|two|five)?(hundred|thousand)", flat)
    if m:
        k = m.group(1)
        k = int(k) if (k and k.isdigit()) else _SMALL.get(k, 1)
        value = k * (100 if m.group(2) == "hundred" else 1000)
    else:
        # whole words only, so "content" or "written" are not read as "ten"
        for word, v in (("fifty", 50), ("twenty", 20), ("ten", 10)):
            if re.search(rf"(^|[^a-z]){word}(note|rupee|rs|[^a-z]|$)", s):
                value = v
                break
    if value in config.DENOMINATIONS:
        return str(value)
    return None


def label_from_path(path: Path, root: Path):
    """Look at every folder between root and the file, deepest first."""
    for part in reversed(path.relative_to(root).parts[:-1]):
        lab = label_from_name(part)
        if lab is not None:
            return lab
    return None


def find_images(root: Path):
    """Yield (path, label) for every image under root. label is None if it can't be read."""
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."):
            yield p, label_from_path(p, root)


def sort_classes(labels):
    """Notes in increasing value, background last."""
    notes = sorted((l for l in labels if l != config.BACKGROUND), key=int)
    return notes + ([config.BACKGROUND] if config.BACKGROUND in labels else [])
