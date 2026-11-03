"""Classify one or more photos from the command line.

Usage:  python predict.py photo.jpg
        python predict.py photo1.jpg photo2.jpg --speak --lang hi
"""
import argparse
from pathlib import Path

from PIL import Image

import config
from currency.predictor import Predictor
from currency.speech import phrase, speak_to_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--speak", action="store_true", help="also save a spoken answer (mp3)")
    ap.add_argument("--lang", default="en", choices=["en", "hi", "mr"])
    args = ap.parse_args()

    pred = Predictor()
    total = 0
    for path in args.images:
        res = pred.predict(Image.open(path))
        top3 = sorted(res.probs.items(), key=lambda kv: -kv[1])[:3]
        top3 = ", ".join(f"{'no note' if c == config.BACKGROUND else 'Rs ' + c}: {p:.1%}" for c, p in top3)
        if res.status == "note":
            total += res.value
            text = phrase("note", args.lang, n=res.value)
        else:
            text = phrase(res.status, args.lang)
        print(f"{path.name}: {res.status.upper():9s} {res.confidence:.1%}   [{top3}]")
        if args.speak:
            audio = speak_to_file(text, args.lang)
            print(f"   voice: {audio or 'no text-to-speech available'}")
    if len(args.images) > 1:
        print(f"\nTotal of recognised notes: Rs {total}")


if __name__ == "__main__":
    main()
