"""Load the trained model once and turn photos into answers. Used by the app, predict.py and evaluate.py."""
from dataclasses import dataclass

import torch
from PIL import Image, ImageOps

import config
from currency.data import eval_transforms
from currency.model import get_device, load_checkpoint


@dataclass
class Result:
    label: str           # "500", "background", ...
    confidence: float    # 0..1
    probs: dict          # every class -> probability
    status: str          # "note", "uncertain" or "no_note"

    @property
    def value(self):
        return int(self.label) if self.status == "note" else None


class Predictor:
    def __init__(self, model_path=config.MODEL_PATH, tta=config.USE_TTA,
                 threshold=config.CONFIDENCE_THRESHOLD):
        if not model_path.exists():
            raise FileNotFoundError(f"No trained model at {model_path}. Run: python train.py")
        self.device = get_device()
        self.model, ckpt = load_checkpoint(model_path, self.device)
        self.classes = ckpt["classes"]
        self.arch = ckpt["arch"]
        self.metrics = ckpt.get("metrics", {})
        self.transform = eval_transforms(ckpt["img_size"])
        self.tta = tta
        self.threshold = threshold

    @staticmethod
    def _prepare(img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((config.STORE_MAX_SIDE, config.STORE_MAX_SIDE))
        return img

    @torch.no_grad()
    def probabilities(self, images, batch_size=32):
        """Return an (N, num_classes) tensor of probabilities for a list of PIL images."""
        out = []
        for start in range(0, len(images), batch_size):
            batch = [self._prepare(im) for im in images[start:start + batch_size]]
            x = torch.stack([self.transform(im) for im in batch]).to(self.device)
            if self.tta:
                # average over the 4 right-angle rotations (the model was trained on all of them)
                p = sum(torch.softmax(self.model(torch.rot90(x, k, dims=(2, 3))), 1) for k in range(4)) / 4
            else:
                p = torch.softmax(self.model(x), 1)
            out.append(p.cpu())
        return torch.cat(out)

    def decide(self, p) -> Result:
        conf, idx = p.max(0)
        label = self.classes[int(idx)]
        conf = float(conf)
        if label == config.BACKGROUND:
            status = "no_note" if conf >= self.threshold else "uncertain"
        else:
            status = "note" if conf >= self.threshold else "uncertain"
        return Result(label, conf, {c: float(v) for c, v in zip(self.classes, p)}, status)

    def predict(self, img) -> Result:
        return self.decide(self.probabilities([img])[0])
