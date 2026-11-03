"""Model: an ImageNet-pretrained network with a new last layer for our classes (transfer learning)."""
import torch
from torch import nn
from torchvision import models


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.mobilenet_v3_large(weights=weights)
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=weights)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m


def set_backbone_trainable(model: nn.Module, trainable: bool):
    for p in model.features.parameters():
        p.requires_grad = trainable


def save_checkpoint(path, model, arch, classes, img_size, metrics=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arch": arch,
        "classes": classes,
        "img_size": img_size,
        "state_dict": model.state_dict(),
        "metrics": metrics or {},
    }, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon Macs
    return torch.device("cpu")
