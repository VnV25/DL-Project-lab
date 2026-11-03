"""Step 2: train the model with transfer learning.

Stage 1 trains only the new last layer (the pretrained network stays frozen).
Stage 2 unfreezes everything and fine-tunes with a small learning rate.
The checkpoint with the best validation accuracy is saved to models/currency_model.pt.

Usage:  python train.py
        python train.py --arch efficientnet_b0 --epochs 15
"""
import argparse
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from currency.data import NoteDataset, class_weights, eval_transforms, read_splits, train_transforms
from currency.labels import sort_classes
from currency.model import build_model, get_device, save_checkpoint, set_backbone_trainable


def run_epoch(model, loader, device, criterion, optimizer=None, freeze_backbone=False):
    training = optimizer is not None
    model.train(training)
    if training and freeze_backbone:
        model.features.eval()  # keep pretrained BatchNorm statistics while the backbone is frozen
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def plot_history(history, path):
    epochs = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(epochs, [h["train_loss"] for h in history], label="train")
    a1.plot(epochs, [h["val_loss"] for h in history], label="validation")
    a1.set_title("Loss"); a1.set_xlabel("epoch"); a1.legend()
    a2.plot(epochs, [h["train_acc"] for h in history], label="train")
    a2.plot(epochs, [h["val_acc"] for h in history], label="validation")
    a2.set_title("Accuracy"); a2.set_xlabel("epoch"); a2.legend()
    head = config.HEAD_EPOCHS
    for a in (a1, a2):
        a.axvline(head + 0.5, color="grey", ls="--", lw=1)
        a.grid(alpha=0.3)
    a2.text(head + 0.6, a2.get_ylim()[0], " fine-tuning starts", color="grey", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=config.ARCH)
    ap.add_argument("--head-epochs", type=int, default=config.HEAD_EPOCHS)
    ap.add_argument("--epochs", type=int, default=config.FINETUNE_EPOCHS, help="fine-tuning epochs")
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--no-pretrained", action="store_true", help="train from scratch (only for testing)")
    args = ap.parse_args()
    config.HEAD_EPOCHS = args.head_epochs

    if not config.SPLITS_FILE.exists():
        raise SystemExit("Run python prepare_data.py first.")
    torch.manual_seed(config.SEED)
    rows = read_splits()
    classes = sort_classes({r["label"] for r in rows})
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    print(f"Classes: {classes}")
    print(f"Train: {len(train_rows)} images, validation: {len(val_rows)} images")

    device = get_device()
    print(f"Device: {device}" + ("  (no GPU found: training on CPU is slower but works)" if device.type == "cpu" else ""))

    kw = dict(batch_size=args.batch_size, num_workers=config.NUM_WORKERS,
              pin_memory=device.type == "cuda", persistent_workers=config.NUM_WORKERS > 0)
    train_dl = DataLoader(NoteDataset(train_rows, classes, train_transforms()), shuffle=True,
                          drop_last=len(train_rows) > args.batch_size, **kw)
    val_dl = DataLoader(NoteDataset(val_rows, classes, eval_transforms()), shuffle=False, **kw)

    model = build_model(args.arch, len(classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows, classes).to(device),
                                    label_smoothing=config.LABEL_SMOOTHING)

    history, best_acc, bad_epochs, epoch = [], -1.0, 0, 0
    stages = [
        ("Stage 1: training the new last layer", args.head_epochs, config.HEAD_LR, False),
        ("Stage 2: fine-tuning the whole network", args.epochs, config.FINETUNE_LR, True),
    ]
    for title, n_epochs, lr, trainable in stages:
        if n_epochs <= 0:
            continue
        print(f"\n{title}")
        set_backbone_trainable(model, trainable)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        bad_epochs = 0
        for _ in range(n_epochs):
            epoch += 1
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(model, train_dl, device, criterion, optimizer, freeze_backbone=not trainable)
            va_loss, va_acc = run_epoch(model, val_dl, device, criterion)
            scheduler.step()
            history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss, val_acc=va_acc))
            mark = ""
            if va_acc > best_acc:
                best_acc, bad_epochs, mark = va_acc, 0, "  <- best so far, saved"
                save_checkpoint(config.MODEL_PATH, model, args.arch, classes, config.IMG_SIZE,
                                metrics={"val_acc": va_acc, "epoch": epoch})
            else:
                bad_epochs += 1
            print(f"epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.3f} acc {va_acc:.3f} | {time.time() - t0:.0f}s{mark}")
            if trainable and bad_epochs >= config.EARLY_STOP_PATIENCE:
                print("Validation accuracy stopped improving, stopping early.")
                break

    config.REPORTS_DIR.mkdir(exist_ok=True)
    with open(config.REPORTS_DIR / "training_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)
    plot_history(history, config.REPORTS_DIR / "training_curves.png")
    print(f"\nBest validation accuracy: {best_acc:.4f}")
    print(f"Model saved to {config.MODEL_PATH}. Next: python evaluate.py")


if __name__ == "__main__":
    main()
