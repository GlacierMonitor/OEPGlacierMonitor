"""U-Net, Dice+BCE loss and IoU for GlacierMonitor Part 4."""
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

from data import CHANNELS, IGNORE


def build_model(encoder="resnet34", weights="imagenet"):
    # smp adapts the pretrained 3-channel first conv to len(CHANNELS) inputs
    return smp.Unet(encoder_name=encoder, encoder_weights=weights,
                    in_channels=len(CHANNELS), classes=1)


def dice_bce_loss(logits, target, dice_weight=0.5):
    """BCE + soft Dice over labelled pixels only (target == IGNORE is skipped)."""
    logits = logits[:, 0].float()
    mask = target != IGNORE
    if not mask.any():
        return logits.sum() * 0.0
    t = target.float()
    bce = F.binary_cross_entropy_with_logits(logits[mask], t[mask])
    p = torch.sigmoid(logits[mask])
    dice = 1 - (2 * (p * t[mask]).sum() + 1) / (p.sum() + t[mask].sum() + 1)
    return (1 - dice_weight) * bce + dice_weight * dice


def confusion(pred, target):
    """(tp, fp, fn) counts for glacier = 1 over labelled pixels; pred is a bool tensor/array."""
    mask = target != IGNORE
    p, t = pred[mask], target[mask] == 1
    return int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())


def iou(tp, fp, fn):
    return tp / (tp + fp + fn) if tp + fp + fn else float("nan")
