import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from common.image_watermark_common import (
    make_image_datasets,
    get_models,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "data"))


def generate_mask_with_ratio(shape, zero_ratio=0.5, seed=2025):
    # mask=1 means protected (unmodifiable), mask=0 means editable
    rng = np.random.RandomState(seed)
    c, h, w = shape
    total = h * w
    mask = np.ones(total, dtype=np.float32)
    num_zero = int(total * zero_ratio)
    idx = rng.choice(total, num_zero, replace=False)
    mask[idx] = 0.0
    mask = torch.tensor(mask.reshape(1, h, w), dtype=torch.float32).repeat(c, 1, 1)
    return mask


def generate_per_client_target_labels(num_clients, clean_labels, num_classes, seed=2025):
    rng = np.random.RandomState(seed)
    clean_np = (clean_labels.detach().cpu().numpy().astype(np.int64)
                if isinstance(clean_labels, torch.Tensor)
                else np.asarray(clean_labels, dtype=np.int64))
    n = clean_np.shape[0]

    out, used = [], set()
    for _ in range(num_clients):
        while True:
            offsets = rng.randint(1, num_classes, size=n).astype(np.int64)
            targets = (clean_np + offsets) % num_classes
            k = tuple(targets.tolist())
            if k not in used:
                used.add(k)
                out.append(torch.tensor(targets, dtype=torch.long))
                break
    return out


def _single_batch_pgd_attack(model_front, model_back, images, target_labels, mask,
                             eps=8/255, alpha=2/255, steps=40, device="cuda"):
    ce = nn.CrossEntropyLoss()
    x = images.clone().detach().to(device)
    ori = x.clone().detach()
    y = target_labels.to(device)
    m = mask.to(device)
    x.requires_grad = True

    for _ in range(steps):
        mixed = x * (1 - m) + ori * m
        logits = model_back(model_front(mixed))
        loss = ce(logits, y)
        loss.backward()
        with torch.no_grad():
            adv = x - alpha * x.grad.sign()
            eta = torch.clamp(adv - ori, -eps, eps)
            x = torch.clamp(ori + eta, 0, 1)
            x = x * (1 - m) + ori * m
        x = x.detach().requires_grad_(True)
    return x.detach()


def masked_pgd_attack(model_front, model_back, images, targets, mask,
                      eps, alpha, steps, device, attack_batch_size):
    outs = []
    for i in range(0, len(images), attack_batch_size):
        v = images[i : i + attack_batch_size]
        t = targets[i : i + attack_batch_size]
        outs.append(_single_batch_pgd_attack(
            model_front, model_back, v, t, mask, eps, alpha, steps, device
        ).cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def targeted_success_rate(model_front, model_back, adv_images, target_labels, device, topk=1):
    total, success = 0, 0
    for i in range(0, len(adv_images), 64):
        x = adv_images[i : i + 64].to(device)
        y = target_labels[i : i + 64].to(device)
        logits = model_back(model_front(x))
        top = logits.topk(topk, dim=1).indices
        if topk == 1:
            success += (top[:, 0] == y).sum().item()
        else:
            success += (top == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.numel()
    return success / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="tiny-imagenet")
    parser.add_argument("--model", type=str, default="vit-b-16", choices=["vit-b-16"])
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts_dir", type=str, default="./artifacts_dir_image")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--attack_batch_size", type=int, default=64)
    parser.add_argument("--cleanset_max", type=int, default=200)
    parser.add_argument("--eps", type=float, default=0.06)
    parser.add_argument("--alpha", type=float, default=0.007)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--client_idx", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    print(f"\n====== Configuration for {os.path.basename(__file__)} ======")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 50)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(os.path.join(args.artifacts_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    num_classes = int(meta["num_classes"])
    model_name = meta.get("model", "vit-b-16")

    _, test_ds, _ = make_image_datasets(args.dataset_name, args.data_dir)
    clean_subset = Subset(test_ds, list(range(min(args.cleanset_max, len(test_ds)))))
    clean_loader = DataLoader(clean_subset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    images, labels = [], []
    for x, y in clean_loader:
        images.append(x)
        labels.append(y)
    clean_images = torch.cat(images, dim=0)
    clean_labels = torch.cat(labels, dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client, server, _ = get_models(model_name, num_classes)
    client.load_state_dict(torch.load(os.path.join(args.artifacts_dir, "client.pt"),
                                      map_location="cpu")["state_dict"])
    server.load_state_dict(torch.load(os.path.join(args.artifacts_dir, "server.pt"),
                                      map_location="cpu")["state_dict"])

    client.to(device)
    server.to(device)
    client.eval()
    server.eval()

    # Generate spatial mask: 50% of pixels are editable
    img_shape = tuple(clean_images[0].shape)
    client_mask = generate_mask_with_ratio(img_shape, zero_ratio=0.5, seed=args.seed)

    all_target_clients = generate_per_client_target_labels(
        args.num_clients, clean_labels, num_classes, seed=args.seed
    )
    target_client = all_target_clients[args.client_idx]

    print("\n=== Client Watermarking (Masked PGD on Image) ===")
    adv_c = masked_pgd_attack(
        client, server, clean_images, target_client, client_mask,
        args.eps, args.alpha, args.steps, device, args.attack_batch_size,
    )

    acc_c1 = targeted_success_rate(client, server, adv_c, target_client, device, topk=1)
    acc_c2 = targeted_success_rate(client, server, adv_c, target_client, device, topk=2)
    print(f"Client verification accuracy (Top-1) = {acc_c1*100:.2f}%")
    print(f"Client verification accuracy (Top-2) = {acc_c2*100:.2f}%")

    torch.save({
        "adv_images": adv_c,
        "targets": target_client,
        "clean_labels": clean_labels,
        "mask": client_mask,
    }, os.path.join(args.artifacts_dir, "client_watermark_image.pt"))


if __name__ == "__main__":
    main()
