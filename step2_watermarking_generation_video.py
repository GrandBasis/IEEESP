import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from mmengine.registry import MODELS
import mmaction.models.data_preprocessors as data_preproc_module

from common.video_watermark_common import (
    VideoClassificationDataset,
    resolve_video_dataset,
    get_models,
)


def generate_mask_with_ratio(shape, zero_ratio=0.5, seed=2025):
    # mask=1 means protected, mask=0 means editable
    rng = np.random.RandomState(seed)
    c, t, h, w = shape
    total = t * h * w
    mask = np.ones(total, dtype=np.float32)
    num_zero = int(total * zero_ratio)
    zero_idx = rng.choice(total, num_zero, replace=False)
    mask[zero_idx] = 0.0
    mask = torch.tensor(mask.reshape(1, t, h, w), dtype=torch.float32).repeat(c, 1, 1, 1)
    return mask


def random_derangement(rng, num_classes):
    base = np.arange(num_classes)
    while True:
        perm = rng.permutation(base)
        if not np.any(perm == base):
            return perm


def generate_label_derangements(num_clients, num_classes, seed=2025):
    rng = np.random.RandomState(seed)
    out = []
    used = set()
    for _ in range(num_clients):
        while True:
            d = random_derangement(rng, num_classes)
            k = tuple(d.tolist())
            if k not in used:
                used.add(k)
                out.append(torch.tensor(d, dtype=torch.long))
                break
    return out


def generate_per_client_target_labels(num_clients, clean_labels, num_classes, seed=2025):
    rng = np.random.RandomState(seed)
    if isinstance(clean_labels, torch.Tensor):
        clean_np = clean_labels.detach().cpu().numpy().astype(np.int64)
    else:
        clean_np = np.asarray(clean_labels, dtype=np.int64)
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


def _single_batch_pgd_attack(model_front, model_back, videos, target_labels, mask, num_classes,
                             eps=0.08, alpha=0.005, steps=40, device="cuda"):
    ce = nn.CrossEntropyLoss()
    x = videos.clone().detach().to(device)
    ori = x.clone().detach()
    y = target_labels.to(device)
    m = mask.to(device)
    x.requires_grad = True
    for _ in range(steps):
        mixed = x * (1 - m) + ori * m
        logits = model_back(model_front(mixed))
        loss = ce(logits[:, :num_classes], y)
        loss.backward()
        with torch.no_grad():
            adv = x - alpha * x.grad.sign()
            eta = torch.clamp(adv - ori, -eps, eps)
            x = torch.clamp(ori + eta, 0, 1)
            x = x * (1 - m) + ori * m
        x = x.detach().requires_grad_(True)
    return x.detach()


def masked_pgd_attack(model_front, model_back, videos, target_labels, mask, num_classes,
                      eps=0.08, alpha=0.005, steps=40, device="cuda", attack_batch_size=8):
    outs = []
    for i in range(0, len(videos), attack_batch_size):
        v = videos[i : i + attack_batch_size]
        t = target_labels[i : i + attack_batch_size]
        outs.append(_single_batch_pgd_attack(
            model_front, model_back, v, t, mask, num_classes,
            eps, alpha, steps, device
        ).cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def targeted_success_rate(model_front, model_back, adv_videos, target_labels, mask,
                          device, num_classes, topk=1):
    total, success = 0, 0
    for i in range(0, len(adv_videos), 8):
        x = adv_videos[i : i + 8].to(device)
        y = target_labels[i : i + 8].to(device)
        logits = model_back(model_front(x))[:, :num_classes]
        top = logits.topk(topk, dim=1).indices
        if topk == 1:
            success += (top[:, 0] == y).sum().item()
        else:
            success += (top == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.numel()
    return success / max(1, total)


def main():
    parser = argparse.ArgumentParser(description="Client-side watermark generation for video Split Learning")
    parser.add_argument("--dataset_name", type=str, default="ucf101")
    parser.add_argument("--model", type=str, required=True, choices=["tsn"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--artifacts_dir", type=str, default="./artifacts_dir_video")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cleanset_max", type=int, default=120)
    parser.add_argument("--eps", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--attack_batch_size", type=int, default=4)
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--client_idx", type=int, default=0,
                        help="Which client target labels to use, in [0, num_clients-1].")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    print(f"\n====== Configuration for {os.path.basename(__file__)} ======")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 50)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    meta_path = os.path.join(args.artifacts_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_classes = int(meta["num_classes"])
    num_frames = int(meta["num_frames"])
    image_size = int(meta["image_size"])
    model_name = meta.get("model", "tsn")
    dataset_name = meta.get("dataset_name", args.dataset_name)
    pretrained = meta.get("pretrained", False)

    _, test_ex, _ = resolve_video_dataset(args.dataset_name, args.data_dir, None, None)
    test_ds = VideoClassificationDataset(test_ex, num_frames=num_frames, image_size=image_size, train=False)
    clean_subset = Subset(test_ds, list(range(min(args.cleanset_max, len(test_ds)))))
    clean_loader = DataLoader(clean_subset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    vids, labels = [], []
    for x, y in clean_loader:
        vids.append(x)
        labels.append(y)
    clean_videos = torch.cat(vids, dim=0)
    clean_labels = torch.cat(labels, dim=0)
    print(f"[INFO] Loaded clean set: {len(clean_videos)} videos")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Register MMAction2 data preprocessor for compatibility
    target_class_name = 'ActionDataPreprocessor'
    if target_class_name not in MODELS.module_dict:
        if hasattr(data_preproc_module, target_class_name):
            target_class = getattr(data_preproc_module, target_class_name)
            MODELS.register_module(name=target_class_name, module=target_class)
        else:
            from mmaction.models.data_preprocessors.data_preprocessor import ActionDataPreprocessor
            MODELS.register_module(name=target_class_name, module=ActionDataPreprocessor)

    client, server, _ = get_models(model_name, num_classes, dataset_name=dataset_name, pretrained=pretrained)

    # Load Step 1 trained weights
    client.load_state_dict(torch.load(os.path.join(args.artifacts_dir, "client.pt"),
                                      map_location="cpu")["state_dict"])
    server.load_state_dict(torch.load(os.path.join(args.artifacts_dir, "server.pt"),
                                      map_location="cpu")["state_dict"])

    client.to(device)
    server.to(device)
    client.eval()
    server.eval()

    clip_shape = tuple(clean_videos[0].shape)
    client_mask = generate_mask_with_ratio(clip_shape, zero_ratio=0.5, seed=args.seed)

    all_target_clients = generate_per_client_target_labels(
        args.num_clients, clean_labels, num_classes, seed=args.seed
    )
    if not (0 <= args.client_idx < len(all_target_clients)):
        raise ValueError(
            f"--client_idx={args.client_idx} out of range. "
            f"Expected 0..{len(all_target_clients) - 1} (num_clients={args.num_clients})."
        )
    target_client = all_target_clients[args.client_idx]

    print(f"\n=== Client Watermarking (Masked PGD on clips using model: {model_name}) ===")
    adv_c = masked_pgd_attack(
        client, server, clean_videos, target_client, client_mask, num_classes,
        eps=args.eps, alpha=args.alpha, steps=args.steps, device=device,
        attack_batch_size=args.attack_batch_size,
    )
    acc_c1 = targeted_success_rate(client, server, adv_c, target_client, client_mask,
                                   device, num_classes, topk=1)
    acc_c2 = targeted_success_rate(client, server, adv_c, target_client, client_mask,
                                   device, num_classes, topk=2)
    print(f"Client watermark verification accuracy (Top-1) = {acc_c1*100:.2f}%")
    print(f"Client watermark verification accuracy (Top-2) = {acc_c2*100:.2f}%")

    torch.save(
        {
            "adv_videos": adv_c,
            "targets": target_client,
            "clean_labels": clean_labels,
            "mask": client_mask,
            "client_verify_acc_top1": acc_c1,
            "client_verify_acc_top2": acc_c2,
        },
        os.path.join(args.artifacts_dir, "client_watermark_video.pt"),
    )


if __name__ == "__main__":
    main()
