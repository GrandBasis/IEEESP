import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from common.audio_watermark_common import (
    make_audio_datasets,
    get_models,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "data"))


def generate_mask_with_ratio(shape, zero_ratio=0.5, seed=2025):
    rng = np.random.RandomState(seed)
    c, t = shape
    total = t
    mask = np.ones(total, dtype=np.float32)
    num_zero = int(total * zero_ratio)
    idx = rng.choice(total, num_zero, replace=False)
    mask[idx] = 0.0
    mask = torch.tensor(mask.reshape(1, t), dtype=torch.float32).repeat(c, 1)
    return mask


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


def _single_batch_pgd_attack(model_front, model_back, waves, target_labels, mask, num_classes,
                             eps=0.05, alpha=0.002, steps=80, device="cuda"):
    ce = nn.CrossEntropyLoss()
    x = waves.clone().detach().to(device)
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
            x = torch.clamp(ori + eta, -1, 1)
            x = x * (1 - m) + ori * m
        x = x.detach().requires_grad_(True)
    return x.detach()


def masked_pgd_attack(model_front, model_back, waves, targets, mask, num_classes,
                      eps=0.05, alpha=0.002, steps=80, device="cuda", attack_batch_size=64):
    outs = []
    for i in range(0, len(waves), attack_batch_size):
        w = waves[i : i + attack_batch_size]
        t = targets[i : i + attack_batch_size]
        outs.append(
            _single_batch_pgd_attack(
                model_front, model_back, w, t, mask, num_classes,
                eps=eps, alpha=alpha, steps=steps, device=device
            ).cpu()
        )
    return torch.cat(outs, dim=0)


@torch.no_grad()
def targeted_success_rate(model_front, model_back, adv_waves, target_labels, mask,
                          device, num_classes, topk=1):
    total, success = 0, 0
    for i in range(0, len(adv_waves), 64):
        x = adv_waves[i : i + 64].to(device)
        y = target_labels[i : i + 64].to(device)
        logits = model_back(model_front(x))[:, :num_classes]
        top = logits.topk(topk, dim=1).indices
        if topk == 1:
            success += (top[:, 0] == y).sum().item()
        else:
            success += (top == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.numel()
    return success / max(1, total)


def main():
    parser = argparse.ArgumentParser(description="Client-side watermark generation for audio Split Learning")
    parser.add_argument("--dataset_name", type=str, default="speechcommands")
    parser.add_argument("--model", type=str, required=True, choices=["resnet34"])
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts_dir", type=str, default="./artifacts_dir_audio")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cleanset_max", type=int, default=400)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.002)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--attack_batch_size", type=int, default=64)
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--client_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    print(f"\n====== Configuration for {os.path.basename(__file__)} ======")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 50)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(os.path.join(args.artifacts_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_classes = int(meta["num_classes"])
    model_name = meta.get("model", "resnet34")
    pretrained = meta.get("pretrained", False)

    _, test_ds, _, _ = make_audio_datasets(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        train_manifest=None,
        test_manifest=None,
        commands_csv=str(meta.get("commands", "")),
        target_sample_rate=int(meta["target_sample_rate"]),
        target_num_samples=int(meta["target_num_samples"]),
        download=False,
    )

    clean_subset = Subset(test_ds, list(range(min(args.cleanset_max, len(test_ds)))))
    loader = DataLoader(clean_subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    waves, labels = [], []
    for x, y in loader:
        waves.append(x)
        labels.append(y)
    clean_waves = torch.cat(waves, dim=0)
    clean_labels = torch.cat(labels, dim=0)
    print(f"[INFO] Loaded clean set: {len(clean_waves)} audio clips")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client, server, _ = get_models(model_name, num_classes,
                                   sample_rate=int(meta["target_sample_rate"]),
                                   pretrained=pretrained)

    client.load_state_dict(torch.load(os.path.join(args.artifacts_dir, "client.pt"),
                                      map_location="cpu")["state_dict"])
    server.load_state_dict(torch.load(os.path.join(args.artifacts_dir, "server.pt"),
                                      map_location="cpu")["state_dict"])

    client.to(device)
    server.to(device)
    client.eval()
    server.eval()

    wav_shape = tuple(clean_waves[0].shape)
    client_mask = generate_mask_with_ratio(wav_shape, zero_ratio=0.4, seed=args.seed)

    all_target_clients = generate_per_client_target_labels(
        args.num_clients, clean_labels, num_classes, seed=args.seed
    )
    target_client = all_target_clients[args.client_idx]

    print(f"\n=== Client Watermarking (Masked PGD on waveforms using model: {model_name}) ===")
    adv_c = masked_pgd_attack(
        client, server, clean_waves, target_client, client_mask, num_classes,
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
            "adv_waves": adv_c,
            "targets": target_client,
            "clean_labels": clean_labels,
            "mask": client_mask,
            "client_verify_acc_top1": acc_c1,
            "client_verify_acc_top2": acc_c2,
        },
        os.path.join(args.artifacts_dir, "client_watermark_audio.pt"),
    )


if __name__ == "__main__":
    main()
