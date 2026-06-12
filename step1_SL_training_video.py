import argparse
import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from common.video_watermark_common import (
    VideoClassificationDataset,
    resolve_video_dataset,
    test_accuracy,
    get_models,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))


def train_rotating_sl(client, server, client_loaders, test_loader, args, device, init_acc=0.0):
    client.to(device)
    server.to(device)

    criterion = nn.CrossEntropyLoss()
    opt_c = optim.Adam(client.parameters(), lr=args.lr)
    opt_s = optim.Adam(server.parameters(), lr=args.lr)

    best_acc = init_acc
    best_client = copy.deepcopy(client.state_dict())
    best_server = copy.deepcopy(server.state_dict())
    if init_acc > 0.0:
        torch.save({"state_dict": best_client}, os.path.join(args.artifacts_dir, "client.pt"))
        torch.save({"state_dict": best_server}, os.path.join(args.artifacts_dir, "server.pt"))

    for r in range(args.rounds):
        for cid, loader in enumerate(client_loaders):
            print(f"\n--- Round {r+1}/{args.rounds} | Client {cid} ---")
            for le in range(args.local_epochs):
                client.train()
                server.train()
                total_loss = 0.0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    logits = server(client(x))
                    loss = criterion(logits, y)
                    opt_c.zero_grad()
                    opt_s.zero_grad()
                    loss.backward()
                    opt_c.step()
                    opt_s.step()
                    total_loss += float(loss.item())

                acc = test_accuracy(client, server, test_loader, device)
                msg = (
                    f"Round {r+1}/{args.rounds} | Client {cid} | "
                    f"Local Epoch {le+1}/{args.local_epochs} | "
                    f"Loss={total_loss/max(1,len(loader)):.4f} | Test Acc={acc*100:.2f}%"
                )
                if acc > best_acc:
                    best_acc = acc
                    best_client = copy.deepcopy(client.state_dict())
                    best_server = copy.deepcopy(server.state_dict())
                    torch.save({"state_dict": best_client}, os.path.join(args.artifacts_dir, "client.pt"))
                    torch.save({"state_dict": best_server}, os.path.join(args.artifacts_dir, "server.pt"))
                    msg += "  <-- New Best! Saved."
                print(msg)

    # Restore best model
    client.load_state_dict(best_client)
    server.load_state_dict(best_server)

    print(f"\nTraining finished. Best Acc={best_acc*100:.2f}%")
    return client, server


def main():
    parser = argparse.ArgumentParser(description="Step1 Split Learning training for video classification")
    parser.add_argument("--dataset_name", type=str, default="ucf101")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts_dir", type=str, default="./artifacts_dir_video")
    parser.add_argument("--model", type=str, required=True, choices=["tsn"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    print(f"\n====== Configuration for {os.path.basename(__file__)} ======")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Register MMAction2 data preprocessor for compatibility
    from mmengine.registry import MODELS
    import mmaction.models.data_preprocessors as data_preproc_module

    target_class_name = 'ActionDataPreprocessor'
    if target_class_name not in MODELS.module_dict:
        if hasattr(data_preproc_module, target_class_name):
            target_class = getattr(data_preproc_module, target_class_name)
            MODELS.register_module(name=target_class_name, module=target_class)
        else:
            from mmaction.models.data_preprocessors.data_preprocessor import ActionDataPreprocessor
            MODELS.register_module(name=target_class_name, module=ActionDataPreprocessor)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    train_ex, test_ex, num_classes = resolve_video_dataset(
        args.dataset_name, args.data_dir, None, None
    )

    train_ds = VideoClassificationDataset(train_ex, num_frames=args.num_frames,
                                          image_size=args.image_size, train=True)
    test_ds = VideoClassificationDataset(test_ex, num_frames=args.num_frames,
                                         image_size=args.image_size, train=False)

    indices = np.arange(len(train_ds))
    np.random.shuffle(indices)
    shards = np.array_split(indices, args.num_clients)

    client_loaders = [
        DataLoader(
            Subset(train_ds, s.tolist()),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )
        for s in shards if len(s) > 0
    ]
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    client, server, _ = get_models(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        pretrained_dir=os.path.join('pretrained_dir', args.dataset_name, args.model),
    )

    # Evaluate pretrained model accuracy
    client.to(device)
    server.to(device)
    client.eval()
    server.eval()
    init_acc = test_accuracy(client, server, test_loader, device)
    print(f"\nInitial accuracy (pretrained): {init_acc*100:.2f}%")

    train_rotating_sl(client, server, client_loaders, test_loader, args, device, init_acc=init_acc)

    meta = {
        "model": args.model,
        "pretrained": args.pretrained,
        "dataset_name": args.dataset_name,
        "num_classes": num_classes,
        "num_frames": args.num_frames,
        "image_size": args.image_size,
    }
    with open(os.path.join(args.artifacts_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Artifacts saved under: {args.artifacts_dir}")


if __name__ == "__main__":
    main()
