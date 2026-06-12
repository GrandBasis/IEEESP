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

from common.image_watermark_common import (
    make_image_datasets,
    get_models,
    test_accuracy,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "data"))


def train_rotating_sl(client, server, client_dataloaders, test_loader, args, device, init_acc=0.0):
    client.to(device)
    server.to(device)

    # ViT fine-tuning benefits from AdamW with weight decay
    criterion = nn.CrossEntropyLoss()
    opt_c = optim.AdamW(client.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_s = optim.AdamW(server.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = init_acc
    best_client = copy.deepcopy(client.state_dict())
    best_server = copy.deepcopy(server.state_dict())
    if init_acc > 0.0:
        torch.save({"state_dict": best_client}, os.path.join(args.artifacts_dir, "client.pt"))
        torch.save({"state_dict": best_server}, os.path.join(args.artifacts_dir, "server.pt"))

    for r in range(args.rounds):
        for cid, loader in enumerate(client_dataloaders):
            print(f"\n--- Round {r + 1}/{args.rounds} | Client {cid} ---")
            for local_epoch in range(args.local_epochs):
                client.train()
                server.train()
                total_loss = 0.0

                for x, y in loader:
                    x, y = x.to(device), y.to(device)

                    smashed = client(x)
                    logits = server(smashed)
                    loss = criterion(logits, y)

                    opt_c.zero_grad()
                    opt_s.zero_grad()
                    loss.backward()
                    opt_c.step()
                    opt_s.step()
                    total_loss += loss.item()

                acc = test_accuracy(client, server, test_loader, device)
                avg_loss = total_loss / max(1, len(loader))
                log_msg = (
                    f"Round {r + 1}/{args.rounds} | Client {cid} | "
                    f"Epoch {local_epoch + 1}/{args.local_epochs} | "
                    f"Loss={avg_loss:.4f} | Test Acc={acc * 100:.2f}%"
                )

                if acc > best_acc:
                    best_acc = acc
                    best_client = copy.deepcopy(client.state_dict())
                    best_server = copy.deepcopy(server.state_dict())
                    torch.save({"state_dict": best_client}, os.path.join(args.artifacts_dir, "client.pt"))
                    torch.save({"state_dict": best_server}, os.path.join(args.artifacts_dir, "server.pt"))
                    log_msg += "  <-- New Best! Saved."

                print(log_msg)

    print(f"\nTraining finished. Best Acc={best_acc * 100:.2f}%")
    # Restore best model
    client.load_state_dict(best_client)
    server.load_state_dict(best_server)
    return client, server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, choices=["tiny-imagenet"], default="tiny-imagenet")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts_dir", type=str, default="./artifacts_dir_image")
    parser.add_argument("--model", type=str, default="vit-b-16", choices=["vit-b-16"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.artifacts_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n====== Configuration for {os.path.basename(__file__)} ======")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 50)

    train_ds, test_ds, num_classes = make_image_datasets(args.dataset_name, args.data_dir)

    indices = np.arange(len(train_ds))
    np.random.shuffle(indices)
    split_indices = np.array_split(indices, args.num_clients)

    client_dataloaders = [
        DataLoader(Subset(train_ds, idxs.tolist()), batch_size=args.batch_size, shuffle=True,
                   num_workers=args.num_workers, pin_memory=True)
        for idxs in split_indices if len(idxs) > 0
    ]
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

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

    train_rotating_sl(client, server, client_dataloaders, test_loader, args, device, init_acc=init_acc)

    meta = {
        "dataset_name": args.dataset_name,
        "model": args.model,
        "num_classes": num_classes,
    }
    with open(os.path.join(args.artifacts_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
