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

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from common.text_watermark_common import (
    build_vocab,
    get_models,
    load_builtin_dataset,
    read_examples,
    test_accuracy,
    TextClassificationDataset,
    PadCollate,
    PAD_IDX,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))


def train_rotating_sl(client, server, client_loaders, test_loader, args, device, init_acc=0.0):
    client.to(device)
    server.to(device)

    ce = nn.CrossEntropyLoss()
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
                for token_ids, lengths, labels in loader:
                    token_ids = token_ids.to(device)
                    lengths = lengths.to(device)
                    labels = labels.to(device)

                    smashed = client(token_ids, lengths)
                    logits = server(smashed, lengths)
                    loss = ce(logits, labels)

                    opt_c.zero_grad()
                    opt_s.zero_grad()
                    loss.backward()
                    opt_c.step()
                    opt_s.step()
                    total_loss += float(loss.item())

                acc = test_accuracy(client, server, test_loader, device)
                msg = (
                    f"Round {r+1}/{args.rounds} | Client {cid} | Local Epoch {le+1}/{args.local_epochs} | "
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
    p = argparse.ArgumentParser(description="Step1 Split Learning training for text classification")
    p.add_argument("--dataset_name", type=str, choices=["ag_news"], default=None,
                    help="Use a builtin dataset instead of local files.")
    p.add_argument("--train_path", type=str, default=None, help="Path to train .csv or .jsonl")
    p.add_argument("--test_path", type=str, default=None, help="Path to test .csv or .jsonl")
    p.add_argument("--text_col", type=str, default="text")
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--num_classes", type=int, default=None,
                    help="Optional for builtin datasets; auto-detected if omitted.")
    p.add_argument("--model", type=str, default="bert", choices=["bert"])
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--num_clients", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--min_freq", type=int, default=2)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--emb_dim", type=int, default=768)
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--artifacts_dir", type=str, default="./artifacts_dir_text")
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--num_workers", type=int, default=16)
    args = p.parse_args()

    print(f"\n====== Configuration for {os.path.basename(__file__)} ======")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 50)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    if args.dataset_name:
        train_examples, test_examples, auto_num_classes = load_builtin_dataset(
            args.dataset_name, data_dir=args.data_dir
        )
        if args.num_classes is None:
            args.num_classes = auto_num_classes
    else:
        if not args.train_path or not args.test_path:
            raise ValueError("For local files, both --train_path and --test_path are required.")
        train_examples = read_examples(args.train_path, args.text_col, args.label_col)
        test_examples = read_examples(args.test_path, args.text_col, args.label_col)
        if args.num_classes is None:
            labels = {ex.label for ex in train_examples}
            args.num_classes = len(labels)

    assert args.num_classes is not None

    vocab = build_vocab(train_examples, min_freq=args.min_freq)
    with open(os.path.join(args.artifacts_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    train_dataset = TextClassificationDataset(train_examples, vocab, max_len=args.max_len)
    test_dataset = TextClassificationDataset(test_examples, vocab, max_len=args.max_len)

    collate_fn = PadCollate(PAD_IDX)

    indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)
    split_indices = np.array_split(indices, args.num_clients)

    client_loaders = [
        DataLoader(
            Subset(train_dataset, idxs.tolist()),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
        for idxs in split_indices
        if len(idxs) > 0
    ]
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    client, server, _ = get_models(
        model_name=args.model,
        num_classes=args.num_classes,
        vocab_size=len(vocab),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        pretrained=args.pretrained,
        pretrained_dir=os.path.join("pretrained_dir", args.dataset_name, args.model) if args.pretrained else None,
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
        "num_classes": args.num_classes,
        "vocab_size": len(vocab),
        "max_len": args.max_len,
        "emb_dim": args.emb_dim,
        "hidden_dim": args.hidden_dim,
    }
    with open(os.path.join(args.artifacts_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Artifacts saved under: {args.artifacts_dir}")


if __name__ == "__main__":
    main()
