import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from common.text_watermark_common import (
    get_models,
    load_builtin_dataset,
    read_examples,
    test_accuracy,
    TextClassificationDataset,
    PadCollate,
    PAD_IDX,
    UNK_IDX,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "data")


class FrontEmbedWrapper(nn.Module):
    """Wrapper to pass embeddings directly to the client model."""
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, emb, lengths):
        return self.base.forward_from_embeddings(emb, lengths)


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


@torch.no_grad()
def targeted_success_rate(model_front, model_back, adv_ids, lengths, target_labels,
                          device, topk=1, batch_size=64):
    total, success = 0, 0
    model_front.eval()
    model_back.eval()

    for i in range(0, len(adv_ids), batch_size):
        x = adv_ids[i : i + batch_size].to(device)
        l = lengths[i : i + batch_size].to(device)
        y = target_labels[i : i + batch_size].to(device)

        logits = model_back(model_front(x, l), l)
        topk_idx = logits.topk(topk, dim=1).indices

        if topk == 1:
            success += (topk_idx[:, 0] == y).sum().item()
        else:
            success += (topk_idx == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.numel()

    return success / max(1, total)


def build_token_masks(lengths, edit_ratio, seed):
    """Build per-sample boolean masks indicating which token positions are editable."""
    rng = np.random.RandomState(seed)
    masks = []
    for l in lengths.tolist():
        l = int(l)
        editable = max(1, int(l * edit_ratio))
        editable = min(editable, l)
        idx = rng.choice(l, size=editable, replace=False)
        m = torch.zeros(l, dtype=torch.bool)
        m[idx] = True
        masks.append(m)
    return masks


def _build_candidate_token_ids(embedding_weight, grad_at_pos, current_token_id, candidate_size):
    """Select candidate token IDs based on gradient direction, excluding special tokens."""
    scores = torch.matmul(embedding_weight, grad_at_pos)
    scores = scores.clone()

    special_tokens = {PAD_IDX, UNK_IDX, 101, 102, 103}
    for st in special_tokens:
        if st < scores.size(0):
            scores[st] = float("inf")

    sorted_ids = torch.argsort(scores, descending=False).tolist()
    chosen = [current_token_id]
    chosen_set = {current_token_id} | special_tokens
    for tok in sorted_ids:
        if tok in chosen_set:
            continue
        chosen.append(int(tok))
        chosen_set.add(int(tok))
        if len(chosen) >= candidate_size:
            break

    while len(chosen) < candidate_size:
        chosen.append(current_token_id)

    return torch.tensor(chosen, dtype=torch.long)


def batched_adaptive_soft_token_optimization_attack(
    model_front, model_back, token_ids, lengths, target_labels, editable_masks,
    candidate_size, steps, temperature_start, temperature_end, soft_lr,
    sparse_weight, entropy_weight, adaptive_sparse_scale, candidate_refresh_interval,
    projection_sweeps, seed, device, attack_batch_size=64,
):
    """Batched adaptive soft-token optimization attack for text watermarking."""
    model_front.eval()
    model_back.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)

    front_mod = model_front
    embedding_weight = front_mod.embedding.weight.detach().to(device)
    adv = token_ids.clone()
    dataset_size = len(adv)

    front_embed_dp = FrontEmbedWrapper(front_mod)

    total_gap = 0.0
    total_max_prob = 0.0
    optimized_samples = 0
    target_hits = 0

    for chunk_start in range(0, dataset_size, attack_batch_size):
        chunk_end = min(chunk_start + attack_batch_size, dataset_size)
        x = adv[chunk_start:chunk_end].clone().to(device)
        l = lengths[chunk_start:chunk_end].clone().to(device)
        y = target_labels[chunk_start:chunk_end].clone().to(device)
        masks = editable_masks[chunk_start:chunk_end]

        B = x.size(0)

        batch_idx, seq_idx = [], []
        for b_i, m in enumerate(masks):
            pos = torch.where(m)[0].tolist()
            batch_idx.extend([b_i] * len(pos))
            seq_idx.extend(pos)

        if len(batch_idx) == 0:
            continue

        N = len(batch_idx)

        def refresh_cands(current_x):
            emb = front_mod.embedding(current_x).detach().clone()
            emb.requires_grad_(True)
            logits = model_back(front_embed_dp(emb, l), l)
            loss = F.cross_entropy(logits, y)

            model_front.zero_grad(set_to_none=True)
            model_back.zero_grad(set_to_none=True)
            loss.backward()
            grad = emb.grad.detach()

            cand_ids_list = []
            for k in range(N):
                b = batch_idx[k]
                s = seq_idx[k]
                cands = _build_candidate_token_ids(
                    embedding_weight, grad[b, s], int(current_x[b, s].item()), candidate_size
                )
                cand_ids_list.append(cands)

            c_ids = torch.stack(cand_ids_list, dim=0).to(device)
            c_embs = embedding_weight[c_ids]
            return c_ids, c_embs

        candidate_ids_t, candidate_embeds = refresh_cands(x)

        init_logits = torch.full((N, candidate_ids_t.size(1)), fill_value=-4.0, device=device)
        init_logits[:, 0] = 4.0
        soft_logits = nn.Parameter(init_logits)
        optimizer = torch.optim.Adam([soft_logits], lr=soft_lr)

        with torch.no_grad():
            initial_logits = model_back(model_front(x, l), l)
            best_target_logit = initial_logits[torch.arange(B), y].clone()
            best_discrete = x.clone()

        final_gap = 1.0
        final_max_prob = 0.0

        print(f"  -> Optimizing Chunk [{chunk_start}:{chunk_end}], editable tokens: {N}")

        for step_idx in range(steps):
            if step_idx > 0 and candidate_refresh_interval > 0 and step_idx % candidate_refresh_interval == 0:
                x = best_discrete.clone()
                candidate_ids_t, candidate_embeds = refresh_cands(x)

                refreshed_logits = torch.full_like(soft_logits.detach(), fill_value=-4.0)
                refreshed_logits[:, 0] = 4.0
                soft_logits = nn.Parameter(refreshed_logits)
                optimizer = torch.optim.Adam([soft_logits], lr=soft_lr)

            progress = float(step_idx + 1) / float(max(1, steps))
            tau = temperature_start + (temperature_end - temperature_start) * progress
            tau = max(temperature_end, tau)

            probs = F.softmax(soft_logits / tau, dim=-1)
            dense_gap = (1.0 - probs.max(dim=-1).values).mean()
            entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=-1).mean()
            adaptive_weight = sparse_weight * (1.0 + adaptive_sparse_scale * progress * float(dense_gap.detach().item()))

            soft_emb = front_mod.embedding(x).detach().clone()
            weighted_emb = torch.einsum("nc,ncd->nd", probs, candidate_embeds)
            soft_emb[batch_idx, seq_idx] = weighted_emb

            logits = model_back(front_embed_dp(soft_emb, l), l)
            cls_loss = F.cross_entropy(logits, y)
            loss = cls_loss + adaptive_weight * dense_gap + entropy_weight * progress * entropy

            optimizer.zero_grad(set_to_none=True)
            model_front.zero_grad(set_to_none=True)
            model_back.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                eval_list = []

                projected_choice = probs.argmax(dim=-1)
                full_argmax_ids = x.clone()
                full_argmax_ids[batch_idx, seq_idx] = candidate_ids_t[torch.arange(N, device=device), projected_choice]
                eval_list.append(full_argmax_ids)

                for mut_rate in [0.1, 0.3, 0.5]:
                    update_mask = torch.rand(N, device=device) < mut_rate
                    blend_ids = best_discrete.clone()
                    blend_ids[batch_idx, seq_idx] = torch.where(
                        update_mask, full_argmax_ids[batch_idx, seq_idx], blend_ids[batch_idx, seq_idx]
                    )
                    eval_list.append(blend_ids)

                for _ in range(2):
                    rand_choice = torch.multinomial(probs, 1).squeeze(-1)
                    rand_ids = x.clone()
                    rand_ids[batch_idx, seq_idx] = candidate_ids_t[torch.arange(N, device=device), rand_choice]

                    update_mask = torch.rand(N, device=device) < 0.2
                    blend_ids = best_discrete.clone()
                    blend_ids[batch_idx, seq_idx] = torch.where(
                        update_mask, rand_ids[batch_idx, seq_idx], blend_ids[batch_idx, seq_idx]
                    )
                    eval_list.append(blend_ids)

                for p_ids in eval_list:
                    p_logits = model_back(model_front(p_ids, l), l)
                    p_target_logits = p_logits[torch.arange(B), y]

                    improved_mask = p_target_logits > best_target_logit
                    if improved_mask.any():
                        best_target_logit[improved_mask] = p_target_logits[improved_mask]
                        best_discrete[improved_mask] = p_ids[improved_mask]

                final_gap = float(dense_gap.item())
                final_max_prob = float(probs.max(dim=-1).values.mean().item())

        # Greedy polish for hard samples
        with torch.no_grad():
            final_logits = model_back(model_front(best_discrete, l), l)
            final_preds = final_logits.argmax(dim=1)
            failed_indices = torch.where(final_preds != y)[0].tolist()

        if failed_indices and projection_sweeps > 0:
            print(f"    -> [Greedy Polish] Refining {len(failed_indices)} hard samples...")
            b_array = np.array(batch_idx)
            s_array = np.array(seq_idx)

            for b_id in failed_indices:
                s_mask = b_array == b_id
                s_seq = s_array[s_mask]
                if len(s_seq) == 0:
                    continue

                cur_x = best_discrete[b_id:b_id+1].clone()
                cur_l = l[b_id:b_id+1].clone()
                cur_y = y[b_id:b_id+1].clone()
                best_score = best_target_logit[b_id].item()

                for _ in range(projection_sweeps):
                    improved_in_sweep = False

                    # Compute gradient for this failed sample
                    emb = front_mod.embedding(cur_x).detach().clone()
                    emb.requires_grad_(True)
                    logits = model_back(front_embed_dp(emb, cur_l), cur_l)
                    loss = F.cross_entropy(logits, cur_y)
                    model_front.zero_grad(set_to_none=True)
                    model_back.zero_grad(set_to_none=True)
                    loss.backward()
                    g = emb.grad.detach()[0]

                    # Per-token greedy search
                    for pos in s_seq:
                        cands = _build_candidate_token_ids(
                            embedding_weight, g[pos], int(cur_x[0, pos].item()), min(candidate_size, 32)
                        )

                        batch_eval = cur_x.repeat(len(cands), 1)
                        batch_eval[:, pos] = cands
                        l_eval = cur_l.repeat(len(cands))

                        with torch.no_grad():
                            eval_logits = model_back(model_front(batch_eval, l_eval), l_eval)
                            scores = eval_logits[:, cur_y[0]]
                            max_idx = scores.argmax().item()

                            if scores[max_idx].item() > best_score:
                                best_score = scores[max_idx].item()
                                cur_x = batch_eval[max_idx:max_idx+1].clone()
                                improved_in_sweep = True

                                if eval_logits[max_idx].argmax().item() == cur_y[0].item():
                                    break

                    # Stop if attack already succeeded for this sample
                    with torch.no_grad():
                        if model_back(model_front(cur_x, cur_l), cur_l).argmax().item() == cur_y[0].item():
                            break

                best_discrete[b_id] = cur_x[0]

        # Final hit rate statistics
        with torch.no_grad():
            final_logits = model_back(model_front(best_discrete, l), l)
            final_preds = final_logits.argmax(dim=1)
            target_hits += (final_preds == y).sum().item()

        adv[chunk_start:chunk_end] = best_discrete.cpu()
        total_gap += final_gap * B
        total_max_prob += final_max_prob * B
        optimized_samples += B

    metrics = {
        "avg_projection_gap": total_gap / max(1, optimized_samples),
        "avg_max_prob": total_max_prob / max(1, optimized_samples),
        "target_hit_rate_during_opt": target_hits / max(1, optimized_samples),
    }
    return adv, metrics


def main():
    parser = argparse.ArgumentParser(description="Client-side text watermark generation")
    parser.add_argument("--dataset_name", type=str, choices=["ag_news"], default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts_dir", type=str, default="./artifacts_dir_text")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--cleanset_max", type=int, default=300)
    parser.add_argument("--attack_batch_size", type=int, default=128)
    parser.add_argument("--model", type=str, default="bert", choices=["bert"])
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--client_idx", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--candidate_size", type=int, default=32)
    parser.add_argument("--client_edit_ratio", type=float, default=0.35)
    parser.add_argument("--attack_mode", type=str, choices=["adaptive_soft"], default="adaptive_soft")
    parser.add_argument("--soft_lr", type=float, default=0.05)
    parser.add_argument("--temperature_start", type=float, default=2.0)
    parser.add_argument("--temperature_end", type=float, default=0.1)
    parser.add_argument("--sparse_weight", type=float, default=1.0)
    parser.add_argument("--entropy_weight", type=float, default=0.05)
    parser.add_argument("--adaptive_sparse_scale", type=float, default=4.0)
    parser.add_argument("--candidate_refresh_interval", type=int, default=5)
    parser.add_argument("--projection_sweeps", type=int, default=2)
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

    os.makedirs(args.artifacts_dir, exist_ok=True)
    meta_path = os.path.join(args.artifacts_dir, "meta.json")
    vocab_path = os.path.join(args.artifacts_dir, "vocab.json")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    num_classes = int(meta["num_classes"])
    max_len = int(meta["max_len"])
    emb_dim = int(meta.get("emb_dim", 768))
    hidden_dim = int(meta.get("hidden_dim", 768))
    model_name = args.model or meta.get("model", "bert")
    pretrained = meta.get("pretrained", False)

    if args.dataset_name:
        _, test_examples, _ = load_builtin_dataset(args.dataset_name, args.data_dir)
    else:
        test_examples = read_examples(args.test_path, args.text_col, args.label_col)

    dataset = TextClassificationDataset(test_examples, vocab=vocab, max_len=max_len)
    clean_subset = Subset(dataset, list(range(min(args.cleanset_max, len(dataset)))))
    collate_fn = PadCollate(PAD_IDX, fixed_max_len=max_len)
    clean_loader = DataLoader(
        clean_subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_ids, all_lengths, all_labels = [], [], []
    for ids, lengths, labels in clean_loader:
        all_ids.append(ids)
        all_lengths.append(lengths)
        all_labels.append(labels)
    clean_ids = torch.cat(all_ids, dim=0)
    clean_lengths = torch.cat(all_lengths, dim=0)
    clean_labels = torch.cat(all_labels, dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client, server, _ = get_models(
        model_name=model_name, num_classes=num_classes, vocab_size=len(vocab),
        emb_dim=emb_dim, hidden_dim=hidden_dim, pretrained=pretrained,
        pretrained_dir=args.artifacts_dir,
    )

    client.to(device)
    server.to(device)
    client.eval()
    server.eval()

    all_target_labels = generate_per_client_target_labels(
        args.num_clients, clean_labels, num_classes, seed=args.seed
    )
    target_labels_client = all_target_labels[args.client_idx]
    client_masks = build_token_masks(clean_lengths, args.client_edit_ratio, seed=args.seed)

    print(f"\n=== Client Watermarking (batched adaptive soft-token optimization) ===")
    adv_ids_c, attack_metrics = batched_adaptive_soft_token_optimization_attack(
        model_front=client,
        model_back=server,
        token_ids=clean_ids,
        lengths=clean_lengths,
        target_labels=target_labels_client,
        editable_masks=client_masks,
        candidate_size=args.candidate_size,
        steps=args.steps,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        soft_lr=args.soft_lr,
        sparse_weight=args.sparse_weight,
        entropy_weight=args.entropy_weight,
        adaptive_sparse_scale=args.adaptive_sparse_scale,
        candidate_refresh_interval=args.candidate_refresh_interval,
        projection_sweeps=args.projection_sweeps,
        seed=args.seed,
        device=device,
        attack_batch_size=args.attack_batch_size,
    )

    acc_c1 = targeted_success_rate(client, server, adv_ids_c, clean_lengths,
                                   target_labels_client, device, topk=1)
    acc_c2 = targeted_success_rate(client, server, adv_ids_c, clean_lengths,
                                   target_labels_client, device, topk=2)
    print(f"Client watermark verification accuracy (Top-1) = {acc_c1 * 100:.2f}%")
    print(f"Client watermark verification accuracy (Top-2) = {acc_c2 * 100:.2f}%")

    for key, value in attack_metrics.items():
        print(f"{key} = {value:.4f}")

    torch.save(
        {
            "adv_token_ids": adv_ids_c.cpu(),
            "lengths": clean_lengths.cpu(),
            "targets": target_labels_client.cpu(),
            "clean_labels": clean_labels.cpu(),
            "editable_masks": [m.cpu() for m in client_masks],
            "client_verify_acc_top1": acc_c1,
            "client_verify_acc_top2": acc_c2,
            "attack_mode": args.attack_mode,
            "attack_metrics": attack_metrics,
        },
        os.path.join(args.artifacts_dir, "client_watermark_text.pt"),
    )


if __name__ == "__main__":
    main()
