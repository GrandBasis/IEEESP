import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# =====================================================================
# Constants (Hardcoded for BERT)
# =====================================================================

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
PAD_IDX = 0
UNK_IDX = 100


# =====================================================================
# Data Helpers
# =====================================================================

@dataclass
class TextExample:
    text: str
    label: int


def read_examples(path: str, text_col: str = "text", label_col: str = "label") -> List[TextExample]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    examples: List[TextExample] = []

    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                examples.append(TextExample(text=str(obj[text_col]), label=int(obj[label_col])))
    elif path.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                examples.append(TextExample(text=str(row[text_col]), label=int(row[label_col])))
    else:
        raise ValueError("Only .csv and .jsonl are supported for text datasets.")

    if not examples:
        raise ValueError(f"No examples loaded from {path}")

    return examples


def load_builtin_dataset(
    dataset_name: str, data_dir: Optional[str] = None
) -> Tuple[List[TextExample], List[TextExample], int]:
    try:
        from datasets import DownloadConfig, load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'datasets'. Install with: pip install datasets"
        ) from exc

    name = dataset_name.lower()
    if name not in {"ag_news"}:
        raise ValueError(f"Unsupported builtin dataset: {dataset_name}. Supported: ['ag_news']")

    cache_dir = data_dir if data_dir else None
    try:
        ds = load_dataset(
            name,
            cache_dir=cache_dir,
            download_config=DownloadConfig(local_files_only=True),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load builtin dataset '{name}' from local cache only. "
            f"Please ensure it already exists under --data_dir ({cache_dir})."
        ) from exc
    
    train_split = ds["train"]
    test_split = ds["test"]

    train_examples = [TextExample(text=str(row["text"]), label=int(row["label"])) for row in train_split]
    test_examples = [TextExample(text=str(row["text"]), label=int(row["label"])) for row in test_split]

    labels = {ex.label for ex in train_examples}
    num_classes = len(labels)
    return train_examples, test_examples, num_classes


def build_vocab(examples: Sequence[TextExample], min_freq: int = 2) -> Dict[str, int]:
    """For BERT, we ignore min_freq and simply return the pretrained vocab to keep the interface identical"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='.model_cache')
    return tokenizer.get_vocab()


def encode_text(text: str, tokenizer: BertTokenizer, max_len: int) -> List[int]:
    # Do not add CLS/SEP to match the original GRU logic sequence behavior
    # and avoid editing special tokens in step 2.
    tokens = tokenizer.encode(text, add_special_tokens=False, max_length=max_len, truncation=True)
    if not tokens:
        tokens = [UNK_IDX]
    return tokens


# =====================================================================
# Datasets
# =====================================================================

class TextClassificationDataset(Dataset):
    def __init__(self, examples: Sequence[TextExample], vocab: Dict[str, int], max_len: int) -> None:
        self.items: List[Tuple[List[int], int]] = []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='.model_cache')
        for ex in examples:
            ids = encode_text(ex.text, tokenizer, max_len)
            self.items.append((ids, ex.label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        return self.items[idx]


class PadCollate:
    def __init__(self, pad_idx: int = PAD_IDX, fixed_max_len: Optional[int] = None) -> None:
        self.pad_idx = pad_idx
        self.fixed_max_len = fixed_max_len

    def __call__(
        self, batch: Sequence[Tuple[List[int], int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids_list, labels = zip(*batch)
        lengths = torch.tensor([len(x) for x in ids_list], dtype=torch.long)
        if self.fixed_max_len is not None:
            max_len = int(self.fixed_max_len)
        else:
            max_len = int(lengths.max().item())

        padded = torch.full((len(ids_list), max_len), self.pad_idx, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            cut = min(len(ids), max_len)
            padded[i, :cut] = torch.tensor(ids[:cut], dtype=torch.long)
        lengths = torch.clamp(lengths, max=max_len)

        labels_t = torch.tensor(labels, dtype=torch.long)
        return padded, lengths, labels_t


# =====================================================================
# Models: BERT-based Text Client / Server / Surrogate
# =====================================================================

class BertClient(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', cache_dir='.model_cache')
        # Expose the word embeddings so the watermark attacks in Step 2 can read/write gradients smoothly
        self.embedding = self.bert.embeddings.word_embeddings

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        max_len = token_ids.size(1)
        # Create attention mask from lengths
        mask = (torch.arange(max_len, device=token_ids.device)[None, :] < lengths[:, None]).long()
        out = self.bert(input_ids=token_ids, attention_mask=mask)
        return out.last_hidden_state

    def forward_from_embeddings(self, emb: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass starting from (possibly perturbed / grad-requiring) embeddings.
        Used by the gradient-guided attack to obtain dL/de_i.
        """
        max_len = emb.size(1)
        mask = (torch.arange(max_len, device=emb.device)[None, :] < lengths[:, None]).long()
        # HF transformers allows passing directly through `inputs_embeds`
        out = self.bert(inputs_embeds=emb, attention_mask=mask)
        return out.last_hidden_state


class BertServer(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        feat_dim = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, smashed: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Mean-pool valid timesteps only, consistent with original logic.
        mask = torch.arange(smashed.size(1), device=smashed.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(-1).float()
        summed = (smashed * mask).sum(dim=1)
        denom = lengths.unsqueeze(1).clamp_min(1).float()
        pooled = summed / denom
        return self.classifier(pooled)


class BertSurrogateClient(nn.Module):
    def __init__(self, client_module):
        super().__init__()
        self.base = client_module
        # Bert hidden state output: [B, seq_len, 768]
        hidden_dim = 768
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        y = self.base(token_ids, lengths)
        return y + self.adapter(y)


# =====================================================================
# Model Factory
# =====================================================================

def get_models(
    model_name: str,
    num_classes: int,
    vocab_size: int,
    emb_dim: int = 768,
    hidden_dim: int = 768,
    pretrained: bool = False,
    pretrained_dir: Optional[str] = None,
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    model_name = model_name.lower()

    if model_name == "bert":
        client = BertClient()
        server = BertServer(hidden_dim=hidden_dim, num_classes=num_classes)
        surrogate = BertSurrogateClient(client)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. Supported: ['bert']")

    # Load locally SL-finetuned weights
    if pretrained and pretrained_dir:
        client_path = os.path.join(pretrained_dir, "client.pt")
        server_path = os.path.join(pretrained_dir, "server.pt")

        if os.path.exists(client_path):
            print(f"Loading local SL Text Client weights from: {client_path}")
            ckpt_c = torch.load(client_path, map_location="cpu")
            client.load_state_dict(ckpt_c.get("state_dict", ckpt_c))
            # Sync surrogate base weights
            if hasattr(surrogate, "base") and surrogate.base is not client:
                surrogate.base.load_state_dict(ckpt_c.get("state_dict", ckpt_c))

        if os.path.exists(server_path):
            print(f"Loading local SL Text Server weights from: {server_path}")
            ckpt_s = torch.load(server_path, map_location="cpu")
            server.load_state_dict(ckpt_s.get("state_dict", ckpt_s))

    return client, server, surrogate


# =====================================================================
# Utilities
# =====================================================================

@torch.no_grad()
def test_accuracy(client: nn.Module, server: nn.Module, loader, device: torch.device) -> float:
    client.eval()
    server.eval()
    correct, total = 0, 0
    for token_ids, lengths, labels in loader:
        token_ids = token_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        smashed = client(token_ids, lengths)
        pred = server(smashed, lengths).argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / max(1, total)
