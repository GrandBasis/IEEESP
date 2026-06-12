import os
import shutil
import zipfile
from typing import Optional

import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm


def download_tiny_imagenet(data_dir):
    """Download and organize Tiny ImageNet dataset"""
    os.makedirs(data_dir, exist_ok=True)
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    extract_path = os.path.join(data_dir, "tiny-imagenet-200")

    if os.path.exists(extract_path) and len(os.listdir(extract_path)) == 3:
        print("Tiny ImageNet dataset already exists and is organized")
        return extract_path

    if not os.path.exists(zip_path):
        print("Downloading Tiny ImageNet dataset...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(zip_path, 'wb') as f, tqdm(
            desc="Download progress",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                pbar.update(len(data))

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    print("Organizing validation set structure...")
    val_dir = os.path.join(extract_path, "val")
    val_images_dir = os.path.join(val_dir, "images")
    val_annotations_file = os.path.join(val_dir, "val_annotations.txt")

    with open(val_annotations_file, 'r') as f:
        val_annotations = f.readlines()

    for line in val_annotations:
        filename, class_name, *_ = line.strip().split('\t')
        class_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        src_path = os.path.join(val_images_dir, filename)
        dst_path = os.path.join(class_dir, filename)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)

    if os.path.exists(val_images_dir):
        shutil.rmtree(val_images_dir)
    if os.path.exists(val_annotations_file):
        os.remove(val_annotations_file)

    print("Tiny ImageNet dataset download and organization complete!")
    return extract_path


def make_image_datasets(dataset_name: str, data_dir: str):
    name = dataset_name.lower()

    # ViT requires 224x224 input resolution
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    if name == "tiny-imagenet":
        root = download_tiny_imagenet(data_dir)
        train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=base_transform)
        test_ds = datasets.ImageFolder(os.path.join(root, "val"), transform=base_transform)
        num_classes = 200
    else:
        raise ValueError(f"Unsupported dataset: {name}. Supported: tiny-imagenet")

    return train_ds, test_ds, num_classes


# =====================================================================
# Models: Vision Transformer (ViT-B-16) Split
# =====================================================================

class ImageViTClient(nn.Module):
    def __init__(self, base_vit, split_layer_idx=3):
        super().__init__()
        # Built-in normalization (compatible with Step 2 PGD attack)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Extract Patch Embedding, Class Token, and Positional Embedding
        self.conv_proj = base_vit.conv_proj
        self.class_token = base_vit.class_token
        self.pos_embedding = base_vit.encoder.pos_embedding
        self.dropout = base_vit.encoder.dropout

        # Extract first N Transformer blocks
        self.client_layers = base_vit.encoder.layers[:split_layer_idx]

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.conv_proj.kernel_size[0]
        torch._assert(h == self.conv_proj.in_channels, f"Wrong input shape {x.shape}")
        torch._assert(h % p == 0 and w % p == 0, f"Image size must be divisible by patch size")
        x = self.conv_proj(x)
        x = x.reshape(n, x.shape[1], -1).permute(0, 2, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize
        x = (x - self.mean) / self.std

        # Patch Embedding
        n = x.shape[0]
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # Add Class Token & Positional Embedding
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        # Forward through client layers
        return self.client_layers(x)


class ImageViTServer(nn.Module):
    def __init__(self, base_vit, num_classes: int, split_layer_idx=3):
        super().__init__()
        # Extract remaining Transformer blocks
        self.server_layers = base_vit.encoder.layers[split_layer_idx:]
        self.ln = base_vit.encoder.ln

        # Replace classification head
        self.heads = nn.Linear(base_vit.heads.head.in_features, num_classes)

    def forward(self, smashed: torch.Tensor) -> torch.Tensor:
        x = self.server_layers(smashed)
        x = self.ln(x)

        # Extract Class Token [CLS] for classification
        x = x[:, 0]
        return self.heads(x)


class ImageViTSurrogateClient(nn.Module):
    def __init__(self, client_module):
        super().__init__()
        self.base = client_module
        hidden_dim = 768  # ViT-B-16 hidden size
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + self.adapter(y)


def get_models(model_name: str, num_classes: int, pretrained: bool = False, pretrained_dir: Optional[str] = None):
    model_name = model_name.lower()

    if model_name == "vit-b-16":
        print("Loading Torchvision ViT-B-16 ImageNet Pretrained Weights...")
        base_vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        client = ImageViTClient(base_vit, split_layer_idx=3)
        server = ImageViTServer(base_vit, num_classes, split_layer_idx=3)
        surrogate = ImageViTSurrogateClient(client)
    else:
        raise ValueError("Unsupported model.")

    if pretrained and pretrained_dir:
        client_path = os.path.join(pretrained_dir, "client.pt")
        server_path = os.path.join(pretrained_dir, "server.pt")

        if os.path.exists(client_path):
            print(f"Loading local SL Client weights from: {client_path}")
            ckpt_c = torch.load(client_path, map_location="cpu")
            client.load_state_dict(ckpt_c.get("state_dict", ckpt_c))
            surrogate.base.load_state_dict(ckpt_c.get("state_dict", ckpt_c))

        if os.path.exists(server_path):
            print(f"Loading local SL Server weights from: {server_path}")
            ckpt_s = torch.load(server_path, map_location="cpu")
            server.load_state_dict(ckpt_s.get("state_dict", ckpt_s))

    return client, server, surrogate


@torch.no_grad()
def test_accuracy(client: nn.Module, server: nn.Module, loader, device: torch.device) -> float:
    client.eval()
    server.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = server(client(x)).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)
