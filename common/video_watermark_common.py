import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.models as models
import decord
import mmaction
from mmaction.registry import MODELS
from mmengine.runner import load_checkpoint


@dataclass
class VideoExample:
    path: str
    label: int


def _default_manifest_paths(dataset_name: str, data_dir: str) -> Tuple[str, str]:
    root = os.path.join(data_dir, dataset_name.lower())
    return os.path.join(root, "train.csv"), os.path.join(root, "test.csv")


def load_video_examples_from_manifest(path: str, path_col: str = "path", label_col: str = "label") -> List[VideoExample]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")
    examples: List[VideoExample] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append(VideoExample(path=str(row[path_col]), label=int(row[label_col])))
    if not examples:
        raise ValueError(f"No rows loaded from {path}")
    return examples


def resolve_video_dataset(
    dataset_name: str,
    data_dir: str,
    train_manifest: Optional[str],
    test_manifest: Optional[str],
) -> Tuple[List[VideoExample], List[VideoExample], int]:
    name = dataset_name.lower()
    if name != "ucf101":
        raise ValueError("dataset_name must be: ucf101")

    tm, vm = _default_manifest_paths(name, data_dir)
    train_manifest = train_manifest or tm
    test_manifest = test_manifest or vm

    train_examples = load_video_examples_from_manifest(train_manifest)
    test_examples = load_video_examples_from_manifest(test_manifest)
    labels = {ex.label for ex in train_examples}
    return train_examples, test_examples, len(labels)


def _sample_or_pad_frames(video: torch.Tensor, num_frames: int, train: bool) -> torch.Tensor:
    # video: [T, H, W, C], uint8/float
    t = video.shape[0]
    if t == 0:
        return torch.zeros((num_frames, 112, 112, 3), dtype=torch.float32)
    if t >= num_frames:
        if train:
            start = int(torch.randint(0, t - num_frames + 1, (1,)).item())
        else:
            start = (t - num_frames) // 2
        out = video[start : start + num_frames]
    else:
        pad = video[-1:].repeat(num_frames - t, 1, 1, 1)
        out = torch.cat([video, pad], dim=0)
    return out


def preprocess_video_tensor(video: torch.Tensor, num_frames: int, image_size: int, train: bool) -> torch.Tensor:
    clip = _sample_or_pad_frames(video, num_frames=num_frames, train=train).float() / 255.0
    # [T,H,W,C] -> [T,C,H,W]
    clip = clip.permute(0, 3, 1, 2)
    clip = F.interpolate(clip, size=(image_size, image_size), mode="bilinear", align_corners=False)
    # [T,C,H,W] -> [C,T,H,W]
    clip = clip.permute(1, 0, 2, 3).contiguous()
    return clip


class VideoClassificationDataset(Dataset):
    def __init__(self, examples: Sequence[VideoExample], num_frames: int = 16, image_size: int = 112, train: bool = True):
        self.examples = list(examples)
        self.num_frames = num_frames
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        try:
            # Load video frames using decord
            vr = decord.VideoReader(ex.path)
            # Sample frame indices: random for training, center for evaluation
            if self.train:
                start_idx = torch.randint(0, len(vr) - self.num_frames + 1, (1,)).item() if len(vr) >= self.num_frames else 0
            else:
                start_idx = (len(vr) - self.num_frames) // 2 if len(vr) >= self.num_frames else 0
            frame_indices = list(range(start_idx, start_idx + self.num_frames))
            # Read frames from decord
            frames = vr.get_batch(frame_indices)  # (T, H, W, C) numpy
            # Convert numpy array to torch tensor
            video = torch.from_numpy(frames.asnumpy())  # (T, H, W, C)
        except Exception as e:
            print(f"Error loading video {ex.path}: {e}")
            # Return a zero tensor on failure to avoid crashing the pipeline
            video = torch.zeros((self.num_frames, self.image_size, self.image_size, 3), dtype=torch.uint8)

        clip = preprocess_video_tensor(video, self.num_frames, self.image_size, self.train)
        return clip, ex.label


# =====================================================================
# Models: MMAction2 TSN-based Video Client / Server / Surrogate
# =====================================================================

class MMActionTSNClient(nn.Module):
    def __init__(self, mmaction_backbone):
        super().__init__()
        # Built-in normalization so external inputs stay in [0, 1], compatible with PGD attack clipping in Step 2
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Dynamically extract stem layers (everything before layer1, e.g. conv1, bn, relu, maxpool).
        # This relies on PyTorch registering submodules in initialization order,
        # avoiding attribute name mismatches across framework versions.
        stem_layers = []
        for name, child in mmaction_backbone.named_children():
            if name == 'layer1':
                break
            stem_layers.append(child)
        self.stem = nn.Sequential(*stem_layers)

        # Extract layer1 and layer2 from the backbone
        self.layer1 = mmaction_backbone.layer1
        self.layer2 = mmaction_backbone.layer2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W] in [0, 1]
        B, C, T, H, W = x.shape
        # Reshape to 2D per-frame processing: [B*T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # Apply built-in normalization
        x = (x - self.mean) / self.std

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)  # ResNet50 layer2 outputs 512 channels

        _, C_out, H_out, W_out = x.shape
        # Restore 3D tensor shape: [B, C_out, T, H_out, W_out]
        return x.view(B, T, C_out, H_out, W_out).permute(0, 2, 1, 3, 4).contiguous()


class MMActionTSNServer(nn.Module):
    def __init__(self, mmaction_backbone, num_classes: int):
        super().__init__()
        # Extract deep feature layers from the MMAction2 ResNet backbone
        self.layer3 = mmaction_backbone.layer3
        self.layer4 = mmaction_backbone.layer4

        # Rebuild the TSN classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(2048, num_classes)  # ResNet50 outputs 2048 channels

    def forward(self, smashed: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = smashed.shape
        x = smashed.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x).flatten(1)
        x = self.dropout(x)
        logits = self.fc(x)

        # TSN Temporal Consensus: average logits across all frames
        logits = logits.view(B, T, -1).mean(dim=1)
        return logits


class MMActionTSNSurrogateClient(nn.Module):
    def __init__(self, client_module):
        super().__init__()
        self.base = client_module
        # Client outputs 512 channels (ResNet50 layer2)
        self.adapter = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + self.adapter(y)


# =====================================================================
# Model Factory
# =====================================================================

def get_models(
    model_name: str,
    num_classes: int,
    dataset_name: str = "ucf101",
    pretrained: bool = False,
    pretrained_dir: Optional[str] = None,
) -> Tuple[nn.Module, nn.Module, nn.Module]:

    model_name = model_name.lower()
    dataset_name = dataset_name.lower()

    if model_name != "tsn":
        raise ValueError(f"Unsupported model_name: {model_name}. Supported: ['tsn']")

    # Pretrained backbone weights URL (K400-pretrained, suitable as a starting point for SL fine-tuning)
    CKPT_URLS = {
        "ucf101": "https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth",
    }

    ckpt_url = CKPT_URLS[dataset_name]
    print(f"Building MMAction2 TSN (ResNet50) and loading backbone weights from: {ckpt_url}")

    cfg = dict(
        type='Recognizer2D',
        backbone=dict(type='ResNet', depth=50, norm_eval=False),
        # num_classes here is arbitrary since we only extract the backbone, not this fc head
        cls_head=dict(type='TSNHead', num_classes=400, in_channels=2048, spatial_type='avg', consensus=dict(type='AvgConsensus', dim=1)),
        data_preprocessor=dict(type='ActionDataPreprocessor', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], format_shape='NCHW')
    )
    mmaction_model = MODELS.build(cfg)

    # Load remote pretrained weights (strict=False allows classification head mismatch)
    load_checkpoint(mmaction_model, ckpt_url, map_location='cpu', strict=False)

    # Build split-learning models; the server reinitializes its fc layer with the real num_classes
    client = MMActionTSNClient(mmaction_model.backbone)
    server = MMActionTSNServer(mmaction_model.backbone, num_classes)
    surrogate = MMActionTSNSurrogateClient(client)

    # Load locally SL-finetuned weights if available
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
        x = x.to(device)
        y = y.to(device)
        pred = server(client(x)).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)
