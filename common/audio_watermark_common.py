import os
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.models as models

DEFAULT_SPEECHCOMMANDS_10 = "yes,no,up,down,left,right,on,off,stop,go"


def _require_torchaudio():
    try:
        import torchaudio  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "torchaudio is required for audio pipeline. Install with: pip install torchaudio"
        ) from exc


def _require_soundfile():
    try:
        import soundfile  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "soundfile is required as fallback backend for audio loading. Install with: pip install soundfile"
        ) from exc


def parse_commands(commands_csv: str) -> List[str]:
    return [x.strip() for x in commands_csv.split(",") if x.strip()]


def _load_and_standardize_waveform(path: str, target_sample_rate: int, target_num_samples: int) -> torch.Tensor:
    import torchaudio

    try:
        wav, sr = torchaudio.load(path, backend="soundfile")  # [C, T]
    except Exception:
        _require_soundfile()
        import soundfile as sf

        arr, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(arr.T)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sample_rate:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sample_rate)
    t = wav.size(1)
    if t >= target_num_samples:
        wav = wav[:, :target_num_samples]
    else:
        wav = torch.nn.functional.pad(wav, (0, target_num_samples - t))
    return wav


class SpeechCommandsDataset(Dataset):
    def __init__(
        self,
        root: str,
        subset: str,
        commands: Sequence[str],
        target_sample_rate: int = 16000,
        target_num_samples: int = 16000,
        download: bool = False,
    ):
        _require_torchaudio()
        self.subset = subset
        self.commands = list(commands)
        self.label_to_idx = {c: i for i, c in enumerate(self.commands)}
        self.target_sample_rate = target_sample_rate
        self.target_num_samples = target_num_samples
        self.samples: List[Tuple[str, str]] = []

        import torchaudio

        _ = torchaudio.datasets.SPEECHCOMMANDS(root=root, subset="training", download=download)
        dataset_root = os.path.join(root, "SpeechCommands", "speech_commands_v0.02")
        if not os.path.exists(dataset_root):
            raise FileNotFoundError(
                f"SpeechCommands root not found: {dataset_root}. "
                "Auto-download is disabled; place dataset files under the provided --data_dir."
            )

        val_list_path = os.path.join(dataset_root, "validation_list.txt")
        test_list_path = os.path.join(dataset_root, "testing_list.txt")
        with open(val_list_path, "r", encoding="utf-8") as f:
            val_set = {line.strip() for line in f if line.strip()}
        with open(test_list_path, "r", encoding="utf-8") as f:
            test_set = {line.strip() for line in f if line.strip()}

        all_rel: List[str] = []
        for label in sorted(os.listdir(dataset_root)):
            d = os.path.join(dataset_root, label)
            if not os.path.isdir(d):
                continue
            if label.startswith("_"):
                continue
            for fn in os.listdir(d):
                if fn.lower().endswith(".wav"):
                    all_rel.append(f"{label}/{fn}")

        if subset == "training":
            rel_list = [r for r in all_rel if r not in val_set and r not in test_set]
        elif subset == "validation":
            rel_list = [r for r in all_rel if r in val_set]
        elif subset == "testing":
            rel_list = [r for r in all_rel if r in test_set]
        else:
            raise ValueError("subset must be one of: training, validation, testing")

        for rel in rel_list:
            label = rel.split("/")[0]
            if label in self.label_to_idx:
                self.samples.append((os.path.join(dataset_root, rel), label))

        if not self.samples:
            raise ValueError("No samples matched selected commands in SpeechCommands dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        wav = _load_and_standardize_waveform(path, self.target_sample_rate, self.target_num_samples)
        return wav, self.label_to_idx[label]


def make_audio_datasets(
    dataset_name: str,
    data_dir: str,
    train_manifest: Optional[str],
    test_manifest: Optional[str],
    commands_csv: str = DEFAULT_SPEECHCOMMANDS_10,
    target_sample_rate: int = 16000,
    target_num_samples: int = 16000,
    download: bool = False,
):
    name = dataset_name.lower()
    if name == "speechcommands":
        commands = parse_commands(commands_csv)
        train_ds = SpeechCommandsDataset(
            root=data_dir,
            subset="training",
            commands=commands,
            target_sample_rate=target_sample_rate,
            target_num_samples=target_num_samples,
            download=download,
        )
        test_ds = SpeechCommandsDataset(
            root=data_dir,
            subset="testing",
            commands=commands,
            target_sample_rate=target_sample_rate,
            target_num_samples=target_num_samples,
            download=download,
        )
        label_to_idx = {c: i for i, c in enumerate(commands)}
        return train_ds, test_ds, len(commands), label_to_idx

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


# =====================================================================
# Models: ResNet34 with Differentiable Mel-Spectrogram Front-End
# =====================================================================

class AudioResNetClient(nn.Module):
    def __init__(self, sample_rate=16000):
        super().__init__()
        import torchaudio.transforms as T

        # Waveform -> Mel spectrogram (fully differentiable)
        self.mel_spec = T.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=64)
        self.amp_to_db = T.AmplitudeToDB(stype='power', top_db=80)

        # ImageNet normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # Extract stem layers (all layers before layer1)
        stem_layers = []
        for name, child in base.named_children():
            if name == 'layer1':
                break
            stem_layers.append(child)
        self.stem = nn.Sequential(*stem_layers)

        self.layer1 = base.layer1
        self.layer2 = base.layer2

    def forward(self, x):
        # x: [B, 1, T] waveform input
        x = self.mel_spec(x)          # -> [B, 1, n_mels, time_frames]
        x = self.amp_to_db(x)
        x = x.expand(-1, 3, -1, -1)   # Expand 1-channel spectrogram to 3 channels for pretrained ResNet34

        x = (x - self.mean) / self.std  # Normalize

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)  # ResNet34 layer2 output: 128 channels
        return x


class AudioResNetServer(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)  # ResNet34 layer4 output: 512 channels

    def forward(self, smashed):
        x = self.layer3(smashed)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


class AudioResNetSurrogateClient(nn.Module):
    def __init__(self, client_module):
        super().__init__()
        self.base = client_module
        # AudioResNetClient output: 128 channels
        self.adapter = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
        )

    def forward(self, x):
        y = self.base(x)
        return y + self.adapter(y)


# =====================================================================
# Model Factory
# =====================================================================

def get_models(
    model_name: str,
    num_classes: int,
    sample_rate: int = 16000,
    pretrained: bool = False,
    pretrained_dir: Optional[str] = None
) -> Tuple[nn.Module, nn.Module, nn.Module]:

    model_name = model_name.lower()

    if model_name == "resnet34":
        client = AudioResNetClient(sample_rate=sample_rate)
        server = AudioResNetServer(num_classes)
        surrogate = AudioResNetSurrogateClient(client)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. Supported: ['resnet34']")

    # Load locally fine-tuned SL weights
    if pretrained and pretrained_dir:
        client_path = os.path.join(pretrained_dir, "client.pt")
        server_path = os.path.join(pretrained_dir, "server.pt")

        if os.path.exists(client_path):
            print(f"Loading local SL Audio Client weights from: {client_path}")
            ckpt_c = torch.load(client_path, map_location="cpu")
            client.load_state_dict(ckpt_c.get("state_dict", ckpt_c))

            # Sync surrogate base with client weights (for architectures where base is not a reference)
            if hasattr(surrogate, 'base') and surrogate.base is not client:
                surrogate.base.load_state_dict(ckpt_c.get("state_dict", ckpt_c))

        if os.path.exists(server_path):
            print(f"Loading local SL Audio Server weights from: {server_path}")
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
