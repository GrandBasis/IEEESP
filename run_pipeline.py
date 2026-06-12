import argparse
import json
import os
import subprocess
import sys
from typing import List

import torch


SCRIPT_MAP = {
    "text": {
        1: "step1_SL_training_text.py",
        2: "step2_watermarking_generation_text.py",
    },
    "audio": {
        1: "step1_SL_training_audio.py",
        2: "step2_watermarking_generation_audio.py",
    },
    "video": {
        1: "step1_SL_training_video.py",
        2: "step2_watermarking_generation_video.py",
    },
    "image": {
        1: "step1_SL_training_image.py",
        2: "step2_watermarking_generation_image.py",
    }
}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))


def split_step_args(raw: List[str]) -> dict:
    """
    Parse passthrough args into:
      --step1-args ...
      --step2-args ...
      --common-args ...
    """
    buckets = {"common": [], "step1": [], "step2": []}
    current = "common"
    markers = {
        "--common-args": "common",
        "--step1-args": "step1",
        "--step2-args": "step2",
    }
    for tok in raw:
        if tok in markers:
            current = markers[tok]
            continue
        buckets[current].append(tok)
    return buckets


def parse_flag_args(tokens: List[str]) -> dict:
    out = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("--"):
            i += 1
            continue
        key = t[2:]
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            out[key] = tokens[i + 1]
            i += 2
        else:
            out[key] = True
            i += 1
    return out


def _save_preview(modality: str, common_args: List[str], preview_count: int) -> None:
    args_map = parse_flag_args(common_args)
    artifacts_dir = args_map.get("artifacts_dir")
    if not artifacts_dir:
        print("Preview export skipped: --artifacts_dir not provided.")
        return
    artifacts_dir = os.path.abspath(artifacts_dir)
    meta_path = os.path.join(artifacts_dir, "meta.json")
    if not os.path.exists(meta_path):
        print(f"Preview export skipped: meta not found at {meta_path}")
        return
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    preview_dir = os.path.join(artifacts_dir, "preview_samples")
    os.makedirs(preview_dir, exist_ok=True)

    if modality == "text":
        wm_path = os.path.join(artifacts_dir, "client_watermark_text.pt")
        if not os.path.exists(wm_path):
            print(f"Preview export skipped: watermark file missing {wm_path}")
            return
        wm = torch.load(wm_path, map_location="cpu")
        adv = wm["adv_token_ids"]
        masks = wm.get("editable_masks")
        targets = wm.get("targets")

        from step2_watermarking_generation_text import (
            TextClassificationDataset,
            load_builtin_dataset,
            read_examples,
        )

        dataset_name = args_map.get("dataset_name")
        data_dir = args_map.get("data_dir", DEFAULT_DATA_DIR)
        if dataset_name:
            _, test_examples, _ = load_builtin_dataset(dataset_name, data_dir=data_dir)
        else:
            test_path = args_map.get("test_path")
            text_col = args_map.get("text_col", "text")
            label_col = args_map.get("label_col", "label")
            if not test_path:
                print("Preview export skipped for text: need --dataset_name or --test_path.")
                return
            test_examples = read_examples(test_path, text_col, label_col)

        vocab_path = os.path.join(artifacts_dir, "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        clean_ds = TextClassificationDataset(test_examples, vocab=vocab, max_len=int(meta["max_len"]))
        n = min(preview_count, len(adv), len(clean_ds))
        clean_ids = []
        clean_labels = []
        for i in range(n):
            ids, y = clean_ds[i]
            row = torch.zeros(int(meta["max_len"]), dtype=torch.long)
            k = min(len(ids), row.numel())
            row[:k] = torch.tensor(ids[:k], dtype=torch.long)
            clean_ids.append(row)
            clean_labels.append(y)
        clean_ids = torch.stack(clean_ids, dim=0)
        clean_labels = torch.tensor(clean_labels, dtype=torch.long)
        out = {
            "clean_samples": clean_ids,
            "clean_labels": clean_labels,
            "watermark_samples": adv[:n],
            "targets": targets[:n] if targets is not None else None,
            "mask": masks[:n] if masks is not None else None,
        }
        out_path = os.path.join(preview_dir, "client_watermark_preview_text.pt")
        torch.save(out, out_path)
        print(f"Saved preview samples: {out_path}")
        return

    if modality == "audio":
        wm_path = os.path.join(artifacts_dir, "client_watermark_audio.pt")
        if not os.path.exists(wm_path):
            print(f"Preview export skipped: watermark file missing {wm_path}")
            return
        wm = torch.load(wm_path, map_location="cpu")
        adv = wm["adv_waves"]
        mask = wm.get("mask")
        targets = wm.get("targets")
        from common.audio_watermark_common import make_audio_datasets
        data_dir = args_map.get("data_dir", DEFAULT_DATA_DIR)
        dataset_name = str(args_map.get("dataset_name", "speechcommands")).lower()
        if dataset_name != "speechcommands":
            print(
                f"Preview export skipped: unsupported audio dataset_name={dataset_name} (only speechcommands)."
            )
            return
        _, test_ds, _, _ = make_audio_datasets(
            dataset_name=dataset_name,
            data_dir=data_dir,
            train_manifest=None,
            test_manifest=None,
            commands_csv=str(meta.get("commands", "")),
            target_sample_rate=int(meta["target_sample_rate"]),
            target_num_samples=int(meta["target_num_samples"]),
            download=False,
        )
        n = min(preview_count, len(adv), len(test_ds))
        clean = []
        clean_labels = []
        for i in range(n):
            x, y = test_ds[i]
            clean.append(x)
            clean_labels.append(y)
        out = {
            "clean_samples": torch.stack(clean, dim=0),
            "clean_labels": torch.tensor(clean_labels, dtype=torch.long),
            "watermark_samples": adv[:n],
            "targets": targets[:n] if targets is not None else None,
            "mask": mask,
        }
        out_path = os.path.join(preview_dir, "client_watermark_preview_audio.pt")
        torch.save(out, out_path)
        print(f"Saved preview samples: {out_path}")
        return

    if modality == "video":
        wm_path = os.path.join(artifacts_dir, "client_watermark_video.pt")
        if not os.path.exists(wm_path):
            print(f"Preview export skipped: watermark file missing {wm_path}")
            return
        wm = torch.load(wm_path, map_location="cpu")
        adv = wm["adv_videos"]
        mask = wm.get("mask")
        targets = wm.get("targets")
        from common.video_watermark_common import VideoClassificationDataset, resolve_video_dataset
        dataset_name = str(args_map.get("dataset_name", "ucf101")).lower()
        if dataset_name != "ucf101":
            print(
                f"Preview export skipped: unsupported video dataset_name={dataset_name} (only ucf101)."
            )
            return
        data_dir = args_map.get("data_dir", DEFAULT_DATA_DIR)
        _, test_ex, _ = resolve_video_dataset(dataset_name, data_dir, None, None)
        test_ds = VideoClassificationDataset(
            test_ex,
            num_frames=int(meta["num_frames"]),
            image_size=int(meta["image_size"]),
            train=False,
        )
        n = min(preview_count, len(adv), len(test_ds))
        clean = []
        clean_labels = []
        for i in range(n):
            x, y = test_ds[i]
            clean.append(x)
            clean_labels.append(y)
        out = {
            "clean_samples": torch.stack(clean, dim=0),
            "clean_labels": torch.tensor(clean_labels, dtype=torch.long),
            "watermark_samples": adv[:n],
            "targets": targets[:n] if targets is not None else None,
            "mask": mask,
        }
        out_path = os.path.join(preview_dir, "client_watermark_preview_video.pt")
        torch.save(out, out_path)
        print(f"Saved preview samples: {out_path}")
        return

    if modality == "image":
        wm_path = os.path.join(artifacts_dir, "client_watermark_image.pt")
        if not os.path.exists(wm_path):
            print(f"Preview export skipped: watermark file missing {wm_path}")
            return
        wm = torch.load(wm_path, map_location="cpu")
        adv = wm["adv_images"]
        mask = wm.get("mask")
        targets = wm.get("targets")
        from common.image_watermark_common import make_image_datasets

        dataset_name = str(args_map.get("dataset_name", "tiny-imagenet")).lower()
        data_dir = args_map.get("data_dir", DEFAULT_DATA_DIR)

        _, test_ds, _ = make_image_datasets(dataset_name, data_dir)

        n = min(preview_count, len(adv), len(test_ds))
        clean = []
        clean_labels = []
        for i in range(n):
            x, y = test_ds[i]
            clean.append(x)
            clean_labels.append(y)
        out = {
            "clean_samples": torch.stack(clean, dim=0),
            "clean_labels": torch.tensor(clean_labels, dtype=torch.long),
            "watermark_samples": adv[:n],
            "targets": targets[:n] if targets is not None else None,
            "mask": mask,
        }
        out_path = os.path.join(preview_dir, "client_watermark_preview_image.pt")
        torch.save(out, out_path)
        print(f"Saved preview samples: {out_path}")
        return


def run_step(
    script_dir: str,
    script_name: str,
    step_num: int,
    args_for_step: List[str],
) -> None:
    script_path = os.path.join(script_dir, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Step script not found: {script_path}")
    cmd = [sys.executable, script_path] + args_for_step
    print(f"\n===== Running Step {step_num}: {script_name} =====")
    print("Command: " + " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        cwd=script_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip("\n"))

    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unified launcher for text/audio/video/image watermark pipelines.\n\n"
            "Example:\n"
            "  python run_pipeline.py --modality image --step all "
            "--common-args --dataset_name tiny-imagenet --artifacts_dir ./artifacts_dir_image_tiny "
            "--step1-args --batch_size 256 --rounds 5 "
            "--step2-args --cleanset_max 300 --steps 40"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--modality", choices=["text", "audio", "video", "image"], required=True)
    parser.add_argument("--step", choices=["1", "2", "all"], required=True)
    parser.add_argument(
        "--save_preview_samples",
        action="store_true",
        help="After watermark step (Step2), save preview of clean samples + client-side watermarks + mask.",
    )
    parser.add_argument("--preview_count", type=int, default=5, help="Number of preview pairs to save.")
    args, passthrough = parser.parse_known_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    step_args = split_step_args(passthrough)

    if args.step == "all":
        steps = [1, 2]
    else:
        steps = [int(args.step)]

    for s in steps:
        script_name = SCRIPT_MAP[args.modality][s]
        final_args = step_args["common"] + step_args[f"step{s}"]
        run_step(script_dir, script_name, s, final_args)

    if args.save_preview_samples and 2 in steps:
        _save_preview(args.modality, step_args["common"], args.preview_count)

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
