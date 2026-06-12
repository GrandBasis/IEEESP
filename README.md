# Multi-Modal Federated Learning Framework

A unified framework for federated learning with multimodal data processing, supporting model training and client-side watermark embedding.

## Features

- **Multi-Modal Support**: Video, Audio, Image, and Text
- **Split Learning Training**: Federated training with multiple clients
- **Client Watermarking**: Embed verifiable watermarks into client models
- **Pre-trained Models**: Ready-to-use pre-trained weights for each modality

## Supported Modalities & Models

| Modality | Dataset | Model | Classes |
|----------|---------|-------|---------|
| Video | UCF101 | TSN (Temporal Segment Networks) | 101 |
| Audio | SpeechCommands | ResNet34 | 35 |
| Image | Tiny-ImageNet | ViT-B-16 (Vision Transformer) | 200 |
| Text | AG News | BERT | 4 |

### Datasets

- **UCF101** (Video, 101 classes): action-recognition dataset from the UCF Center for Research in Computer Vision — <https://www.crcv.ucf.edu/data/UCF101.php>
- **SpeechCommands v0.02** (Audio, 35 classes): Google’s short spoken-keyword corpus, mirrored on torchaudio — <https://datasets.readthedocs.io/en/latest/api/torchaudio.html#torchaudio.datasets.SPEECHCOMMANDS> (paper: <https://arxiv.org/abs/1804.03209>)
- **Tiny-ImageNet** (Image, 200 classes): 200-class subset of ImageNet from Stanford CS231N — <http://cs231n.stanford.edu/tiny-imagenet-200.zip>
- **AG News** (Text, 4 classes): topic-classification news corpus distributed via the HuggingFace `datasets` hub — <https://huggingface.co/datasets/ag_news>

## Project Structure

```
.
├── run_pipeline.py                    # Main entry point
├── requirements.txt                   # Project dependencies
├── step1_SL_training_*.py             # Step 1: Split Learning Training
├── step2_watermarking_generation_*.py # Step 2: Watermark Embedding
└── common/                            # Common utilities for each modality
    ├── audio_watermark_common.py
    ├── image_watermark_common.py
    ├── text_watermark_common.py
    └── video_watermark_common.py
```

## Installation

To keep the four modalities reproducible and to avoid dependency
clashes, the project relies on **two isolated conda environments** —
one for the lightweight audio / text / image stack, and another for
the video stack, which depends on a locally compiled `mmaction2` and
its matching `mmcv` / `mmengine` versions. **All commands below assume
you have conda installed and that you activate the appropriate
environment before running the pipeline.**

| Modality | Conda env | Requirements file |
|----------|-----------|-------------------|
| Audio / Text / Image | `DEMO` | `requirements_audio_text_image.txt` |
| Video | `DEMO_video` | `requirements_video.txt` |

### Prerequisites

- `conda` (Miniconda or Anaconda)
- Python >= 3.10
- NVIDIA GPU with CUDA driver
  - `DEMO`        -> tested with CUDA 12.4 (PyTorch 2.6.0+cu124)
  - `DEMO_video`  -> tested with CUDA 11.7 (PyTorch 2.0.1+cu117)

### Environment 1 — Audio / Text / Image (`DEMO`)

This env covers the `audio`, `text` and `image` pipelines
(SpeechCommands, AG News, Tiny-ImageNet). It is built on a recent
PyTorch (2.6.0) with CUDA 12.4 wheels.

**Step 1**: Create the conda environment and activate it

```bash
conda create -n DEMO python=3.10 -y
conda activate DEMO
```

**Step 2**: Install PyTorch with CUDA 12.4 support

```bash
pip install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
            --index-url https://download.pytorch.org/whl/cu124
```

**Step 3**: Install the remaining dependencies

```bash
pip install -r requirements_audio_text_image.txt
```

### Environment 2 — Video (`DEMO_video`)

This env is built around **mmaction2 v1.2.0**, fetched from GitHub and
built from source. It matches the CUDA 11.7 / cu117 prebuilt wheels of
PyTorch 2.0.1.

**Step 1**: Create the conda environment and activate it

```bash
conda create -n DEMO_video python=3.10 -y
conda activate DEMO_video
```

**Step 2**: Install PyTorch / torchvision / torchaudio with CUDA 11.7
support (all three come from the same index URL)

```bash
pip install --index-url https://download.pytorch.org/whl/cu117 \
    "torch==2.0.1+cu117" \
    "torchvision==0.15.2+cu117" \
    "torchaudio==2.0.2+cu117"
```

**Step 3**: Install mmengine and mmcv

```bash
pip install -U openmim
mim install "mmengine==0.10.7"
mim install "mmcv==2.1.0"
```

**Step 4**: Fetch the `mmaction2` source code from GitHub and check
out the v1.2.0 tag.

```bash
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2 && git checkout v1.2.0 && cd ..
```

**Step 5**: Build and install `mmaction2` from source

```bash
pip install -r mmaction2/requirements/build.txt
pip install -v -e ./mmaction2
```

**Step 6**: Install the two extras that the previous steps do **not**
pull in automatically.

```bash
pip install -r requirements_video.txt
```

## Usage

### Command Line Interface

```bash
python run_pipeline.py --modality <MODALITY> --step <STEP> [options]
```

**Parameters**:
- `--modality`: Choose from `video`, `audio`, `image`, `text`
- `--step`: Choose `1` (training only), `2` (watermarking only), or `all` (complete pipeline)
- `--save_preview_samples`: Save preview samples after watermarking

### Run Complete Pipeline

> **Note**: If no pre-trained models are available, you can appropriately increase the learning rate and number of training rounds. The pre-trained models in the experiments were obtained by running Step 1.

#### Video Modality

> **Note**: `--data_dir ./data/ucf101/out_dir` refers to the directory containing `train.csv` and `test.csv` when preparing the UCF101 dataset.

```bash
python run_pipeline.py \
  --modality video --step all \
  --common-args \
    --artifacts_dir ./artifacts_dir_video \
    --data_dir ./data/ucf101/out_dir \
    --dataset_name ucf101 \
    --model tsn \
    --num_clients 5 \
    --batch_size 16 \
  --step1-args \
    --num_frames 16 \
    --image_size 224 \
    --local_epochs 1 \
    --rounds 5 \
    --lr 0.000001 \
    --num_workers 8 \
    --pretrained \
  --step2-args \
    --attack_batch_size 16 \
    --cleanset_max 120 \
    --steps 40
```

#### Audio Modality

```bash
python run_pipeline.py \
  --modality audio --step all \
  --common-args \
    --artifacts_dir ./artifacts_dir_audio \
    --data_dir ./data \
    --dataset_name speechcommands \
    --model resnet34 \
    --num_clients 5 \
    --batch_size 64 \
  --step1-args \
    --local_epochs 1 \
    --rounds 5 \
    --lr 0.00001 \
    --num_workers 8 \
    --pretrained \
  --step2-args \
    --attack_batch_size 64 \
    --cleanset_max 200 \
    --steps 100
```

#### Image Modality

```bash
python3 run_pipeline.py \
  --modality image --step all \
  --common-args \
    --artifacts_dir ./artifacts_dir_image \
    --data_dir ./data \
    --dataset_name tiny-imagenet \
    --model vit-b-16 \
    --num_clients 5 \
    --batch_size 64 \
  --step1-args \
    --local_epochs 1 \
    --rounds 5 \
    --lr 0.000001 \
    --num_workers 8 \
    --pretrained \
  --step2-args \
    --attack_batch_size 64 \
    --cleanset_max 1000 \
    --steps 100
```

#### Text Modality

```bash
python3 run_pipeline.py \
  --modality text --step all \
  --common-args \
    --artifacts_dir ./artifacts_dir_text \
    --data_dir ./data \
    --dataset_name ag_news \
    --model bert \
    --num_clients 5 \
    --batch_size 64 \
  --step1-args \
    --local_epochs 1 \
    --rounds 5 \
    --lr 0.00001 \
    --num_workers 8 \
    --pretrained \
  --step2-args \
    --client_edit_ratio 0.8 \
    --steps 50 \
    --candidate_size 128 \
    --projection_sweeps 5 \
    --cleanset_max 1000
```

### Run Single Step

**Step 1 Only** (Split Learning Training):

```bash
python run_pipeline.py --modality <MODALITY> --step 1 [other args]
```

**Step 2 Only** (Watermark Embedding):

```bash
python run_pipeline.py --modality <MODALITY> --step 2 [other args]
```

## Workflow

### Step 1: Split Learning Training

1. **Data Partitioning**: Split dataset among multiple clients
2. **Local Training**: Each client trains on local data
3. **Model Aggregation**: Server collects and aggregates model parameters
4. **Global Update**: Distribute aggregated parameters to all clients
5. **Iterative Optimization**: Repeat until specified rounds completed

### Step 2: Client Watermark Embedding

1. **Watermark Generation**: Generate watermark information
2. **Model Selection**: Select client models for watermarking
3. **Watermark Embedding**: Embed watermark into client models
4. **Verification**: Validate watermark effectiveness and robustness
5. **Result Saving**: Save watermarked models and results

## Output Results

After completing the pipeline, the program will output the following key metrics:

- **Top-1 / Top-2 Validation Accuracy**: Used as watermark embedding success rate indicator
- **Watermark Verification Accuracy (Top-1)**: Closer to 100% indicates more successful client watermark embedding

These metrics are printed in the terminal output at the end of the pipeline execution.

## Notes

- Ensure sufficient GPU memory for training
- Pre-trained models will be downloaded automatically on first run
- Use `python run_pipeline.py --help` for detailed parameter descriptions
- Different modalities require corresponding data and model parameters
- **When training with more clients**: It is recommended to use GPU parallelism to avoid out of memory errors, and appropriately increase the batch size
