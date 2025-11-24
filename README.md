# AhoSPHiTZ - Aholab/HiTZ Speech Processing Pipeline (Diarization + ASR + C&P)

This repository provides a full end-to-end pipeline for **speaker
diarization**, **speech-to-text transcription**, **optional translation
/ correction & paraphrasing (C&P)**, and **subtitle generation**.\
It integrates:

-   **Pyannote** for Voice Activity Detection (VAD) or Speaker
    Diarization\
-   **NVIDIA NeMo** for ASR with word-level timestamps\
-   **MarianMT** (HuggingFace) for translation / C&P\
-   Multiple output formats: `.rttm`, `.xml`, `.json`, `.txt`, `.vtt`,
    `.srt`, `.png` waveform overlays

The pipeline can be run in three modes:

  -----------------------------------------------------------------------
  Mode           Description                     Outputs
  -------------- ------------------------------- ------------------------
  `diar`         Only diarization                `.rttm`, `.png`

  `all`          Diarization + ASR +             `.xml`, `.rttm`,
                 segmentation + C&P + subtitle   `.json`, `.txt`, `.srt`,
                 files                           `.vtt`, and optionally
                                                 `_cp.*`

  `cp`           Only text C&P via MarianMT      text result
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Features

-   **Speaker diarization** using Pyannote with configurable YAML
    pipeline.
-   **ASR with timestamps**, confidence scores, word-level metadata.
-   **Smart segmentation**: split long utterances by duration and
    character count.
-   **GPU-safe inference** with recursive segment splitting when CUDA
    OOM happens.
-   **Subtitle-ready postprocessing**: padding, speaker overlap
    avoidance.
-   **Multiple formats**: XML, RTTM, JSON lines, TXT, SRT, VTT.
-   **Plotting**: generate waveform + speaker bar visualization.
-   **Compression**: automatically packages results into a ZIP.

------------------------------------------------------------------------

## Installation

### 1. Create environment

``` bash
conda create -n audio-pipeline python=3.10
conda activate audio-pipeline
```

### 2. Install required libraries

- Install torch, torchvision and torchaudio using the [PyTorch](https://pytorch.org/get-started/locally/) recommendations depending on your CUDA system.
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```
- Install the NeMo toolkit via pip:
```bash
pip install nemo_toolkit[asr]
```
- Install pyannote.audio via pip:
```bash
pip install pyannote-audio
```
- Install transformers:
```bash
pip install transformers
```
- Additional packages:
yaml, json, shutil, time, os, xml, matplotlib, tqdm, omegaconf

> Pyannote ≥3.x and NeMo ASR both benefit from a CUDA-enabled GPU.

------------------------------------------------------------------------

## Project Structure (functions in script)

-   `pyannote_seg()` --- diarization or VAD\
-   `nemo_asr()` / `nemo_inference()` --- ASR with timestamps\
-   `marianmt_cp()` --- capitalization and punctuation\
-   `divide_segments()` --- split long utterances\
-   `add_padding()` --- avoid subtitle overlap\
-   `map_speaker_color()` --- assign colors\
-   Writers:
    -   `to_rttm()`, `to_xml()`, `to_json()`, `to_vtt()`, `to_srt()`,
        `to_txt()`
-   Runners:
    -   `run_diar()`, `run_cp()`, `run_all()`

------------------------------------------------------------------------

## Input Requirements

### Audio files

-   Any format supported by `torchaudio` (`.wav`, `.mp3`, `.flac`)\
-   Auto-converted to **mono 16 kHz** (required by NeMo)

### Text (for `run_mode="cp"`)

Plain text input (string or file content).

------------------------------------------------------------------------

## Output Files

  Format      Description
  ----------- ------------------------------
  `.png`      Diarization visualization
  `.rttm`     Standard diarization file
  `.xml`      Segment + word timing export
  `.json`     JSONL with metadata
  `.txt`      Transcript
  `_cp.txt`   C&P/translated text
  `.srt`      Subtitle file
  `_cp.srt`   C&P subtitle
  `.vtt`      WebVTT
  `_cp.vtt`   C&P WebVTT
  `.zip`      Compressed final output

------------------------------------------------------------------------

## Command-Line Usage

Run the script:

``` bash
python script.py --run_mode <mode> [arguments...]
```

### Arguments

  Flag                 Description
  -------------------- ----------------------------------
  `--run_mode`         `all`, `diar`, or `cp`
  `--audio_file`       Path to audio (for `diar`/`all`)
  `--input_text`       Text (for `cp`)
  `--out_path`         Output directory
  `--seg_model`        Pyannote model checkpoint
  `--seg_config_yml`   Pyannote pipeline YAML
  `--seg_option`       `diar` or `vad`
  `--stt_model`        NeMo ASR checkpoint
  `--cp_model`         MarianMT model
  `--device`           `cuda` or `cpu`

------------------------------------------------------------------------

## Examples

### 1. Speaker diarization only

``` bash
python script.py \
    --run_mode diar \
    --audio_file input.wav \
    --seg_model /models/pyannote/model.ckpt \
    --seg_config_yml config.yaml \
    --out_path results \
    --device cuda
```

### 2. Full pipeline

``` bash
python script.py \
    --run_mode all \
    --audio_file input.wav \
    --seg_model /models/pyannote/model.ckpt \
    --seg_config_yml config.yaml \
    --seg_option diar \
    --stt_model /models/nemo/stt_es.nemo \
    --cp_model /models/mt/eu_norm-eu \
    --out_path results \
    --device cuda
```

### 3. Only Capitalization and Punctuation (C&P)

``` bash
python script.py \
    --run_mode cp \
    --input_text "kaixo mundua" \
    --cp_model /models/mt/eu_norm-eu \
    --device cuda
```

------------------------------------------------------------------------

## Language Handling

Output language is inferred from the ASR model filename:

-   contains `"eu"` → Basque (`eu`)
-   contains `"es"` → Spanish (`es`)

------------------------------------------------------------------------

## Output Organization

Results are stored at:

    <out_path>/<audio_filename>/

A ZIP file is generated:

    result_<timestamp>.zip

------------------------------------------------------------------------

## Visualization

`plot_diarization_sample()` outputs a waveform + speaker visualization:

    audio.png

------------------------------------------------------------------------

## GPU & Memory Safety

-   All large models run on GPU if `--device cuda`.
-   If ASR runs out of GPU memory:
    -   The segment is split recursively
    -   ASR is retried
    -   Results are merged using confidence-aware stitching

------------------------------------------------------------------------

## License

TODO

------------------------------------------------------------------------
