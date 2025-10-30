# (NAME) - Speech Processing Pipeline

(NAME) is a speech processing pipeline for the diarization and transcription of large audio files.

## Installation

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