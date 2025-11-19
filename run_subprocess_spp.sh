#!/bin/bash

# Wrapper taht activates a minimal venv for executing the speech_processing_pipeline from another script using subprocess
# The .py script prints a json so it can be captured later.

source .venvs/spp_venv/bin/activate
python speech_processing_pipeline.py \
    --run_mode="$1" \
    --audio_file="$2" \
    --input_text="$3" \
    --out_path="$4" \
    --seg_model="$5" \
    --seg_config_yml="$6" \
    --seg_option="$7" \
    --stt_model="$8" \
    --cp_model="$9" \
    --device="$10"
