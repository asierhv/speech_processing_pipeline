#!/bin/bash

start_time=$(date +%s)

python speech_processing_pipeline.py \
    --audio_file="audio.wav" \
    --out_path="./results" \
    --seg_model="seg_CONF.ckpt" \
    --seg_config_yml="config.yaml" \
    --seg_option="diar" \
    --stt_model="stt_eu_conformer_transducer_large.nemo" \
    --cp_model="eu_norm-eu" \
    --device="cuda"

echo

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

days=$((elapsed_time/86400))
hours=$(( (elapsed_time%86400)/3600 ))
minutes=$(( (elapsed_time%3600)/60 ))
seconds=$(( elapsed_time%60 ))

echo "Elapsed time: ${elapsed_time}s | ${days} days ${hours} hr ${minutes} min ${seconds} sec"
echo
