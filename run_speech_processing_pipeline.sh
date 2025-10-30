#!/bin/bash

start_time=$(date +%s)

python speech_processing_pipeline.py \
    --audio_file="/mnt/corpus/Common_Voice_v18.0/eu/wavs/common_voice_eu_39359564.wav" \
    --out_path="./results" \
    --seg_model="/mnt/aholab/asierhv/vad_for_asr/seg_CONF.ckpt" \
    --seg_config_yml="seg_config.yaml" \
    --seg_option="diar" \
    --stt_model="/mnt/aholab/asierhv/ASR_eu_test_files/results/models/finalmodels/stt_eu_conformer_transducer_large_v2.nemo" \
    --cp_model="" \
    --device="cpu"

echo

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

days=$((elapsed_time/86400))
hours=$(( (elapsed_time%86400)/3600 ))
minutes=$(( (elapsed_time%3600)/60 ))
seconds=$(( elapsed_time%60 ))

echo "Elapsed time: ${elapsed_time}s | ${days} days ${hours} hr ${minutes} min ${seconds} sec"
echo
