import os
import torch
import torchaudio
import yaml
import argparse
import uuid
import json
import numpy as np
import nemo.collections.asr.models as nemo_models
import xml.etree.ElementTree as ET
from omegaconf import open_dict
from xml.dom import minidom
from tqdm import tqdm
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core.utils.helper import get_class_by_name
from transformers import pipeline

def pyannote_seg(inputs, model_path, option, config_yml, device):
    """
    Perform segmentation (VAD or Diarization) using pyannote.
    - inputs: {'waveform': torch.Tensor (1,T), 'sample_rate': int}
    - model: path to pyannote model checkpoint
    - option: 'vad' or 'diar'
    - config_yml: path to configuration .yaml file
    - device: 'cuda' or 'cpu'
    Returns list of dicts: [{'start', 'end', 'speaker'}]
    """    
    # Load configuration
    with open(config_yml, "r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)
        
    config["pipeline"]["params"]["segmentation"] = model_path
    print(f"- Pyannote Configuration:\n{yaml.dump(config)}")
    
    if option == "diar":
        print("Option mode: Speaker Diarization\n")
        Klass = get_class_by_name(config["pipeline"]["name"], default_module_name="pyannote.pipeline.blocks")
        params = config["pipeline"].get("params", {})
        pyannote_pipeline = Klass(**params)
        pyannote_pipeline.instantiate(config["params"])
        
        # Perform segmentation
        pyannote_pipeline.to(torch.device(device))
        result = pyannote_pipeline(inputs)  
        result_tracks = result.exclusive_speaker_diarization.itertracks(yield_label=True)
        
    elif option == "vad":
        print("Option mode: Voice Activity Detection\n")
        model_path = Model.from_pretrained(model_path)
        pyannote_pipeline = VoiceActivityDetection(segmentation=model_path)
        config["params"]["segmentation"]["min_duration_on"] = 0.0
        pyannote_pipeline.instantiate(config["params"]["segmentation"])      
         
        # Perform segmentation
        pyannote_pipeline.to(torch.device(device))
        result = pyannote_pipeline(inputs)  
        result_tracks = result.itertracks(yield_label=True)
    
    segment_item_list = []    
    for segment, _, speaker in result_tracks: #version 4.x
        segment_item = {
            'start' : round(float(segment.start),3),
            'end' : round(float(segment.end),3),
            'speaker' : str(speaker), # VAD -> 'SPEECH' | Diarization -> 'SPEAKER_n'
        }
        segment_item_list.append(segment_item)
    
    # Free GPU memory before returning
    del pyannote_pipeline
    torch.cuda.empty_cache()
    
    return segment_item_list    

def merge_subsegments(subsegment_item_list):
    """
    Merge two sub_segments into original segment using "conf" for overlaped words.
    - sub_segment_item_list: list of dicts with 'start', 'end', 'speaker', 'words'
    Returns a dict: {'start', 'end', 'speaker', 'words': [{'start', 'end', 'word', 'conf'}]}
    """
    # Define the padd window
    padd_start = subsegment_item_list[1]["start"]
    padd_end = subsegment_item_list[0]["end"]
    
    words_1 = [word for word in subsegment_item_list[0]["words"]]
    words_2 = [word for word in subsegment_item_list[1]["words"]]

    words_before_padd = [] # Words that start before padd_start
    words_after_padd = []  # Words that end after padd_end
    words_in_padd = [] # Words that are within the padd window
    for word in words_1:
        if word["start"] < padd_start:
            words_before_padd.append(word)
        else:
            words_in_padd.append(word)
    for word in words_2:
        if word["end"] > padd_end:
            words_after_padd.append(word)
        else:
            words_in_padd.append(word)
    
    # Re-define padd_start based on last word_before_padd's end time
    if words_before_padd[-1]["end"] > padd_start:
        padd_start = words_before_padd[-1]["end"]
    
    # Re-define padd_end based on first word_after_padd's start time
    if words_after_padd[0]["start"] < padd_end:
        padd_end = words_after_padd[0]["start"]

    if padd_start < padd_end:
        # Merge words in padd window based on max "conf"
        words_in_padd = sorted(words_in_padd, key=lambda x: x["start"])
        acc_words_in_padd = [words_in_padd[0]] # Accepted words in padd window, initialized to the first one
        for word in words_in_padd[1:]:
            acc_word = acc_words_in_padd[-1]
            if word["start"] < acc_word["end"]:
                # Overlap detected, keep word with higher "conf"
                if word["conf"] > acc_word["conf"]:
                    acc_words_in_padd[-1] = word
            else:
                acc_words_in_padd.append(word)
        words_merged = words_before_padd + acc_words_in_padd + words_after_padd
    else:
        # Padd windows is negative, there is an overlap between accepted words, just adjust its times
        padd_mid = (padd_start + padd_end) / 2
        words_before_padd[-1]["end"] = round(padd_mid,3)
        words_after_padd[0]["start"] = round(padd_mid,3)
        words_merged = words_before_padd + words_after_padd
    
    words_merged = sorted(words_merged, key=lambda x: x["start"])
    
    segment_item = {
        'start' : subsegment_item_list[0]['start'],
        'end' : subsegment_item_list[1]['end'],
        'speaker' : subsegment_item_list[0]['speaker'],
        'words' : words_merged,
        'language' : subsegment_item_list[0]['language']
        }
    
    return segment_item            

def nemo_inference(asr_model, audio, sr, time_stride, padd, segment_item_list, lang, device):
    """
    Perform STT inference segment by segment using nemo.
    - asr_model: loaded nemo ASR model
    - audio: numpy array of the full audio
    - sr: sample rate
    - time_stride: time stride of the model
    - padd: padding in seconds to use when splitting segments due to CUDA OOM
    - segment_item_list: list of dicts with 'start' and 'end' keys
    - lang: 'es' or 'eu'
    - device: 'cuda' or 'cpu'
    Returns list of dicts: [{'start', 'end', 'speaker', 'language', 'words': [{'start', 'end', 'word', 'conf'}]}]
    """    
    
    with torch.no_grad():
        new_segment_item_list = []
        for i, segment_item in enumerate(tqdm(segment_item_list)):
            # Extract segment
            seg_start = segment_item["start"]
            seg_end = segment_item["end"]            
            start = int(round(seg_start * sr))
            end = int(round(seg_end * sr))
            audio_seg = audio[start:end]
            
            seg_tensor = torch.tensor(audio_seg, dtype=torch.float32, device=device).unsqueeze(0)

            try:
                # Preprocess to features
                features, features_len = asr_model.preprocessor(
                    input_signal = seg_tensor,
                    length = torch.tensor([seg_tensor.shape[1]], device=device)
                )

                # Encoder forward
                encoded, encoded_len = asr_model.encoder(
                    audio_signal = features, 
                    length = features_len
                )

                # RNNT decoding
                hypothesis = asr_model.decoding.rnnt_decoder_predictions_tensor(
                    encoded, 
                    encoded_len, 
                    return_hypotheses = True
                )
                hypothesis = hypothesis[0]
                
                # Extract words and timestamps
                word_info_list = hypothesis.timestamp["word"]
                word_confidence_list = hypothesis.word_confidence
                word_timestamp_list = []
                for i, word_info in enumerate(word_info_list):
                    word_timestamp = {
                        "start": round(float(word_info["start_offset"]) * time_stride + seg_start, 3),
                        "end": round(float(word_info["end_offset"]) * time_stride + seg_start, 3),
                        "word": word_info["word"],
                        "conf": round(word_confidence_list[i].item(),3)
                    }
                    word_timestamp_list.append(word_timestamp)
                
                segment_item["words"] = word_timestamp_list
                segment_item["language"] = lang
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"\nWARNING: CUDA OOM during STT inference on segment {i}.\nDividing segment in half using a {padd}s padding and retrying.\n")
                    mid_time = (seg_start + seg_end) / 2
                    
                    # Create two new segments with padding
                    subsegment_item_list = [
                        {"start": seg_start, "end": round(mid_time + padd, 3), "speaker": segment_item["speaker"]},
                        {"start": round(mid_time - padd, 3), "end": seg_end, "speaker": segment_item["speaker"]},
                    ]
                    
                    # Recursive call to process the two new segments
                    subsegment_item_list = nemo_inference(asr_model=asr_model, audio=audio, sr=sr,
                                                          time_stride=time_stride, padd=padd,
                                                          subsegment_item_list=segment_item_list,
                                                          lang=lang, device=device)
                    
                    # Merge sub segments in one original segment
                    segment_item = merge_subsegments(subsegment_item_list)
                    
                else:
                    print(f"\nWARNING: {e}\nSkipping segment {i}.\n")
                    segment_item["language"] = ""
                    segment_item["words"] = []

            new_segment_item_list.append(segment_item)
    
    return new_segment_item_list

def nemo_asr(inputs, model_path, segment_item_list, lang, device):
    """
    Perform ASR using nemo.
    - inputs: {'waveform': torch.Tensor (1,T), 'sample_rate': int}
    - model_path: path to nemo model checkpoint
    - segment_item_list: list of dicts with 'start' and 'end' keys
    - lang: 'es', 'eu' or 'bi'
    - device: 'cuda' or 'cpu'
    Returns list of dicts: [{'start', 'end', 'speaker', 'language', 'words': [{'start', 'end', 'word', 'conf'}]}]
    """
    # Load nemo ASR model
    asr_model = nemo_models.ASRModel.restore_from(model_path)
    decoding_cfg = asr_model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.greedy.preserve_alignments = True
        decoding_cfg.greedy.compute_timestamps = True
        decoding_cfg.greedy.preserve_frame_confidence = True
        decoding_cfg.confidence_cfg.preserve_frame_confidence = True
        decoding_cfg.confidence_cfg.preserve_token_confidence = True
        decoding_cfg.confidence_cfg.preserve_word_confidence = True
        decoding_cfg.compute_timestamps = True
    asr_model.change_decoding_strategy(decoding_cfg)
    asr_model.eval().to(device)

    # Convert audio to numpy for segmentation
    audio = inputs["waveform"].squeeze(0).numpy()
    sr = inputs["sample_rate"]
    
    time_stride = 4*asr_model.cfg.preprocessor.window_stride # '4' for Conformer models
    padd = 0.3 # seconds
    
    # Perform inference segment by segment
    segment_item_list = nemo_inference(asr_model=asr_model, audio=audio, sr=sr,
                                       time_stride=time_stride, padd=padd,
                                       segment_item_list=segment_item_list,
                                       lang=lang, device=device)
            
    # Free GPU memory before returning
    del asr_model
    torch.cuda.empty_cache()
    
    segment_item_list = [segment_item for segment_item in segment_item_list if len(segment_item["words"])>0]

    return segment_item_list

# Work in progess
def segments_for_subtitles(segment_item_list, max_chars: int=40, max_dur: float=5.0, min_dur: float=1.0):
    """
    Segment or merge sentences for creating better subtitulation files like .vtt or .srt
    - segment_item_list: list of dicts with 'words' key
    - max_chars: max characters allowed per sentence
    - max_dur: max duration per sentence
    - min_dur: min duration per sentence
    """
    for i, segment_item in enumerate(segment_item_list):
        seg_start = segment_item["start"]
        seg_end = segment_item["end"]
        seg_dur = round(seg_end-seg_start,3)
        words = segment_item["words"]
        for j, word in enumerate(words):
            pass 
    
def marianmt_cp(segment_item_list, model_path, device):
    """
    Perform C&P of segments using transformers pipeline with MarianMT model.
    - segment_item_list: list of dicts with 'words' key
    - model_path: path to MarianMt translation model checkpoint
    - device: 'cuda' or 'cpu'
    Returns two lists:
    - cp_segment_list: list of translated texts
    - segment_list: list of original texts
    """
    if "cuda" in device:
        device_id = 0
    else:
        device_id = -1
        
    segment_list = []
    for segment_item in segment_item_list:
        text = " ".join([words['word'] for words in segment_item['words']])
        segment_list.append(text.strip())
    
    if not model_path=="":
        # Esto esta fallando, revisar por que:
        translator = pipeline(task="translation", model=model_path, tokenizer=model_path, device=device_id)
        result_list = translator(segment_list)
        cp_segment_list = [result["translation_text"] for result in result_list]
        del translator
        torch.cuda.empty_cache()
    else:
        cp_segment_list = segment_list
    
    return cp_segment_list, segment_list   

"""
Add support for language detection in each segment and multiple use of stt models
"""

def to_rttm(segment_item_list, out_filepath):
    _, name = os.path.split(out_filepath)
    with open(out_filepath, "w", encoding="utf-8") as rttm_file:
        for segment_item in segment_item_list:
            start = segment_item["start"]
            end = segment_item["end"]
            duration = round(end-start,3)
            rttm_file.write(f"SPEAKER {name.replace('.rttm','')} 1 {start:.3f} {duration:.3f} <NA> <NA> {segment_item['speaker']} <NA> <NA>\n")
    print("End Writing RTTM:", out_filepath)

def to_xml(segment_item_list, out_filepath):
    root = ET.Element("root")
    for segment_item in segment_item_list:
        segment = ET.SubElement(root, "segment", speaker=segment_item['speaker'], language=segment_item['language'], start="{:.3f}".format(segment_item['start']), end="{:.3f}".format(segment_item['end']))
        for words in segment_item['words']:
            word = ET.SubElement(segment, "word", start="{:.3f}".format(words['start']), end="{:.3f}".format(words['end']), conf="{:.3f}".format(words['conf']))
            word.text = words['word']
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ")
    with open(out_filepath, 'w', encoding='utf-8') as xml_file:
        xml_file.write(pretty_xml)
    print("End Writing XML:", out_filepath)
    
def to_trs():
    pass

def to_json(segment_list, cp_segment_list, segment_item_list, out_filepath):
    with open(out_filepath, 'w', encoding='utf-8') as json_file:
        for i, segment_item in enumerate(segment_item_list):
            start = segment_item["start"]
            end = segment_item["end"]
            item = {
                "audio_filepath": "",
                "text": "",
                "pred_text": segment_list[i],
                "cp_pred_text": cp_segment_list[i],
                "duration": round(end-start,3),
                "start": start,
                "end": end,
                "speaker": segment_item['speaker'],
                "language": segment_item['language'],
                "words": segment_item['words']
            }
            json_file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("End Writing JSON:", out_filepath)  

def to_txt(segment_list, out_filepath):
    with open(out_filepath,"w",encoding="utf-8") as txt_file:
        for segment in segment_list:
            txt_file.write(segment.strip()+"\n")
    print("End Writing TXT:", out_filepath)
    
def to_vtt(segment_list, segment_item_list, out_filepath):
    with open(out_filepath,"w",encoding="utf-8") as vtt_file:
        vtt_file.write("WEBVTT\n\n")
        for i, segment_item in enumerate(segment_item_list):
            vtt_file.write(str(i+1)+"\n")
            start_time=f"{(int(segment_item['start'])//3600):02}:{((int(segment_item['start'])%3600)//60):02}:{(int(segment_item['start'])%60):02}.{(int(((segment_item['start'])-int(segment_item['start']))*1000)):03}"
            end_time=f"{(int(segment_item['end'])//3600):02}:{((int(segment_item['end'])%3600)//60):02}:{(int(segment_item['end'])%60):02}.{(int(((segment_item['end'])-int(segment_item['end']))*1000)):03}"
            vtt_file.write(f"{start_time} --> {end_time} line:16 position:50% align:center\n")
            text = segment_list[i]
            vtt_file.write(f"<c.white.bg_black>{text.strip()}</c>\n\n")
    print("End Writing VTT:", out_filepath)

def to_srt(segment_list, segment_item_list, out_filepath):
    with open(out_filepath,"w",encoding="utf-8") as srt_file:
        for i, segment_item in enumerate(segment_item_list):
            srt_file.write(str(i+1)+"\n")
            start_time=f"{(int(segment_item['start'])//3600):02}:{((int(segment_item['start'])%3600)//60):02}:{(int(segment_item['start'])%60):02},{(int(((segment_item['start'])-int(segment_item['start']))*1000)):03}"
            end_time=f"{(int(segment_item['end'])//3600):02}:{((int(segment_item['end'])%3600)//60):02}:{(int(segment_item['end'])%60):02},{(int(((segment_item['end'])-int(segment_item['end']))*1000)):03}"
            srt_file.write(f"{start_time} --> {end_time}\n")
            text = segment_list[i]
            srt_file.write(f"<font  color=\"white\" back=\"black\" line=\"16\" position=\"50%\" align=\"center\">{text.strip()}</font>\n\n")
    print("End Writing SRT:", out_filepath)


########################################################################################
######################################### MAIN #########################################
########################################################################################
   
def main(args):
    audio_file = args.audio_file
    audio_name, _ = os.path.splitext(os.path.basename(audio_file))
    seg_model = args.seg_model
    config_yml = args.seg_config_yml
    seg_option = args.seg_option
    stt_model = args.stt_model
    cp_model = args.cp_model
    out_path = f"{args.out_path}/{audio_name}"
    device = args.device
    target_sr = 16000 # All nemo models work with 16kHz audio

    # Load audio and resample if needed
    waveform, sample_rate = torchaudio.load(audio_file)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        sample_rate = target_sr
        
    inputs = {'waveform': waveform, 'sample_rate': sample_rate}

    # Perform Segmentation 'diar' or 'vad' using Pyannote
    segment_item_list = pyannote_seg(inputs=inputs, model_path=seg_model,
                                     option=seg_option, config_yml=config_yml,
                                     device=device)
    
    # Assuming model name contains language code, in the future need to improve this
    if "eu" in os.path.basename(stt_model):
        lang="eu"
    elif "es" in os.path.basename(stt_model):
        lang="es"
    else:
        lang="bi"
    
    # Perform ASR using NeMo
    segment_item_list = nemo_asr(inputs=inputs, model_path=stt_model,
                            segment_item_list=segment_item_list,
                            lang=lang, device=device)
    
    # Perform C&P using MarianMT
    cp_segment_list, segment_list = marianmt_cp(segment_item_list=segment_item_list,
                                                model_path=cp_model, device=device)
    
    # Write outputs
    os.makedirs(out_path, exist_ok=True)
    
    to_rttm(segment_item_list, f"{out_path}/{audio_name}.rttm")
    to_xml(segment_item_list, f"{out_path}/{audio_name}.xml")
    to_json(segment_list, cp_segment_list, segment_item_list, f"{out_path}/{audio_name}.json")
    
    to_txt(segment_list, f"{out_path}/{audio_name}.txt")
    to_vtt(segment_list, segment_item_list, f"{out_path}/{audio_name}.vtt")
    to_srt(segment_list, segment_item_list, f"{out_path}/{audio_name}.srt")
    
    if not cp_model=="":
        to_txt(cp_segment_list, f"{out_path}/{audio_name}_cp.txt")
        to_vtt(cp_segment_list, segment_item_list, f"{out_path}/{audio_name}_cp.vtt")
        to_srt(cp_segment_list, segment_item_list, f"{out_path}/{audio_name}_cp.srt")

if __name__=="__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--audio_file", help="(str): path to audio file", required=True, type=str)
    parser.add_argument("--out_path", help="(str): path of the destination folder", default="./results", type=str)
    parser.add_argument("--seg_model", help="(str): path to segmentation model (Pyannote)", default="seg_CONF.ckpt", type=str)
    parser.add_argument("--seg_config_yml", help="(str): path configuration .yaml file for segmentation model", default="config_diarization-3.1.yaml", type=str)
    parser.add_argument("--seg_option", help="(str): how to perform the segmentation: 'vad' or 'diar'", default="vad", type=str)
    parser.add_argument("--stt_model", help="(str): path to stt model (NeMo)", default="stt_eu_conformer_transducer_v1.7", type=str)
    parser.add_argument("--cp_model", help="(str): path to capitalization & punctuation model (MarianMT)", default="", type=str)
    parser.add_argument("--device", help="(str): 'cuda' or 'cpu'", default="cpu", type=str)
    args = parser.parse_args()
    main(args)