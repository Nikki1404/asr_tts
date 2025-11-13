import pandas as pd
import requests
import json
import time
import werpy
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch
from pydub.utils import make_chunks
from pydub import AudioSegment
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

device = "cuda"
repo_id = "/home/CORP/Models/Voxtral-Mini-3B-2507"
processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map=device)


df = pd.read_excel(r"ground_th.xlsx")

base_path = '/home/CORP/om224604/Converted Files'

def get_vad_timestamps(file_path):
    model = load_silero_vad()
    wav = read_audio(file_path)
    speech_timestamps = get_speech_timestamps(wav, model)
    sample_rate = 16000

    proper_timestamps = [
        {'start': segment['start'] / sample_rate, 'end': segment['end'] / sample_rate}
        for segment in speech_timestamps
    ]
    return proper_timestamps

def group_vad_timestamps(vad_timestamps, max_duration=120):
    chunks = []
    current_chunk = []
    current_duration = 0

    for ts in vad_timestamps:
        segment_duration = ts['end'] - ts['start']
        if current_duration + segment_duration <= max_duration:
            current_chunk.append(ts)
            current_duration += segment_duration
        else:
            chunks.append(current_chunk)
            current_chunk = [ts]
            current_duration = segment_duration
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def get_transcription(file_path):
    vad_timestamps = get_vad_timestamps(file_path)
    chunk_groups = group_vad_timestamps(vad_timestamps, max_duration=120)
    audio = AudioSegment.from_file(file_path)
    chunk_files = []

    for i, chunk_group in enumerate(chunk_groups):
        start_ms = int(chunk_group[0]['start'] * 1000)
        end_ms = int(chunk_group[-1]['end'] * 1000)
        chunk = audio[start_ms:end_ms]
        chunk_name = f"{file_path}_chunk{i}.wav"
        chunk.export(chunk_name, format="wav")
        chunk_files.append(chunk_name)

    inputs = processor.apply_transcrition_request(language="en", audio=chunk_files, model_id=repo_id, sampling_rate=8000)
    inputs = inputs.to(device, dtype=torch.bfloat16)
    outputs = model.generate(**inputs, max_new_tokens=500)
    decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    final_transcript = " ".join(decoded_outputs)
    return final_transcript


def get_wer_score(ground_truth, transcription):
    return werpy.wer(ground_truth, transcription)


for index, row in df.iterrows():
    file_name = row['File Name']
    # Skip rows where file_name is NaN or empty
    if pd.isna(file_name) or file_name == '':
        print(f"Skipping row {index} due to missing file name")
        continue
    
    file_name = str(file_name) 
    ground_truth = row['Ground_truth']
    
    if not file_name.endswith('.wav'):
        file_name += '.wav'
    
    # Construct the full file path
    file_path = f"{base_path}/{file_name}"

    start_time_transcription = time.time()
    
    transcription = get_transcription(file_path)
    
    end_time_transcription = time.time()
    
    inference_time_transcription = end_time_transcription - start_time_transcription
    print(ground_truth)
    print("----------")
    print(transcription)
    

    # Get WER score
    wer_score = get_wer_score(ground_truth, transcription)

    print(wer_score)
    
   
    df.at[index, 'transcription'] = transcription
    df.at[index, 'WER_new_score'] = round(wer_score, 2)
    
    df.at[index, 'voxtral_inference_time'] = round(inference_time_transcription,2)
 



df.to_excel('output_file_voxtral.xlsx', index=False)

print("Process completed successfully.")
