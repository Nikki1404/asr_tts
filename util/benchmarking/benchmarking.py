import pandas as pd
import requests
import json
import time


df = pd.read_excel(r"ground_th.xlsx")

# Define the paths and headers
base_path = '/home/CORP/om224604/Converted Files'
upload_url = "http://10.90.126.61:3000/upload_file"
# wer_url = "https://whisperstream.exlservice.com/wer_score"
wer_url = "http://10.90.126.61:3000/wer_score"
  
headers_upload = {
    'api_key': '12345',
    'api_name': 'exl_asr',
    'asrPipeline': 'riva'
}
headers_wer = {
    'api_key': '12345',
    'api_name': 'exl_asr',
    'Content-Type': 'application/json'
}



# Function to upload file and get transcription
def upload_file_and_get_transcription(file_path):
    print(f"Attempting to open file: {file_path}")
    files = [
        ('file', (file_path.split('/')[-1], open(file_path, 'rb'), 'wav'))
    ]
    response = requests.request("POST", upload_url, headers=headers_upload, files=files)
    return response.text

# Function to get WER score
def get_wer_score(ground_truth, transcription):
    payload = json.dumps({
        "ground_truth": ground_truth,
        "transcription": transcription
    })
    response = requests.request("POST", wer_url, headers=headers_wer, data=payload)
    return response.text

# Iterate over each row in the DataFrame
for index, row in df.iterrows():

    # time.sleep(2)
    file_name = row['File Name']
    
    # Skip rows where file_name is NaN or empty
    if pd.isna(file_name) or file_name == '':
        print(f"Skipping row {index} due to missing file name")
        continue
    
    file_name = str(file_name)  # Convert to string
    ground_truth = row['Ground_truth']
    
    # Ensure the file name has the .wav extension
    if not file_name.endswith('.wav'):
        file_name += '.wav'
    
    # Construct the full file path
    file_path = f"{base_path}/{file_name}"

    # Measure the inference time for transcription
    start_time_transcription = time.time()
    
    # Upload the file and get transcription
    transcription = upload_file_and_get_transcription(file_path)
    
    end_time_transcription = time.time()
    
    inference_time_transcription = end_time_transcription - start_time_transcription
    print(ground_truth)
    print("----------")
    print(transcription)
    

    # Get WER score
    wer_score = get_wer_score(ground_truth, transcription)

    print(wer_score)
    
    # Write transcription and WER score to DataFrame
    df.at[index, 'transcription'] = transcription
    df.at[index, 'WER_new_score'] = wer_score
    
    # Write inference times to the appropriate columns based on header name
    if headers_upload['asrPipeline'] == 'riva':
        df.at[index, 'riva_inference_time'] = inference_time_transcription
        #df.at[index, 'riva_wer'] = inference_time_wer
    elif headers_upload['asrPipeline'] == 'whisper':
        df.at[index, 'whisper_inference_time'] = inference_time_transcription
    elif headers_upload['asrPipeline'] == 'parakeet_tdt':
        df.at[index, 'parakeet_tdt_inference_time'] = inference_time_transcription
    elif headers_upload['asrPipeline'] == 'canary':
        df.at[index, 'canary_inference_time'] = inference_time_transcription
    elif headers_upload['asrPipeline'] == 'parakeet_tdt_110':
        df.at[index, 'parakeet_tdt_110_inference_time'] = inference_time_transcription
        #df.at[index, 'distil_wer'] = inference_time_wer

# Save the updated DataFrame to a new Excel file
print(df)
df.to_excel('output_file_rnnt.xlsx', index=False)

print("Process completed successfully.")
