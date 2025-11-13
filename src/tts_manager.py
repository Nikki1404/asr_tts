import azure.cognitiveservices.speech as speechsdk
import json
import requests
import logging
import io, wave, time
import riva.client
import base64
import boto3
import os
from google.cloud import texttospeech

from .config import ALL_CONFIG
from .azure_openai_prompt import classify_emotion

#TODO: Make all TTS functions async

# Create a logger
logger = logging.getLogger(__name__)

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.INFO)

# Create a file handler to write logs to a file
file_handler = logging.FileHandler(ALL_CONFIG["PATH"]["log_file_global"])
logger.addHandler(file_handler)


# Initialize azure
azure_subscription_key=ALL_CONFIG['Credentials']['azure']['subscription_key']
azure_service_region=ALL_CONFIG['Credentials']['azure']['service_region']
azure_speech_config = speechsdk.SpeechConfig(subscription=azure_subscription_key, region=azure_service_region)


# Initialize Riva TTS service
riva_auth = riva.client.Auth(uri=ALL_CONFIG['Urls']['riva'])
riva_tts_service = riva.client.SpeechSynthesisService(riva_auth)
language_code = 'en-US'
sample_rate_hz = 16000
nchannels = 1
sampwidth = 2

# Initialize Amazon Polly
polly_client = boto3.Session(
    region_name='us-east-1'  
    ).client('polly')

# Initialize Google
google_client = texttospeech.TextToSpeechClient() 

google_audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate_hz
    )


    
def riva_tts(text, output_file, tts_emotion_detection=False, voice = None):
    if tts_emotion_detection:
            
        emotion = classify_emotion(text)
        if emotion:
            emotion = str(emotion).capitalize()
        
        else: emotion = "1"

        print(f"detected emotion is: {emotion}")
        
        if "Female" in voice:
            voice = f"English-US.Female-{emotion}"
        else:
            if emotion not in ["Fearful", "Sad"]:
                voice = f"English-US.Male-{emotion}"
    print(f"voice chosen:{voice}")
    
    resp = riva_tts_service.synthesize(text, language_code=language_code, sample_rate_hz=sample_rate_hz, voice_name=voice)
    audio = resp.audio
    

    # Save the audio to a file
    with wave.open(output_file, 'wb') as out_f:
        out_f.setnchannels(nchannels)
        out_f.setsampwidth(sampwidth)
        out_f.setframerate(sample_rate_hz)
        out_f.writeframesraw(audio)


def polly_neural_tts(text, output_file = "polly_neural_sample3.wav",voice="Joanna", output_format = "pcm"):
    
    try:
        response = polly_client.synthesize_speech(
            Text=text,
            VoiceId=voice,  
            OutputFormat=output_format,  # Output file format (MP3, OGG, PCM)
            Engine='neural' 
        )
        
      
        sample_rate_hz = 16000
        nchannels = 1
        sampwidth = 2
        with wave.open(output_file, 'wb') as out_f:
            out_f.setnchannels(nchannels)
            out_f.setsampwidth(sampwidth)
            out_f.setframerate(sample_rate_hz)
            out_f.writeframesraw(response['AudioStream'].read())
    
    
    except Exception as e:
        print(f"Error in Amazon Polly TTS: {e}")



def google_tts(
    text,
    output_file,
    voice: str = "en-US-Chirp3-HD-Achernar",
):
    
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    voice_params = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=voice,
    )
    
    response = google_client.synthesize_speech(
        input=synthesis_input,
        voice=voice_params,
        audio_config=google_audio_config
    )
    
    audio_bytes = response.audio_content

    # Save the audio to a file using the wave module
    with wave.open(output_file, 'wb') as out_f:
        out_f.setnchannels(nchannels)
        out_f.setsampwidth(sampwidth)
        out_f.setframerate(sample_rate_hz)
        out_f.writeframes(audio_bytes)




def save_azure_tts_to_file(text, output_file,voice=None):
    
    
    if voice:
        azure_speech_config.speech_synthesis_voice_name = voice
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=azure_speech_config, audio_config=audio_config)
    

    import time
    start_time = time.time()

    result = synthesizer.speak_text_async(text).get()
    
    end_time = time.time()

    # print(f"Time taken for azure tts result: {end_time - start_time} seconds")
    
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Speech synthesized for text: {text}")
        end_time2 = time.time()

        
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {cancellation_details.error_details}")
        return None
    
async def save_azure_tts_to_file_async(text, output_file, voice=None):
    start_time = time.time()
    
    print("inside azure tts")
    audio_buffer = io.BytesIO()
    still_synthesizing = True

    def synthesis_callback(evt):
        nonlocal audio_buffer

        header_offset = 46
        chunk_size = len(evt.result.audio_data) - header_offset

        if evt.result.reason == speechsdk.ResultReason.SynthesizingAudio:
            audio_buffer.write(evt.result.audio_data[-chunk_size:])
        elif evt.result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesis completed.")

    def completed_callback(evt):
        nonlocal still_synthesizing

        print("Synthesis completed")
        still_synthesizing = False

    pull_stream = speechsdk.audio.PullAudioOutputStream()
    stream_config = speechsdk.audio.AudioOutputConfig(stream=pull_stream)
    
    
    if voice:
        azure_speech_config.speech_synthesis_voice_name = voice
    
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=azure_speech_config, audio_config=stream_config)

    speech_synthesizer.synthesis_started.connect(lambda evt: print("Synthesis started"))
    speech_synthesizer.synthesizing.connect(synthesis_callback)
    speech_synthesizer.synthesis_completed.connect(completed_callback)

    result = speech_synthesizer.speak_text_async(text)

    # Give it time to work asynchronously
    while still_synthesizing:
        time.sleep(.05)

    # Save the PCM data to a WAV file
    audio_buffer.seek(0)
    with wave.open(output_file, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)   # 8-bit
        wav_file.setframerate(16000)  # Sample rate
        wav_file.writeframes(audio_buffer.getvalue())

    end_time = time.time()

    print(f"Time taken for azure tts result: {end_time - start_time} seconds")
    
    print(f"Audio saved to {output_file}")
    

    

async def save_tts_to_file(text, output_file,tts_engine="riva", tts_emotion_detection=False, voice = None):
    
    if tts_engine=="riva":
        riva_tts(text, output_file, tts_emotion_detection, voice)
    elif tts_engine=="polly":
        polly_neural_tts(text, output_file,voice)

    elif tts_engine== "azure":
        save_azure_tts_to_file(text, output_file,voice)
    elif tts_engine=="google":
        google_tts(text, output_file, voice)
        
        
        