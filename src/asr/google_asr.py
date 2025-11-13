from ..config import ALL_CONFIG
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions
from google.cloud import speech_v1p1beta1 as speech_v1
from google.cloud import speech_v2
import json
import logging
import os

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(ALL_CONFIG["PATH"]["log_file_global"])
logger.addHandler(file_handler)


class GoogleASRClient:
    def __init__(self):
        self.speech_client = speech_v2.SpeechClient(
            client_options=ClientOptions(api_endpoint=ALL_CONFIG['Urls']['google_asr'])
        )
        self.v1_client = speech_v1.SpeechClient()
        
        self.config = speech_v2.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=["en-US"],
            model='latest_short'
        )
     
    async def transcribe_v2(self, client):
        """
        Transcribes the audio file for a given client using Google's Speech API v2.
        """
        file_path = os.path.join(ALL_CONFIG['PATH']['audio_dir'], client.get_file_name())
        try:
            with open(file_path, "rb") as f:
                audio_content = f.read()

            request = cloud_speech.RecognizeRequest(
                recognizer= ALL_CONFIG['Credentials']['google_asr_recognizer'],
                config=self.config,
                content=audio_content
            )

            response = self.speech_client.recognize(request=request)
            transcriptions = [
                result.alternatives[0].transcript for result in response.results
            ]
            concatenated_transcription = " ".join(transcriptions).strip()
            
            if concatenated_transcription in ["", ".", ". ", "None", None]:
                concatenated_transcription = ""
            
            os.remove(file_path)
            return {"text": concatenated_transcription}
        
        except Exception as e:
            logger.error("Error in GOOGLE ASR pipeline: %s", e)
            return {"text": ""}
        
    async def transcribe(self, client):
        """
        Transcribes the audio file using Speech API v1beta1.
        """
       
        file_path = os.path.join(ALL_CONFIG['PATH']['audio_dir'], client.get_file_name())
        try:
            
            with open(file_path, "rb") as f:
                audio_content = f.read()
            
            audio = speech_v1.RecognitionAudio(content=audio_content)
            config = speech_v1.RecognitionConfig(
                encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=client.sampling_rate,
                language_code="en-US",
                enable_automatic_punctuation=True,
            )

            response = self.v1_client.recognize(config=config, audio=audio)
            transcriptions = [
                result.alternatives[0].transcript for result in response.results
                if result.alternatives
            ]
            concatenated_transcription = " ".join(transcriptions).strip()
            if concatenated_transcription in ["", ".", ". ", "None", None]:
                concatenated_transcription = ""
                
            os.remove(file_path)
            return {"text": concatenated_transcription}
        except Exception as e:
            logger.error("Error in Speech v1 transcription: %s", e)
            return {"text": ""}