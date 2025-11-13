import azure.cognitiveservices.speech as speechsdk
import os
import logging
from ..config import ALL_CONFIG

logger = logging.getLogger(__name__)

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.INFO)

# Create a file handler to write logs to a file
file_handler = logging.FileHandler(ALL_CONFIG["PATH"]["log_file_global"])
logger.addHandler(file_handler)



class AzureASRClient:
    def __init__(self, subscription_key=ALL_CONFIG['Credentials']['azure']['subscription_key'], service_region=ALL_CONFIG['Credentials']['azure']['service_region']):
        
        self.speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=service_region)

    async def transcribe(self, client):

            file_path = os.path.join(ALL_CONFIG['PATH']['audio_dir'], client.get_file_name())
            audio_config = speechsdk.audio.AudioConfig(filename=file_path)
            recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)

            result = recognizer.recognize_once()
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:

                if result.text not in ["","No speech could be recognized", None]:

                    os.remove(file_path)
                    
                    return {"text": result.text}


            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.info("No speech could be recognized in Azure ASR")
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                logger.error (f"Speech Recognition canceled in Azure ASR: {cancellation_details.reason}. Error details: {cancellation_details.error_details}")