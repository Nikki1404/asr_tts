import riva.client
import os

import logging
import requests, json
from ..post_processing_utils import post_process_itn_output
from ..config import ALL_CONFIG

from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
inverse_normalizer = InverseNormalizer(lang='en')

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(ALL_CONFIG["PATH"]["log_file_global"])
logger.addHandler(file_handler)


class RivaASRClient:
    def __init__(self, uri=f"{ALL_CONFIG['Urls']['riva']}", **kwargs):
        self.auth = riva.client.Auth(uri=uri)
        self.asr_service = riva.client.ASRService(self.auth)
        self.riva_client = riva.client
        self.offline_config = riva.client.RecognitionConfig(
            language_code="en-US",
            max_alternatives=1,
            enable_automatic_punctuation=True,
            verbatim_transcripts=False,
            audio_channel_count=1,
            sample_rate_hertz=16000, 
            # model = "parakeet-1.1b-en-US-asr-offline-silero-vad-asr-bls-ensemble",
            # model = "parakeet-1.1b-en-US-asr-offline-asr-bls-ensemble"
            # model = "canary-0.6b-turbo-multi-asr-offline-asr-bls-ensemble",
            # model = "conformer-en-US-asr-offline-asr-bls-ensemble"
        )
        

    def normalize_transcription(self, text: str) -> str:
        if not text or text.strip(". ").strip() == "":
            return ""
        text = text.replace(",", "")
        text = inverse_normalizer.inverse_normalize(text, verbose=False)
        return post_process_itn_output(text)
    
    async def transcribe(self, client):
    
        try:
       
            file_path = os.path.join(ALL_CONFIG['PATH']['audio_dir'], client.get_file_name())
            with open(file_path, 'rb') as fh:
                data = fh.read()
                
            if client.sampling_rate != 16000:
                self.offline_config.sample_rate_hertz = client.sampling_rate
            
            response = self.asr_service.offline_recognize(data, self.offline_config)
            
            
            transcriptions = [result.alternatives[0].transcript.strip() for result in response.results]
           
            concatenated_transcription = self.normalize_transcription(
                str(" ".join(t for t in transcriptions if t and t.strip()))
            )
          
            os.remove(file_path)
            
            return {"text": concatenated_transcription}
        
        except Exception as e:
            logger.error("Error in RIVA ASR pipeline: {}".format(e))
            return {"text": ""}
        
    def update_word_boosting(self, word_boosting_dict=None, domain= "global"):
        
        if word_boosting_dict:
            try:
                for key, value in word_boosting_dict.items():
                    self.riva_client.add_word_boosting_to_config(self.offline_config, [key], value)
                    
                logger.info(f"word boosting completed for RIVA ASR")
            except Exception as e:
                logger.error("Error in updating word boosting in RIVA ASR pipeline : {}".format(e))
                
            
