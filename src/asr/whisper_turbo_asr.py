import os
import logging
import torch
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


from .asr_interface import ASRInterface
from ..config import ALL_CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(ALL_CONFIG["PATH"]["log_file_global"])
logger.addHandler(file_handler)

class WhisperTurbo(ASRInterface):
    def __init__(self, model_id="openai/whisper-large-v3-turbo" , device=None):
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
        # self.model.generation_config.language = "en"
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device
            
        )

    async def transcribe(self, client):
        try:
            file_path = os.path.join(ALL_CONFIG['PATH']['audio_dir'], client.get_file_name())
            
            result = self.pipe(file_path)
            
            return result
        
        except Exception as e:
            logger.error("Error in Whisper Turbo ASR pipeline: {}".format(e))
            

