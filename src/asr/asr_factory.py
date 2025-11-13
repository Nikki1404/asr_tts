from .whisper_turbo_asr import WhisperTurbo
from .riva_asr import RivaASRClient
from .azure_asr import AzureASRClient
from .google_asr import GoogleASRClient

class ASRFactory:
    @staticmethod
    def create_asr_pipeline(asr_type, **kwargs):
        if asr_type == "whisper_turbo":
            return WhisperTurbo()
        if asr_type == "riva_asr":
            return RivaASRClient(**kwargs)
        if asr_type == "azure_asr":
            return AzureASRClient()
        if asr_type == "google_asr":
            return GoogleASRClient()
        else:
            raise ValueError(f"Unknown ASR pipeline type: {asr_type}")
