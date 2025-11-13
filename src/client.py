# isort: skip_file
from datetime import datetime
import pytz
import json
import requests

from src.asr.asr_factory import ASRFactory
from src.buffering_strategy.buffering_strategy_factory import (
    BufferingStrategyFactory,
)

class Client:
    """
    Represents a client connected to the VoiceStreamAI server.

    This class maintains the state for each connected client, including their
    unique identifier, audio buffer, configuration, and a counter for processed
    audio files.

    Attributes:
        client_id (str): A unique identifier for the client.
        buffer (bytearray): A buffer to store incoming audio data.
        config (dict): Configuration settings for the client, like chunk length
                       and offset.
        file_counter (int): Counter for the number of audio files processed.
        total_samples (int): Total number of audio samples received from this
                             client.
        sampling_rate (int): The sampling rate of the audio data in Hz.
        samples_width (int): The width of each audio sample in bits.
    """

    def __init__(self, client_id, sampling_rate, samples_width):
        self.client_id = client_id
        self.service = "asr" # can be "asr", "s2s"
        self.session_id = ""
        self.amelia_token = ""
        self.auth_config = {}
        self.buffer = bytearray()
        self.scratch_buffer = bytearray()
        self.contact_id = None
        self.channel = None
        self.asr_engine = "riva"
        self.nlp_engine = "chatgpt"
        self.nlp_engine_config = {}
        self.user_speaking = True
        self.user_input_txt = ""
        self.extracted_entity_dict = {}
        self.tts_response = ""
        self.received_initial_config = False
        self.chunk_length_seconds = 1.8
        self.chunk_offset_seconds = 0.6
        ist = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist)
        self.file_counter = current_time.strftime("%Y%m%d_%H%M%S")
        self.total_samples = 0
        self.sampling_rate = sampling_rate
        self.samples_width = samples_width
        self.asr_pipeline_riva = ASRFactory.create_asr_pipeline("riva_asr")
        self.config = {
            "language": None,
            "processing_strategy": "silence_at_end_of_chunk",
            "processing_args": {
                "chunk_length_seconds": 1.8,
                "chunk_offset_seconds": 0.6,
            },
        }
        self.buffering_strategy = (
            BufferingStrategyFactory.create_buffering_strategy(
                self.config["processing_strategy"],
                self,
                **self.config["processing_args"],
            )
        )         
    def update_client_details(self, kwargs):
        self.service = kwargs.get("service", self.service)
        self.asr_engine = kwargs.get("asrPipeline", self.asr_engine)
        self.nlp_engine = kwargs.get("nlpEngine", self.nlp_engine)
        self.tts_engine = kwargs.get("ttsEngine",self.asr_engine)
        self.nlp_engine_config = kwargs.get("nlpEngine_config",self.nlp_engine_config)
        self.tts_voice = kwargs.get("ttsVoice")
        self.tts_emotion_detection = kwargs.get("tts_emotion_detection", False)
        self.sampling_rate = kwargs.get("sampling_rate", self.sampling_rate) 
        self.user_speaking = kwargs.get("user_speaking", True) 
        self.chunk_length_seconds = kwargs.get("chunk_length_seconds",self.chunk_length_seconds)
        self.chunk_offset_seconds = kwargs.get("chunk_offset_seconds",self.chunk_offset_seconds)
        if self.session_id == "":
            self.session_id = kwargs.get("session_id", "") 
            
        user_input = kwargs.get("user_input")
        if user_input:
            self.user_input_txt = user_input 
            
        if self.asr_engine=="google":
            self.chunk_length_seconds = 4
       
        if self.tts_voice in [None, ""] :
            default_voices = {
            "riva": "English-US.Female-1",
            # "azure": "en-US-Emma:DragonHDLatestNeural",
            "polly":"Joanna"}
            
            self.tts_voice = default_voices.get(self.tts_engine)
        
        self.channel = kwargs.get("channel") 
        if self.channel in ["CUSTOMER", "AGENT"]:
            self.sampling_rate = 8000 

    def append_audio_data(self, audio_data):
        self.buffer.extend(audio_data)
        self.total_samples += len(audio_data) / self.samples_width

    def clear_buffer(self):
        self.buffer.clear()

    def increment_file_counter(self):
        # self.file_counter += 1
        ist = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist)
        # Format the timestamp as YYYYMMDD_HHMMSS
        self.file_counter = current_time.strftime("%Y%m%d_%H%M%S")

    def get_file_name(self):
        if self.contact_id:
            return f"{self.contact_id}_{self.channel}_{self.file_counter}.wav" 
        else:
            return f"{self.client_id}_{self.channel}_{self.file_counter}.wav"

    def process_audio(self, websocket, vad_pipeline, asr_pipeline):
        self.buffering_strategy.process_audio(
            websocket, vad_pipeline, asr_pipeline
        )
