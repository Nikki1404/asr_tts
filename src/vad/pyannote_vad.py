import os
from os import remove

from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

from src.audio_utils import save_audio_to_file

from .vad_interface import VADInterface


class PyannoteVAD(VADInterface):
    def __init__(self, **kwargs):
        """
        Initializes Pyannote's VAD pipeline.

        Args:
            model_name (str): The model name for Pyannote.
            auth_token (str, optional): Authentication token for Hugging Face.
        """

        model_name = kwargs.get("model_name", "pyannote/segmentation")

        auth_token = kwargs.get("auth_token")

        if auth_token is None:
            raise ValueError(
                "Missing required argument "
                "in --vad-args: 'auth_token'"
            )

        pyannote_args = kwargs.get(
            "pyannote_args",
            {
                "onset": 0.5,
                "offset": 0.5,
                "min_duration_on": 0.3,
                "min_duration_off": 0.3,
            },
        )
        self.model = Model.from_pretrained(model_name, use_auth_token=auth_token)
        self.vad_pipeline = VoiceActivityDetection(segmentation=self.model)
        self.vad_pipeline.instantiate(pyannote_args)

    async def detect_activity(self, client):
        audio_file_path = await save_audio_to_file(
            audio_data = client.scratch_buffer, file_name = client.get_file_name(), sampling_rate = client.sampling_rate
        )
        vad_results = self.vad_pipeline(audio_file_path)
        
        vad_segments = []
        if len(vad_results) > 0:
            vad_segments = [
                {"start": segment.start, "end": segment.end, "confidence": 1.0}
                for segment in vad_results.itersegments()
            ]
        else:
            pass
        
            
        return vad_segments
