import asyncio
import json
import os
import time
from datetime import datetime

from .buffering_strategy_interface import BufferingStrategyInterface
from ..send_response_with_speech import send_dm_response_with_tts
from ..config import ALL_CONFIG


import logging

# Create a logger
logger = logging.getLogger(__name__)

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.INFO)

# Create a file handler to write logs to a file
file_handler = logging.FileHandler(ALL_CONFIG["PATH"]["log_file_global"])
logger.addHandler(file_handler)



class SilenceAtEndOfChunk(BufferingStrategyInterface):
    """
    A buffering strategy that processes audio at the end of each chunk with
    silence detection.

    This class is responsible for handling audio chunks, detecting silence at
    the end of each chunk, and initiating the transcription process for the
    chunk.

    Attributes:
        client (Client): The client instance associated with this buffering
                         strategy.
        chunk_length_seconds (float): Length of each audio chunk in seconds.
        chunk_offset_seconds (float): Offset time in seconds to be considered
                                      for processing audio chunks.
    """

    def __init__(self, client, **kwargs):
        """
        Initialize the SilenceAtEndOfChunk buffering strategy.

        Args:
            client (Client): The client instance associated with this buffering
                             strategy.
            **kwargs: Additional keyword arguments, including
                      'chunk_length_seconds' and 'chunk_offset_seconds'.
        """
        self.client = client
        self.processing_flag = False

    def process_audio(self, websocket, vad_pipeline, asr_pipeline):
        """
        Process audio chunks by checking their length and scheduling
        asynchronous processing.

        This method checks if the length of the audio buffer exceeds the chunk
        length and, if so, it schedules asynchronous processing of the audio.

        Args:
            websocket: The WebSocket connection for sending transcriptions.
            vad_pipeline: The voice activity detection pipeline.
            asr_pipeline: The automatic speech recognition pipeline.
        """
        if (self.client.service == "s2s") and (self.client.user_input_txt):
            
            
            asyncio.create_task(
                    send_dm_response_with_tts(self.client, websocket)
                )
            self.client.scratch_buffer.clear()
            
        else:
        
            chunk_length_in_bytes = (
                self.client.chunk_length_seconds
                * self.client.sampling_rate
                * self.client.samples_width
            )
            
            
            if len(self.client.buffer) > chunk_length_in_bytes:
                if self.processing_flag:
                    logger.error(
                        "Error in realtime processing: tried processing a new "
                        "chunk while the previous one was still being processed"
                    )


                self.client.scratch_buffer += self.client.buffer
                self.client.buffer.clear()
                self.processing_flag = True
                # Schedule the processing in a separate task
                asyncio.create_task(
                    self.process_audio_async(websocket, vad_pipeline, asr_pipeline)
                )
       


    async def process_audio_async(self, websocket, vad_pipeline, asr_pipeline):
        """
        Asynchronously process audio for activity detection and transcription.

        This method performs heavy processing, including voice activity
        detection and transcription of the audio data. It sends the
        transcription results through the WebSocket connection.

        Args:
            websocket (Websocket): The WebSocket connection for sending
                                   transcriptions.
            vad_pipeline: The voice activity detection pipeline.
            asr_pipeline: The automatic speech recognition pipeline.
        """
        
        vad_results = await vad_pipeline.detect_activity(self.client)
        
        # logger.info("vad results found ------{}".format(vad_results))
        # logger.info("scratch buffer length: %s", len(self.client.scratch_buffer))
        # logger.info("scratch buffer length in seconds: %s",len(self.client.scratch_buffer) / (self.client.sampling_rate * self.client.samples_width))

        if len(vad_results) == 0:
            os.remove(os.path.join(ALL_CONFIG['PATH']['audio_dir'], self.client.get_file_name()))
            self.client.scratch_buffer.clear()
            self.client.buffer.clear()
            self.client.increment_file_counter()
            self.processing_flag = False
            return

        last_segment_should_end_before = (
            len(self.client.scratch_buffer)
            / (self.client.sampling_rate * self.client.samples_width)
        ) - self.client.chunk_offset_seconds

        
        if vad_results[-1]["end"] < last_segment_should_end_before:
            
            start = time.time()
            transcription = await asr_pipeline.transcribe(self.client)

            
            
            if transcription and transcription.get("text") != "":

                end = time.time()
                transcription["processing_time"] = end - start
                logger.info(f"reference audio file:{self.client.get_file_name()} ")
                logger.info(f"time taken for {self.client.asr_engine} ASR : {end - start}")
                logger.info("{}_{} : {}".format(self.client.contact_id, self.client.channel,transcription.get("text")))
                

                if self.client.service=="s2s" and self.client.user_speaking:
                    
                    self.client.user_input_txt=str(transcription.get("text"))
                    self.client.user_speaking = False
                    await websocket.send_json({"type":"user_transcript","text":str(self.client.user_input_txt)})
                    
                    asyncio.create_task(send_dm_response_with_tts(self.client, websocket))
                    
                    self.client.scratch_buffer.clear()
                
                if self.client.service=="asr":
                    
                    await websocket.send_json(transcription)
                    
                
            self.client.scratch_buffer.clear()
            self.client.increment_file_counter()

        self.processing_flag = False
