import logging
import time
import json
import os
from .config import ALL_CONFIG
from src.dialogue_management import dialogue_manager
from src.tts_manager import save_tts_to_file



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(ALL_CONFIG["PATH"]["log_file_global"])
logger.addHandler(file_handler)

async def send_dm_response_with_tts(client, websocket):
    try:
        start_time = time.time()
        response = await dialogue_manager(client)
        end_time = time.time()

        logger.info(f"Time taken by {client.nlp_engine} LLM : {end_time-start_time} seconds")

        if isinstance(response, dict):
            await websocket.send_json(response)
            response = response["text"]
        elif isinstance(response, str):
            await websocket.send_json({"type":"server_transcript","text":str(response), "session_id":client.session_id})
            
        
        output_file = f"{client.client_id}_tts.wav"
        await websocket.send_json({"type":"config","audio_bytes_status":"start"})
        
        
        if client.tts_response not in ["",None]:
            response = client.tts_response
            
        start_time = time.time()
        
        logger.info(f"tts response string is: {response}")
        
        await save_tts_to_file(text= str(response), output_file= output_file,tts_engine=client.tts_engine, tts_emotion_detection=client.tts_emotion_detection, voice = client.tts_voice)
        
        client.tts_response = ""
        
        end_time = time.time()

        logger.info(f"Time taken by {client.tts_engine} TTS : {end_time-start_time} seconds")
        
        with open(output_file, 'rb') as f:
            while True:
                data = f.read(1024)
                if not data:
                    break
                await websocket.send_bytes(data)
        await websocket.send_json({"type":"config","audio_bytes_status":"end"})
        os.remove(output_file)
        
        client.user_input_txt= ""
        
    except Exception as e:
        logger.error(f"Error in send_dm_response_with_tts {e}")