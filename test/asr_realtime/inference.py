import asyncio
import websockets
import json
import pyaudio
import numpy as np
import uuid
 
# WebSocket address
# websocket_address = "ws://10.90.126.61:3000"
websocket_address = "wss://whisperstream.exlservice.com"
 
# Audio configuration
sample_rate = 16000
channels = 1
 
# WebSocket connection
websocket = None
 
# Audio stream
stream = None
 
# Recording flag
is_recording = False
 
async def receive_data():
    global websocket
    while True:
        try:
            data = await websocket.recv()
            if isinstance(data, str):
                json_data = json.loads(data)
                print(f"Data received from server: {json_data}")
                if json_data.get("type") == "config" and json_data.get("audio_bytes_status") == "end":
                    await send_audio_config()
           
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
            # exit()
            # break
 
 
async def connect_websocket():
    global websocket
    websocket = await websockets.connect(websocket_address)
    print("WebSocket connection established")
 
 
#ASR pipeline can be: "whisper" (default), "riva", "azure", "google"
#In speech to speech pipeline nlpEngine is the Responder, can be: chatgpt, different agents
 
async def send_audio_config():
 
    audio_config = {
 "service": "asr", #"asr" (default), "s2s" (speech to speech)
 "asrPipeline":"riva", #riva, azure, google, default is whisper
 "nlpEngine":"healthcare-agent", #utility-agent, insurance-agent, banking-agent, healthcare-agent
 "ttsEngine":"polly", #polly, riva, azure, google
    # "ttsVoice":"English-US.Male-1",
    "tts_emotion_detection": False,
    "user_speaking": True,
     #"sampling_rate":16000,
    "chunk_offset_seconds": 0.6,
    "chunk_length_seconds": 1.8,
    # "session_id":"AA-1b577f5f-0e19-4f3c-b423-dbe98197d352",
 
 }
    await websocket.send(json.dumps(audio_config))
 
async def start_recording():
    global stream, is_recording
    is_recording = True
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=sample_rate, input=True, frames_per_buffer=512)
    print("Recording started")
 
async def stop_recording():
    global stream, is_recording
    is_recording = False
    stream.stop_stream()
    stream.close()
    print("Recording stopped")
   
async def process_audio(sample_data):
    try:
        await websocket.send(sample_data.tobytes())
    
    except Exception as e:
        print(f'Error in process audio:*-{e}-*')
     
 
async def main():
    await connect_websocket()
    await send_audio_config()
    await start_recording()
   
    receive_task = asyncio.create_task(receive_data())
 
    while is_recording:
       
        sample_data = np.frombuffer(stream.read(1024), dtype=np.int16)
           
        # print(sample_data)
       
        await process_audio(sample_data)
 
    await stop_recording()
   
    receive_task.cancel()
 
 
asyncio.run(main())
 
