from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Dict
from pydantic import BaseModel
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from tts_utils import polly_neural_tts, azure_tts, get_file_name

import riva.client
import wave
import uuid
import requests
import json
import httpx
import os
import csv
import zipfile


app = FastAPI()


origins = ["*"]
# origins = [
#     "http://localhost",
#     "http://localhost:8000",
#     # Add more origins as needed
# ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)






# Initialize Riva TTS service
uri = "10.90.126.61:5000"
auth = riva.client.Auth(uri=uri)
tts_service = riva.client.SpeechSynthesisService(auth)
language_code = 'en-US'
sample_rate_hz = 16000
nchannels = 1
sampwidth = 2

class ttsTextRequest(BaseModel):
    text: str
    emotion_detection: bool
    # tts_engine: Optional[str] = None

class TextRequest(BaseModel):
    text: str

class WordBoostingDict(BaseModel):
    word_boosting_dict: dict

class WordBoostingStatus(BaseModel):
    word_boosting_status: str

@app.middleware("http")
async def strip_tts_prefix(request: Request, call_next):
    # Check if the path starts with "/tts"
    if request.url.path.startswith("/tts"):
        # If the path is exactly "/tts", do not strip it
        if request.url.path in ["/tts","/tts/"]:
            request.scope["path"] = "/tts"
        else:
            # Strip the "/tts" prefix for other paths
            request.scope["path"] = request.url.path[len("/tts") :]

    if request.url.path.startswith("/speech-api"):

        # Rewrite the path by removing '/speech-api' prefix
        new_path = request.url.path.replace("/speech-api", "", 1)
        # Remove trailing slash if present
        if new_path.endswith("/"):
            new_path = new_path[:-1]
        # Update the request scope with the new path
        request.scope["path"] = new_path

        print(request.scope["path"])

    response = await call_next(request)
    return response


@app.get("/health_tts")
async def health_check():
    return JSONResponse(content={"status": "ok"}, status_code=200)

@app.get("/health_gpt")
async def health_check():
    try:
        # Make an asynchronous request to localhost:5000/health
        async with httpx.AsyncClient() as client:
            response = await client.get("http://127.0.0.1:6000/health",timeout=10.0)
        # response = requests.get("http://localhost:5000/health",timeout=10.0)
        # If the response is 200, return "ok"
        if response.status_code == 200:
            return JSONResponse(content={"status": "ok"}, status_code=200)
        else:
            # If the response is not 200, return a failure status
            return JSONResponse(content={"status": "failed", "detail": "Health check failed"}, status_code=500)

    except httpx.RequestError as e:
        # Handle connection or other request errors
        return JSONResponse(content={"status": "failed", "detail": f"Error: {e}"}, status_code=500)


# TODO Add openai azure gpt function
@app.post("/emotion_detection")
async def emotion_classify(request: TextRequest):
    try:
        text = request.text
        response = emotion_detection_riva(text)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
async def synthesize(request: ttsTextRequest, voice: str = Header(default=""),engine: str = Header(default="riva")):
    try:

        text = request.text
        emotion_detection = request.emotion_detection
        print(f"emotion_detection is:{emotion_detection}")

        file_name = get_file_name(engine)


        default_voices = {
        "riva": "English-US.Female-1",
        "azure": "en-US-Emma:DragonHDLatestNeural",
        "polly":"Joanna"}

        if voice in ["",None]:
            voice = default_voices[engine]

        if engine == "polly":
            polly_neural_tts(text, file_name,voice_id=voice)

        if engine=="azure":
            azure_tts(text, file_name, voice_id=voice)

        # Synthesize speech


        if engine == "riva":

            if emotion_detection:

                emotion = emotion_detection_riva(text)
                if emotion != None:
                    emotion = str(emotion.get("emotion")).capitalize()

                else: emotion = "1"

                print(f"detected emotion is: {emotion}")

                if "Female" in voice:
                    voice = f"English-US.Female-{emotion}"
                else:
                    if emotion not in ["Fearful", "Sad"]:
                        voice = f"English-US.Male-{emotion}"

                print(f"voice chosen:{voice}")

            resp = tts_service.synthesize(text, language_code=language_code, sample_rate_hz=sample_rate_hz, voice_name=voice)
            audio = resp.audio

            # Generate filename
            # unique_id = uuid.uuid4()
            # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            # filename = f"/home/CORP/RIVA/audios/{engine}_{unique_id}_{current_time}.wav"


            # Save the audio to a file
            with wave.open(file_name, 'wb') as out_f:
                out_f.setnchannels(nchannels)
                out_f.setsampwidth(sampwidth)
                out_f.setframerate(sample_rate_hz)
                out_f.writeframesraw(audio)

        return FileResponse(file_name, media_type='audio/wav', filename=file_name)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="An unexpected error occurred: " + str(e))

@app.get("/available_voices/")
async def get_voices():
    # List of available voices
    voices = [
        "English-US.Female-1",
        "English-US.Male-1",
        "English-US.Female-Calm",
        "English-US.Female-Neutral",
        "English-US.Female-Happy",
        "English-US.Female-Angry",
        "English-US.Female-Fearful",
        "English-US.Female-Sad",
        "English-US.Male-Calm",
        "English-US.Male-Neutral",
        "English-US.Male-Happy",
        "English-US.Male-Angry"
    ]
    return JSONResponse(content={"voices": voices})


@app.get("/available_voices/{engine:path}")
async def get_voices(engine:str):
    # List of available voices
    print(engine)
    voices_dict =   {"riva":[
            "English-US.Female-1",
            "English-US.Male-1",
            "English-US.Female-Calm",
            "English-US.Female-Neutral",
            "English-US.Female-Happy",
            "English-US.Female-Angry",
            "English-US.Female-Fearful",
            "English-US.Female-Sad",
            "English-US.Male-Calm",
            "English-US.Male-Neutral",
            "English-US.Male-Happy",
            "English-US.Male-Angry",
        ],
        "riva/magpie":[
            "Magpie-Multilingual.EN-US.Female.Neutral",
            "Magpie-Multilingual.EN-US.Female.Calm",
            "Magpie-Multilingual.EN-US.Female.Fearful",
            "Magpie-Multilingual.EN-US.Female.Happy",
            "Magpie-Multilingual.EN-US.Female.Angry",
            "Magpie-Multilingual.EN-US.Female.Female-1",
            "Magpie-Multilingual.EN-US.Male.Calm",
            "Magpie-Multilingual.EN-US.Male.Fearful",
            "Magpie-Multilingual.EN-US.Male.Happy",
            "Magpie-Multilingual.EN-US.Male.Neutral",
            "Magpie-Multilingual.EN-US.Male.Angry",
            "Magpie-Multilingual.EN-US.Male.Disgusted",
            "Magpie-Multilingual.EN-US.Male.Male-1",
            "Magpie-Multilingual.FR-FR.Male.Male-1",
            "Magpie-Multilingual.FR-FR.Female.Female-1",
            "Magpie-Multilingual.FR-FR.Female.Angry",
            "Magpie-Multilingual.FR-FR.Female.Calm",
            "Magpie-Multilingual.FR-FR.Female.Disgust",
            "Magpie-Multilingual.FR-FR.Female.Sad",
            "Magpie-Multilingual.FR-FR.Female.Happy",
            "Magpie-Multilingual.FR-FR.Female.Fearful",
            "Magpie-Multilingual.FR-FR.Female.Neutral",
            "Magpie-Multilingual.FR-FR.Male.Neutral",
            "Magpie-Multilingual.FR-FR.Male.Angry",
            "Magpie-Multilingual.FR-FR.Male.Calm",
            "Magpie-Multilingual.FR-FR.Male.Sad",
            "Magpie-Multilingual.ES-US.Male.Male-1",
            "Magpie-Multilingual.ES-US.Female.Female-1",
            "Magpie-Multilingual.ES-US.Female.Neutral",
            "Magpie-Multilingual.ES-US.Male.Neutral",
            "Magpie-Multilingual.ES-US.Male.Angry",
            "Magpie-Multilingual.ES-US.Female.Angry",
            "Magpie-Multilingual.ES-US.Female.Happy",
            "Magpie-Multilingual.ES-US.Male.Happy",
            "Magpie-Multilingual.ES-US.Female.Calm",
            "Magpie-Multilingual.ES-US.Male.Calm",
            "Magpie-Multilingual.ES-US.Female.Pleasant_Surprise",
            "Magpie-Multilingual.ES-US.Male.Pleasant_Surprise",
            "Magpie-Multilingual.ES-US.Female.Sad",
            "Magpie-Multilingual.ES-US.Male.Sad",
            "Magpie-Multilingual.ES-US.Male.Disgust"
],
    "polly":
            [
            "Joanna",
            "Matthew",
            "Ruth",
            "Stephen",
            "Amy",
            "Brian",
            "Olivia",
            "Kajal",
            "Niamh",
            "Aria",
            "Jasmine",
            "Ayanda",
            "Danielle"
            ]}

    if engine not in voices_dict:
        raise HTTPException(status_code=404, detail="TTS engine not found")

    # Get the voices based on the key
    voices = voices_dict[engine]

    return JSONResponse(content={"voices": voices})


INTERNAL_CX_SPEECH_TTS_API_URL = "http://0.0.0.0:8080/v1/audio/speech"
@app.post("/v1/audio/speech")
async def cx_speech_tts(request: Request):
    try:
        body = await request.json()
        print("Incoming body:", body)

        internal_resp = requests.post(INTERNAL_CX_SPEECH_TTS_API_URL, json=body)
        internal_resp.raise_for_status()

        return Response(
            content=internal_resp.content,
            media_type=internal_resp.headers.get("Content-Type", "audio/wav")
        )

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Internal TTS service error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=443)
