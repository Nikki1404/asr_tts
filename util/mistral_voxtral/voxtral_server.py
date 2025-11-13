from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch
import soundfile as sf
import uvicorn

app = FastAPI()

device = "cuda"
repo_id = "/home/CORP/Models/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map=device)

class TranscriptionRequest(BaseModel):
    filepath: str

@app.post("voxtral/transcribe/")
def transcribe_audio(request: TranscriptionRequest):
    try:
        audio_file = request.filepath
        audio, sample_rate = sf.read(audio_file)
        inputs = processor.apply_transcrition_request(
            language="en",
            audio=audio,
            model_id=repo_id
        )
        inputs = inputs.to(device, dtype=torch.bfloat16)
        outputs = model.generate(**inputs, max_new_tokens=500)
        decoded_outputs = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return {"transcription": "".join(decoded_outputs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("voxtral_server:app", host="0.0.0.0", port=5005, reload=True)