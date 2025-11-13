import soundfile as sf
import glob
import os
import riva.client

os.environ.pop("https_proxy", None)
os.environ.pop("http_proxy", None)

RIVA_URI = os.getenv("RIVA_URI", "localhost:5000")

auth = riva.client.Auth(uri=RIVA_URI)
asr_service = riva.client.ASRService(auth)

for wav_path in glob.glob("audio/*.wav"):
    print(f"\n[+] Transcribing {wav_path}")
    _, sr = sf.read(wav_path)

    config = riva.client.RecognitionConfig(
            language_code="en-US",
            max_alternatives=1,
            enable_automatic_punctuation=True,
            verbatim_transcripts=False,
            audio_channel_count=1,
            sample_rate_hertz=sr, 
        
    )
    with open(wav_path, 'rb') as fh:
        audio = fh.read()
    
                                                        
    response = asr_service.offline_recognize(audio, config)
    transcriptions = [result.alternatives[0].transcript.strip() for result in response.results]
    final_transcription = " ".join(transcriptions)
    
    print("Transcript:", final_transcription)
