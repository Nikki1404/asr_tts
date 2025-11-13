import argparse
import asyncio
from .config import ALL_CONFIG
import json
import logging
import os

log_file_path = ALL_CONFIG["PATH"]["log_file_global"]
log_dir = os.path.dirname(log_file_path)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

from src.asr.asr_factory import ASRFactory
from src.vad.vad_factory import VADFactory
from .server import Server


def parse_args():
    parser = argparse.ArgumentParser(
        description="VoiceStreamAI Server: Real-time audio transcription "
        "using self-hosted Whisper and WebSocket."
    )
    parser.add_argument(
        "--vad-type",
        type=str,
        default="pyannote",
        help="Type of VAD pipeline to use (e.g., 'pyannote')",
    )
    parser.add_argument(
        "--vad-args",
        type=str,
        default=json.dumps({
            "auth_token": ALL_CONFIG["Credentials"]["hf_token"]
        }),
        help="JSON string of additional arguments for VAD pipeline",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=ALL_CONFIG["HOST"]["host"],
        help="Host for the WebSocket server",
    )
    parser.add_argument(
        "--port", type=int, default=ALL_CONFIG["HOST"]["port"], help="Port for the WebSocket server"
    )
    parser.add_argument(
        "--certfile",
        type=str,
        default=None,
        help="The path to the SSL certificate (cert file) if using secure "
        "websockets",
    )
    parser.add_argument(
        "--keyfile",
        type=str,
        default=None,
        help="The path to the SSL key file if using secure websockets",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="error",
        choices=["debug", "info", "warning", "error"],
        help="Logging level: debug, info, warning, error. default: error",
    )
    return parser.parse_args()




def main():
    args = parse_args()

    logging.basicConfig()
    logging.getLogger().setLevel(args.log_level.upper())

    try:
        vad_args = json.loads(args.vad_args)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON arguments: {e}")
        return
    
    proxy = ALL_CONFIG.get('Urls', {}).get('proxy', '')

    if proxy:
        os.environ["https_proxy"] = os.environ["http_proxy"] = proxy
    
    
    vad_pipeline = VADFactory.create_vad_pipeline(args.vad_type, **vad_args)
    asr_pipeline = ASRFactory.create_asr_pipeline("whisper_turbo")
    
    if proxy:
        os.environ.pop("https_proxy", None)
        os.environ.pop("http_proxy", None)
    
    asr_pipeline_riva = ASRFactory.create_asr_pipeline("riva_asr")
    asr_pipeline_azure = ASRFactory.create_asr_pipeline("azure_asr")
    asr_pipeline_google = ASRFactory.create_asr_pipeline("google_asr")

    server = Server(
        vad_pipeline,
        asr_pipeline,
        asr_pipeline_riva,
        asr_pipeline_azure,
        asr_pipeline_google,
        host=args.host,
        port=args.port,
        sampling_rate=16000,
        samples_width=2,
        certfile=args.certfile,
        keyfile=args.keyfile,
    )

    asyncio.get_event_loop().run_until_complete(server.start())
    asyncio.get_event_loop().run_forever()
    

if __name__ == "__main__":
    main()
