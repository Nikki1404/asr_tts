import argparse
import asyncio
from .config import ALL_CONFIG
import json
import logging
import os

from .config import ALL_CONFIG
from src.utils.logger import get_logger

from src.asr.asr_factory import ASRFactory
from src.vad.vad_factory import VADFactory
from .server import Server


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "VoiceStreamAI Server: Real-time audio transcription "
            "using self-hosted Whisper and WebSocket."
        )
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
        default=json.dumps({"auth_token": ALL_CONFIG["Credentials"]["hf_token"]}),
        help="JSON string of additional arguments for VAD pipeline",
    )

    parser.add_argument(
        "--host",
        type=str,
        default=ALL_CONFIG["HOST"]["host"],
        help="Host for the WebSocket server",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=ALL_CONFIG["HOST"]["port"],
        help="Port for the WebSocket server",
    )

    parser.add_argument(
        "--certfile",
        type=str,
        default=None,
        help="Path to SSL certificate (if using secure websockets)",
    )

    parser.add_argument(
        "--keyfile",
        type=str,
        default=None,
        help="Path to SSL key (if using secure websockets)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="error",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: error)",
    )

    return parser.parse_args()



def main():
    args = parse_args()

    # Apply centralized log level
    logger.setLevel(args.log_level.upper())

    # Parse VAD args safely
    try:
        vad_args = json.loads(args.vad_args)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing VAD JSON arguments: {e}")
        return

    # Proxy setup (optional)
    proxy = ALL_CONFIG.get("Urls", {}).get("proxy", "")
    if proxy:
        os.environ["https_proxy"] = proxy
        os.environ["http_proxy"] = proxy
        logger.info(f"Proxy enabled -> {proxy}")

    # Create pipelines
    vad_pipeline = VADFactory.create_vad_pipeline(args.vad_type, **vad_args)
    asr_pipeline = ASRFactory.create_asr_pipeline("whisper_turbo")

    # Remove proxy after Whisper download (prevent external calls later)
    if proxy:
        os.environ.pop("https_proxy", None)
        os.environ.pop("http_proxy", None)

    asr_pipeline_riva = ASRFactory.create_asr_pipeline("riva_asr")
    asr_pipeline_azure = ASRFactory.create_asr_pipeline("azure_asr")
    asr_pipeline_google = ASRFactory.create_asr_pipeline("google_asr")

    # Initialize server
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

    loop = asyncio.get_event_loop()
    loop.run_until_complete(server.start())
    loop.run_forever()



if __name__ == "__main__":
    main()
