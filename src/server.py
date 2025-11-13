import json
import logging
import ssl
import uuid
import sys
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from uvicorn import Config, Server as UvicornServer

from src.client import Client
from .config import ALL_CONFIG

# ---------- Logging ----------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(ALL_CONFIG["PATH"]["log_file_global"])
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


# ---------- Main Server ----------
class Server:
    """
    Represents the WebSocket + REST server for handling real-time audio transcription
    with in-memory word boosting.
    """

    def __init__(
        self,
        vad_pipeline,
        asr_pipeline,
        asr_pipeline_riva,
        asr_pipeline_azure,
        asr_pipeline_google,
        host="localhost",
        port=3000,
        sampling_rate=16000,
        samples_width=2,
        certfile=None,
        keyfile=None,
        word_boosting_config_path: Optional[str] = None,
    ):
        self.vad_pipeline = vad_pipeline
        self.asr_pipeline = asr_pipeline
        self.asr_pipeline_riva = asr_pipeline_riva
        self.asr_pipeline_azure = asr_pipeline_azure
        self.asr_pipeline_google = asr_pipeline_google
        self.host = host
        self.port = port
        self.sampling_rate = sampling_rate
        self.samples_width = samples_width
        self.certfile = certfile
        self.keyfile = keyfile
        self.connected_clients = {}

        # ---------- In-memory word boosting store ----------
        # Structure: { domain_name: { word: boost_value, ... }, ... }
        # domain_name may be None or "global" for config that applies generally
        self.word_boosting_dict: Dict[Optional[str], Dict[str, float]] = {}

        # Determine config path: use provided, else ALL_CONFIG path if present
        if word_boosting_config_path:
            self._word_boosting_config_path = word_boosting_config_path
        else:
            self._word_boosting_config_path = (
                ALL_CONFIG.get("PATH", {}).get("word_boosting_config")
            )

        # Try load initial config file at startup (reload from file at startup)
        try:
            with open(self._word_boosting_config_path, "r") as fh:
                data = json.load(fh)
                # Accept either:
                # 1) dict of domain -> {word: boost}
                # 2) flat dict of word -> boost (apply as "global")
                if isinstance(data, dict):
                    # check if values are dicts (per-domain)
                    if all(isinstance(v, dict) for v in data.values()):
                        for dmn, mapping in data.items():
                            self.word_boosting_dict[dmn] = mapping
                        logger.info("Loaded per-domain word boosting config from %s", self._word_boosting_config_path)
                    else:
                        # treat as global mapping
                        self.word_boosting_dict["global"] = data
                        logger.info("Loaded global word boosting config from %s", self._word_boosting_config_path)
                else:
                    logger.warning("Word boosting config file %s is not a JSON object; ignoring.", self._word_boosting_config_path)
        except FileNotFoundError:
            logger.info("No initial word boosting config file found at %s. Starting empty.", self._word_boosting_config_path)
        except Exception as e:
            logger.exception("Error loading word boosting config: %s", e)

        
        self.app = FastAPI(title="FastAPI + WebSocket ASR Server")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # REST endpoints
        self.app.post("/update_custom_words")(self.update_word_boosting_dict)
        self.app.get("/current_custom_words")(self.get_word_boosting_dict)
        self.app.post("/delete_custom_words")(self.clear_word_boosting_dict)
        self.app.get("/domains")(self.get_domain_list)
        self.app.get("/health")(self.health_check)
        self.app.get("/")(self.health_check)

        # WebSocket endpoint
        self.app.websocket("/")(self.handle_websocket)

   
    async def handle_audio(self, client, websocket):
        while True:
            msg = await websocket.receive()
            
            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("bytes"):
                message = msg["bytes"]
                client.append_audio_data(message)

            elif msg.get("text"):
                try:
                    payload = json.loads(msg["text"])
                except Exception:
                    payload = None

                if payload is not None:
                    logger.info("received config : {}".format(str(payload)))
                    client.update_client_details(kwargs=payload)

                    if client.asr_engine == "riva" and not client.received_initial_config:
                        # Prefer domain-specific boosting if available, else fallback to "global"
                        dmn = getattr(client, "domain", None) or "global"
                        word_boosting_dict = self.word_boosting_dict.get(dmn) or self.word_boosting_dict.get("global", {})
                        try:
                            client.asr_pipeline_riva.update_word_boosting(domain=dmn, word_boosting_dict=word_boosting_dict)
                            client.received_initial_config = True
                            logger.info(f"updated word boosting in riva for client {client.client_id} for domain {dmn}")
                        except Exception:
                            logger.exception("Failed to update word boosting on client's riva pipeline for client %s", client.client_id)
                else:
                    logger.debug("Received non-json text frame from client %s: %s", getattr(client, "client_id", "<unknown>"), msg.get("text"))

            else:
                logger.warning(f"Unexpected message type from {getattr(client, 'client_id', '<unknown>')} - {msg}")

         
            try:
                if client.asr_engine in ["riva"]:
                    client.process_audio(websocket, self.vad_pipeline, client.asr_pipeline_riva)
                elif client.asr_engine in ["azure"]:
                    client.process_audio(websocket, self.vad_pipeline, self.asr_pipeline_azure)
                elif client.asr_engine in ["google"]:
                    client.process_audio(websocket, self.vad_pipeline, self.asr_pipeline_google)
                    
                #default to whisper if no asr engine is specified
                else:
                    client.process_audio(websocket, self.vad_pipeline, self.asr_pipeline)
            except Exception as e:
                #TODO
                #break
                # kill connection if process_audio raises (error raising has to be implemented on component level)
                logger.exception("Error while calling client.process_audio for client %s", getattr(client, "client_id", "<unknown>"))

    async def handle_websocket(self, websocket: WebSocket):
        await websocket.accept()
        client_id = str(uuid.uuid4())
        client = Client(client_id, self.sampling_rate, self.samples_width)
        self.connected_clients[client_id] = client

        logger.info(f"Client {client_id} connected")

        try:
            await self.handle_audio(client, websocket)
        except WebSocketDisconnect:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.exception(f"Error with client {client_id}: {e}")
        finally:
            self.connected_clients.pop(client_id, None)

    # ---------- REST Endpoints ----------
    async def update_word_boosting_dict(self, request: Request, domain: str = Header(default=None)):
        """
        Update Riva word boosting dictionary for a given domain (Header).
        Body expected: {"word_boosting_dict": { "word": boost_value, ... }}
        """
        data = await request.json()
        word_boosting_dict = data.get("word_boosting_dict", {})

        dmn_key = domain or "global"
        
        if dmn_key not in self.word_boosting_dict:
            self.word_boosting_dict[dmn_key] = {}

        for k, v in word_boosting_dict.items():
            self.word_boosting_dict[dmn_key][k] = v

        logger.info(f"Updated Riva word boosting dict for domain {dmn_key} with {len(word_boosting_dict)} entries")
        return JSONResponse(content={"status": "ok"}, status_code=200)

    async def get_word_boosting_dict(self, domain: str = Header(default=None)):
        dmn_key = domain or "global"
        dict_val = self.word_boosting_dict.get(dmn_key, {})
        return JSONResponse(content={dmn_key: dict_val}, status_code=200)

    async def clear_word_boosting_dict(self, domain: str = Header(default=None)):
        dmn_key = domain or "global"
        self.word_boosting_dict.pop(dmn_key, None)
        
        logger.info(f"Cleared word boosting dict for domain {dmn_key}")
        return JSONResponse(content={"status": "ok"}, status_code=200)
    
    async def get_domain_list(self):
        dmn_key = domain or "global"
        key_names = list(self.word_boosting_dict.keys())
        return JSONResponse(content={"domains": key_names}, status_code=200)

    async def health_check(self):
        return {"status": "ok"}

    # ---------- Start ----------
    async def start(self):
        """
        Start FastAPI server with REST + WebSocket on the configured port (default 3000).
        If certfile/keyfile provided, uvicorn will run with TLS.
        """
        if self.certfile and self.keyfile:
            config = Config(
                self.app,
                host=self.host,
                port=self.port,
                ssl_certfile=self.certfile,
                ssl_keyfile=self.keyfile,
            )
            server = UvicornServer(config)
            await server.serve()
            return server 
        else:
            logger.info(
                f"WebSocket server ready to accept secure connections on "
                f"{self.host}:{self.port}"
            )
            config = uvicorn.Config(
                self.app,
                host=self.host,
                port=self.port
            )
            server = uvicorn.Server(config)
            await server.serve()
            return server 
