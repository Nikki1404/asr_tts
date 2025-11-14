import json
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.client import Client
from .config import ALL_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Server:
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
        certfile: Optional[str] = None,
        keyfile: Optional[str] = None,
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
        self.connected_clients: Dict[str, Client] = {}

        self.word_boosting_dict: Dict[str, Dict[str, float]] = {}

        if word_boosting_config_path:
            self._word_boosting_config_path = word_boosting_config_path
        else:
            self._word_boosting_config_path = (
                ALL_CONFIG.get("PATH", {}).get("word_boosting_config")
            )

        try:
            with open(self._word_boosting_config_path, "r") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    if all(isinstance(v, dict) for v in data.values()):
                        for dmn, mapping in data.items():
                            self.word_boosting_dict[dmn] = mapping
                        logger.info(
                            "Loaded per-domain word boosting config from %s",
                            self._word_boosting_config_path,
                        )
                    else:
                        self.word_boosting_dict["global"] = data
                        logger.info(
                            "Loaded global word boosting config from %s",
                            self._word_boosting_config_path,
                        )
        except FileNotFoundError:
            logger.info(
                "No initial word boosting config file found at %s. Starting empty.",
                self._word_boosting_config_path,
            )
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

        self.app.post("/update_custom_words")(self.update_word_boosting_dict)
        self.app.get("/current_custom_words")(self.get_word_boosting_dict)
        self.app.post("/delete_custom_words")(self.clear_word_boosting_dict)
        self.app.get("/domains")(self.get_domain_list)
        self.app.get("/health")(self.health_check)
        self.app.get("/")(self.health_check)

        self.app.websocket("/")(self.handle_websocket)

    async def handle_audio(self, client, websocket):
        while True:
            msg = await websocket.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("bytes"):
                client.append_audio_data(msg["bytes"])

            elif msg.get("text"):
                try:
                    payload = json.loads(msg["text"])
                except Exception:
                    payload = None

                if payload:
                    logger.info("received config : %s", str(payload))
                    client.update_client_details(kwargs=payload)

                    if client.asr_engine == "riva" and not client.received_initial_config:
                        dmn = getattr(client, "domain", None) or "global"
                        word_boosting_dict = (
                            self.word_boosting_dict.get(dmn)
                            or self.word_boosting_dict.get("global", {})
                        )
                        try:
                            client.asr_pipeline_riva.update_word_boosting(
                                domain=dmn, word_boosting_dict=word_boosting_dict
                            )
                            client.received_initial_config = True
                            logger.info(
                                "updated word boosting in riva for client %s (domain %s)",
                                client.client_id,
                                dmn,
                            )
                        except Exception:
                            logger.exception(
                                "Failed to update word boosting for client %s",
                                client.client_id,
                            )

                else:
                    logger.debug(
                        "Received non-json text frame from client %s: %s",
                        getattr(client, "client_id", "<unknown>"),
                        msg.get("text"),
                    )

            else:
                logger.warning(
                    "Unexpected message type from %s - %s",
                    getattr(client, "client_id", "<unknown>"),
                    msg,
                )

            # -------------------------------------------------------
            # OPTIMIZATION: Faster ASR routing via dictionary dispatch
            # -------------------------------------------------------
            engine_map = {
                "riva": client.asr_pipeline_riva,
                "azure": self.asr_pipeline_azure,
                "google": self.asr_pipeline_google,
            }
            pipeline = engine_map.get(client.asr_engine, self.asr_pipeline)

            try:
                client.process_audio(websocket, self.vad_pipeline, pipeline)
            except Exception:
                logger.exception(
                    "Error during process_audio for client %s",
                    getattr(client, "client_id", "<unknown>"),
                )

    async def handle_websocket(self, websocket: WebSocket):
        await websocket.accept()

        # -------------------------------------------------------
        # OPTIMIZATION: Use faster unique client id (hex pointer)
        # -------------------------------------------------------
        client_id = hex(id(websocket))

        client = Client(client_id, self.sampling_rate, self.samples_width)
        self.connected_clients[client_id] = client

        logger.info("Client %s connected", client_id)

        try:
            await self.handle_audio(client, websocket)
        except WebSocketDisconnect:
            logger.info("Client %s disconnected", client_id)
        except Exception as e:
            logger.exception("Error with client %s: %s", client_id, e)
        finally:
            self.connected_clients.pop(client_id, None)

    async def update_word_boosting_dict(self, request: Request, domain: Optional[str] = None):
        data = await request.json()
        word_boosting_dict = data.get("word_boosting_dict", {})
        dmn_key = domain or "global"

        if dmn_key not in self.word_boosting_dict:
            self.word_boosting_dict[dmn_key] = {}

        for k, v in word_boosting_dict.items():
            self.word_boosting_dict[dmn_key][k] = v

        logger.info(
            "Updated boosting dict for domain %s with %d entries",
            dmn_key,
            len(word_boosting_dict),
        )
        return JSONResponse(content={"status": "ok"}, status_code=200)

    async def get_word_boosting_dict(self, domain: Optional[str] = None):
        dmn_key = domain or "global"
        return JSONResponse(
            content={dmn_key: self.word_boosting_dict.get(dmn_key, {})},
            status_code=200,
        )

    async def clear_word_boosting_dict(self, domain: Optional[str] = None):
        dmn_key = domain or "global"
        self.word_boosting_dict.pop(dmn_key, None)

        logger.info("Cleared boosting dict for %s", dmn_key)
        return JSONResponse(content={"status": "ok"}, status_code=200)

    async def get_domain_list(self):
        return JSONResponse(
            content={"domains": list(self.word_boosting_dict.keys())},
            status_code=200,
        )

    async def health_check(self):
        return {"status": "ok"}

    async def start(self):
        if self.certfile and self.keyfile:
            config = uvicorn.Config(
                self.app,
                host=self.host,
                port=self.port,
                ssl_certfile=self.certfile,
                ssl_keyfile=self.keyfile,
            )
            server = uvicorn.Server(config)
            await server.serve()
            return server

        logger.info("WebSocket server ready on %s:%s", self.host, self.port)

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
        )
        server = uvicorn.Server(config)
        await server.serve()
        return server
