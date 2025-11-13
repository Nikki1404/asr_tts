import os
import requests
import logging

# from ..post_processing_utils import post_process_itn_output
logger = logging.getLogger(__name__)

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.INFO)

# Create a file handler to write logs to a file
file_handler = logging.FileHandler('binary_strings.log')
logger.addHandler(file_handler)


class VoxtralASRClient:
    def __init__(self, uri="http://10.90.126.78:5001", **kwargs):
        self.url = f"{uri}/voxtral/transcribe"


    async def transcribe(self, client):

        try:
            file_name = client.get_file_name()
            file_path = os.path.join("audio_files", file_name)


            with open(file_path, "rb") as f:
                files = {"file": (file_name, f, "audio/wav")}

                response = requests.post(self.url, files=files)

            os.remove(file_path)

            if response.status_code == 200:

                response_data = response.json()

                transcript = response_data.get("transcript","")
                to_return = {"text": transcript}
                return to_return

            else:
                logger.error({"error": f"Voxtral trancription API failed with status code {response.status_code}"})


        except Exception as e:
            logger.error("Error in Voxtral ASR pipeline: {}".format(e))