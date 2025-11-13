## Run
- We can run this in our local machine using the following commands:
    - python -m venv venv
    - source venv/bin/activate
    - pip install -r requirements.txt
    - python inference.py
- The above code connects to the src directory deployed using tmux in 10.90.126.61 (usually named like asr-109)
- The above tmux session connects to a running docker image-container nvcr.io/nvidia/riva/riva-speech for the transcriptions