FROM python:3.12-slim

ENV https_proxy="http://163.116.128.80:8080"
ENV http_proxy="http://163.116.128.80:8080"


RUN pip install --no-cache-dir \
    nvidia-riva-client==2.21.1 \
    soundfile


ENV http_proxy=
ENV https_proxy=
ENV RIVA_URI=localhost:50051


WORKDIR /app


COPY transcribe.py /app/
COPY audio/ /app/audio/

CMD ["python", "transcribe.py"]


