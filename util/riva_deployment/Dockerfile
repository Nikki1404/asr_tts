FROM nvcr.io/nvidia/riva/riva-speech:2.19.0

WORKDIR /opt/riva

# Copy your models (fine-tuned)
COPY /home/CORP/RIVA/riva_models_19_stable/models/ /data/models

EXPOSE 50051

# Start Riva
CMD ["/opt/riva/bin/start-riva"]
