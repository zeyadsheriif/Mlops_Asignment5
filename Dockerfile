FROM python:3.10-slim

ARG RUN_ID

ENV RUN_ID=${RUN_ID}

RUN echo "Downloading model for Run ID: ${RUN_ID} from MLflow Tracking Server..."

ENTRYPOINT ["python", "-c", "print('Model Container is Ready!')"]