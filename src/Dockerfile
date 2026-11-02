# RecProbe execution environment.
# GPU image so that vLLM can be used for review generation; for CPU-only use,
# replace the base image with python:3.11-slim and drop vllm.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /code
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && pip3 install --no-cache-dir vllm

COPY src/ /code/src/
# all commands run from src/, where main.py and the relative paths of the YAML files live
WORKDIR /code/src

EXPOSE 8501
# default: web interface. Override to run the pipeline, e.g.
#   docker run --gpus all -v ./src/:/code/src/ recprobe python3 main.py --data ... --injection ... --evaluation ...
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.address=0.0.0.0"]
