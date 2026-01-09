FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip

# Cartella di lavoro
WORKDIR /code
COPY requirements.txt .

# Aggiorna pip e installa Flower + librerie ML comuni
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html

# Assicurati che i binari pip siano nel PATH
ENV PATH="/root/.local/bin:${PATH}"


# Avvia una shell interattiva di default
CMD ["/bin/bash"]
