# ==================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1


RUN python -m pip install --upgrade pip

WORKDIR /code/src
ENV PYTHONPATH=/code
COPY . /code
COPY requirements.txt .

# the following to let recbole and cognac coexist, feel free to move to requirements if only one of them is needed.
RUN pip install --no-cache-dir Cython numpy==1.25.2 scipy
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt
RUN pip install --no-cache-dir dgl -f https://data.dgl.ai/wheels/cu124/repo.html
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124


ENV PATH="/root/.local/bin:${PATH}"

CMD ["/bin/bash"]
