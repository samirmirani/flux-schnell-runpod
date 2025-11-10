FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONPATH=/workspace/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /workspace/.cache/huggingface

COPY builder/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY src ./src
COPY handler.py ./handler.py
COPY test_input.json ./test_input.json

CMD ["python", "-u", "handler.py"]
