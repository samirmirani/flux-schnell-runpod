FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DISABLE_SDP_FASTPATH=1 \
    PYTHONPATH=/workspace/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /workspace/.cache/huggingface

COPY builder/requirements.txt /tmp/requirements.txt
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --index-url ${PYTORCH_INDEX_URL} \
        torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY src ./src
COPY rp_handler.py ./rp_handler.py
COPY test_input.json ./test_input.json

CMD ["python", "-u", "rp_handler.py"]
