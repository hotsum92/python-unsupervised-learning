FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

WORKDIR /usr/src/

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt
