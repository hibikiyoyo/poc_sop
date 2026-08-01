# AI SOP Monitor — CPU image (works on any host OS with Docker).
#
#   docker build -t sop-monitor .
#   docker run --rm -p 8000:8000 -v sop_data:/app/monitor_data sop-monitor
#
# For NVIDIA GPU inference instead, install the NVIDIA Container Toolkit on
# the host, change the pip install line below to use the CUDA extra
# (`.[gpu] --extra-index-url https://download.pytorch.org/whl/cu126`), and
# run with `--gpus all`.

FROM python:3.12-slim

# libgl1 + libglib2.0-0 are runtime deps of opencv-python;
# ffmpeg provides H.264 encoding for browser-playable result videos.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so Docker layer caching survives code edits.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .[cpu]

COPY weights ./weights
COPY sop_config.example.json SOP.md ./

ENV SOP_MONITOR_HOST=0.0.0.0
EXPOSE 8000

CMD ["sop-monitor"]
