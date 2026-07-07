# The official vLLM image ships CUDA + PyTorch + vLLM already version-matched —
# the fragile part is solved for you. Pin a real release (check Docker Hub tags
# at https://hub.docker.com/r/vllm/vllm-openai/tags for anything newer).
FROM vllm/vllm-openai:v0.11.0

# The base image's ENTRYPOINT launches the vLLM server directly; clear it so our
# dual-mode launcher (CMD, below) runs instead.
ENTRYPOINT []

# Add ONLY the extras we need. vLLM, torch and CUDA are already in the base image
# — do not reinstall them or you risk breaking the matched CUDA/torch build.
RUN pip install --no-cache-dir runpod==1.10.0 requests==2.32.3

WORKDIR /app
COPY handler.py      /app/handler.py
COPY test_input.json /app/test_input.json

# Weights are NOT baked into the image. HF_HOME points the Hugging Face cache at a
# mount path; vLLM downloads the model on first run and reuses the cache after.
#   - Local:      mount a persistent volume at /models (see docker-compose.yml)
#   - Serverless: attach a RunPod network volume (mounts at /runpod-volume) and set
#                 HF_HOME=/runpod-volume/hf
# The cache format is identical across environments, so the same weights work in both.

# ---- Defaults. Override with `docker run -e KEY=VAL` or in the RunPod template. ----
ENV MODE=local \
    MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-AWQ \
    HF_HOME=/models \
    DTYPE=auto \
    MAX_MODEL_LEN=8192 \
    GPU_MEMORY_UTILIZATION=0.90 \
    ENABLE_PREFIX_CACHING=true \
    ENABLE_AUTO_TOOL_CHOICE=true \
    TOOL_CALL_PARSER=hermes \
    ENABLE_STREAMING=false \
    VLLM_PORT=8000


# Local mode serves the OpenAI API here. (Serverless keeps the server internal.)
EXPOSE 8000

CMD ["python3", "/app/handler.py"]