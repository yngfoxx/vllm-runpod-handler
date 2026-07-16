# Custom Inference Image (vLLM · dual-mode)

One image, two run modes, selected by the `MODE` environment variable:

- **`MODE=local`** — runs the vLLM OpenAI-compatible server in the foreground. Point your Bun app at `http://localhost:8000/v1`.
- **`MODE=serverless`** — runs the same vLLM server internally and wraps it in a RunPod serverless worker that proxies each job to the OpenAI endpoint.

Inference behaviour is identical in both modes, so local dev matches production.

## Weights: cached on a volume, never baked

The image does **not** contain model weights. `HF_HOME` points the Hugging Face cache at a mounted volume; vLLM downloads the model on first run and reuses the cache after that. The cache format is identical everywhere, so the same weights work locally and in production — only the mount path differs:

| Environment            | Persistent storage                          | `HF_HOME`                 |
| ---------------------- | ------------------------------------------- | ------------------------- |
| Local (docker compose) | named volume (or bind mount) at `/models`   | `/models` (image default) |
| RunPod Serverless      | network volume — mounts at `/runpod-volume` | `/runpod-volume/hf`       |

## Local development (docker compose)

```bash
docker compose up --build      # first run downloads the model into the volume
docker compose up              # later runs reuse the cached weights
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local-model","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

The named volume `hf-cache` persists across `docker compose down/up`. Weights are pulled once. (Requires the NVIDIA Container Toolkit.)

### Plain `docker run` (equivalent)

```bash
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  -v hf-cache:/models -e HF_HOME=/models -e MODE=local \
  local-model-vllm:latest
```

## Deploy to RunPod Serverless

1. Build and push the image to a registry (Docker Hub, GHCR, ...).
2. Create a **network volume** and, one time, populate it with the model (e.g. run a cheap Pod with the volume attached and `huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ`, or just let the first serverless request download it into the volume).
3. Create a Serverless endpoint from the image and attach the network volume (Edit Endpoint → Advanced → Network Volumes).
4. Set these endpoint environment variables:
   - `MODE=serverless`
   - `HF_HOME=/runpod-volume/hf` ← points the cache at the mounted network volume
   - plus any overrides from the table below.

The base image's entrypoint is overridden, so the container start command is already `python3 /app/handler.py`; no custom command needed.

### Test the serverless handler locally

```bash
# requires a local GPU + the model, since the handler proxies to a real vLLM server
MODE=serverless HF_HOME=/models python3 handler.py --test_input "$(cat test_input.json)"
```

## Request formats (serverless)

The handler accepts any of these inside the job `input`:

```jsonc
// 1. RunPod OpenAI convention — recommended.
//    Works with RunPod's built-in /openai/v1 passthrough, so your Bun app can use
//    the standard OpenAI SDK pointed at:
//    https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1
{ "input": { "openai_route": "/v1/chat/completions",
             "openai_input": { "model": "local-model", "messages": [ ... ], "max_tokens": 128 } } }

// 2. Chat shorthand
{ "input": { "messages": [ ... ], "max_tokens": 128 } }

// 3. Completion shorthand
{ "input": { "prompt": "…", "max_tokens": 128 } }
```

The handler honors the **per-request** `stream` flag. A non-stream request always
returns the full OpenAI JSON response (`chat.completion` with `choices[0].message`
and `usage`) — this is true even when `ENABLE_STREAMING=true`, so an OpenAI
non-stream client (e.g. LangChain's `ChatOpenAI.invoke`) is never handed
`chat.completion.chunk` deltas. Setting `ENABLE_STREAMING=true` only *enables*
token streaming for clients that opt in with `stream: true` (consume via the
endpoint's `/stream` or the OpenAI passthrough); it no longer forces streaming on
every request.

## Environment variables

| Variable                 | Default                        | Purpose                                                         |
| ------------------------ | ------------------------------ | --------------------------------------------------------------- |
| `MODE`                   | `local`                        | `local` (foreground server) or `serverless` (RunPod worker)     |
| `MODEL_NAME`             | `Qwen/Qwen2.5-7B-Instruct-AWQ` | HF repo id (AWQ Int4). Also try a Mistral 7B v0.3 AWQ build     |
| `SERVED_MODEL_NAME`      | `local-model`                  | Friendly alias clients pass as `model`                          |
| `HF_HOME`                | `/models`                      | Hugging Face cache dir — point at your mounted volume           |
| `QUANTIZATION`           | _(empty)_                      | Empty = auto-detect AWQ → Marlin. Set `awq_marlin` to force     |
| `MAX_MODEL_LEN`          | `8192`                         | Context length. Raise for long transcripts (uses more VRAM/seq) |
| `GPU_MEMORY_UTILIZATION` | `0.90`                         | Fraction of VRAM for weights + KV cache                         |
| `MAX_NUM_SEQS`           | _(empty)_                      | Optional cap on concurrent sequences                            |
| `ENABLE_PREFIX_CACHING`  | `true`                         | KV-cache prefix reuse across the 4 pipeline stages              |
| `ENABLE_STREAMING`       | `false`                        | Allow `stream: true` clients to stream. Non-stream requests still get full JSON either way |
| `VLLM_API_KEY`           | _(empty)_                      | If set, the server requires this bearer token                   |
| `VLLM_PORT`              | `8000`                         | Server port (local mode also exposes it)                        |
| `TRUST_REMOTE_CODE`      | `false`                        | Only needed for models that require it (Qwen2.5 does not)       |
| `EXTRA_VLLM_ARGS`        | _(empty)_                      | Any extra raw vLLM flags, space-separated                       |

## Notes

- Prefix caching is on by default — the four back-to-back pipeline stages reuse the shared transcript's KV cache instead of recomputing it.
- Cold start on a scaled-to-zero worker is the model-load time (~20–60s for a 7B). With the model already on the network volume, that's load-only (no download). Mitigate further with FlashBoot (default), a warm-up call at session start, or a minimum active worker during business hours.
- One network volume + one concurrent writer is safest; RunPod warns that many workers writing the same volume at once can corrupt it. Downloading the weights once up front avoids concurrent-write races on cold scale-ups.
- Pin the base image tag and the two pip packages for reproducible builds; bump deliberately.
