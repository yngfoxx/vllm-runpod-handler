#!/usr/bin/env python3
"""
Dual-mode entrypoint for an inference container.

  MODE=local       Run the vLLM OpenAI-compatible server in the foreground.
                   Exposes the OpenAI API on 0.0.0.0:$VLLM_PORT for local dev,
                   or for hosting on an always-on box later.

  MODE=serverless  Start the same vLLM server as an internal subprocess, wait
                   until it is healthy, then run the RunPod serverless worker.
                   The handler forwards each job to the local OpenAI endpoint,
                   so inference behaviour is identical to local mode.

Both modes serve the same model with the same engine flags (prefix caching,
AWQ/Marlin, context length, ...), so what you test locally is exactly what runs
on RunPod. The model loads once per worker; RunPod/FlashBoot keeps it warm.
"""

import os
import sys
import time
import signal
import subprocess

import requests

# --------------------------------------------------------------------------- #
# Configuration — everything is overridable via environment variables.
# --------------------------------------------------------------------------- #
MODE = os.getenv("MODE", "local").strip().lower()

MODEL_NAME              = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-AWQ")
SERVED_MODEL_NAME       = os.getenv("SERVED_MODEL_NAME", "").strip()      # friendly alias clients call, e.g. "sales-coach"
# QUANTIZATION="" lets vLLM auto-detect AWQ from the checkpoint and pick the
# Marlin kernel automatically on Ampere/Ada. Set "awq_marlin" to force it.
QUANTIZATION            = os.getenv("QUANTIZATION", "").strip()
DTYPE                   = os.getenv("DTYPE", "auto").strip()
MAX_MODEL_LEN           = os.getenv("MAX_MODEL_LEN", "8192").strip()      # raise for longer transcripts (more VRAM/seq)
GPU_MEMORY_UTILIZATION  = os.getenv("GPU_MEMORY_UTILIZATION", "0.90").strip()
MAX_NUM_SEQS            = os.getenv("MAX_NUM_SEQS", "").strip()           # optional concurrency cap
ENABLE_PREFIX_CACHING   = os.getenv("ENABLE_PREFIX_CACHING", "true").strip().lower() in ("1", "true", "yes")
TRUST_REMOTE_CODE       = os.getenv("TRUST_REMOTE_CODE", "false").strip().lower() in ("1", "true", "yes")
ENABLE_AUTO_TOOL_CHOICE = os.getenv("ENABLE_AUTO_TOOL_CHOICE", "false").strip().lower() in ("1", "true", "yes")
TOOL_CALL_PARSER        = os.getenv("TOOL_CALL_PARSER", "").strip()       # e.g. "hermes" for Qwen2.5
EXTRA_VLLM_ARGS         = os.getenv("EXTRA_VLLM_ARGS", "").strip()        # any extra raw flags, space-separated

VLLM_PORT               = os.getenv("VLLM_PORT", "8000").strip()
VLLM_API_KEY            = os.getenv("VLLM_API_KEY", "").strip()          # if set, the server requires this bearer token
ENABLE_STREAMING        = os.getenv("ENABLE_STREAMING", "false").strip().lower() in ("1", "true", "yes")

STARTUP_TIMEOUT_S       = int(os.getenv("STARTUP_TIMEOUT_S", "600"))     # first model load can take minutes
REQUEST_TIMEOUT_S       = int(os.getenv("REQUEST_TIMEOUT_S", "600"))

# Local mode binds publicly; serverless keeps the server internal-only.
BIND_HOST   = "0.0.0.0" if MODE == "local" else "127.0.0.1"
LOCAL_BASE  = f"http://127.0.0.1:{VLLM_PORT}"
MODEL_ALIAS = SERVED_MODEL_NAME or MODEL_NAME


def build_vllm_command():
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--host", BIND_HOST,
        "--port", VLLM_PORT,
        "--dtype", DTYPE,
        "--max-model-len", MAX_MODEL_LEN,
        "--gpu-memory-utilization", GPU_MEMORY_UTILIZATION,
    ]
    if SERVED_MODEL_NAME:
        cmd += ["--served-model-name", SERVED_MODEL_NAME]
    if QUANTIZATION:
        cmd += ["--quantization", QUANTIZATION]
    if ENABLE_PREFIX_CACHING:
        cmd += ["--enable-prefix-caching"]
    if MAX_NUM_SEQS:
        cmd += ["--max-num-seqs", MAX_NUM_SEQS]
    if TRUST_REMOTE_CODE:
        cmd += ["--trust-remote-code"]
    if ENABLE_AUTO_TOOL_CHOICE:
        cmd += ["--enable-auto-tool-choice"]
    if TOOL_CALL_PARSER:
        cmd += ["--tool-call-parser", TOOL_CALL_PARSER]
    if VLLM_API_KEY:
        cmd += ["--api-key", VLLM_API_KEY]
    if EXTRA_VLLM_ARGS:
        cmd += EXTRA_VLLM_ARGS.split()
    return cmd


def server_healthy():
    try:
        return requests.get(f"{LOCAL_BASE}/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def wait_until_healthy(timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if server_healthy():
            return True
        time.sleep(2)
    return False


# --------------------------------------------------------------------------- #
# LOCAL mode — replace this process with the vLLM server (logs stream through).
# --------------------------------------------------------------------------- #
def run_local():
    cmd = build_vllm_command()
    print(f"[entrypoint] LOCAL mode — exec: {' '.join(cmd)}", flush=True)
    os.execvp(cmd[0], cmd)


# --------------------------------------------------------------------------- #
# SERVERLESS mode — vLLM subprocess + RunPod worker acting as an OpenAI proxy.
# --------------------------------------------------------------------------- #
def start_vllm_subprocess():
    if server_healthy():
        print("[entrypoint] vLLM already healthy — reusing existing server.", flush=True)
        return None
    cmd = build_vllm_command()
    print(f"[entrypoint] SERVERLESS mode — launching vLLM: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd)
    if not wait_until_healthy(STARTUP_TIMEOUT_S):
        proc.terminate()
        raise RuntimeError(f"vLLM did not become healthy within {STARTUP_TIMEOUT_S}s")
    print("[entrypoint] vLLM healthy — starting RunPod worker.", flush=True)
    return proc


def _internal_headers():
    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    return headers


def _resolve_request(job_input):
    """
    Accepts three input shapes and returns (route, body):

      1. RunPod OpenAI convention (recommended — works with RunPod's /openai/v1):
         { "openai_route": "/v1/chat/completions", "openai_input": { ...OpenAI body... } }
      2. Chat shorthand:      { "messages": [...], "max_tokens": ... }
      3. Completion shorthand:{ "prompt": "...", "max_tokens": ... }
    """
    route = job_input.get("openai_route", "/v1/chat/completions")
    if "openai_input" in job_input:
        body = dict(job_input["openai_input"])
    elif "messages" in job_input:
        route, body = "/v1/chat/completions", dict(job_input)
    elif "prompt" in job_input:
        route, body = "/v1/completions", dict(job_input)
    else:
        raise ValueError("input must contain one of: 'openai_input', 'messages', or 'prompt'")
    body.setdefault("model", MODEL_ALIAS)
    return route, body


def _wants_stream(body):
    """Whether THIS request asked to stream. OpenAI clients send `stream: true`
    to opt in; everything else (including LangChain's ChatOpenAI.invoke) is
    non-stream and must receive a single complete response."""
    return bool(body.get("stream", False))


def _forward_full(route, body):
    """POST to the local OpenAI server in non-stream mode and return the full
    OpenAI JSON response (a chat.completion / completion object), or an error dict."""
    body = dict(body)
    body["stream"] = False
    try:
        resp = requests.post(f"{LOCAL_BASE}{route}", json=body,
                             headers=_internal_headers(), timeout=REQUEST_TIMEOUT_S)
    except requests.RequestException as exc:
        return {"error": f"upstream request failed: {exc}"}
    if resp.status_code != 200:
        return {"error": "vllm_error", "status_code": resp.status_code, "body": resp.text}
    return resp.json()


def handler(job):
    """Non-streaming handler: always returns the full OpenAI JSON response.
    Registered when ENABLE_STREAMING is false."""
    try:
        route, body = _resolve_request(job.get("input", {}) or {})
    except ValueError as exc:
        return {"error": str(exc)}
    return _forward_full(route, body)


def streaming_handler(job):
    """Generator handler registered when ENABLE_STREAMING=true.

    It honors the PER-REQUEST `stream` flag instead of streaming every request —
    this is the fix for the `undefined ... generations[0][0].message` crash. A
    non-stream request (which is every request the analysis app makes) yields
    exactly ONE full chat.completion object, so an OpenAI-compatible non-stream
    client receives a valid `choices[0].message` — not a lone chat.completion.chunk
    delta. Only a client that explicitly sends `stream: true` gets SSE chunks.
    """
    try:
        route, body = _resolve_request(job.get("input", {}) or {})
    except ValueError as exc:
        yield {"error": str(exc)}
        return

    if not _wants_stream(body):
        # Non-stream request: return the full response as a single yield.
        yield _forward_full(route, body)
        return

    body = dict(body)
    body["stream"] = True
    try:
        with requests.post(f"{LOCAL_BASE}{route}", json=body, headers=_internal_headers(),
                           stream=True, timeout=REQUEST_TIMEOUT_S) as resp:
            if resp.status_code != 200:
                yield {"error": "vllm_error", "status_code": resp.status_code, "body": resp.text}
                return
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore")
                if line.startswith("data: "):
                    line = line[len("data: "):]
                if line.strip() == "[DONE]":
                    break
                yield line  # already-JSON OpenAI chunk, forwarded as-is
    except requests.RequestException as exc:
        yield {"error": f"upstream request failed: {exc}"}


def run_serverless():
    import runpod  # imported lazily so LOCAL mode needs neither runpod nor a GPU

    # Fail fast at startup if the GPU is missing (production workers only).
    try:
        @runpod.serverless.register_fitness_check
        def _gpu_present():
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("no CUDA GPU visible to the worker")
    except AttributeError:
        pass  # older runpod SDK without fitness-check support

    proc = start_vllm_subprocess()

    def _shutdown(*_):
        if proc is not None:
            proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Both registrations serve non-stream clients correctly:
    #   ENABLE_STREAMING=false → `handler` returns the full JSON object.
    #   ENABLE_STREAMING=true  → `streaming_handler` streams ONLY when the request
    #                            set `stream: true`, and otherwise yields one full
    #                            chat.completion. So a non-stream client (the app's
    #                            ChatOpenAI.invoke path) is never handed SSE chunks,
    #                            regardless of this flag.
    chosen = streaming_handler if ENABLE_STREAMING else handler
    config = {"handler": chosen}
    if ENABLE_STREAMING:
        config["return_aggregate_stream"] = True  # also expose streamed output via /run and /runsync
    runpod.serverless.start(config)


if __name__ == "__main__":
    if MODE == "serverless":
        run_serverless()
    else:
        run_local()