# Serve GLiNER2.x using vLLM

Serve [GLiNER2.x](https://github.com/fastino-ai/GLiNER2) extraction and classification models through vLLM's HTTP `/pooling` endpoint. Send text and a task schema; get back structured entities or classification labels.

This repository adds the DeBERTa model, pooler, and request-processing plugins to **vLLM 0.31.0**, with Docker images for NVIDIA GPUs and x86-64 based CPUs.

## Quick start

Clone the repository and start **one** of the following:

```sh
git clone https://github.com/maxdswain/vllm-gliner.git
cd vllm-gliner
```

**NVIDIA GPU** (default: GLiNER2.5, `bfloat16`):

```sh
docker compose pull
docker compose up -d
docker compose logs -f gliner
```

**CPU** (default: GLiNER2.5, `float32`):

```sh
docker compose -f compose.cpu.yaml pull
docker compose -f compose.cpu.yaml up -d
docker compose -f compose.cpu.yaml logs -f gliner
```

Once the logs show the server is ready, try a request from [Examples](#examples). The first startup downloads the checkpoint from Hugging Face; the `huggingface-cache` Docker volume retains those downloads across restarts.

## Choose a model

Edit `MODEL_ID`, `MODEL_PLUGIN`, and `IO_PLUGIN` **together** in the Compose file you use:

| Model | `MODEL_ID` | `MODEL_PLUGIN` | `IO_PLUGIN` |
| --- | --- | --- | --- |
| GLiNER2.5 boundary (default) | `fastino/gliner2.5-multi-v1` | `deberta_gliner25` | `deberta_gliner25_io` |
| GLiNER2 span | `fastino/gliner2-large-v1` | `deberta_gliner2` | `deberta_gliner2_io` |
| [GLiNER2.5 Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) (English classification) | `fastino/GLiNER2.5-Decide` | `deberta_gliner2` | `deberta_gliner2_io` |

Then restart with `docker compose up -d` (GPU) or `docker compose -f compose.cpu.yaml up -d` (CPU). The `DTYPE` setting defaults to `bfloat16` on GPU and `float32` on CPU; your GPU must support the selected dtype. Both plugins accept the example's `data.text` and `data.schema` format with the `classic` runtime. The GLiNER2.5 boundary plugin also supports `constrained_classification` and `joint_ie` runtimes with schemas appropriate to those runtimes. Request flags `include_spans` and `include_confidence` control the returned details.

## Examples

The same `/pooling` endpoint works with either Compose file. When using Compose, set `model` to `/models/current` as shown below.

### Extract entities (GLiNER2.5 boundary)

```sh
curl -fsS http://localhost:8000/pooling \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/current","data":{"text":"Alice works at Acme in Paris.","schema":{"entities":{"person":"","organization":"","location":""}},"runtime":"classic","include_spans":true,"include_confidence":true}}'
```

The response's `data.entities` contains matches under `person`, `organization`, and `location` (for example, `Alice`, `Acme`, and `Paris`), each with text, character offsets, and confidence.

### Classify with GLiNER2.5 Decide

```sh
curl -fsS http://localhost:8000/pooling \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/current","data":{"text":"Please cancel my subscription.","schema":{"classifications":[{"task":"intent","labels":["cancel","refund","technical_support","other"]}]},"threshold":0}}'
```

The response contains `"data":{"intent":"cancel"}`. `threshold: 0` disables the optional single-label confidence cutoff; omit it to use the default threshold.

## Build an image locally

The Compose files use fixed GHCR image names. Tag a local build with the matching name to use it without editing Compose:

```sh
# GPU (requires an NVIDIA host to serve, but not to build):
docker build -t ghcr.io/maxdswain/vllm-gliner:latest .

# Or CPU (x86-64):
docker build --build-arg VLLM_BASE_IMAGE=vllm/vllm-openai-cpu:v0.31.0-x86_64 \
  -t ghcr.io/maxdswain/vllm-gliner:latest-cpu .
```

Run the corresponding `docker compose up -d` command from [Quick start](#quick-start). Omit the `pull` step so Compose uses your locally tagged image.

## Install without Docker

Use Python 3.11–3.13 with vLLM 0.31.0 and `pip install .`. Prepare a model and then start the server:

```sh
vllm-gliner-prep --model fastino/gliner2.5-multi-v1 \
  --plugin deberta_gliner25 --output "$PWD/gliner25-vllm"
vllm serve "$PWD/gliner25-vllm" --host 0.0.0.0 --port 8000 \
  --runner pooling --trust-remote-code --dtype float32 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --io-processor-plugin deberta_gliner25_io
```

For GLiNER2, use its row from the table above for `--model`, `--plugin`, and `--io-processor-plugin`. When installing directly, set the request's `model` field to the path passed to `vllm serve` (here the absolute path to `gliner25-vllm`).

## Development and contributing

Issues and pull requests are welcome. The `dev` dependency group supplies Poe, Ruff, and Pyrefly; `test` supplies pytest and httpx2. With vLLM installed locally:

```sh
uv sync --group dev --group test
uv run poe lint
uv run poe types
uv run poe test
```

On a CPU-only development machine, run these checks in the CPU image to avoid installing the CUDA build of vLLM:

```sh
docker build --build-arg VLLM_BASE_IMAGE=vllm/vllm-openai-cpu:v0.31.0-x86_64 \
  -t vllm-gliner:cpu .
docker run --rm -v "$PWD:/workspace" -w /workspace \
  --entrypoint /bin/sh vllm-gliner:cpu -ec \
  'uv pip install --python /opt/venv/bin/python --group dev --group test && poe lint && poe types && poe test'
```
