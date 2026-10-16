# Serve GLiNER2.x using vLLM

Serve [GLiNER2.x](https://github.com/fastino-ai/GLiNER2) extraction and classification models through vLLM's HTTP `/pooling` endpoint. Send text and a task schema; get back structured entities or classification labels.

This repository adds the DeBERTa model, pooler, and request-processing plugins to **vLLM 0.31.0**, with Docker images for NVIDIA GPUs and x86-64 based CPUs.

## Quick start


**NVIDIA GPU** (`bfloat16`):

```sh
docker pull ghcr.io/maxdswain/vllm-gliner:latest
docker run --gpus all -p 8000:8000 \
  --entrypoint /bin/sh ghcr.io/maxdswain/vllm-gliner:latest -ec '
    vllm-gliner-prep --model fastino/gliner2.5-multi-v1 \
      --plugin deberta_gliner25 --output /models/current --force &&
    exec vllm serve /models/current --host 0.0.0.0 --port 8000 \
      --runner pooling --trust-remote-code --dtype bfloat16 \
      --no-enable-prefix-caching --no-enable-chunked-prefill \
      --io-processor-plugin deberta_gliner25_io'
```

**CPU** (`float32`; x86-64):

```sh
docker pull ghcr.io/maxdswain/vllm-gliner:latest-cpu
docker run -p 8000:8000 \
  --entrypoint /bin/sh ghcr.io/maxdswain/vllm-gliner:latest-cpu -ec '
    vllm-gliner-prep --model fastino/gliner2.5-multi-v1 \
      --plugin deberta_gliner25 --output /models/current --force &&
    exec vllm serve /models/current --host 0.0.0.0 --port 8000 \
      --runner pooling --trust-remote-code --dtype float32 \
      --no-enable-prefix-caching --no-enable-chunked-prefill \
      --io-processor-plugin deberta_gliner25_io'
```

For a Compose deployment instead, clone this repository and use [compose.yaml](compose.yaml) (`docker compose up -d`) or [compose.cpu.yaml](compose.cpu.yaml) (`docker compose -f compose.cpu.yaml up -d`).

## Choose a model

Select the model ID, preparation plugin, and IO processor **together**. For `docker run`, change `--model`, `--plugin`, and `--io-processor-plugin` in the command above. For Compose, edit `MODEL_ID`, `MODEL_PLUGIN`, and `IO_PLUGIN` in the corresponding file:

| Model | `MODEL_ID` | `MODEL_PLUGIN` | `IO_PLUGIN` |
| --- | --- | --- | --- |
| GLiNER2.5 boundary (default) | `fastino/gliner2.5-multi-v1` | `deberta_gliner25` | `deberta_gliner25_io` |
| GLiNER2 span | `fastino/gliner2-large-v1` | `deberta_gliner2` | `deberta_gliner2_io` |
| [GLiNER2.5 Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) (English classification) | `fastino/GLiNER2.5-Decide` | `deberta_gliner2` | `deberta_gliner2_io` |

Restart the container after changing models (or run `docker compose up -d` for GPU / `docker compose -f compose.cpu.yaml up -d` for CPU). The examples use `bfloat16` on GPU and `float32` on CPU; your GPU must support the selected dtype. Both plugins accept the example's `data.text` and `data.schema` format with the `classic` runtime. The GLiNER2.5 boundary plugin also supports `constrained_classification` and `joint_ie` runtimes with schemas appropriate to those runtimes. Request flags `include_spans` and `include_confidence` control the returned details.

## Examples

The same `/pooling` endpoint works with either Docker command or Compose file. Set `model` to `/models/current` as shown below.

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

Clone the repository and tag a local build with the name used above (and in the Compose files):

```sh
git clone https://github.com/maxdswain/vllm-gliner.git
cd vllm-gliner

# GPU (requires an NVIDIA host to serve, but not to build):
docker build -t ghcr.io/maxdswain/vllm-gliner:latest .

# Or CPU (x86-64):
docker build --build-arg VLLM_BASE_IMAGE=vllm/vllm-openai-cpu:v0.31.0-x86_64 \
  -t ghcr.io/maxdswain/vllm-gliner:latest-cpu .
```

Run the corresponding `docker run` command from [Quick start](#quick-start), omitting `docker pull` so it uses your locally tagged image. The same tags also work with the linked Compose files.

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
