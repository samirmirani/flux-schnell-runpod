[![Runpod](https://api.runpod.io/badge/samirmirani/flux-schnell-runpod)](https://console.runpod.io/hub/samirmirani/flux-schnell-runpod)
# Flux Schnell Runpod Worker

Serverless worker that exposes [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) behind the same JSON contract that Runpod uses for its Public Endpoint. It is ready to publish on the Runpod Hub (`.runpod/hub.json` + `.runpod/tests.json`) and ships with a Dockerfile that targets CUDA 12.1 GPUs.

> **Important:** FLUX.1-schnell is a gated Hugging Face model. Make sure the account that builds or runs this worker has been granted access and export a token as `HF_TOKEN`.

## Features

- Mirrors the Flux Schnell request schema (`prompt`, `width`, `height`, `num_inference_steps`, `guidance`, `seed`, `image_format`).
- Lazy-loads the `FluxPipeline` once per container and reuses it for subsequent jobs.
- Returns base64-encoded images plus metadata (seed, resolution, reported cost, guidance, steps).
- Configurable defaults via environment variables that are surfaced in `.runpod/hub.json` presets.

## Project layout

```
repo-root/
├── Dockerfile                 # CUDA 12.1 + PyTorch 2.4 runtime image
├── rp_handler.py              # Runpod handler entrypoint
├── src/flux_schnell_worker    # Generation logic + validation helpers
├── builder/requirements.txt   # Python dependencies (Torch comes from the base image)
├── test_input.json            # Handy for `python rp_handler.py --test_input ...`
├── .runpod/hub.json           # Hub metadata + environment schema
├── .runpod/tests.json         # Hub smoke test definition
└── docs/                      # Reference docs (Hub + Serverless guides)
    ├── hub/
    └── serverless/
```

All Hub-specific configuration lives inside `.runpod/`, while the worker code stays at the repo root (as required by the Runpod Hub contract). Documentation remains under `docs/`.

## Inputs and outputs

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `prompt` | string | ✅ | – | Text that describes the desired image. |
| `negative_prompt` | string | | – | Optional negative guidance. |
| `width` | integer | | `DEFAULT_WIDTH` (1024) | 256–1536, divisible by 64. |
| `height` | integer | | `DEFAULT_HEIGHT` (1024) | 256–1536, divisible by 64. |
| `num_inference_steps` | integer | | `DEFAULT_INFERENCE_STEPS` (4) | 1–8 for Schnell. |
| `guidance` | float | | `DEFAULT_GUIDANCE` (1.0) | 0.0–10.0 guidance scale. |
| `seed` | integer | | random | Use -1 or omit for random seeds. |
| `image_format` | string | | `DEFAULT_IMAGE_FORMAT` (`jpeg`) | `jpeg` or `png`. |

Typical response payload:

```json
{
  "output": {
    "image_base64": "...",
    "mime_type": "image/png",
    "seed": 170688142,
    "cost": 0.00251658,
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 4,
    "guidance": 1.0,
    "image_format": "png"
  },
  "metrics": {
    "execution_ms": 3920.41
  }
}
```

The `cost` field mirrors the price table from the Public Endpoint reference using `PRICE_PER_MEGAPIXEL` (defaults to USD 0.0024 for Flux Schnell).

## Local development

1. Accept the FLUX.1-schnell license on Hugging Face and create a [personal access token](https://huggingface.co/settings/tokens).
2. Export the token before running any commands:
   ```bash
   export HF_TOKEN="hf_xxx"
   ```
3. Create a virtual environment and install the dependencies:
   ```bash
   cd flux-schnell-runpod
   python -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip -r builder/requirements.txt
   ```
   (The base image supplies PyTorch, so you only need these packages for local CPU debugging.)
4. Run the handler against the canned payload:
   ```bash
   python rp_handler.py --test_input test_input.json
   ```
   Use `python rp_handler.py --test_input '{"input": {"prompt": "..."}}'` for ad-hoc prompts.
5. Developing on CPU hardware? Set `USE_MOCK_PIPELINE=1` (and optionally `DEVICE=cpu`) before running the handler to skip the multi-gigabyte model download. The worker will emit placeholder images but exercises the same validation/response flow. If you need to run on a GPU with tight VRAM headroom, set `ENABLE_CPU_OFFLOAD=1` so Diffusers offloads layers back to system memory during inference.

## Docker build

```bash
cd flux-schnell-runpod
export HF_TOKEN="hf_xxx"   # required at runtime for the first download
docker build --platform linux/amd64 -t your-docker-user/flux-schnell:latest .
docker run --gpus all -e HF_TOKEN -it your-docker-user/flux-schnell:latest
```

The first run pulls the model weights into `/workspace/.cache/huggingface`. Subsequent cold starts reuse the cache layer.

## Hub configuration

- `.runpod/hub.json` exposes the mandatory files (`rp_handler.py`, `Dockerfile`, `README.md`) and surfaces environment variables so Hub users can supply their own `HF_TOKEN`, tweak defaults, or pick from the included presets (`Fast 768px`, `Quality 1024px`).
- `.runpod/tests.json` defines a 512×512 smoke test that mirrors the Public Endpoint contract.
- Remember to set `HF_TOKEN` (and any other overrides) inside the Hub UI before triggering a build so the gated weights can be downloaded.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `HF_TOKEN` (required) | Authenticates downloads from the gated Hugging Face repo. |
| `MODEL_ID` | Alternate repo to load (defaults to `black-forest-labs/FLUX.1-schnell`). |
| `DEFAULT_WIDTH`, `DEFAULT_HEIGHT` | Fallback resolution when request omits dimensions. |
| `DEFAULT_INFERENCE_STEPS`, `DEFAULT_GUIDANCE`, `DEFAULT_IMAGE_FORMAT` | Additional fallbacks. |
| `PRICE_PER_MEGAPIXEL` | Controls the reported `cost` field. |
| `MAX_RESOLUTION`, `MIN_RESOLUTION`, `MAX_INFERENCE_STEPS`, `MIN_INFERENCE_STEPS` | Guard rails for validation. |
| `HUGGINGFACE_HUB_CACHE` | Where weights are stored inside the container (`/workspace/.cache/huggingface`). |
| `DEVICE`, `TORCH_DTYPE` | Override compute target (e.g., `cpu`, `float32`) when running locally. |
| `USE_MOCK_PIPELINE` | When set to `1`, skips HF downloads and returns placeholder images for fast local validation. |
| `ENABLE_CPU_OFFLOAD` | When `1`, calls `pipeline.enable_model_cpu_offload()` to shrink VRAM needs (handy on busy GPUs). |

## Publishing checklist

1. Update `README.md` with any project-specific notes.
2. Ensure `HF_TOKEN` is configured inside the Hub publish flow.
3. Tag a GitHub release – the Hub indexer watches releases, not commits.
4. Wait for the automated build + smoke test defined in `.runpod/tests.json` to pass.
