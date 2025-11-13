[![Runpod](https://api.runpod.io/badge/samirmirani/flux-schnell-runpod)](https://console.runpod.io/hub/samirmirani/flux-schnell-runpod)
# Qwen Image Runpod Worker

Serverless worker that exposes [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) behind the same JSON contract that Runpod uses for its public image endpoints. It is ready to publish on the Runpod Hub (`.runpod/hub.json` + `.runpod/tests.json`) and ships with a CUDA 12.1 Docker image that installs PyTorch 2.5, Diffusers, and the rest of the Qwen Image stack.

> **Heads-up:** Qwen-Image is open, but the initial download is large (~20 GB). Export `HF_TOKEN` to avoid anonymous rate limits on the Hugging Face Hub, or point `MODEL_ID` to a local checkout.

## Features

- Mirrors the existing Runpod image schema (`prompt`, `width`, `height`, `num_inference_steps`, `guidance`, `seed`, `image_format`) so existing clients keep working.
- Lazy-loads the `QwenImagePipeline` once per container and reuses it for subsequent jobs.
- Returns base64-encoded images plus metadata (seed, resolution, reported cost, guidance, steps) and exposes every knob via environment variables surfaced in `.runpod/hub.json`.

## Project layout

```
repo-root/
├── Dockerfile                 # CUDA 12.1 + PyTorch 2.5 runtime image
├── rp_handler.py              # Runpod handler entrypoint
├── src/qwen_image_worker      # Generation logic + validation helpers
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
| `width` | integer | | `DEFAULT_WIDTH` (1024) | 512–1664, divisible by `RESOLUTION_STEP` (default 16). |
| `height` | integer | | `DEFAULT_HEIGHT` (1024) | 512–1664, divisible by `RESOLUTION_STEP` (default 16). |
| `num_inference_steps` | integer | | `DEFAULT_INFERENCE_STEPS` (30) | 1–50 recommended for Qwen Image. |
| `guidance` | float | | `DEFAULT_GUIDANCE` (4.0) | Passed to Qwen’s `true_cfg_scale`. |
| `seed` | integer | | random | Use -1 or omit for random seeds. |
| `image_format` | string | | `DEFAULT_IMAGE_FORMAT` (`jpeg`) | `jpeg` or `png`. |

Typical response payload:

```json
{
  "output": {
    "image_base64": "...",
    "mime_type": "image/png",
    "seed": 170688142,
    "cost": 0.00418879,
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 20,
    "guidance": 4.0,
    "image_format": "png"
  },
  "metrics": {
    "execution_ms": 3920.41
  }
}
```

The `cost` field mirrors Runpod’s Public Endpoint calculator using `PRICE_PER_MEGAPIXEL` (defaults to USD 0.004 for Qwen Image).

## Local development

### Standard Qwen-Image weights

1. (Optional) Create a [Hugging Face access token](https://huggingface.co/settings/tokens) and export it to skip rate limits:
   ```bash
   export HF_TOKEN="hf_xxx"
   ```
3. Create a virtual environment and install the dependencies:
   ```bash
   cd flux-schnell-runpod
   python -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip -r builder/requirements.txt
   ```
   (The base image supplies PyTorch, so you only need these packages for local CPU testing.)
4. Run the handler against the canned payload:
   ```bash
   python rp_handler.py --test_input test_input.json
   ```
   Use `python rp_handler.py --test_input '{"input": {"prompt": "..."}}'` for ad-hoc prompts.
5. Developing on CPU hardware? Set `USE_MOCK_PIPELINE=1` (and optionally `DEVICE=cpu`) before running the handler to skip the multi-gigabyte model download. The worker will emit placeholder images but exercises the same validation/response flow. If you need to run on a GPU with tight VRAM headroom, set `ENABLE_CPU_OFFLOAD=1` so Diffusers offloads layers back to system memory during inference.

### FP8 single-file local testing

If you have a single-file transformer checkpoint such as `qwen_image_fp8_e4m3fn_scaled.safetensors`, you can point the worker at it to avoid downloading the full 20 GB bf16 transformer:

```bash
export TRANSFORMER_SINGLE_FILE_PATH=$HOME/Downloads/qwen_image_fp8_e4m3fn_scaled.safetensors
python rp_handler.py --test_input test_input.json
```

The rest of the pipeline (VAE, text encoder, scheduler) continues to load from `Qwen/Qwen-Image`, but the transformer weights now come from that FP8 file. Combine this with smaller `DEFAULT_WIDTH/HEIGHT` values if you’re testing on tight VRAM.

## Docker build

```bash
cd flux-schnell-runpod
export HF_TOKEN="hf_xxx"   # required at runtime for the first download
docker build --platform linux/amd64 -t your-docker-user/qwen-image-worker:latest .
docker run --gpus all -e HF_TOKEN -it your-docker-user/qwen-image-worker:latest
```

The first run pulls the model weights into `/workspace/.cache/huggingface`. Subsequent cold starts reuse the cache layer.

## Hub configuration

- `.runpod/hub.json` exposes the mandatory files (`rp_handler.py`, `Dockerfile`, `README.md`) and surfaces environment variables so Hub users can supply their own `HF_TOKEN`, tweak defaults, or pick from the included presets (for example, `Quality 1K`).
- `.runpod/tests.json` defines a 1024×1024 smoke test that mirrors the contract used in the README.
- Remember to set `HF_TOKEN` (and any other overrides) inside the Hub UI before triggering a build so the initial download succeeds.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `HF_TOKEN` | (Recommended) Authenticates downloads from Hugging Face. Required if the repo is gated. |
| `MODEL_ID` | Alternate repo to load (defaults to `Qwen/Qwen-Image`). Paths work too. |
| `DEFAULT_WIDTH`, `DEFAULT_HEIGHT` | Fallback resolution when request omits dimensions. |
| `DEFAULT_INFERENCE_STEPS`, `DEFAULT_GUIDANCE`, `DEFAULT_IMAGE_FORMAT` | Additional fallbacks (guidance feeds `true_cfg_scale`). |
| `PRICE_PER_MEGAPIXEL` | Controls the reported `cost` field. |
| `MAX_RESOLUTION`, `MIN_RESOLUTION`, `RESOLUTION_STEP`, `MAX_INFERENCE_STEPS`, `MIN_INFERENCE_STEPS` | Guard rails for validation. |
| `HUGGINGFACE_HUB_CACHE` | Where weights are stored inside the container (`/workspace/.cache/huggingface`). |
| `DEVICE`, `TORCH_DTYPE` | Override compute target (e.g., `cpu`, `float32`) when running locally. |
| `USE_MOCK_PIPELINE` | When set to `1`, skips HF downloads and returns placeholder images for fast local validation. |
| `ENABLE_CPU_OFFLOAD` | When `1`, calls `pipeline.enable_model_cpu_offload()` to shrink VRAM needs. |
| `TRANSFORMER_SINGLE_FILE_PATH` | Optional path to a standalone transformer checkpoint (e.g., `qwen_image_fp8_e4m3fn_scaled.safetensors`) for local FP8/quantized testing. |

## Publishing checklist

1. Update `README.md` with any project-specific notes.
2. Ensure `HF_TOKEN` is configured inside the Hub publish flow.
3. Tag a GitHub release – the Hub indexer watches releases, not commits.
4. Wait for the automated build + smoke test defined in `.runpod/tests.json` to pass.
