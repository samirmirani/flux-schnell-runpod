export HF_TOKEN="hf_xxx"  
  export CACHE_DIR="$PWD/.cache/huggingface"
export FP8_PATH="$HOME/Downloads/qwen_image_fp8_e4m3fn_scaled.safetensors"
  mkdir -p "$CACHE_DIR"
 
 docker run --rm --gpus all \
    -e HF_TOKEN="$HF_TOKEN" \
     -e TRANSFORMER_SINGLE_FILE_PATH=/runpod-data/qwen_image_fp8_e4m3fn_scaled.safetensors \
    -e DISABLE_SDP_FASTPATH=1 \
    -e ENABLE_CPU_OFFLOAD=0 \
    -e RUNPOD_TEST_INPUT="$(cat test_input.json)" \
    -v "$CACHE_DIR":/workspace/.cache/huggingface \
    -v "$FP8_PATH":/runpod-data/qwen_image_fp8_e4m3fn_scaled.safetensors:ro \
    qwen-image-worker:local \
    python -u rp_handler.py
