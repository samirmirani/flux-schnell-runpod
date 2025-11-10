"""Hub entrypoint ensuring Runpod detects the handler."""

from rp_handler import handler

if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
