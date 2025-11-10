"""Hub-specific entrypoint that reuses the repository root handler."""

from handler import handler as handler  # re-export for Runpod detection

if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
