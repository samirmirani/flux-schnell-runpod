from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import runpod

ROOT = Path(__file__).parent.resolve()
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qwen_image_worker import QwenImageGenerator, QwenInputError

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

GENERATOR = QwenImageGenerator()


def _extract_input(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job.get("input")
    if job_input is None:
        return {}
    if not isinstance(job_input, dict):
        raise QwenInputError("`input` must be a JSON object.")
    return job_input


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    start_time = time.perf_counter()
    try:
        job_input = _extract_input(job)
        result = GENERATOR.generate(job_input)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "output": result.to_dict(),
            "metrics": {
                "execution_ms": duration_ms,
            },
        }
    except QwenInputError as error:
        LOGGER.warning("User input error: %s", error)
        return {"error": str(error)}
    except Exception:  # pragma: no cover
        LOGGER.exception("Unhandled error while running inference")
        raise


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
