import os
from pathlib import Path

import runpod


def handler(job):
    cache_root = Path(
        "/runpod-volume/huggingface-cache/hub"
    )

    medgemma_dir = (
        cache_root
        / "models--google--medgemma-1.5-4b-it"
    )

    snapshots_dir = (
        medgemma_dir
        / "snapshots"
    )

    snapshot_names = []

    if snapshots_dir.exists():
        snapshot_names = [
            item.name
            for item in snapshots_dir.iterdir()
            if item.is_dir()
        ]

    return {
        "diagnostic": True,

        "cache_root": str(
            cache_root
        ),

        "cache_root_exists":
            cache_root.exists(),

        "medgemma_dir": str(
            medgemma_dir
        ),

        "medgemma_dir_exists":
            medgemma_dir.exists(),

        "snapshots_dir_exists":
            snapshots_dir.exists(),

        "snapshots":
            snapshot_names,

        "environment": {
            "RUNPOD_MODEL_PATH":
                os.getenv(
                    "RUNPOD_MODEL_PATH"
                ),

            "RUNPOD_MODEL_NAME":
                os.getenv(
                    "RUNPOD_MODEL_NAME"
                ),

            "MODEL_PATH":
                os.getenv(
                    "MODEL_PATH"
                ),

            "HF_HOME":
                os.getenv(
                    "HF_HOME"
                ),

            "HF_HUB_CACHE":
                os.getenv(
                    "HF_HUB_CACHE"
                ),

            "TRANSFORMERS_CACHE":
                os.getenv(
                    "TRANSFORMERS_CACHE"
                ),
        },
    }


runpod.serverless.start(
    {
        "handler": handler
    }
)