import base64
import binascii
from typing import Any

import runpod

from model_manager import model_manager


MAX_IMAGES = 8
DEFAULT_MAX_TOKENS = 512


def decode_image(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError(
            "Image must be a non-empty base64 string."
        )

    if value.startswith("data:"):
        if "," not in value:
            raise ValueError(
                "Invalid image data URL."
            )

        _, value = value.split(",", 1)

    try:
        return base64.b64decode(
            value,
            validate=True,
        )

    except (binascii.Error, ValueError) as exc:
        raise ValueError(
            "Invalid base64 image."
        ) from exc


def validate_input(
    job_input: dict[str, Any],
) -> tuple[str, list[str], int]:
    prompt = str(
        job_input.get("prompt", "")
    ).strip()

    raw_images = (
        job_input.get("images")
        or []
    )

    max_tokens = int(
        job_input.get(
            "max_tokens",
            DEFAULT_MAX_TOKENS,
        )
    )

    if not prompt:
        raise ValueError(
            "prompt is required"
        )

    if not isinstance(raw_images, list):
        raise ValueError(
            "images must be a list"
        )

    if not raw_images:
        raise ValueError(
            "at least one image is required"
        )

    if len(raw_images) > MAX_IMAGES:
        raise ValueError(
            f"maximum {MAX_IMAGES} images per request"
        )

    if max_tokens < 1 or max_tokens > 2048:
        raise ValueError(
            "max_tokens must be between 1 and 2048"
        )

    return (
        prompt,
        raw_images,
        max_tokens,
    )


def handler(job):
    try:
        job_input = (
            job.get("input")
            or {}
        )

        (
            prompt,
            raw_images,
            max_tokens,
        ) = validate_input(job_input)

        images = [
            decode_image(value)
            for value in raw_images
        ]

        model_results = (
            model_manager.analyze_all(
                images=images,
                prompt=prompt,
                max_tokens=max_tokens,
            )
        )

        consensus = (
            model_manager.build_consensus(
                model_results
            )
        )

        if consensus.get("error"):
            return {
                "error": consensus["error"],
                "models": model_results,
                "image_count": len(images),
            }

        return {
            "result": {
                "findings": consensus["findings"],
                "severity": consensus["severity"],
                "confidence": consensus["confidence"],
                "flags": consensus["flags"],
                "evidence": consensus["evidence"],
                "limitations": consensus["limitations"],
                "supporting_labels": consensus[
                    "supporting_labels"
                ],
            },
            "models": model_results,
            "image_count": len(images),
        }

    except Exception as exc:
        return {
            "error": str(exc)
        }


runpod.serverless.start(
    {
        "handler": handler
    }
)