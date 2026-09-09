import base64

import runpod

from model_manager import model


def decode_image(image_value: str) -> bytes:
    if image_value.startswith("data:"):
        _, encoded = image_value.split(",", 1)
    else:
        encoded = image_value

    return base64.b64decode(encoded)


def handler(job):
    job_input = job.get("input", {})

    prompt = job_input.get("prompt", "")
    raw_images = job_input.get("images", [])
    max_tokens = int(job_input.get("max_tokens", 512))

    if not prompt:
        return {
            "error": "prompt is required"
        }

    if not raw_images:
        return {
            "error": "at least one image is required"
        }

    images = [
        decode_image(image)
        for image in raw_images
    ]

    result = model.generate(
        images=images,
        prompt=prompt,
        max_tokens=max_tokens,
    )

    return {
        "result": result,
        "model": "google/medgemma-4b-it",
    }


runpod.serverless.start({
    "handler": handler
})