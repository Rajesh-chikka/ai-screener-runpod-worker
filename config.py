import os
from dataclasses import dataclass


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class Settings:
    api_key: str
    max_images: int
    max_image_bytes: int
    device: str


def load_settings() -> Settings:
    max_image_mb = _positive_int("RUNPOD_MAX_IMAGE_MB", 8)
    return Settings(
        api_key=os.getenv("RUNPOD_SERVER_API_KEY", "").strip(),
        max_images=_positive_int("RUNPOD_MAX_IMAGES", 8),
        max_image_bytes=max_image_mb * 1024 * 1024,
        device=os.getenv("RUNPOD_DEVICE", "cuda").strip() or "cuda",
    )
