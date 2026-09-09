class UnsupportedModelError(ValueError):
    pass


class ModelNotLoadedError(RuntimeError):
    pass


class ModelManager:
    def __init__(self, device: str) -> None:
        self.device = device
        self._loaded_models: set[str] = set()

    @property
    def loaded_models(self) -> list[str]:
        return sorted(self._loaded_models)

    async def generate(
        self,
        model_id: str,
        images: list[bytes],
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if not self._is_supported_model_id(model_id):
            raise UnsupportedModelError(f"Unsupported model: {model_id}")
        raise ModelNotLoadedError(f"Model is not loaded: {model_id}")

    @staticmethod
    def _is_supported_model_id(model_id: str) -> bool:
        normalized = model_id.strip()
        return (
            normalized == "medgemma"
            or normalized.startswith("hf:") and len(normalized) > 3
            or normalized.startswith("ollama:") and len(normalized) > 7
        )
