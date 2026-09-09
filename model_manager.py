import io
import os

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


MODEL_ID = os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it")


class MedGemmaModel:
    def __init__(self):
        self.processor = None
        self.model = None

    def load(self):
        if self.model is not None:
            return

        token = os.getenv("HF_TOKEN")

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            token=token,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token,
        )

        self.model.eval()

    def generate(
        self,
        images: list[bytes],
        prompt: str,
        max_tokens: int = 512,
    ) -> str:
        self.load()

        pil_images = [
            Image.open(io.BytesIO(data)).convert("RGB")
            for data in images
        ]

        content = []

        for image in pil_images:
            content.append(
                {
                    "type": "image",
                    "image": image,
                }
            )

        content.append(
            {
                "type": "text",
                "text": prompt,
            }
        )

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.model.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        input_length = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        generated = output[0][input_length:]

        return self.processor.decode(
            generated,
            skip_special_tokens=True,
        )


model = MedGemmaModel()