import gc
import io
import json
import os
import re
from typing import Any

import open_clip
import torch
from PIL import Image
from transformers import (
    AutoModelForMultimodalLM,
    AutoModelForZeroShotImageClassification,
    AutoProcessor,
)


MEDGEMMA_MODEL = "google/medgemma-1.5-4b-it"
MEDSIGLIP_MODEL = "google/medsiglip-448"
BIOMEDCLIP_MODEL = (
    "hf-hub:"
    "microsoft/"
    "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

SEVERITY_ORDER = {
    "normal": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
}


class MedicalModelManager:
    def __init__(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    @staticmethod
    def _hf_token():
        return os.getenv("HF_TOKEN")

    @staticmethod
    def _images(images: list[bytes]):
        return [
            Image.open(
                io.BytesIO(image)
            ).convert("RGB")
            for image in images
        ]

    @staticmethod
    def _analysis_prompt(prompt: str) -> str:
        return f"""
Analyze this medical screening image.

This is screening support only, not a confirmed diagnosis.

Return ONLY valid JSON:

{{
  "findings": "concise visible findings",
  "severity": "normal|low|moderate|high",
  "confidence": "low|medium|high",
  "flags": ["short visible concern"],
  "evidence": ["visible observation"],
  "limitations": ["important limitation"]
}}

Do not invent medical history.
Only use visible evidence.

Context:
{prompt}
""".strip()

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(
                "MedGemma did not return JSON."
            )

        data = json.loads(
            text[start:end + 1]
        )

        severity = str(
            data.get("severity", "normal")
        ).lower()

        if severity not in SEVERITY_ORDER:
            severity = "normal"

        confidence = str(
            data.get("confidence", "low")
        ).lower()

        if confidence not in {
            "low",
            "medium",
            "high",
        }:
            confidence = "low"

        return {
            "findings": str(
                data.get("findings", "")
            ).strip(),
            "severity": severity,
            "confidence": confidence,
            "flags": data.get("flags", []),
            "evidence": data.get(
                "evidence",
                [],
            ),
            "limitations": data.get(
                "limitations",
                [],
            ),
        }

    def run_medgemma(
        self,
        images: list[bytes],
        prompt: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        token = self._hf_token()

        processor = AutoProcessor.from_pretrained(
            MEDGEMMA_MODEL,
            token=token,
        )

        model = AutoModelForMultimodalLM.from_pretrained(
            MEDGEMMA_MODEL,
            token=token,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        pil_images = self._images(images)

        content = [
            {
                "type": "image",
                "image": image,
            }
            for image in pil_images
        ]

        content.append({
            "type": "text",
            "text": self._analysis_prompt(
                prompt
            ),
        })

        messages = [{
            "role": "user",
            "content": content,
        }]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            key: (
                value.to(model.device)
                if hasattr(value, "to")
                else value
            )
            for key, value in inputs.items()
        }

        input_length = (
            inputs["input_ids"].shape[-1]
        )

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        generated = output[
            0,
            input_length:
        ]

        text = processor.decode(
            generated,
            skip_special_tokens=True,
        )

        result = self._parse_json(text)

        del model
        del processor

        gc.collect()
        torch.cuda.empty_cache()

        return result

    def run_medsiglip(
        self,
        images: list[bytes],
    ) -> dict[str, Any]:
        token = self._hf_token()

        processor = AutoProcessor.from_pretrained(
            MEDSIGLIP_MODEL,
            token=token,
        )

        model = (
            AutoModelForZeroShotImageClassification
            .from_pretrained(
                MEDSIGLIP_MODEL,
                token=token,
            )
            .to(self.device)
        )

        labels = [
            "normal medical appearance",
            "visible inflammation",
            "visible swelling",
            "visible redness",
            "visible lesion",
            "visible abnormal tissue",
            "poor image quality",
        ]

        image = self._images(images)[0]

        inputs = processor(
            images=image,
            text=labels,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)

        probabilities = (
            outputs.logits_per_image
            .softmax(dim=-1)[0]
        )

        ranked = sorted(
            zip(
                labels,
                probabilities.tolist(),
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        result = {
            "top_labels": [
                {
                    "label": label,
                    "score": round(
                        score,
                        4,
                    ),
                }
                for label, score
                in ranked[:3]
            ]
        }

        del model
        del processor

        gc.collect()
        torch.cuda.empty_cache()

        return result

    def run_biomedclip(
        self,
        images: list[bytes],
    ) -> dict[str, Any]:
        model, _, preprocess = (
            open_clip.create_model_and_transforms(
                BIOMEDCLIP_MODEL
            )
        )

        tokenizer = open_clip.get_tokenizer(
            BIOMEDCLIP_MODEL
        )

        model = model.to(self.device)
        model.eval()

        labels = [
            "normal medical image",
            "inflammation",
            "swelling",
            "redness",
            "lesion",
            "abnormal tissue",
            "poor quality image",
        ]

        image = preprocess(
            self._images(images)[0]
        ).unsqueeze(0).to(self.device)

        text = tokenizer(
            labels
        ).to(self.device)

        with torch.inference_mode():
            image_features = (
                model.encode_image(image)
            )

            text_features = (
                model.encode_text(text)
            )

            image_features /= (
                image_features.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

            text_features /= (
                text_features.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

            scores = (
                100.0
                * image_features
                @ text_features.T
            ).softmax(dim=-1)[0]

        ranked = sorted(
            zip(
                labels,
                scores.tolist(),
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        result = {
            "top_labels": [
                {
                    "label": label,
                    "score": round(
                        score,
                        4,
                    ),
                }
                for label, score
                in ranked[:3]
            ]
        }

        del model

        gc.collect()
        torch.cuda.empty_cache()

        return result

    def analyze_all(
        self,
        images: list[bytes],
        prompt: str,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        results = {}

        try:
            results["medgemma"] = (
                self.run_medgemma(
                    images,
                    prompt,
                    max_tokens,
                )
            )
        except Exception as exc:
            results["medgemma"] = {
                "error": str(exc)
            }

        try:
            results["medsiglip"] = (
                self.run_medsiglip(
                    images
                )
            )
        except Exception as exc:
            results["medsiglip"] = {
                "error": str(exc)
            }

        try:
            results["biomedclip"] = (
                self.run_biomedclip(
                    images
                )
            )
        except Exception as exc:
            results["biomedclip"] = {
                "error": str(exc)
            }

        return results

    @staticmethod
    def build_consensus(
        results: dict[str, Any],
    ) -> dict[str, Any]:
        medgemma = results.get(
            "medgemma",
            {},
        )

        if medgemma.get("error"):
            return {
                "error":
                    "Primary MedGemma analysis failed.",
                "models": results,
            }

        supporting_labels = []

        for name in [
            "medsiglip",
            "biomedclip",
        ]:
            model_result = results.get(
                name,
                {},
            )

            for item in model_result.get(
                "top_labels",
                [],
            ):
                if item["score"] >= 0.25:
                    supporting_labels.append(
                        item["label"]
                    )

        supporting_labels = list(
            dict.fromkeys(
                supporting_labels
            )
        )

        confidence = medgemma[
            "confidence"
        ]

        if (
            medgemma["severity"]
            != "normal"
            and not supporting_labels
        ):
            confidence = "low"

        return {
            "findings":
                medgemma["findings"],
            "severity":
                medgemma["severity"],
            "confidence":
                confidence,
            "flags":
                medgemma["flags"],
            "evidence":
                medgemma["evidence"],
            "limitations":
                medgemma["limitations"],
            "supporting_labels":
                supporting_labels,
            "model_results":
                results,
        }


model_manager = MedicalModelManager()