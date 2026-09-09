import gc
import io
import json
import os
import re
from typing import Any

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModelForImageTextRiText,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


MEDGEMMA_MODEL = "google/medgemma-1.5-4b-it"
LINGSHU_MODEL = "lingshu-medical-mllm/Lingshu-7B"
OCTOMED_MODEL = "OctoMed/OctoMed-7B"

MODEL_IDS = [
    MEDGEMMA_MODEL,
    LINGSHU_MODEL,
    OCTOMED_MODEL,
]

SEVERITY_ORDER = {
    "normal": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
}

CONFIDENCE_WEIGHT = {
    "low": 1.0,
    "medium": 1.5,
    "high": 2.0,
}


class MedicalModelManager:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.loaded_model_id: str | None = None

    def _release_model(self) -> None:
        self.model = None
        self.processor = None
        self.loaded_model_id = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _hf_token() -> str | None:
        return os.getenv("HF_TOKEN")

    @staticmethod
    def _prepare_images(images: list[bytes]) -> list[Image.Image]:
        prepared = []

        for image_bytes in images:
            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            prepared.append(image)

        return prepared

    @staticmethod
    def _analysis_prompt(user_prompt: str) -> str:
        return f"""
You are analyzing medical screening imagery.

This output is screening support only and is not a confirmed diagnosis.

Analyze only the visible information in the supplied image or images.

Return ONLY valid JSON using exactly this structure:

{{
  "findings": "concise medically relevant visible findings",
  "severity": "normal|low|moderate|high",
  "confidence": "low|medium|high",
  "flags": ["short relevant finding"],
  "evidence": ["specific visible observation"],
  "limitations": ["important limitation"]
}}

Rules:
- Do not invent medical history.
- Do not infer facts that are not visually supported.
- Mention normal visible findings when relevant.
- If no concerning abnormality is visible, use severity "normal".
- If image quality is poor, lower confidence.
- Evidence must describe visible observations.
- Keep findings concise.
- Do not include markdown.
- Do not include text outside the JSON object.

Screening context:
{user_prompt}
""".strip()

    def _load_medgemma(self) -> None:
        token = self._hf_token()

        self.processor = AutoProcessor.from_pretrained(
            MEDGEMMA_MODEL,
            token=token,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            MEDGEMMA_MODEL,
            token=token,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.model.eval()
        self.loaded_model_id = MEDGEMMA_MODEL

    def _load_qwen_medical_model(
        self,
        model_id: str,
    ) -> None:
        token = self._hf_token()

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            token=token,
        )

        self.model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                model_id,
                token=token,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        )

        self.model.eval()
        self.loaded_model_id = model_id

    def _load_model(
        self,
        model_id: str,
    ) -> None:
        if (
            self.model is not None
            and self.loaded_model_id == model_id
        ):
            return

        self._release_model()

        if model_id == MEDGEMMA_MODEL:
            self._load_medgemma()
            return

        if model_id in {
            LINGSHU_MODEL,
            OCTOMED_MODEL,
        }:
            self._load_qwen_medical_model(
                model_id
            )
            return

        raise ValueError(
            f"Unsupported model: {model_id}"
        )

    def _generate_medgemma(
        self,
        images: list[bytes],
        prompt: str,
        max_tokens: int,
    ) -> str:
        self._load_model(
            MEDGEMMA_MODEL
        )

        pil_images = self._prepare_images(
            images
        )

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
                "text": self._analysis_prompt(
                    prompt
                ),
            }
        )

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        inputs = (
            self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        )

        inputs = {
            key: (
                value.to(self.model.device)
                if hasattr(value, "to")
                else value
            )
            for key, value in inputs.items()
        }

        input_length = (
            inputs["input_ids"].shape[-1]
        )

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        output_ids = generated_ids[
            0,
            input_length:
        ]

        return self.processor.decode(
            output_ids,
            skip_special_tokens=True,
        )

    def _generate_qwen_model(
        self,
        model_id: str,
        images: list[bytes],
        prompt: str,
        max_tokens: int,
    ) -> str:
        self._load_model(
            model_id
        )

        pil_images = self._prepare_images(
            images
        )

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
                "text": self._analysis_prompt(
                    prompt
                ),
            }
        )

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        text = (
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        image_inputs, video_inputs = (
            process_vision_info(
                messages
            )
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(
            self.model.device
        )

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        trimmed_ids = [
            output_ids[
                len(input_ids):
            ]
            for input_ids, output_ids
            in zip(
                inputs.input_ids,
                generated_ids,
            )
        ]

        output_text = (
            self.processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )

        return output_text[0]

    def _generate_single(
        self,
        model_id: str,
        images: list[bytes],
        prompt: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        if model_id == MEDGEMMA_MODEL:
            text = self._generate_medgemma(
                images=images,
                prompt=prompt,
                max_tokens=max_tokens,
            )

        elif model_id in {
            LINGSHU_MODEL,
            OCTOMED_MODEL,
        }:
            text = self._generate_qwen_model(
                model_id=model_id,
                images=images,
                prompt=prompt,
                max_tokens=max_tokens,
            )

        else:
            raise ValueError(
                f"Unsupported model: {model_id}"
            )

        result = self._parse_json(
            text
        )

        result["model"] = model_id

        return result

    @staticmethod
    def _parse_json(
        text: str,
    ) -> dict[str, Any]:
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
                "Model did not return a JSON object."
            )

        data = json.loads(
            text[start:end + 1]
        )

        severity = str(
            data.get(
                "severity",
                "normal",
            )
        ).lower()

        if severity not in SEVERITY_ORDER:
            severity = "normal"

        confidence = str(
            data.get(
                "confidence",
                "low",
            )
        ).lower()

        if confidence not in CONFIDENCE_WEIGHT:
            confidence = "low"

        flags = data.get(
            "flags",
            [],
        )

        evidence = data.get(
            "evidence",
            [],
        )

        limitations = data.get(
            "limitations",
            [],
        )

        return {
            "findings": str(
                data.get(
                    "findings",
                    "",
                )
            ).strip(),

            "severity": severity,

            "confidence": confidence,

            "flags": [
                str(item).strip()
                for item in flags
                if str(item).strip()
            ],

            "evidence": [
                str(item).strip()
                for item in evidence
                if str(item).strip()
            ],

            "limitations": [
                str(item).strip()
                for item in limitations
                if str(item).strip()
            ],
        }

    def analyze_all(
        self,
        images: list[bytes],
        prompt: str,
        max_tokens: int = 512,
    ) -> list[dict[str, Any]]:
        results = []

        for model_id in MODEL_IDS:
            try:
                result = (
                    self._generate_single(
                        model_id=model_id,
                        images=images,
                        prompt=prompt,
                        max_tokens=max_tokens,
                    )
                )

                results.append(
                    result
                )

            except Exception as exc:
                results.append(
                    {
                        "model": model_id,
                        "error": str(exc),
                    }
                )

            finally:
                self._release_model()

        return results

    @staticmethod
    def build_consensus(
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        valid_results = [
            result
            for result in results
            if not result.get("error")
        ]

        if not valid_results:
            return {
                "error":
                    "All medical models failed."
            }

        severity_scores = {
            severity: 0.0
            for severity
            in SEVERITY_ORDER
        }

        for result in valid_results:
            severity = result[
                "severity"
            ]

            confidence = result[
                "confidence"
            ]

            severity_scores[
                severity
            ] += CONFIDENCE_WEIGHT[
                confidence
            ]

        winning_severity = max(
            severity_scores,
            key=lambda severity: (
                severity_scores[
                    severity
                ],
                SEVERITY_ORDER[
                    severity
                ],
            ),
        )

        matching_results = [
            result
            for result in valid_results
            if result["severity"]
            == winning_severity
        ]

        agreement_score = (
            len(matching_results)
            / len(valid_results)
        )

        representative = max(
            matching_results
            or valid_results,
            key=lambda result: (
                CONFIDENCE_WEIGHT[
                    result["confidence"]
                ],
                len(
                    result.get(
                        "evidence",
                        [],
                    )
                ),
            ),
        )

        flag_counts = {}

        for result in valid_results:
            for flag in result.get(
                "flags",
                [],
            ):
                key = flag.lower().strip()

                if not key:
                    continue

                if key not in flag_counts:
                    flag_counts[key] = {
                        "text": flag,
                        "count": 0,
                    }

                flag_counts[
                    key
                ]["count"] += 1

        supported_flags = [
            item["text"]
            for item in flag_counts.values()
            if (
                item["count"] >= 2
                or len(valid_results) == 1
            )
        ]

        evidence_counts = {}

        for result in valid_results:
            for evidence in result.get(
                "evidence",
                [],
            ):
                key = (
                    evidence
                    .lower()
                    .strip()
                )

                if not key:
                    continue

                if key not in evidence_counts:
                    evidence_counts[
                        key
                    ] = {
                        "text": evidence,
                        "count": 0,
                    }

                evidence_counts[
                    key
                ]["count"] += 1

        supported_evidence = [
            item["text"]
            for item
            in evidence_counts.values()
            if item["count"] >= 2
        ]

        if not supported_evidence:
            supported_evidence = (
                representative.get(
                    "evidence",
                    [],
                )
            )

        limitations = []

        for result in valid_results:
            for limitation in result.get(
                "limitations",
                [],
            ):
                if (
                    limitation
                    not in limitations
                ):
                    limitations.append(
                        limitation
                    )

        if len(valid_results) == 1:
            consensus_confidence = (
                valid_results[0][
                    "confidence"
                ]
            )

        elif agreement_score >= 0.67:
            consensus_confidence = "high"

        elif agreement_score >= 0.5:
            consensus_confidence = (
                "medium"
            )

        else:
            consensus_confidence = "low"

        return {
            "findings":
                representative[
                    "findings"
                ],

            "severity":
                winning_severity,

            "confidence":
                consensus_confidence,

            "flags":
                supported_flags,

            "evidence":
                supported_evidence,

            "limitations":
                limitations,

            "agreement_score":
                round(
                    agreement_score,
                    3,
                ),

            "successful_models":
                len(valid_results),

            "total_models":
                len(results),
        }


model_manager = MedicalModelManager()