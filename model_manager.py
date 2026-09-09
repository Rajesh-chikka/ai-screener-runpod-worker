import gc
import io
import json
import os
import re
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


MODEL_IDS = [
    "google/medgemma-1.5-4b-it",
    "lingshu-medical-mllm/Lingshu-7B",
    "OctoMed/OctoMed-7B",
]


SEVERITY_ORDER = {
    "normal": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
}


class MedicalModelManager:
    def __init__(self):
        self.model = None
        self.processor = None
        self.loaded_model_id = None

    def _release_model(self):
        self.model = None
        self.processor = None
        self.loaded_model_id = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self, model_id: str):
        if self.loaded_model_id == model_id and self.model is not None:
            return

        self._release_model()

        token = os.getenv("HF_TOKEN")

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            token=token,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.eval()
        self.loaded_model_id = model_id

    @staticmethod
    def _prepare_images(images: list[bytes]) -> list[Image.Image]:
        return [
            Image.open(io.BytesIO(image)).convert("RGB")
            for image in images
        ]

    @staticmethod
    def _analysis_prompt(user_prompt: str) -> str:
        return f"""
You are reviewing medical screening imagery.

This is a screening-support task, not a confirmed diagnosis.

Analyze only what is visible in the provided image or images.

Return ONLY valid JSON with this structure:

{{
  "findings": "short clinically relevant description",
  "severity": "normal|low|moderate|high",
  "confidence": "low|medium|high",
  "flags": ["short flag"],
  "evidence": ["specific visible observation"],
  "limitations": ["important limitation"]
}}

Rules:
- Do not invent patient history.
- Do not claim certainty where the image is ambiguous.
- Describe abnormal and normal visible findings.
- Use "normal" severity when no concerning visual abnormality is seen.
- Keep findings concise.
- Evidence must describe visible observations.
- If image quality is inadequate, lower confidence and explain why.
- Do not include markdown.
- Do not include text outside the JSON.

Screening context:
{user_prompt}
""".strip()

    def _generate_single(
        self,
        model_id: str,
        images: list[bytes],
        prompt: str,
        max_tokens: int,
    ) -> dict[str, Any]:

        self._load_model(model_id)

        pil_images = self._prepare_images(images)

        content = []

        for image in pil_images:
            content.append({
                "type": "image",
                "image": image,
            })

        content.append({
            "type": "text",
            "text": self._analysis_prompt(prompt),
        })

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

        text = self.processor.decode(
            generated,
            skip_special_tokens=True,
        )

        parsed = self._parse_json(text)

        parsed["model"] = model_id

        return parsed

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
                "Model did not return a JSON object."
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

            "flags": [
                str(value).strip()
                for value in data.get("flags", [])
                if str(value).strip()
            ],

            "evidence": [
                str(value).strip()
                for value in data.get("evidence", [])
                if str(value).strip()
            ],

            "limitations": [
                str(value).strip()
                for value in data.get("limitations", [])
                if str(value).strip()
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
                result = self._generate_single(
                    model_id=model_id,
                    images=images,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )

                results.append(result)

            except Exception as exc:
                results.append({
                    "model": model_id,
                    "error": str(exc),
                })

            finally:
                self._release_model()

        return results

    @staticmethod
    def build_consensus(
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:

        valid = [
            result
            for result in results
            if not result.get("error")
        ]

        if not valid:
            return {
                "error": "All medical models failed."
            }

        confidence_weight = {
            "low": 1.0,
            "medium": 1.5,
            "high": 2.0,
        }

        severity_scores = {
            severity: 0.0
            for severity in SEVERITY_ORDER
        }

        for result in valid:
            severity = result["severity"]

            severity_scores[severity] += (
                confidence_weight[
                    result["confidence"]
                ]
            )

        winning_severity = max(
            severity_scores,
            key=lambda severity: (
                severity_scores[severity],
                SEVERITY_ORDER[severity],
            ),
        )

        agreement_count = sum(
            1
            for result in valid
            if result["severity"] == winning_severity
        )

        agreement_score = (
            agreement_count / len(valid)
        )

        matching = [
            result
            for result in valid
            if result["severity"] == winning_severity
        ]

        representative = max(
            matching or valid,
            key=lambda result: (
                confidence_weight[
                    result["confidence"]
                ],
                len(result["evidence"]),
            ),
        )

        all_flags = []

        for result in valid:
            for flag in result["flags"]:
                if flag not in all_flags:
                    all_flags.append(flag)

        evidence_support = {}

        for result in valid:
            for evidence in result["evidence"]:
                key = evidence.lower().strip()

                evidence_support.setdefault(
                    key,
                    {
                        "text": evidence,
                        "count": 0,
                    },
                )

                evidence_support[key]["count"] += 1

        supported_evidence = [
            item["text"]
            for item in evidence_support.values()
            if item["count"] >= 2
        ]

        if not supported_evidence:
            supported_evidence = (
                representative["evidence"]
            )

        limitations = []

        for result in valid:
            for limitation in result["limitations"]:
                if limitation not in limitations:
                    limitations.append(limination)

        if agreement_score >= 0.67:
            consensus_confidence = "high"

        elif agreement_score >= 0.5:
            consensus_confidence = "medium"

        else:
            consensus_confidence = "low"

        findings = representative["findings"]

        if supported_evidence:
            findings = (
                findings
                + " Supporting observations: "
                + "; ".join(supported_evidence)
            )

        return {
            "findings": findings.strip(),
            "severity": winning_severity,
            "confidence": consensus_confidence,
            "flags": all_flags,
            "evidence": supported_evidence,
            "limitations": limitations,
            "agreement_score": round(
                agreement_score,
                3,
            ),
            "successful_models": len(valid),
            "total_models": len(results),
        }


model_manager = MedicalModelManager()