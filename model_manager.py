import gc
import io
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import open_clip
import torch
from PIL import Image
from transformers import (
    AutoModelForMultimodalLM,
    AutoModelForZeroShotImageClassification,
    AutoProcessor,
)


MEDGEMMA_MODEL_ID = "google/medgemma-1.5-4b-it"

MEDSIGLIP_MODEL = "google/medsiglip-448"

BIOMEDCLIP_MODEL = (
    "hf-hub:"
    "microsoft/"
    "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

RUNPOD_HF_CACHE_ROOT = Path(
    "/runpod-volume/huggingface-cache/hub"
)

SEVERITY_ORDER = {
    "unknown": -1,
    "normal": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
}


def print_disk_usage(label: str):
    try:
        total, used, free = shutil.disk_usage("/")
        gb = 1024 ** 3

        print(
            f"[DISK] {label}: "
            f"total={total / gb:.2f} GB, "
            f"used={used / gb:.2f} GB, "
            f"free={free / gb:.2f} GB",
            flush=True,
        )

    except Exception as exc:
        print(
            f"[DISK] Unable to read disk usage: {exc}",
            flush=True,
        )


def resolve_cached_medgemma_path() -> str:
    model_directory = (
        RUNPOD_HF_CACHE_ROOT
        / "models--google--medgemma-1.5-4b-it"
    )

    refs_main = (
        model_directory
        / "refs"
        / "main"
    )

    snapshots_directory = (
        model_directory
        / "snapshots"
    )

    if not model_directory.exists():
        raise RuntimeError(
            "RunPod cached MedGemma directory was not found at "
            f"{model_directory}"
        )

    if refs_main.exists():
        snapshot_hash = (
            refs_main
            .read_text()
            .strip()
        )

        snapshot_path = (
            snapshots_directory
            / snapshot_hash
        )

        if snapshot_path.exists():
            print(
                "[MEDGEMMA CACHE] "
                f"Using snapshot: {snapshot_path}",
                flush=True,
            )

            return str(snapshot_path)

    if snapshots_directory.exists():
        snapshots = [
            path
            for path in snapshots_directory.iterdir()
            if path.is_dir()
        ]

        if snapshots:
            snapshots.sort(
                key=lambda path:
                    path.stat().st_mtime,
                reverse=True,
            )

            snapshot_path = snapshots[0]

            print(
                "[MEDGEMMA CACHE] "
                f"Using latest snapshot: {snapshot_path}",
                flush=True,
            )

            return str(snapshot_path)

    raise RuntimeError(
        "MedGemma cache directory exists, "
        "but no snapshot was found."
    )


class MedicalModelManager:
    def __init__(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"[MODEL] Device: {self.device}",
            flush=True,
        )

        print_disk_usage(
            "MODEL MANAGER START"
        )

        self.medgemma_model = None
        self.medgemma_processor = None

        self.medsiglip_model = None
        self.medsiglip_processor = None

        self.biomedclip_model = None
        self.biomedclip_preprocess = None
        self.biomedclip_tokenizer = None

    @staticmethod
    def _hf_token():
        return os.getenv(
            "HF_TOKEN"
        )

    @staticmethod
    def _images(
        images: list[bytes],
    ):
        return [
            Image.open(
                io.BytesIO(image)
            ).convert("RGB")
            for image in images
        ]

    @staticmethod
    def _analysis_prompt(
        prompt: str,
    ) -> str:

        return f"""
You are analyzing a medical screening image.

This is screening support only and is not a confirmed diagnosis.

Your entire response MUST be exactly one valid JSON object.

Do not use Markdown.
Do not use ```json.
Do not write anything before the JSON.
Do not write anything after the JSON.

Use exactly this schema:

{{
  "findings": "concise description of visible findings",
  "severity": "unknown",
  "confidence": "low",
  "flags": [],
  "evidence": [],
  "limitations": []
}}

Allowed severity values:
"normal", "low", "moderate", "high", "unknown"

Allowed confidence values:
"low", "medium", "high"

Output limits:
- findings must be maximum 2 concise sentences
- flags must be maximum 6 unique items
- evidence must be maximum 5 unique items
- limitations must be maximum 3 items
- do not include duplicate entries

If the image is not medically meaningful:
- severity must be "normal"
- confidence must be "low"
- explain that in findings
- add the reason to limitations

Do not invent patient history.
Do not infer facts that are not visible.
Only use visible image evidence.

Context:
{prompt}

Return JSON only.
""".strip()

    @staticmethod
    def _json_string_field(
        text: str,
        field_name: str,
    ) -> str | None:

        match = re.search(
            rf'"{re.escape(field_name)}"\s*:\s*',
            text,
        )

        if not match:
            return None

        decoder = json.JSONDecoder()

        try:
            value, _ = decoder.raw_decode(
                text[
                    match.end():
                ].lstrip()
            )

        except json.JSONDecodeError:
            return None

        if isinstance(
            value,
            str,
        ):
            return value

        return str(value)

    @staticmethod
    def _unique_string_list(
        value: Any,
    ) -> list[str]:

        if value is None:
            return []

        if not isinstance(
            value,
            list,
        ):
            value = [
                value
            ]

        items = []
        seen = set()

        for item in value:
            text = str(
                item
            )

            if text in seen:
                continue

            seen.add(
                text
            )

            items.append(
                text
            )

        return items

    @classmethod
    def _parse_malformed_json(
        cls,
        text: str,
    ) -> dict[str, Any]:

        findings = cls._json_string_field(
            text,
            "findings",
        )

        if findings is not None:
            return {
                "findings":
                    findings.strip(),

                "severity":
                    "unknown",

                "confidence":
                    "low",

                "flags":
                    [],

                "evidence":
                    [],

                "limitations": [
                    "The model response was incomplete or malformed; "
                    "severity could not be reliably determined."
                ],
            }

        return {
            "findings":
                "The image analysis completed, but the structured "
                "result could not be parsed reliably.",

            "severity":
                "unknown",

            "confidence":
                "low",

            "flags":
                [],

            "evidence":
                [],

            "limitations": [
                "The model response was incomplete or malformed."
            ],
        }

    @classmethod
    def _parse_json(
        cls,
        text: str,
    ) -> dict[str, Any]:

        if not text:
            raise ValueError(
                "MedGemma returned an empty response."
            )

        original_text = text.strip()

        print(
            f"[MEDGEMMA RAW OUTPUT] {original_text}",
            flush=True,
        )

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            original_text,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        data = None

        if (
            start != -1
            and end != -1
            and end > start
        ):
            json_text = cleaned[
                start:end + 1
            ]

            try:
                data = json.loads(
                    json_text
                )

            except json.JSONDecodeError:
                data = None

        if data is None:
            lower_text = (
                original_text.lower()
            )

            non_medical_phrases = [
                "not a medical image",
                "no medical information",
                "does not contain medical",
                "does not provide medical",
                "no meaningful medical",
                "cannot assess",
                "unable to assess",
            ]

            is_non_medical = any(
                phrase in lower_text
                for phrase in non_medical_phrases
            )

            if not is_non_medical:
                return cls._parse_malformed_json(
                    cleaned,
                )

            return {
                "findings":
                    original_text,

                "severity":
                    "normal",

                "confidence":
                    "low",

                "flags":
                    [],

                "evidence":
                    [],

                "limitations": [
                    "MedGemma returned unstructured text "
                    "instead of the requested JSON format."
                ],
            }

        severity = str(
            data.get(
                "severity",
                "unknown",
            )
        ).lower()

        if severity not in SEVERITY_ORDER:
            severity = "unknown"

        confidence = str(
            data.get(
                "confidence",
                "low",
            )
        ).lower()

        if confidence not in {
            "low",
            "medium",
            "high",
        }:
            confidence = "low"

        flags = cls._unique_string_list(
            data.get(
                "flags",
                [],
            )
        )

        evidence = cls._unique_string_list(
            data.get(
                "evidence",
                [],
            )
        )

        limitations = cls._unique_string_list(
            data.get(
                "limitations",
                [],
            )
        )

        return {
            "findings": str(
                data.get(
                    "findings",
                    "",
                )
            ).strip(),

            "severity":
                severity,

            "confidence":
                confidence,

            "flags":
                flags,

            "evidence":
                evidence,

            "limitations":
                limitations,
        }

    def _cleanup_gpu(self):
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _load_medgemma(self):
        if (
            self.medgemma_model is not None
            and self.medgemma_processor is not None
        ):
            print(
                "[MEDGEMMA] Reusing already-loaded model",
                flush=True,
            )

            return (
                self.medgemma_model,
                self.medgemma_processor,
            )

        print(
            "[MEDGEMMA] Loading model into memory for first time",
            flush=True,
        )

        print_disk_usage(
            "BEFORE MEDGEMMA LOAD"
        )

        local_model_path = (
            resolve_cached_medgemma_path()
        )

        print(
            "[MEDGEMMA] "
            f"Loading from cached path: {local_model_path}",
            flush=True,
        )

        self.medgemma_processor = (
            AutoProcessor.from_pretrained(
                local_model_path,
                local_files_only=True,
            )
        )

        self.medgemma_model = (
            AutoModelForMultimodalLM
            .from_pretrained(
                local_model_path,
                dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
            )
        )

        self.medgemma_model.eval()

        print_disk_usage(
            "AFTER MEDGEMMA LOAD"
        )

        return (
            self.medgemma_model,
            self.medgemma_processor,
        )

    def _load_medsiglip(self):
        if (
            self.medsiglip_model is not None
            and self.medsiglip_processor is not None
        ):
            print(
                "[MEDSIGLIP] Reusing already-loaded model",
                flush=True,
            )

            return (
                self.medsiglip_model,
                self.medsiglip_processor,
            )

        token = self._hf_token()

        print(
            "[MEDSIGLIP] Loading model into memory for first time",
            flush=True,
        )

        print_disk_usage(
            "MEDSIGLIP BEFORE LOAD"
        )

        self.medsiglip_processor = (
            AutoProcessor.from_pretrained(
                MEDSIGLIP_MODEL,
                token=token,
            )
        )

        self.medsiglip_model = (
            AutoModelForZeroShotImageClassification
            .from_pretrained(
                MEDSIGLIP_MODEL,
                token=token,
            )
            .to(
                self.device
            )
        )

        self.medsiglip_model.eval()

        print_disk_usage(
            "MEDSIGLIP AFTER LOAD"
        )

        return (
            self.medsiglip_model,
            self.medsiglip_processor,
        )

    def _load_biomedclip(self):
        if (
            self.biomedclip_model is not None
            and self.biomedclip_preprocess is not None
            and self.biomedclip_tokenizer is not None
        ):
            print(
                "[BIOMEDCLIP] Reusing already-loaded model",
                flush=True,
            )

            return (
                self.biomedclip_model,
                self.biomedclip_preprocess,
                self.biomedclip_tokenizer,
            )

        print(
            "[BIOMEDCLIP] Loading model into memory for first time",
            flush=True,
        )

        print_disk_usage(
            "BIOMEDCLIP BEFORE LOAD"
        )

        (
            self.biomedclip_model,
            _,
            self.biomedclip_preprocess,
        ) = (
            open_clip
            .create_model_and_transforms(
                BIOMEDCLIP_MODEL
            )
        )

        self.biomedclip_tokenizer = (
            open_clip.get_tokenizer(
                BIOMEDCLIP_MODEL
            )
        )

        self.biomedclip_model = (
            self.biomedclip_model
            .to(
                self.device
            )
        )

        self.biomedclip_model.eval()

        print_disk_usage(
            "BIOMEDCLIP AFTER LOAD"
        )

        return (
            self.biomedclip_model,
            self.biomedclip_preprocess,
            self.biomedclip_tokenizer,
        )

    def run_medgemma(
        self,
        images: list[bytes],
        prompt: str,
        max_tokens: int,
    ) -> dict[str, Any]:

        print(
            "[MEDGEMMA] Starting",
            flush=True,
        )

        model, processor = self._load_medgemma()

        pil_images = self._images(
            images
        )

        content = [
            {
                "type": "image",
                "image": image,
            }
            for image in pil_images
        ]

        content.append(
            {
                "type": "text",
                "text":
                    self._analysis_prompt(
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
            processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        )

        model_device = next(
            model.parameters()
        ).device

        inputs = {
            key: (
                value.to(
                    model_device
                )
                if hasattr(
                    value,
                    "to",
                )
                else value
            )
            for key, value
            in inputs.items()
        }

        input_length = (
            inputs[
                "input_ids"
            ].shape[-1]
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

        result = (
            self._parse_json(
                text
            )
        )

        del output
        del inputs
        del generated
        del pil_images

        return result

    def run_medsiglip(
        self,
        images: list[bytes],
    ) -> dict[str, Any]:
        print(
            "[MEDSIGLIP] Starting",
            flush=True,
        )

        model, processor = self._load_medsiglip()

        labels = [
            "normal medical appearance",
            "visible inflammation",
            "visible swelling",
            "visible redness",
            "visible lesion",
            "visible abnormal tissue",
            "poor image quality",
        ]

        image = (
            self._images(
                images
            )[0]
        )

        inputs = processor(
            images=image,
            text=labels,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(
                self.device
            )
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(
                **inputs
            )

        probabilities = (
            outputs
            .logits_per_image
            .softmax(
                dim=-1
            )[0]
        )

        ranked = sorted(
            zip(
                labels,
                probabilities.tolist(),
            ),
            key=lambda item:
                item[1],
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

        del outputs
        del inputs
        del image

        return result

    def run_biomedclip(
        self,
        images: list[bytes],
    ) -> dict[str, Any]:

        print(
            "[BIOMEDCLIP] Starting",
            flush=True,
        )

        (
            model,
            preprocess,
            tokenizer,
        ) = (
            self._load_biomedclip()
        )

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
            self._images(
                images
            )[0]
        ).unsqueeze(
            0
        ).to(
            self.device
        )

        text = tokenizer(
            labels
        ).to(
            self.device
        )

        with torch.inference_mode():

            image_features = (
                model.encode_image(
                    image
                )
            )

            text_features = (
                model.encode_text(
                    text
                )
            )

            image_features = (
                image_features
                / image_features.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

            text_features = (
                text_features
                / text_features.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

            scores = (
                100.0
                * image_features
                @ text_features.T
            ).softmax(
                dim=-1
            )[0]

        ranked = sorted(
            zip(
                labels,
                scores.tolist(),
            ),
            key=lambda item:
                item[1],
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

        del image_features
        del text_features
        del image
        del text

        return result

    def analyze_all(
        self,
        images: list[bytes],
        prompt: str,
        max_tokens: int = 512,
    ) -> dict[str, Any]:

        results = {}
        timings = {}
        total_start = time.perf_counter()

        print_disk_usage(
            "ANALYSIS START"
        )

        model_start = time.perf_counter()

        try:
            results[
                "medgemma"
            ] = self.run_medgemma(
                images=images,
                prompt=prompt,
                max_tokens=max_tokens,
            )

        except Exception as exc:
            results[
                "medgemma"
            ] = {
                "error": str(exc)
            }

            self._cleanup_gpu()

        timings[
            "medgemma_seconds"
        ] = round(
            time.perf_counter()
            - model_start,
            3,
        )

        print_disk_usage(
            "AFTER MEDGEMMA"
        )

        model_start = time.perf_counter()

        try:
            results[
                "medsiglip"
            ] = self.run_medsiglip(
                images=images
            )

        except Exception as exc:
            results[
                "medsiglip"
            ] = {
                "error": str(exc)
            }

            self._cleanup_gpu()

        timings[
            "medsiglip_seconds"
        ] = round(
            time.perf_counter()
            - model_start,
            3,
        )

        print_disk_usage(
            "AFTER MEDSIGLIP"
        )

        model_start = time.perf_counter()

        try:
            results[
                "biomedclip"
            ] = self.run_biomedclip(
                images=images
            )

        except Exception as exc:
            results[
                "biomedclip"
            ] = {
                "error": str(exc)
            }

            self._cleanup_gpu()

        timings[
            "biomedclip_seconds"
        ] = round(
            time.perf_counter()
            - model_start,
            3,
        )

        timings[
            "total_seconds"
        ] = round(
            time.perf_counter()
            - total_start,
            3,
        )

        results[
            "_timings"
        ] = timings

        print_disk_usage(
            "ANALYSIS COMPLETE"
        )

        return results

    @staticmethod
    def build_consensus(
        results: dict[str, Any],
    ) -> dict[str, Any]:

        medgemma = results.get(
            "medgemma",
            {},
        )

        if medgemma.get(
            "error"
        ):
            return {
                "error":
                    "Primary MedGemma analysis failed.",
                "models":
                    results,
            }

        findings = (
            medgemma.get(
                "findings",
                "",
            )
        )

        severity = (
            medgemma.get(
                "severity",
                "unknown",
            )
        )

        confidence = (
            medgemma.get(
                "confidence",
                "low",
            )
        )

        findings_lower = (
            findings.lower()
        )

        limitations_text = " ".join(
            str(item)
            for item in medgemma.get(
                "limitations",
                [],
            )
        ).lower()

        medical_context = (
            findings_lower
            + " "
            + limitations_text
        )

        non_medical_phrases = [
            "does not provide any medical information",
            "no medical information",
            "not a medical image",
            "no interpretable medical",
            "no visible medical",
            "does not contain medical",
            "does not provide medical",
            "no meaningful medical",
        ]

        is_non_medical = any(
            phrase in medical_context
            for phrase in non_medical_phrases
        )

        if is_non_medical:
            return {
                "findings":
                    findings,

                "severity":
                    "normal",

                "confidence":
                    "low",

                "flags":
                    [],

                "evidence":
                    medgemma.get(
                        "evidence",
                        [],
                    ),

                "limitations":
                    medgemma.get(
                        "limitations",
                        [],
                    ),

                "supporting_labels":
                    [],

                "model_agreement":
                    "not_applicable",

                "model_results":
                    results,
            }

        lightweight_results = {}

        for model_name in [
            "medsiglip",
            "biomedclip",
        ]:

            model_result = (
                results.get(
                    model_name,
                    {},
                )
            )

            if model_result.get(
                "error"
            ):
                continue

            valid_labels = []

            for item in (
                model_result.get(
                    "top_labels",
                    [],
                )
            ):

                try:
                    score = float(
                        item.get(
                            "score",
                            0,
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                label = (
                    item.get(
                        "label"
                    )
                )

                if (
                    label
                    and score >= 0.40
                ):
                    valid_labels.append(
                        {
                            "label":
                                label,

                            "score":
                                score,
                        }
                    )

            lightweight_results[
                model_name
            ] = valid_labels

        medsiglip_labels = {
            item["label"]
            for item in (
                lightweight_results.get(
                    "medsiglip",
                    [],
                )
            )
        }

        biomedclip_labels = {
            item["label"]
            for item in (
                lightweight_results.get(
                    "biomedclip",
                    [],
                )
            )
        }

        agreed_labels = (
            medsiglip_labels
            & biomedclip_labels
        )

        supporting_labels = sorted(
            agreed_labels
        )

        if supporting_labels:
            model_agreement = (
                "agreement"
            )
        else:
            model_agreement = (
                "limited"
            )

        if (
            severity != "normal"
            and not supporting_labels
            and confidence == "high"
        ):
            confidence = "medium"

        return {
            "findings":
                findings,

            "severity":
                severity,

            "confidence":
                confidence,

            "flags":
                medgemma.get(
                    "flags",
                    [],
                ),

            "evidence":
                medgemma.get(
                    "evidence",
                    [],
                ),

            "limitations":
                medgemma.get(
                    "limitations",
                    [],
                ),

            "supporting_labels":
                supporting_labels,

            "model_agreement":
                model_agreement,

            "model_results":
                results,
        }


model_manager = MedicalModelManager()
