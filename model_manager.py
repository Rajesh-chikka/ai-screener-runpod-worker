import gc
import io
import json
import os
import re
import shutil
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
            f"[DISK] Could not read disk usage: {exc}",
            flush=True,
        )


def print_runpod_model_env():
    """
    Print possible RunPod / Hugging Face model cache paths.

    This is temporary diagnostic logging so we can determine
    where RunPod places the cached MedGemma model.
    """

    keys = [
        "RUNPOD_MODEL_PATH",
        "RUNPOD_MODEL_NAME",
        "MODEL_PATH",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ]

    print(
        "\n"
        "==============================\n"
        "RUNPOD MODEL ENVIRONMENT\n"
        "==============================",
        flush=True,
    )

    for key in keys:
        print(
            f"{key}={os.getenv(key)}",
            flush=True,
        )

    print(
        "==============================\n",
        flush=True,
    )


class MedicalModelManager:
    def __init__(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print_runpod_model_env()

        print(
            f"[MODEL] Device: {self.device}",
            flush=True,
        )

        print_disk_usage(
            "MODEL MANAGER START"
        )

    @staticmethod
    def _hf_token():
        return os.getenv("HF_TOKEN")

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
                "MedGemma did not return JSON."
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

        if confidence not in {
            "low",
            "medium",
            "high",
        }:
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

        if not isinstance(
            flags,
            list,
        ):
            flags = [
                str(flags)
            ]

        if not isinstance(
            evidence,
            list,
        ):
            evidence = [
                str(evidence)
            ]

        if not isinstance(
            limitations,
            list,
        ):
            limitations = [
                str(limitations)
            ]

        return {
            "findings": str(
                data.get(
                    "findings",
                    "",
                )
            ).strip(),
            "severity": severity,
            "confidence": confidence,
            "flags": flags,
            "evidence": evidence,
            "limitations": limitations,
        }

    def _cleanup_gpu(self):
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def run_medgemma(
        self,
        images: list[bytes],
        prompt: str,
        max_tokens: int,
    ) -> dict[str, Any]:

        token = self._hf_token()

        print(
            "\n[MEDGEMMA] Starting",
            flush=True,
        )

        print(
            f"[MEDGEMMA] Model source: {MEDGEMMA_MODEL}",
            flush=True,
        )

        print(
            "[MEDGEMMA] Loading processor...",
            flush=True,
        )

        print_disk_usage(
            "MEDGEMMA BEFORE PROCESSOR"
        )

        processor = AutoProcessor.from_pretrained(
            MEDGEMMA_MODEL,
            token=token,
        )

        print(
            "[MEDGEMMA] Processor loaded",
            flush=True,
        )

        print(
            "[MEDGEMMA] Loading model...",
            flush=True,
        )

        print_disk_usage(
            "MEDGEMMA BEFORE MODEL LOAD"
        )

        model = (
            AutoModelForMultimodalLM
            .from_pretrained(
                MEDGEMMA_MODEL,
                token=token,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        )

        print(
            "[MEDGEMMA] Model loaded",
            flush=True,
        )

        print_disk_usage(
            "MEDGEMMA AFTER MODEL LOAD"
        )

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

        print(
            "[MEDGEMMA] Generating...",
            flush=True,
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

        print(
            "[MEDGEMMA] Generation complete",
            flush=True,
        )

        result = self._parse_json(
            text
        )

        del output
        del inputs
        del model
        del processor
        del pil_images

        self._cleanup_gpu()

        print_disk_usage(
            "MEDGEMMA AFTER CLEANUP"
        )

        return result

    def run_medsiglip(
        self,
        images: list[bytes],
    ) -> dict[str, Any]:

        token = self._hf_token()

        print(
            "\n[MEDSIGLIP] Starting",
            flush=True,
        )

        print(
            "[MEDSIGLIP] Loading processor...",
            flush=True,
        )

        print_disk_usage(
            "MEDSIGLIP BEFORE PROCESSOR"
        )

        processor = (
            AutoProcessor.from_pretrained(
                MEDSIGLIP_MODEL,
                token=token,
            )
        )

        print(
            "[MEDSIGLIP] Processor loaded",
            flush=True,
        )

        print(
            "[MEDSIGLIP] Loading model...",
            flush=True,
        )

        model = (
            AutoModelForZeroShotImageClassification
            .from_pretrained(
                MEDSIGLIP_MODEL,
                token=token,
            )
            .to(
                self.device
            )
        )

        model.eval()

        print(
            "[MEDSIGLIP] Model loaded",
            flush=True,
        )

        print_disk_usage(
            "MEDSIGLIP AFTER MODEL LOAD"
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

        image = self._images(
            images
        )[0]

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

        del outputs
        del inputs
        del model
        del processor
        del image

        self._cleanup_gpu()

        print_disk_usage(
            "MEDSIGLIP AFTER CLEANUP"
        )

        return result

    def run_biomedclip(
        self,
        images: list[bytes],
    ) -> dict[str, Any]:

        print(
            "\n[BIOMEDCLIP] Starting",
            flush=True,
        )

        print_disk_usage(
            "BIOMEDCLIP BEFORE LOAD"
        )

        model, _, preprocess = (
            open_clip
            .create_model_and_transforms(
                BIOMEDCLIP_MODEL
            )
        )

        tokenizer = (
            open_clip.get_tokenizer(
                BIOMEDCLIP_MODEL
            )
        )

        model = model.to(
            self.device
        )

        model.eval()

        print(
            "[BIOMEDCLIP] Model loaded",
            flush=True,
        )

        print_disk_usage(
            "BIOMEDCLIP AFTER LOAD"
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

        del image_features
        del text_features
        del image
        del text
        del model

        self._cleanup_gpu()

        print_disk_usage(
            "BIOMEDCLIP AFTER CLEANUP"
        )

        return result

    def analyze_all(
        self,
        images: list[bytes],
        prompt: str,
        max_tokens: int = 512,
    ) -> dict[str, Any]:

        results = {}

        print(
            "\n"
            "==============================\n"
            "STARTING MEDICAL ENSEMBLE\n"
            "==============================",
            flush=True,
        )

        print_disk_usage(
            "ANALYSIS START"
        )

        # --------------------------------
        # 1. MedGemma
        # --------------------------------

        try:
            print(
                "\n[1/3] Starting MedGemma",
                flush=True,
            )

            print_disk_usage(
                "BEFORE MEDGEMMA"
            )

            results["medgemma"] = (
                self.run_medgemma(
                    images=images,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
            )

            print(
                "[1/3] MedGemma SUCCESS",
                flush=True,
            )

            print_disk_usage(
                "AFTER MEDGEMMA"
            )

        except Exception as exc:

            print(
                f"[1/3] MedGemma FAILED: {exc}",
                flush=True,
            )

            results["medgemma"] = {
                "error": str(exc)
            }

            self._cleanup_gpu()

            print_disk_usage(
                "MEDGEMMA FAILED"
            )

        # --------------------------------
        # 2. MedSigLIP
        # --------------------------------

        try:
            print(
                "\n[2/3] Starting MedSigLIP",
                flush=True,
            )

            print_disk_usage(
                "BEFORE MEDSIGLIP"
            )

            results["medsiglip"] = (
                self.run_medsiglip(
                    images=images
                )
            )

            print(
                "[2/3] MedSigLIP SUCCESS",
                flush=True,
            )

            print_disk_usage(
                "AFTER MEDSIGLIP"
            )

        except Exception as exc:

            print(
                f"[2/3] MedSigLIP FAILED: {exc}",
                flush=True,
            )

            results["medsiglip"] = {
                "error": str(exc)
            }

            self._cleanup_gpu()

            print_disk_usage(
                "MEDSIGLIP FAILED"
            )

        # --------------------------------
        # 3. BiomedCLIP
        # --------------------------------

        try:
            print(
                "\n[3/3] Starting BiomedCLIP",
                flush=True,
            )

            print_disk_usage(
                "BEFORE BIOMEDCLIP"
            )

            results["biomedclip"] = (
                self.run_biomedclip(
                    images=images
                )
            )

            print(
                "[3/3] BiomedCLIP SUCCESS",
                flush=True,
            )

            print_disk_usage(
                "AFTER BIOMEDCLIP"
            )

        except Exception as exc:

            print(
                f"[3/3] BiomedCLIP FAILED: {exc}",
                flush=True,
            )

            results["biomedclip"] = {
                "error": str(exc)
            }

            self._cleanup_gpu()

            print_disk_usage(
                "BIOMEDCLIP FAILED"
            )

        print(
            "\n"
            "==============================\n"
            "ENSEMBLE COMPLETE\n"
            "==============================",
            flush=True,
        )

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

        supporting_labels = []

        for name in [
            "medsiglip",
            "biomedclip",
        ]:

            model_result = (
                results.get(
                    name,
                    {},
                )
            )

            if model_result.get(
                "error"
            ):
                continue

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

                if score >= 0.25:

                    label = item.get(
                        "label"
                    )

                    if label:
                        supporting_labels.append(
                            label
                        )

        supporting_labels = list(
            dict.fromkeys(
                supporting_labels
            )
        )

        confidence = (
            medgemma.get(
                "confidence",
                "low",
            )
        )

        severity = (
            medgemma.get(
                "severity",
                "normal",
            )
        )

        if (
            severity != "normal"
            and not supporting_labels
        ):
            confidence = "low"

        return {
            "findings":
                medgemma.get(
                    "findings",
                    "",
                ),

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

            "model_results":
                results,
        }


model_manager = MedicalModelManager()