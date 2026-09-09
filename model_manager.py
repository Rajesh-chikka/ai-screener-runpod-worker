import gc
import io
import json
import os
import re
import shutil
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


def print_directory_status(path: Path):
    print(
        f"[PATH] Checking: {path}",
        flush=True,
    )

    print(
        f"[PATH] Exists: {path.exists()}",
        flush=True,
    )

    if path.exists():
        try:
            entries = list(path.iterdir())

            print(
                f"[PATH] Items: {len(entries)}",
                flush=True,
            )

            for entry in entries[:10]:
                print(
                    f"[PATH]   {entry}",
                    flush=True,
                )

        except Exception as exc:
            print(
                f"[PATH] Unable to list directory: {exc}",
                flush=True,
            )


def resolve_cached_medgemma_path() -> str:
    """
    Find MedGemma inside RunPod's cached Hugging Face model storage.

    Expected Hugging Face cache layout:

    /runpod-volume/huggingface-cache/hub/
        models--google--medgemma-1.5-4b-it/
            refs/
            snapshots/
                <commit hash>/
    """

    model_directory = (
        RUNPOD_HF_CACHE_ROOT
        / "models--google--medgemma-1.5-4b-it"
    )

    print(
        "\n"
        "==============================\n"
        "RESOLVING CACHED MEDGEMMA\n"
        "==============================",
        flush=True,
    )

    print_directory_status(
        RUNPOD_HF_CACHE_ROOT
    )

    print_directory_status(
        model_directory
    )

    if not model_directory.exists():
        raise RuntimeError(
            "RunPod cached MedGemma directory was not found at "
            f"{model_directory}"
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

    # Preferred method:
    # read the Hugging Face refs/main pointer.
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

            return str(
                snapshot_path
            )

    # Fallback:
    # use an available snapshot directory.
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
                "refs/main unavailable. "
                f"Using snapshot: {snapshot_path}",
                flush=True,
            )

            return str(
                snapshot_path
            )

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
            "\n"
            "==============================\n"
            "MEDICAL MODEL MANAGER\n"
            "==============================",
            flush=True,
        )

        print(
            f"[MODEL] Device: {self.device}",
            flush=True,
        )

        print(
            f"[MODEL] MedGemma ID: {MEDGEMMA_MODEL_ID}",
            flush=True,
        )

        print(
            f"[MODEL] RunPod cache root: "
            f"{RUNPOD_HF_CACHE_ROOT}",
            flush=True,
        )

        print_disk_usage(
            "MODEL MANAGER START"
        )

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
            text[
                start:end + 1
            ]
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

        print(
            "\n"
            "==============================\n"
            "MEDGEMMA\n"
            "==============================",
            flush=True,
        )

        print_disk_usage(
            "BEFORE MEDGEMMA"
        )

        # IMPORTANT:
        # MedGemma is loaded from RunPod's
        # pre-cached model directory.
        local_model_path = (
            resolve_cached_medgemma_path()
        )

        print(
            "[MEDGEMMA] "
            f"Loading locally from: {local_model_path}",
            flush=True,
        )

        processor = (
            AutoProcessor.from_pretrained(
                local_model_path,
                local_files_only=True,
            )
        )

        print(
            "[MEDGEMMA] Processor loaded",
            flush=True,
        )

        print_disk_usage(
            "MEDGEMMA BEFORE MODEL LOAD"
        )

        model = (
            AutoModelForMultimodalLM
            .from_pretrained(
                local_model_path,
                dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
            )
        )

        model.eval()

        print(
            "[MEDGEMMA] Model loaded from cache",
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

        result = (
            self._parse_json(
                text
            )
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
            "\n"
            "==============================\n"
            "MEDSIGLIP\n"
            "==============================",
            flush=True,
        )

        print(
            "[MEDSIGLIP] "
            "This model is downloaded at runtime.",
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

        print_disk_usage(
            "MEDSIGLIP BEFORE MODEL LOAD"
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
            "\n"
            "==============================\n"
            "BIOMEDCLIP\n"
            "==============================",
            flush=True,
        )

        print(
            "[BIOMEDCLIP] "
            "This model is downloaded at runtime.",
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

        # -------------------------
        # 1. MedGemma
        # -------------------------

        try:
            print(
                "\n[1/3] Starting MedGemma",
                flush=True,
            )

            results[
                "medgemma"
            ] = self.run_medgemma(
                images=images,
                prompt=prompt,
                max_tokens=max_tokens,
            )

            print(
                "[1/3] MedGemma SUCCESS",
                flush=True,
            )

        except Exception as exc:
            print(
                f"[1/3] MedGemma FAILED: {exc}",
                flush=True,
            )

            results[
                "medgemma"
            ] = {
                "error": str(exc)
            }

            self._cleanup_gpu()

        print_disk_usage(
            "AFTER MEDGEMMA ATTEMPT"
        )

        # -------------------------
        # 2. MedSigLIP
        # -------------------------

        try:
            print(
                "\n[2/3] Starting MedSigLIP",
                flush=True,
            )

            results[
                "medsiglip"
            ] = self.run_medsiglip(
                images=images
            )

            print(
                "[2/3] MedSigLIP SUCCESS",
                flush=True,
            )

        except Exception as exc:
            print(
                f"[2/3] MedSigLIP FAILED: {exc}",
                flush=True,
            )

            results[
                "medsiglip"
            ] = {
                "error": str(exc)
            }

            self._cleanup_gpu()

        print_disk_usage(
            "AFTER MEDSIGLIP ATTEMPT"
        )

        # -------------------------
        # 3. BiomedCLIP
        # -------------------------

        try:
            print(
                "\n[3/3] Starting BiomedCLIP",
                flush=True,
            )

            results[
                "biomedclip"
            ] = self.run_biomedclip(
                images=images
            )

            print(
                "[3/3] BiomedCLIP SUCCESS",
                flush=True,
            )

        except Exception as exc:
            print(
                f"[3/3] BiomedCLIP FAILED: {exc}",
                flush=True,
            )

            results[
                "biomedclip"
            ] = {
                "error": str(exc)
            }

            self._cleanup_gpu()

        print_disk_usage(
            "ANALYSIS COMPLETE"
        )

        print(
            "\n"
            "==============================\n"
            "ENSEMBLE COMPLETE\n"
            "==============================",
            flush=True,
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

                    label = (
                        item.get(
                            "label"
                        )
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