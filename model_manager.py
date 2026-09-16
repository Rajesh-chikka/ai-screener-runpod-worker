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

EAR_CHECKLIST_PROMPT_TERMS = (
    "ear_checklist",
    "earwax",
    "eardrum",
    "tympanic membrane",
)

GUMS_CHECKLIST_PROMPT_TERMS = (
    "gums_checklist",
    "gums",
    "gingiva",
    "lips",
)

JAW_CHECKLIST_PROMPT_TERMS = (
    "jaw_checklist",
    "tmj",
    "jaw",
    "dentition",
)

THROAT_CHECKLIST_PROMPT_TERMS = (
    "throat_checklist",
    "tongue",
    "throat",
    "uvula",
    "tonsil",
)

OPTIONAL_CHECKLIST_KEYS = (
    "ear_checklist",
    "gums_checklist",
    "jaw_checklist",
    "throat_checklist",
)

PRESENT_ABSENT_VALUES = {
    "present",
    "absent",
    "ungradable",
}

YES_NO_VALUES = {
    "yes",
    "no",
    "ungradable",
}

EARWAX_AMOUNT_VALUES = {
    "none",
    "minimal",
    "visible",
    "occluding",
    "ungradable",
}

EARWAX_DEPOSIT_VALUES = {
    "yes",
    "no",
    "ungradable",
}

EARWAX_TYPE_VALUES = {
    "wet",
    "dry",
    "flaky",
    "mixed",
    "ungradable",
    "not_applicable",
}

EAR_CANAL_VALUES = {
    "present",
    "absent",
    "ungradable",
}

EARDRUM_VISIBILITY_VALUES = {
    "yes",
    "partial",
    "no",
    "ungradable",
}

EARDRUM_INTACTNESS_VALUES = {
    "intact",
    "not_intact",
    "ungradable",
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
    def _needs_ear_checklist(
        prompt: str,
    ) -> bool:

        lower_prompt = prompt.lower()

        return any(
            term in lower_prompt
            for term in EAR_CHECKLIST_PROMPT_TERMS
        )

    @staticmethod
    def _requested_checklists(
        prompt: str,
    ) -> list[str]:

        lower_prompt = prompt.lower()

        marker_match = re.search(
            r"\bchecklist_type\s*=\s*"
            r"(ear_checklist|gums_checklist|jaw_checklist|throat_checklist)\b",
            lower_prompt,
        )

        if marker_match:
            return [
                marker_match.group(1)
            ]

        matches = []

        if MedicalModelManager._needs_ear_checklist(
            prompt
        ):
            matches.append(
                "ear_checklist"
            )

        term_groups = [
            (
                "gums_checklist",
                GUMS_CHECKLIST_PROMPT_TERMS,
            ),
            (
                "jaw_checklist",
                JAW_CHECKLIST_PROMPT_TERMS,
            ),
            (
                "throat_checklist",
                THROAT_CHECKLIST_PROMPT_TERMS,
            ),
        ]

        for key, terms in term_groups:
            if any(
                term in lower_prompt
                for term in terms
            ):
                matches.append(
                    key
                )

        if len(matches) == 1:
            return matches

        return []

    @staticmethod
    def _ear_checklist_prompt() -> str:

        return """
Because the request includes an ear checklist, include the ear_checklist key.

Assess only what is visibly present in the provided ear frames.
Evaluate the ear canal and tympanic membrane using the requested checklist.
You MUST complete the ear_checklist object with earwax, ear_canal, and eardrum fields.
If a feature cannot be reliably assessed, return ungradable.
Evaluate every checklist field independently.
Do not mark the entire checklist ungradable simply because some structures are not visible.
Do not infer findings that are not clearly visible.
Do not diagnose disease.
Return structured JSON only.
""".strip()

    @staticmethod
    def _gums_checklist_prompt() -> str:

        return """
Return ONE valid JSON object matching EXACTLY the schema below.
Do not add keys. Do not rename keys. Do not add anatomy not listed in the schema.
Assess ONLY what is visible in the supplied images.
Evaluate each field independently.
Use ungradable only when that specific feature cannot be assessed.
Do not diagnose disease.
Do not output markdown or commentary before or after JSON.

SCHEMA:
{
  "findings": "<short 1-3 sentence visual summary>",
  "severity": "normal|low|moderate|high|unknown",
  "confidence": "low|medium|high",
  "flags": [],
  "evidence": [],
  "limitations": [],
  "gums_checklist": {
    "lips": {
      "colour_abnormality": "present|absent|ungradable",
      "pigmentation": "present|absent|ungradable",
      "dryness_cracking": "present|absent|ungradable",
      "swelling": "present|absent|ungradable",
      "ulcer_erosion": "present|absent|ungradable",
      "focal_lesion_lump": "present|absent|ungradable"
    },
    "gums": {
      "abnormal_redness_swelling": "present|absent|ungradable",
      "pigmentation_focal_colour_change": "present|absent|ungradable",
      "visible_plaque_calculus": "present|absent|ungradable",
      "bleeding_ulceration_crypts": "present|absent|ungradable"
    }
  }
}
""".strip()

    @staticmethod
    def _jaw_checklist_prompt() -> str:

        return """
Return ONE valid JSON object matching EXACTLY the schema below.
Do not add keys. Do not rename keys. Do not add anatomy not listed in the schema.
Assess ONLY what is visible in the supplied images.
Evaluate each field independently.
Use ungradable only when that specific feature cannot be assessed.
A visible_cavity_defect requires an actual visible defect or cavitation, not merely discoloration.
Do not diagnose disease.
Do not output markdown or commentary before or after JSON.

SCHEMA:
{
  "findings": "<short 1-3 sentence visual summary>",
  "severity": "normal|low|moderate|high|unknown",
  "confidence": "low|medium|high",
  "flags": [],
  "evidence": [],
  "limitations": [],
  "jaw_checklist": {
    "missing_teeth": "present|absent|ungradable",
    "visible_tooth_discoloration": "present|absent|ungradable",
    "visible_cavity_defect": "present|absent|ungradable",
    "broken_chipped_tooth": "present|absent|ungradable",
    "significant_tooth_wear_erosion": "present|absent|ungradable",
    "crowding": "present|absent|ungradable",
    "misalignment": "present|absent|ungradable",
    "significant_spacing_gaps": "present|absent|ungradable",
    "plaque_calculus": "present|absent|ungradable",
    "gum_redness_swelling_recession": "present|absent|ungradable"
  }
}
""".strip()

    @staticmethod
    def _throat_checklist_prompt() -> str:

        return """
Return ONE valid JSON object matching EXACTLY the schema below.
Do not add keys. Do not rename keys. Do not add anatomy not listed in the schema.
Assess ONLY what is visible in the supplied images.
Evaluate each field independently.
Use ungradable only when that specific feature cannot be assessed.
Do not diagnose disease.
Do not output markdown or commentary before or after JSON.

SCHEMA:
{
  "findings": "<short 1-3 sentence visual summary>",
  "severity": "normal|low|moderate|high|unknown",
  "confidence": "low|medium|high",
  "flags": [],
  "evidence": [],
  "limitations": [],
  "throat_checklist": {
    "tongue": {
      "adequately_visible": "yes|partial|no|ungradable",
      "abnormal_colour_pigmentation": "present|absent|ungradable",
      "coating": "present|absent|ungradable",
      "fissures_irregular_surface": "present|absent|ungradable",
      "ulcer_erosion": "present|absent|ungradable",
      "focal_lesion_swelling": "present|absent|ungradable",
      "asymmetry_deviation": "present|absent|ungradable"
    },
    "throat": {
      "uvula_visible": "yes|partial|no|ungradable",
      "uvula_approximately_midline": "yes|no|ungradable",
      "tonsils_visible": "yes|partial|no|ungradable",
      "tonsillar_asymmetry_swelling": "present|absent|ungradable",
      "white_yellow_material_on_tonsils": "present|absent|ungradable",
      "throat_redness": "present|absent|ungradable",
      "focal_lesion_mass": "present|absent|ungradable"
    }
  }
}
""".strip()

    @staticmethod
    def _checklist_prompt(
        checklist_key: str,
    ) -> str:

        prompts = {
            "ear_checklist":
                MedicalModelManager._ear_checklist_prompt,
            "gums_checklist":
                MedicalModelManager._gums_checklist_prompt,
            "jaw_checklist":
                MedicalModelManager._jaw_checklist_prompt,
            "throat_checklist":
                MedicalModelManager._throat_checklist_prompt,
        }

        prompt_factory = prompts.get(
            checklist_key
        )

        if prompt_factory is None:
            return ""

        return prompt_factory()

    @staticmethod
    def _analysis_prompt(
        prompt: str,
    ) -> str:

        keys = (
            "findings, severity, confidence, flags, evidence, limitations"
        )

        checklist_prompts = []

        requested_checklists = MedicalModelManager._requested_checklists(
            prompt
        )

        for checklist_key in requested_checklists:
            keys = (
                keys
                + f", {checklist_key}"
            )

            checklist_prompts.append(
                MedicalModelManager._checklist_prompt(
                    checklist_key
                )
            )

        checklist_prompt = ""

        if checklist_prompts:
            checklist_prompt = (
                "\n\n"
                + "\n\n".join(
                    checklist_prompts
                )
            )

        return f"""
You are analyzing a medical screening image.

This is screening support only and is not a confirmed diagnosis.

Your entire response MUST be exactly one valid JSON object with these keys:
{keys}.

Do not use Markdown.
Do not use ```json.
Do not write anything before the JSON.
Do not write anything after the JSON.
Do not include chain-of-thought.
Do not include explanations outside the JSON object.
Do not use headings such as FINDINGS:.
Do not repeat the schema.

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
{checklist_prompt}

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

    @staticmethod
    def _normalize_ear_enum(
        value: Any,
        allowed_values: set[str],
    ) -> str:

        normalized = str(
            value
            if value is not None
            else ""
        ).strip().lower()

        if normalized in allowed_values:
            return normalized

        return "ungradable"

    @staticmethod
    def _normalize_ear_colour(
        value: Any,
    ) -> str:

        normalized = " ".join(
            str(
                value
                if value is not None
                else ""
            ).split()
        )

        if normalized:
            return normalized

        return "ungradable"

    @staticmethod
    def _ungradable_ear_checklist() -> dict[str, Any]:

        return {
            "earwax": {
                "amount":
                    "ungradable",
                "deposit_on_eardrum":
                    "ungradable",
                "colour":
                    "ungradable",
                "type":
                    "ungradable",
            },
            "ear_canal": {
                "redness_swelling":
                    "ungradable",
                "bleeding_trauma":
                    "ungradable",
                "foreign_body_visualised":
                    "ungradable",
            },
            "eardrum": {
                "visibility":
                    "ungradable",
                "colour":
                    "ungradable",
                "intactness":
                    "ungradable",
                "bulging":
                    "ungradable",
                "discharge":
                    "ungradable",
                "handle_of_malleus_visibility":
                    "ungradable",
                "cone_of_light_visibility":
                    "ungradable",
            },
        }

    @classmethod
    def _normalize_ear_checklist(
        cls,
        value: Any,
    ) -> dict[str, Any]:

        if not isinstance(
            value,
            dict,
        ):
            return cls._ungradable_ear_checklist()

        earwax = value.get(
            "earwax",
            {},
        )

        if not isinstance(
            earwax,
            dict,
        ):
            earwax = {}

        ear_canal = value.get(
            "ear_canal",
            {},
        )

        if not isinstance(
            ear_canal,
            dict,
        ):
            ear_canal = {}

        eardrum = value.get(
            "eardrum",
            {},
        )

        if not isinstance(
            eardrum,
            dict,
        ):
            eardrum = {}

        earwax_amount = cls._normalize_ear_enum(
            earwax.get(
                "amount"
            ),
            EARWAX_AMOUNT_VALUES,
        )

        if earwax_amount == "none":
            earwax_colour = "not_applicable"
            earwax_type = "not_applicable"
        else:
            earwax_colour = cls._normalize_ear_colour(
                earwax.get(
                    "colour"
                )
            )

            earwax_type = cls._normalize_ear_enum(
                earwax.get(
                    "type"
                ),
                EARWAX_TYPE_VALUES,
            )

            if earwax_colour.lower() == "not_applicable":
                earwax_colour = "ungradable"

            if earwax_type == "not_applicable":
                earwax_type = "ungradable"

        eardrum_visibility = cls._normalize_ear_enum(
            eardrum.get(
                "visibility"
            ),
            EARDRUM_VISIBILITY_VALUES,
        )

        if eardrum_visibility in {
            "no",
            "ungradable",
        }:
            eardrum_colour = "ungradable"
        else:
            eardrum_colour = cls._normalize_ear_colour(
                eardrum.get(
                    "colour"
                )
            )

        return {
            "earwax": {
                "amount":
                    earwax_amount,
                "deposit_on_eardrum":
                    cls._normalize_ear_enum(
                        earwax.get(
                            "deposit_on_eardrum"
                        ),
                        EARWAX_DEPOSIT_VALUES,
                    ),
                "colour":
                    earwax_colour,
                "type":
                    earwax_type,
            },
            "ear_canal": {
                "redness_swelling":
                    cls._normalize_ear_enum(
                        ear_canal.get(
                            "redness_swelling"
                        ),
                        EAR_CANAL_VALUES,
                    ),
                "bleeding_trauma":
                    cls._normalize_ear_enum(
                        ear_canal.get(
                            "bleeding_trauma"
                        ),
                        EAR_CANAL_VALUES,
                    ),
                "foreign_body_visualised":
                    cls._normalize_ear_enum(
                        ear_canal.get(
                            "foreign_body_visualised"
                        ),
                        EAR_CANAL_VALUES,
                    ),
            },
            "eardrum": {
                "visibility":
                    eardrum_visibility,
                "colour":
                    eardrum_colour,
                "intactness":
                    cls._normalize_ear_enum(
                        eardrum.get(
                            "intactness"
                        ),
                        EARDRUM_INTACTNESS_VALUES,
                    ),
                "bulging":
                    cls._normalize_ear_enum(
                        eardrum.get(
                            "bulging"
                        ),
                        EAR_CANAL_VALUES,
                    ),
                "discharge":
                    cls._normalize_ear_enum(
                        eardrum.get(
                            "discharge"
                        ),
                        EAR_CANAL_VALUES,
                    ),
                "handle_of_malleus_visibility":
                    cls._normalize_ear_enum(
                        eardrum.get(
                            "handle_of_malleus_visibility"
                        ),
                        EARDRUM_VISIBILITY_VALUES,
                    ),
                "cone_of_light_visibility":
                    cls._normalize_ear_enum(
                        eardrum.get(
                            "cone_of_light_visibility"
                        ),
                        EARDRUM_VISIBILITY_VALUES,
                    ),
            },
        }

    @classmethod
    def _normalize_present_absent_group(
        cls,
        value: dict[str, Any],
        fields: list[str],
    ) -> dict[str, str]:

        return {
            field:
                cls._normalize_ear_enum(
                    value.get(
                        field
                    ),
                    PRESENT_ABSENT_VALUES,
                )
            for field in fields
        }

    @classmethod
    def _normalize_gums_checklist(
        cls,
        value: Any,
    ) -> dict[str, Any]:

        if not isinstance(
            value,
            dict,
        ):
            value = {}

        lips = value.get(
            "lips",
            {},
        )

        if not isinstance(
            lips,
            dict,
        ):
            lips = {}

        gums = value.get(
            "gums",
            {},
        )

        if not isinstance(
            gums,
            dict,
        ):
            gums = {}

        return {
            "lips":
                cls._normalize_present_absent_group(
                    lips,
                    [
                        "colour_abnormality",
                        "pigmentation",
                        "dryness_cracking",
                        "swelling",
                        "ulcer_erosion",
                        "focal_lesion_lump",
                    ],
                ),
            "gums":
                cls._normalize_present_absent_group(
                    gums,
                    [
                        "abnormal_redness_swelling",
                        "pigmentation_focal_colour_change",
                        "visible_plaque_calculus",
                        "bleeding_ulceration_crypts",
                    ],
                ),
        }

    @classmethod
    def _normalize_jaw_checklist(
        cls,
        value: Any,
    ) -> dict[str, str]:

        if not isinstance(
            value,
            dict,
        ):
            value = {}

        return cls._normalize_present_absent_group(
            value,
            [
                "missing_teeth",
                "visible_tooth_discoloration",
                "visible_cavity_defect",
                "broken_chipped_tooth",
                "significant_tooth_wear_erosion",
                "crowding",
                "misalignment",
                "significant_spacing_gaps",
                "plaque_calculus",
                "gum_redness_swelling_recession",
            ],
        )

    @classmethod
    def _normalize_throat_checklist(
        cls,
        value: Any,
    ) -> dict[str, Any]:

        if not isinstance(
            value,
            dict,
        ):
            value = {}

        tongue = value.get(
            "tongue",
            {},
        )

        if not isinstance(
            tongue,
            dict,
        ):
            tongue = {}

        throat = value.get(
            "throat",
            {},
        )

        if not isinstance(
            throat,
            dict,
        ):
            throat = {}

        return {
            "tongue": {
                "adequately_visible":
                    cls._normalize_ear_enum(
                        tongue.get(
                            "adequately_visible"
                        ),
                        EARDRUM_VISIBILITY_VALUES,
                    ),
                **cls._normalize_present_absent_group(
                    tongue,
                    [
                        "abnormal_colour_pigmentation",
                        "coating",
                        "fissures_irregular_surface",
                        "ulcer_erosion",
                        "focal_lesion_swelling",
                        "asymmetry_deviation",
                    ],
                ),
            },
            "throat": {
                "uvula_visible":
                    cls._normalize_ear_enum(
                        throat.get(
                            "uvula_visible"
                        ),
                        EARDRUM_VISIBILITY_VALUES,
                    ),
                "uvula_approximately_midline":
                    cls._normalize_ear_enum(
                        throat.get(
                            "uvula_approximately_midline"
                        ),
                        YES_NO_VALUES,
                    ),
                "tonsils_visible":
                    cls._normalize_ear_enum(
                        throat.get(
                            "tonsils_visible"
                        ),
                        EARDRUM_VISIBILITY_VALUES,
                    ),
                **cls._normalize_present_absent_group(
                    throat,
                    [
                        "tonsillar_asymmetry_swelling",
                        "white_yellow_material_on_tonsils",
                        "throat_redness",
                        "focal_lesion_mass",
                    ],
                ),
            },
        }

    @classmethod
    def _normalize_checklist(
        cls,
        checklist_key: str,
        value: Any,
    ) -> dict[str, Any]:

        normalizers = {
            "ear_checklist":
                cls._normalize_ear_checklist,
            "gums_checklist":
                cls._normalize_gums_checklist,
            "jaw_checklist":
                cls._normalize_jaw_checklist,
            "throat_checklist":
                cls._normalize_throat_checklist,
        }

        normalizer = normalizers.get(
            checklist_key
        )

        if normalizer is None:
            return {}

        return normalizer(
            value
        )

    @staticmethod
    def _expected_checklist_keys(
        checklist_key: str,
    ) -> set[str]:

        expected_keys = {
            "ear_checklist": {
                "earwax",
                "ear_canal",
                "eardrum",
            },
            "gums_checklist": {
                "lips",
                "gums",
            },
            "jaw_checklist": {
                "missing_teeth",
                "visible_tooth_discoloration",
                "visible_cavity_defect",
                "broken_chipped_tooth",
                "significant_tooth_wear_erosion",
                "crowding",
                "misalignment",
                "significant_spacing_gaps",
                "plaque_calculus",
                "gum_redness_swelling_recession",
            },
            "throat_checklist": {
                "tongue",
                "throat",
            },
        }

        return expected_keys.get(
            checklist_key,
            set(),
        )

    @staticmethod
    def _log_structured_parse_failure(
        requested_checklists: list[str],
        reason: str,
    ) -> None:

        for checklist_key in requested_checklists:
            print(
                "[STRUCTURED] "
                f"requested={checklist_key} "
                "json_parsed=false "
                "fallback=true "
                f"reason={reason}",
                flush=True,
            )

    @classmethod
    def _include_ear_checklist(
        cls,
        result: dict[str, Any],
        include_ear_checklist: bool,
        value: Any = None,
    ) -> dict[str, Any]:

        if include_ear_checklist:
            result[
                "ear_checklist"
            ] = cls._normalize_ear_checklist(
                value
            )

        return result

    @classmethod
    def _include_requested_checklists(
        cls,
        result: dict[str, Any],
        requested_checklists: list[str] | None,
        source: Any = None,
        json_parsed: bool = False,
        fallback_used: bool = False,
    ) -> dict[str, Any]:

        if not requested_checklists:
            return result

        if not isinstance(
            source,
            dict,
        ):
            source = {}

        for checklist_key in requested_checklists:
            checklist_found = (
                checklist_key in source
                and isinstance(
                    source.get(
                        checklist_key
                    ),
                    dict,
                )
            )

            extra_checklist_keys = 0

            if checklist_found:
                extra_checklist_keys = len(
                    set(
                        source[
                            checklist_key
                        ].keys()
                    )
                    - cls._expected_checklist_keys(
                        checklist_key
                    )
                )

            all_ungradable_fallback = (
                fallback_used
                or not checklist_found
            )

            print(
                "[STRUCTURED] "
                f"requested={checklist_key} "
                f"json_parsed={str(json_parsed).lower()} "
                f"expected_checklist={str(checklist_found).lower()} "
                f"extra_checklist_keys={extra_checklist_keys} "
                f"fallback={str(all_ungradable_fallback).lower()}",
                flush=True,
            )

            result[
                checklist_key
            ] = cls._normalize_checklist(
                checklist_key,
                source.get(
                    checklist_key
                ),
            )

        return result

    @staticmethod
    def _limited_unique_string_list(
        values: list[str],
        limit: int,
    ) -> list[str]:

        items = []
        seen = set()

        for value in values:
            text = str(
                value
            ).strip()

            if not text or text == "[]":
                continue

            key = text.lower()

            if key in seen:
                continue

            seen.add(
                key
            )

            items.append(
                text
            )

            if len(items) >= limit:
                break

        return items

    @staticmethod
    def _is_demographic_limitation(
        text: str,
    ) -> bool:

        lower_text = text.lower()

        demographic_terms = [
            "age",
            "ethnicity",
            "race",
            "racial",
            "demographic",
        ]

        return any(
            term in lower_text
            for term in demographic_terms
        )

    @staticmethod
    def _plain_text_sections(
        text: str,
    ) -> dict[str, str]:

        header_pattern = re.compile(
            r"(?im)^\s*"
            r"(findings|severity|confidence|flags|evidence|limitations)"
            r"\s*:\s*"
        )

        matches = list(
            header_pattern.finditer(
                text
            )
        )

        if not matches:
            return {}

        sections = {}

        for index, match in enumerate(
            matches
        ):
            name = (
                match.group(1)
                .lower()
            )

            start = match.end()

            if index + 1 < len(matches):
                end = matches[
                    index + 1
                ].start()
            else:
                end = len(text)

            sections[name] = text[
                start:end
            ].strip()

        return sections

    @staticmethod
    def _plain_text_list(
        text: str,
    ) -> list[str]:

        if not text or text.strip() == "[]":
            return []

        lines = [
            line.strip()
            for line in text.splitlines()
        ]

        items = []
        current_item = None

        for line in lines:
            if not line:
                continue

            bullet = re.match(
                r"^[-*]\s+(.*)$",
                line,
            )

            if bullet:
                if current_item:
                    items.append(
                        current_item
                    )

                current_item = (
                    bullet.group(1)
                    .strip()
                )

            elif current_item:
                current_item = (
                    current_item
                    + " "
                    + line
                ).strip()

            else:
                items.append(
                    line
                )

        if current_item:
            items.append(
                current_item
            )

        return items

    @staticmethod
    def _plain_text_scalar(
        text: str,
    ) -> str:

        for line in text.splitlines():
            value = (
                line.strip()
                .strip("-* ")
                .strip()
            )

            if value:
                return value

        return ""

    @classmethod
    def _parse_sectioned_plain_text(
        cls,
        text: str,
        include_ear_checklist: bool,
        requested_checklists: list[str] | None = None,
    ) -> dict[str, Any] | None:

        sections = cls._plain_text_sections(
            text,
        )

        if not sections:
            return None

        section_names = set(
            sections
        )

        if "findings" not in section_names or not (
            section_names
            & {
                "severity",
                "confidence",
                "flags",
                "evidence",
                "limitations",
            }
        ):
            return None

        severity = (
            cls._plain_text_scalar(
                sections.get(
                    "severity",
                    "",
                )
            ).lower()
        )

        if severity not in SEVERITY_ORDER:
            severity = "unknown"

        confidence = (
            cls._plain_text_scalar(
                sections.get(
                    "confidence",
                    "",
                )
            ).lower()
        )

        if confidence not in {
            "low",
            "medium",
            "high",
        }:
            confidence = "low"

        return cls._include_requested_checklists(
            {
                "findings":
                    sections.get(
                        "findings",
                        "",
                    ).strip(),

                "severity":
                    severity,

                "confidence":
                    confidence,

                "flags":
                    cls._limited_unique_string_list(
                        cls._plain_text_list(
                            sections.get(
                                "flags",
                                "",
                            )
                        ),
                        6,
                    ),

                "evidence":
                    cls._limited_unique_string_list(
                        cls._plain_text_list(
                            sections.get(
                                "evidence",
                                "",
                            )
                        ),
                        5,
                    ),

                "limitations":
                    cls._limited_unique_string_list(
                        [
                            item
                            for item in cls._plain_text_list(
                                sections.get(
                                    "limitations",
                                    "",
                                )
                            )
                            if not cls._is_demographic_limitation(
                                item
                            )
                        ],
                        3,
                    ),
            },
            requested_checklists
            if requested_checklists is not None
            else (
                ["ear_checklist"]
                if include_ear_checklist
                else []
            ),
            json_parsed=False,
            fallback_used=True,
        )

    @classmethod
    def _parse_malformed_json(
        cls,
        text: str,
        include_ear_checklist: bool,
        requested_checklists: list[str] | None = None,
    ) -> dict[str, Any]:

        checklist_keys = (
            requested_checklists
            if requested_checklists is not None
            else (
                ["ear_checklist"]
                if include_ear_checklist
                else []
            )
        )

        findings = cls._json_string_field(
            text,
            "findings",
        )

        if findings is not None:
            return cls._include_requested_checklists(
                {
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
                },
                checklist_keys,
                json_parsed=False,
                fallback_used=True,
            )

        return cls._include_requested_checklists(
            {
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
            },
            checklist_keys,
            json_parsed=False,
            fallback_used=True,
        )

    @classmethod
    def _parse_json(
        cls,
        text: str,
        include_ear_checklist: bool = False,
        requested_checklists: list[str] | None = None,
    ) -> dict[str, Any]:

        checklist_keys = (
            requested_checklists
            if requested_checklists is not None
            else (
                ["ear_checklist"]
                if include_ear_checklist
                else []
            )
        )

        if not text:
            raise ValueError(
                "MedGemma returned an empty response."
            )

        original_text = text.strip()

        if checklist_keys:
            print(
                "[MEDGEMMA OUTPUT] "
                f"structured=true chars={len(original_text)}",
                flush=True,
            )
        else:
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
            ]

            is_non_medical = any(
                phrase in lower_text
                for phrase in non_medical_phrases
            )

            looks_like_json = (
                start != -1
                or end != -1
            )

            if (
                not is_non_medical
                and looks_like_json
            ):
                cls._log_structured_parse_failure(
                    checklist_keys,
                    "truncated_json",
                )

                return cls._parse_malformed_json(
                    cleaned,
                    include_ear_checklist,
                    checklist_keys,
                )

            sectioned_data = cls._parse_sectioned_plain_text(
                cleaned,
                include_ear_checklist,
                checklist_keys,
            )

            if sectioned_data is not None:
                return sectioned_data

            if not is_non_medical:
                cls._log_structured_parse_failure(
                    checklist_keys,
                    "unstructured_output",
                )

                return cls._parse_malformed_json(
                    cleaned,
                    include_ear_checklist,
                    checklist_keys,
                )

            return cls._include_requested_checklists(
                {
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
                },
                checklist_keys,
                json_parsed=False,
                fallback_used=True,
            )

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

        result = {
            "findings":
                str(
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

        return cls._include_requested_checklists(
            result,
            checklist_keys,
            data,
            json_parsed=True,
            fallback_used=False,
        )

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

        requested_checklists = self._requested_checklists(
            prompt
        )

        include_ear_checklist = (
            "ear_checklist"
            in requested_checklists
        )

        generation_max_tokens = max_tokens

        if (
            requested_checklists
            and not include_ear_checklist
        ):
            generation_max_tokens = max(
                max_tokens,
                768,
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

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=generation_max_tokens,
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
                text,
                include_ear_checklist=include_ear_checklist,
                requested_checklists=requested_checklists,
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
            consensus = {
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

            for checklist_key in OPTIONAL_CHECKLIST_KEYS:
                if checklist_key in medgemma:
                    consensus[
                        checklist_key
                    ] = medgemma[
                        checklist_key
                    ]

            return consensus

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

        consensus = {
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

        for checklist_key in OPTIONAL_CHECKLIST_KEYS:
            if checklist_key in medgemma:
                consensus[
                    checklist_key
                ] = medgemma[
                    checklist_key
                ]

        return consensus


model_manager = MedicalModelManager()
