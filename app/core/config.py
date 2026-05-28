from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Oxford-IIIT Pet 37 breed labels
OXFORD_PET_LABELS = [
    "Abyssinian", "american_bulldog", "american_pit_bull_terrier", "basset_hound",
    "beagle", "Bengal", "Birman", "Bombay", "boxer", "British_Shorthair",
    "chihuahua", "Egyptian_Mau", "english_cocker_spaniel", "english_setter",
    "german_shorthaired", "great_pyrenees", "havanese", "japanese_chin", "keeshond",
    "leonberger", "Maine_Coon", "miniature_pinscher", "newfoundland", "Persian",
    "pomeranian", "pug", "Ragdoll", "Russian_Blue", "saint_bernard", "samoyed",
    "scottish_terrier", "shiba_inu", "Siamese", "Sphynx", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier",
]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    architecture: str
    source: str
    num_classes: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "efficientnet_b0": ModelSpec(
        key="efficientnet_b0",
        display_name="EfficientNet-B0",
        architecture="efficientnet_b0",
        source=os.getenv("PET_MODEL_SOURCE_EFFICIENTNET_B0", "hf://google/efficientnet-b0"),
        num_classes=1000,
    ),
    "resnet50": ModelSpec(
        key="resnet50",
        display_name="ResNet50",
        architecture="resnet50",
        source=os.getenv("PET_MODEL_SOURCE_RESNET50", "hf://flaviodell/oxford-pets-resnet50#best_model.pth"),
        num_classes=37,
    ),
    "mobilenetv3": ModelSpec(
        key="mobilenetv3",
        display_name="MobileNetV3",
        architecture="mobilenet_v3_large",
        source=os.getenv("PET_MODEL_SOURCE_MOBILENETV3", "hf://timm/mobilenetv3_large_100.ra_in1k"),
        num_classes=1000,
    ),
    "tmp": ModelSpec(
        key="tmp",
        display_name="TMP (tmp.pth)",
        architecture="resnet50",
        source=os.getenv("PET_MODEL_SOURCE_TMP", "local://tmp.pth"),
        num_classes=37,
    ),
    "stage-2": ModelSpec(
        key="stage-2",
        display_name="stage-2 (stage-2.pth)",
        architecture="resnet50",
        source=os.getenv("PET_MODEL_SOURCE_STAGE_2", "local://stage-2.pth"),
        num_classes=37,
    ),
    "vit_b32": ModelSpec(
        key="vit_b32",
        display_name="ViT-B/32",
        architecture="vit_b_32",
        source=os.getenv("PET_MODEL_SOURCE_VIT_B32", "local://model.safetensors"),
        num_classes=37,
    ),
}
