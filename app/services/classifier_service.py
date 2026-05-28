from __future__ import annotations

import time
from pathlib import Path

import torch
import torchvision.models as tv_models
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from app.core.config import MODEL_SPECS, OXFORD_PET_LABELS, TRAINED_MODELS_DIR, ModelSpec

try:
    from safetensors.torch import load_file as safetensors_load_file
except Exception:
    safetensors_load_file = None


class LoadedModel:
    def __init__(self, spec: ModelSpec, model: torch.nn.Module, class_names: list[str], transform: transforms.Compose):
        self.spec = spec
        self.model = model
        self.class_names = class_names
        self.transform = transform


class PetClassifierService:
    def __init__(self):
        self._cache: dict[str, LoadedModel] = {}
        self._imagenet_labels = self._load_imagenet_labels()

    @staticmethod
    def _load_imagenet_labels() -> list[str]:
        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
        return list(weights.meta["categories"])

    def available_models(self) -> list[str]:
        return list(MODEL_SPECS.keys())

    def _weights_path(self, spec: ModelSpec) -> Path:
        if spec.source.startswith("local://"):
            source_path = spec.source.removeprefix("local://")
            path = Path(source_path)
            if not path.is_absolute():
                path = TRAINED_MODELS_DIR / source_path
            return path

        suffix = ".safetensors" if spec.architecture == "vit_b_32" else ".pth"
        return TRAINED_MODELS_DIR / f"{spec.key}{suffix}"

    def _download_if_needed(self, spec: ModelSpec) -> Path | None:
        target_path = self._weights_path(spec)
        if target_path.exists():
            return target_path

        if spec.source.startswith("local://"):
            return None

        if not spec.source.startswith("hf://"):
            return None

        source = spec.source.removeprefix("hf://")
        if "#" not in source:
            return None

        repo_id, filename = source.split("#", 1)
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(TRAINED_MODELS_DIR))
        downloaded_path = Path(downloaded)
        if downloaded_path.resolve() != target_path.resolve():
            downloaded_path.replace(target_path)
        return target_path

    @staticmethod
    def _extract_state_dict(checkpoint: dict | torch.Tensor) -> dict:
        if isinstance(checkpoint, dict):
            return checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
        return checkpoint

    @staticmethod
    def _load_checkpoint_compat(ckpt_path: Path):
        if ckpt_path.suffix.lower() == ".safetensors":
            if safetensors_load_file is None:
                raise RuntimeError(
                    "El modelo ViT-B/32 requiere 'safetensors'. Instala dependencias con: pip install -r requirements.txt"
                )
            return safetensors_load_file(str(ckpt_path), device="cpu")

        try:
            return torch.load(ckpt_path, map_location="cpu")
        except Exception:
            return torch.load(ckpt_path, map_location="cpu", weights_only=False)

    @staticmethod
    def _infer_resnet_classes(state_dict: dict, fallback: int) -> int:
        if "fc.weight" in state_dict:
            return int(state_dict["fc.weight"].shape[0])
        if "fc.1.weight" in state_dict:
            return int(state_dict["fc.1.weight"].shape[0])
        return fallback

    @staticmethod
    def _infer_vit_classes(state_dict: dict, fallback: int) -> int:
        if "heads.head.weight" in state_dict:
            return int(state_dict["heads.head.weight"].shape[0])
        if "head.weight" in state_dict:
            return int(state_dict["head.weight"].shape[0])
        return fallback

    @staticmethod
    def _build_resnet_model(num_classes: int, use_dropout_head: bool) -> torch.nn.Module:
        model = tv_models.resnet50(weights=None)
        in_features = model.fc.in_features
        if use_dropout_head:
            model.fc = torch.nn.Sequential(torch.nn.Dropout(p=0.3), torch.nn.Linear(in_features, num_classes))
        else:
            model.fc = torch.nn.Linear(in_features, num_classes)
        return model

    @staticmethod
    def _build_vit_b32_model(num_classes: int) -> torch.nn.Module:
        model = tv_models.vit_b_32(weights=None)
        in_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(in_features, num_classes)
        return model

    def _build_model(self, spec: ModelSpec) -> tuple[torch.nn.Module, list[str]]:
        if spec.architecture == "resnet50":
            ckpt_path = self._download_if_needed(spec)
            if ckpt_path is None or not ckpt_path.exists():
                raise RuntimeError(f"Checkpoint not found for model '{spec.key}'. Expected: {self._weights_path(spec)}")

            checkpoint = self._load_checkpoint_compat(ckpt_path)
            state_dict = self._extract_state_dict(checkpoint)
            class_count = self._infer_resnet_classes(state_dict, spec.num_classes)

            model = self._build_resnet_model(class_count, use_dropout_head=False)
            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception:
                model = self._build_resnet_model(class_count, use_dropout_head=True)
                model.load_state_dict(state_dict, strict=False)

            class_names = checkpoint.get("classes", OXFORD_PET_LABELS) if isinstance(checkpoint, dict) else OXFORD_PET_LABELS
            return model.eval(), class_names

        if spec.architecture == "vit_b_32":
            ckpt_path = self._download_if_needed(spec)
            if ckpt_path is None or not ckpt_path.exists():
                raise RuntimeError(f"Checkpoint not found for model '{spec.key}'. Expected: {self._weights_path(spec)}")

            checkpoint = self._load_checkpoint_compat(ckpt_path)
            state_dict = self._extract_state_dict(checkpoint)
            class_count = self._infer_vit_classes(state_dict, spec.num_classes)

            model = self._build_vit_b32_model(class_count)
            model.load_state_dict(state_dict, strict=False)
            class_names = checkpoint.get("classes", OXFORD_PET_LABELS) if isinstance(checkpoint, dict) else OXFORD_PET_LABELS
            return model.eval(), class_names

        if spec.architecture == "efficientnet_b0":
            model = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            return model.eval(), self._imagenet_labels

        if spec.architecture == "mobilenet_v3_large":
            model = tv_models.mobilenet_v3_large(weights=tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
            return model.eval(), self._imagenet_labels

        raise RuntimeError(f"Unsupported architecture: {spec.architecture}")

    def _transform_for(self, spec: ModelSpec) -> transforms.Compose:
        if spec.architecture == "mobilenet_v3_large":
            return tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2.transforms()
        if spec.architecture == "vit_b_32":
            return tv_models.ViT_B_32_Weights.IMAGENET1K_V1.transforms()
        return tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()

    def _get_loaded_model(self, model_key: str) -> LoadedModel:
        if model_key in self._cache:
            return self._cache[model_key]

        if model_key not in MODEL_SPECS:
            raise ValueError(f"Unsupported model '{model_key}'. Available: {', '.join(MODEL_SPECS)}")

        spec = MODEL_SPECS[model_key]
        model, class_names = self._build_model(spec)
        loaded = LoadedModel(spec=spec, model=model, class_names=class_names, transform=self._transform_for(spec))
        self._cache[model_key] = loaded
        return loaded

    def predict(self, image: Image.Image, model_key: str) -> dict:
        loaded = self._get_loaded_model(model_key)
        input_tensor = loaded.transform(image.convert("RGB")).unsqueeze(0)

        start = time.perf_counter()
        with torch.no_grad():
            logits = loaded.model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0)
        elapsed = int((time.perf_counter() - start) * 1000)

        top_probs, top_idxs = torch.topk(probs, k=min(5, probs.shape[0]))
        top5 = []
        for p, idx in zip(top_probs.tolist(), top_idxs.tolist()):
            label = loaded.class_names[idx] if idx < len(loaded.class_names) else f"class_{idx}"
            top5.append({"label": label, "confidence": round(float(p), 4)})

        return {
            "breed": top5[0]["label"],
            "probability": top5[0]["confidence"],
            "top5": top5,
            "inference_time_ms": elapsed,
            "model_used": loaded.spec.display_name,
        }


_classifier_service: PetClassifierService | None = None


def get_classifier_service() -> PetClassifierService:
    global _classifier_service
    if _classifier_service is None:
        _classifier_service = PetClassifierService()
    return _classifier_service
