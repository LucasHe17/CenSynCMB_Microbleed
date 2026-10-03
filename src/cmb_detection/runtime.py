from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet, BasicUNet, DynUNet, SegResNet, SegResNetDS, SwinUNETR, UNETR

try:
    from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
except ImportError:  # pragma: no cover - optional dependency
    MedNeXt = None


@dataclass
class RuntimeConfig:
    backbone: str = "attentionunet"
    aux_heads: str = "c1"
    train_crop_size: tuple[int, int, int] = (96, 96, 96)
    roi_size: tuple[int, int, int] = (128, 128, 128)
    sw_batch_size: int = 1
    sw_overlap: float = 0.6
    amp: bool = True


def tuple3(value: object, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if value is None:
        return default
    if isinstance(value, str):
        parts = [int(part.strip()) for part in value.replace("(", "").replace(")", "").split(",") if part.strip()]
    else:
        parts = [int(part) for part in value]  # type: ignore[arg-type]
    if len(parts) != 3:
        return default
    return tuple(parts)  # type: ignore[return-value]


def enabled_aux_heads(config: RuntimeConfig) -> set[str]:
    value = config.aux_heads.lower().replace("-", "_")
    if value in {"", "h0", "none"}:
        return set()
    if value in {"c1", "centre", "center"}:
        return {"center"}
    if value in {"d1", "distance"}:
        return {"distance"}
    if value in {"dc", "distance_center", "distance+center", "distance,center", "distance_centre"}:
        return {"distance", "center"}
    if value == "h1":
        return {"vessel"}
    if value == "h2":
        return {"mars"}
    if value == "h3":
        return {"vessel", "mars"}
    return {part.strip() for part in value.replace("+", ",").split(",") if part.strip()}


def output_channels_for_config(config: RuntimeConfig) -> int:
    heads = enabled_aux_heads(config)
    return (
        2
        + (1 if "vessel" in heads else 0)
        + (4 if "mars" in heads else 0)
        + (1 if "distance" in heads else 0)
        + (1 if "center" in heads else 0)
    )


def build_dynunet(out_channels: int = 2) -> DynUNet:
    strides = [1, 2, 2, 2, 2]
    kernels = [3, 3, 3, 3, 3]
    return DynUNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=out_channels,
        strides=strides,
        kernel_size=kernels,
        upsample_kernel_size=strides[1:],
        norm_name="INSTANCE",
        res_block=True,
        deep_supervision=True,
        deep_supr_num=3,
    )


def build_model(config: RuntimeConfig) -> nn.Module:
    backbone = config.backbone.lower().replace("-", "_")
    out_channels = output_channels_for_config(config)
    if backbone == "dynunet":
        return build_dynunet(out_channels)
    if backbone in {"attentionunet", "attention_unet"}:
        return AttentionUnet(
            spatial_dims=3,
            in_channels=3,
            out_channels=out_channels,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            dropout=0.1,
        )
    if backbone in {"swinunetr", "swinunetr_v2"}:
        return SwinUNETR(
            spatial_dims=3,
            in_channels=3,
            out_channels=out_channels,
            feature_size=48,
            use_checkpoint=True,
            use_v2=True,
        )
    if backbone == "unetr":
        return UNETR(
            spatial_dims=3,
            in_channels=3,
            out_channels=out_channels,
            img_size=config.train_crop_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            res_block=True,
        )
    if backbone == "segresnet":
        return SegResNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=out_channels,
            init_filters=32,
            dropout_prob=0.1,
        )
    if backbone in {"segresnet_ds", "segresnetds"}:
        return SegResNetDS(
            spatial_dims=3,
            in_channels=3,
            out_channels=out_channels,
            init_filters=32,
            norm="instance",
            blocks_down=(1, 2, 2, 4),
            dsdepth=3,
            upsample_mode="deconv",
        )
    if backbone == "basicunet":
        return BasicUNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=out_channels,
            features=(32, 32, 64, 128, 256, 32),
        )
    if backbone == "mednext":
        if MedNeXt is None:
            raise ImportError("MedNeXt is not installed in this environment.")
        return MedNeXt(
            in_channels=3,
            n_channels=32,
            n_classes=out_channels,
            exp_r=2,
            kernel_size=3,
            deep_supervision=True,
            do_res=True,
            do_res_up_down=True,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        )
    raise ValueError(f"Unknown backbone: {config.backbone}")


def unpack_outputs(outputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    if isinstance(outputs, (list, tuple)):
        return list(outputs)
    if not torch.is_tensor(outputs):
        raise RuntimeError(f"Unexpected output type: {type(outputs)}")
    if outputs.dim() == 5:
        return [outputs]
    if outputs.dim() == 6:
        return [item for item in outputs.unbind(dim=1)]
    raise RuntimeError(f"Unexpected output shape: {tuple(outputs.shape)}")


def pick_main_output(outputs: list[torch.Tensor]) -> torch.Tensor:
    sizes = [int(np.prod(output.shape[2:])) for output in outputs]
    return outputs[int(np.argmax(sizes))]


def segmentation_logits(output: torch.Tensor) -> torch.Tensor:
    return output[:, :2]


def runtime_config_from_checkpoint(checkpoint: dict[str, object]) -> RuntimeConfig:
    raw = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    raw = raw if isinstance(raw, dict) else {}
    config = RuntimeConfig()
    config.backbone = str(raw.get("backbone", config.backbone))
    config.aux_heads = str(raw.get("aux_heads", config.aux_heads))
    config.train_crop_size = tuple3(raw.get("train_crop_size"), config.train_crop_size)
    config.roi_size = tuple3(raw.get("roi_size"), config.roi_size)
    config.sw_batch_size = int(raw.get("sw_batch_size", config.sw_batch_size))
    config.sw_overlap = float(raw.get("sw_overlap", config.sw_overlap))
    config.amp = bool(raw.get("amp", config.amp))
    return config
