"""Standard LoRA for CLIP attention layers."""

from __future__ import annotations

from typing import Iterable

import torch.nn as nn

from ..loralib.layers import LinearLoRA, PlainMultiheadAttentionLoRA


_TEXT_POSITIONS = {
    "bottom": range(0, 4),
    "mid": range(4, 8),
    "up": range(8, 12),
    "half-up": range(6, 12),
    "half-bottom": range(0, 6),
    "top3": range(9, 12),
    "all": range(0, 12),
}

_VISION_DEPTHS = {
    "ViT-B/16": 12,
    "ViT-B/32": 12,
    "ViT-L/14": 24,
}

_PROJECTIONS = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "o": "proj",
}


def _vision_positions(backbone: str, position: str) -> Iterable[int]:
    depth = _VISION_DEPTHS.get(backbone)
    if depth is None:
        raise ValueError(f"Unsupported CLIP backbone: {backbone}")
    if position == "all":
        return range(depth)
    if position == "top3":
        return range(depth - 3, depth)
    if position == "bottom":
        return range(0, 4)
    if position == "mid":
        start = (depth - 4) // 2
        return range(start, start + 4)
    if position == "up":
        return range(depth - 4, depth)
    if position == "half-up":
        return range(depth // 2, depth)
    if position == "half-bottom":
        return range(0, depth // 2)
    raise ValueError(f"Unsupported adapter position: {position}")


def projection_names(layer: PlainMultiheadAttentionLoRA, args) -> list[str]:
    if layer.encoder_type == "text":
        enabled = args.params_text or args.params
    elif layer.encoder_type == "vision":
        enabled = args.params_vision or args.params
    else:
        raise ValueError("LoRA layer is missing its encoder_type tag")
    return [_PROJECTIONS[name] for name in enabled]


def attach_adapters(model, args, *, use_m: bool) -> list[PlainMultiheadAttentionLoRA]:
    """Replace selected CLIP attention modules with LoRA-aware modules."""
    layers: list[PlainMultiheadAttentionLoRA] = []

    if args.encoder in {"text", "both"}:
        indices = set(_TEXT_POSITIONS[args.position])
        rank = args.r_text or args.rank
        enabled = args.params_text or args.params
        for index, block in enumerate(model.transformer.resblocks):
            if index not in indices:
                continue
            for name, module in block.named_children():
                if isinstance(module, nn.MultiheadAttention):
                    wrapped = PlainMultiheadAttentionLoRA(
                        module,
                        enable_lora=enabled,
                        r=rank,
                        lora_alpha=args.alpha,
                        dropout_rate=args.dropout,
                        lora_m=use_m,
                    )
                    wrapped.encoder_type = "text"
                    wrapped.layer_idx = index
                    setattr(block, name, wrapped)
                    layers.append(wrapped)

    if args.encoder in {"vision", "both"}:
        indices = set(_vision_positions(args.backbone, args.position))
        rank = args.r_vision or args.rank
        enabled = args.params_vision or args.params
        for index, block in enumerate(model.visual.transformer.resblocks):
            if index not in indices:
                continue
            for name, module in block.named_children():
                if isinstance(module, nn.MultiheadAttention):
                    wrapped = PlainMultiheadAttentionLoRA(
                        module,
                        enable_lora=enabled,
                        r=rank,
                        lora_alpha=args.alpha,
                        dropout_rate=args.dropout,
                        lora_m=use_m,
                    )
                    wrapped.encoder_type = "vision"
                    wrapped.layer_idx = index
                    setattr(block, name, wrapped)
                    layers.append(wrapped)

    if not layers:
        raise RuntimeError("No CLIP attention layers were selected for LoRA")
    return layers


def collect_adapters(model) -> list[PlainMultiheadAttentionLoRA]:
    return [
        module
        for module in model.modules()
        if isinstance(module, PlainMultiheadAttentionLoRA)
    ]


def mark_trainable(model, *, train_a: bool, train_b: bool, train_m: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for name, parameter in model.named_parameters():
        parameter.requires_grad = (
            (train_a and "lora_A" in name)
            or (train_b and "lora_B" in name)
            or (train_m and "lora_M" in name)
        )


def reset_merge_state(layers: list[PlainMultiheadAttentionLoRA]) -> None:
    """A saved checkpoint already contains merged weights and zero adapters."""
    for layer in layers:
        for proj_name in projection_names(layer, _ArgsProxy(layer)):
            projection = getattr(layer, proj_name)
            if isinstance(projection, LinearLoRA):
                projection.merged = False


class _ArgsProxy:
    """Expose the projections present in a loaded wrapper to projection_names."""

    params = ("q", "k", "v", "o")
    params_text = None
    params_vision = None

    def __init__(self, _layer):
        pass


class LoRA:
    name = "lora"
    use_m = False

    def build(self, model, args):
        return attach_adapters(model, args, use_m=False)

    def initialize(self, layers, args) -> None:
        # LinearLoRA already uses Kaiming A and zero B initialization.
        return None

    def set_trainable(self, model) -> None:
        mark_trainable(model, train_a=True, train_b=True, train_m=False)
