"""MiLoRA: initialize trainable LoRA factors from minor singular components."""

from __future__ import annotations

import torch

from .lora import LoRA, projection_names


class MiLoRA(LoRA):
    name = "milora"

    @torch.no_grad()
    def initialize(self, layers, args) -> None:
        for layer in layers:
            for proj_name in projection_names(layer, args):
                projection = getattr(layer, proj_name)
                weight = projection.weight.detach().float()
                u, singular_values, vh = torch.linalg.svd(
                    weight, full_matrices=False
                )
                rank = min(projection.r, singular_values.numel())
                u_tail = u[:, -rank:]
                s_tail = singular_values[-rank:]
                vh_tail = vh[-rank:, :]
                sqrt_s = torch.sqrt(s_tail)

                projection.w_lora_A.zero_()
                projection.w_lora_B.zero_()
                projection.w_lora_A[:rank].copy_(
                    (sqrt_s[:, None] * vh_tail).to(projection.w_lora_A.dtype)
                )
                projection.w_lora_B[:, :rank].copy_(
                    (u_tail * sqrt_s[None, :]).to(projection.w_lora_B.dtype)
                )
