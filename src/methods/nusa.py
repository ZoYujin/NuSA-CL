"""NuSA-CL: persistent adaptation inside an intrinsic null space."""

from __future__ import annotations

import torch

from .lora import attach_adapters, mark_trainable, projection_names


class NuSA:
    name = "nusa"
    use_m = True

    def build(self, model, args):
        return attach_adapters(model, args, use_m=True)

    @torch.no_grad()
    def initialize(self, layers, args) -> None:
        for layer in layers:
            for proj_name in projection_names(layer, args):
                projection = getattr(layer, proj_name)
                weight = projection.weight.detach().float()
                u, singular_values, vh = torch.linalg.svd(
                    weight, full_matrices=False
                )

                cumulative_energy = torch.cumsum(singular_values.square(), dim=0)
                energy_ratio = cumulative_energy / cumulative_energy[-1]
                principal_rank = int(
                    torch.searchsorted(energy_ratio, args.cutoff).item()
                ) + 1
                null_rank = singular_values.numel() - principal_rank
                effective_rank = min(null_rank, projection.r)

                projection.w_lora_A.zero_()
                projection.w_lora_B.zero_()
                projection.w_lora_M.zero_()
                if effective_rank == 0:
                    continue

                projection.w_lora_A[:effective_rank].copy_(
                    vh[-effective_rank:, :].to(projection.w_lora_A.dtype)
                )
                projection.w_lora_B[:, :effective_rank].copy_(
                    u[:, -effective_rank:].to(projection.w_lora_B.dtype)
                )

                print(
                    f"[{layer.encoder_type}:{layer.layer_idx}:{proj_name}] "
                    f"principal={principal_rank}, null={null_rank}, "
                    f"effective={effective_rank}"
                )

    def set_trainable(self, model) -> None:
        mark_trainable(model, train_a=False, train_b=False, train_m=True)
