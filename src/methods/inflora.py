"""InfLoRA-CLIP: GPM-initialized, B-only LoRA adaptation for CLIP."""

from __future__ import annotations

import math

import torch

from ..loralib.layers import LinearLoRA
from .lora import attach_adapters, mark_trainable, projection_names


class InfLoRA:
    """CLIP adaptation of InfLoRA under the shared Q/K/V/O protocol.

    K/V bases are initialized from the current-task feature subspace after
    projecting out the stored Gradient Projection Memory (GPM).  Q/O retain
    their frozen random bases, matching the legacy CLIP experiment.  Only B
    matrices are optimized.
    """

    name = "inflora"
    use_m = False

    def build(self, model, args):
        layers = attach_adapters(model, args, use_m=False)
        for layer in layers:
            layer.register_buffer(
                "inflora_basis",
                torch.empty(layer.embed_dim, 0, dtype=torch.float32),
                persistent=True,
            )
        return layers

    def initialize(self, layers, args) -> None:
        return None

    def set_trainable(self, model) -> None:
        mark_trainable(model, train_a=False, train_b=True, train_m=False)

    @staticmethod
    def _energy_rank(singular_values: torch.Tensor, threshold: float) -> int:
        energy = singular_values.square()
        total = energy.sum()
        if total <= 0:
            return 0
        return int((torch.cumsum(energy / total, dim=0) < threshold).sum().item())

    def _reset_statistics(self, layers) -> None:
        self._sums = {
            id(layer): torch.zeros(
                layer.embed_dim,
                layer.embed_dim,
                device=layer.inflora_basis.device,
                dtype=torch.float32,
            )
            for layer in layers
        }
        self._counts = {id(layer): 0 for layer in layers}

    def _install_capture_hooks(self, layers) -> None:
        self._hooks = []

        def make_hook(layer):
            def hook(_module, inputs):
                query = inputs[0].detach()
                if not layer.batch_first:
                    query = query.transpose(0, 1)
                features = query.reshape(-1, query.shape[-1]).float()
                self._sums[id(layer)].add_(features.T @ features)
                self._counts[id(layer)] += features.shape[0]

            return hook

        for layer in layers:
            self._hooks.append(layer.register_forward_pre_hook(make_hook(layer)))

    def _remove_capture_hooks(self) -> None:
        for handle in getattr(self, "_hooks", []):
            handle.remove()
        self._hooks = []

    @torch.no_grad()
    def _collect_task_features(self, model, train_loader, texts, device) -> None:
        # Keep training mode: the legacy implementation collected features in
        # this mode, including the configured LoRA dropout behavior.
        for images, _ in train_loader:
            model(images.to(device), texts)

    def _covariances(self, layers) -> dict[int, torch.Tensor]:
        covariances = {}
        for layer in layers:
            count = self._counts[id(layer)]
            if count == 0:
                raise RuntimeError("InfLoRA feature capture collected no activations")
            covariances[id(layer)] = self._sums[id(layer)] / count
        return covariances

    @torch.no_grad()
    def _initialize_kv_bases(self, layers, args, covariances) -> None:
        for layer in layers:
            covariance = covariances[id(layer)]
            basis = layer.inflora_basis.to(covariance.device)
            residual = covariance
            if basis.numel():
                residual = covariance - basis @ (basis.T @ covariance)
            u, _, _ = torch.linalg.svd(residual, full_matrices=False)

            for proj_name in projection_names(layer, args):
                projection = getattr(layer, proj_name)
                if isinstance(projection, LinearLoRA):
                    projection.w_lora_B.zero_()
                if proj_name not in {"k_proj", "v_proj"}:
                    continue
                if not isinstance(projection, LinearLoRA):
                    continue
                rank = min(projection.r, u.shape[1])
                projection.w_lora_A.zero_()
                projection.w_lora_A[:rank].copy_(
                    (u[:, :rank].T / math.sqrt(3)).to(projection.w_lora_A.dtype)
                )

    @torch.no_grad()
    def _update_gpm(self, layers, args, covariances) -> None:
        for layer in layers:
            covariance = covariances[id(layer)]
            basis = layer.inflora_basis.to(covariance.device)
            if basis.numel() == 0:
                u, singular_values, _ = torch.linalg.svd(
                    covariance, full_matrices=False
                )
                rank = self._energy_rank(singular_values, args.gpm_threshold)
                updated = u[:, :max(rank, 1)]
            else:
                _, before_singular_values, _ = torch.linalg.svd(
                    covariance, full_matrices=False
                )
                total_energy = before_singular_values.square().sum()
                residual = covariance - basis @ (basis.T @ covariance)
                u, singular_values, _ = torch.linalg.svd(
                    residual, full_matrices=False
                )
                retained_energy = 1.0 - singular_values.square().sum() / total_energy
                rank = 0
                for value in singular_values.square() / total_energy:
                    if retained_energy >= args.gpm_threshold:
                        break
                    retained_energy += value
                    rank += 1
                updated = basis if rank == 0 else torch.cat((basis, u[:, :rank]), dim=1)
                updated = updated[:, : updated.shape[0]]

            layer.inflora_basis = updated.detach().to(
                device=layer.inflora_basis.device, dtype=torch.float32
            )
            print(
                f"[{layer.encoder_type}:{layer.layer_idx}] "
                f"GPM rank={layer.inflora_basis.shape[1]}"
            )

    def before_train(self, model, layers, args, train_loader, texts) -> None:
        self._reset_statistics(layers)
        self._install_capture_hooks(layers)
        self._collect_task_features(model, train_loader, texts, args.device)
        self._initialize_kv_bases(layers, args, self._covariances(layers))

        # Preserve the legacy behavior: accumulate current-task activations
        # while training, then add one full pass before updating GPM.
        self._reset_statistics(layers)

    def after_train(self, model, layers, args, train_loader, texts) -> None:
        self._collect_task_features(model, train_loader, texts, args.device)
        covariances = self._covariances(layers)
        self._remove_capture_hooks()
        self._update_gpm(layers, args, covariances)
