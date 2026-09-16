"""Shared MTIL training loop for the methods in this release."""

from __future__ import annotations

import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import clip.clip as clip

from . import datasets, templates, utils
from .loralib.layers import LinearLoRA
from .methods import get_method
from .methods.lora import collect_adapters
from .models.evaluation import evaluate


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _torch_load(path: str, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        return torch.load(path, map_location=device)


def _load_model(args, method):
    if args.load:
        print(f"Loading previous task checkpoint: {args.load}")
        model = _torch_load(args.load, args.device)
        _, train_preprocess, val_preprocess = clip.load(args.model, jit=False)
        layers = collect_adapters(model)
        if method.uses_adapters and not layers:
            raise RuntimeError("The checkpoint does not contain LoRA adapters")
        if not method.uses_adapters and layers:
            raise RuntimeError("A LoRA checkpoint cannot be resumed as full fine-tuning")

        # Our checkpoints contain already-merged base weights and reset adapters.
        for layer in layers:
            for projection in (layer.q_proj, layer.k_proj, layer.v_proj, layer.proj):
                if isinstance(projection, LinearLoRA):
                    projection.merged = False
    else:
        model, train_preprocess, val_preprocess = clip.load(args.model, jit=False)
        layers = method.build(model, args)

    return model.to(args.device), layers, train_preprocess, val_preprocess


def _few_shot_loader(train_loader, shots: int, batch_size: int) -> DataLoader:
    examples: dict[int, list[torch.Tensor]] = {}
    for images, labels in train_loader:
        for image, label in zip(images, labels):
            class_id = int(label)
            bucket = examples.setdefault(class_id, [])
            if len(bucket) < shots:
                bucket.append(image)

    images = []
    labels = []
    for class_id in sorted(examples):
        images.extend(examples[class_id])
        labels.extend([class_id] * len(examples[class_id]))
    if not images:
        raise RuntimeError("The few-shot sampler found no examples")
    return DataLoader(
        TensorDataset(torch.stack(images), torch.tensor(labels)),
        batch_size=batch_size,
        shuffle=True,
    )


def _merge_adapters(model) -> None:
    for module in model.modules():
        if isinstance(module, LinearLoRA) and not module.merged:
            module.merge_lora_param()
            module.merged = True
            module.init_lora_param()


def train(args):
    set_random_seed(args.seed)
    method = get_method(args.method)
    model, layers, train_preprocess, val_preprocess = _load_model(args, method)

    if method.uses_adapters:
        has_m = all(
            hasattr(projection, "w_lora_M")
            for layer in layers
            for projection in (layer.q_proj, layer.k_proj, layer.v_proj, layer.proj)
            if isinstance(projection, LinearLoRA)
        )
        expects_m = args.method == "nusa"
        if has_m != expects_m:
            raise ValueError(
                f"Checkpoint adapter type does not match --method {args.method}"
            )

    method.initialize(layers, args)
    method.set_trainable(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        num_workers=args.num_workers,
    )
    train_loader = dataset.train_loader
    if args.few_shot is not None:
        train_loader = _few_shot_loader(
            dataset.train_loader, args.few_shot, args.batch_size
        )

    template = (
        getattr(templates, args.template)[0] if args.template else dataset.template
    )
    texts = clip.tokenize([template(name) for name in dataset.classnames]).to(args.device)

    if hasattr(method, "before_train"):
        method.before_train(model, layers, args, train_loader, texts)
    print(f"Method: {method.name}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    total_steps = (
        args.epochs * len(train_loader)
        if args.epochs is not None
        else args.iterations
    )
    if total_steps is None or total_steps <= 0:
        raise ValueError("Specify a positive --iterations or --epochs value")
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps is not None
        else round(total_steps * args.warmup_ratio)
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, args.beta2),
    )
    scheduler = utils.cosine_lr(optimizer, args.lr, warmup_steps, total_steps)

    started = time.time()
    data_iterator = iter(train_loader)
    for step in tqdm(range(total_steps), desc=f"Training {args.method}"):
        try:
            images, labels = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            images, labels = next(data_iterator)

        model.train()
        scheduler(step)
        images = images.to(args.device)
        labels = labels.to(args.device)
        text_features = F.normalize(model(None, texts), dim=-1)
        image_features = F.normalize(model(images, None), dim=-1)
        logits = model.logit_scale.exp() * image_features @ text_features.t()
        loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % args.log_interval == 0 or step == 0:
            accuracy = (logits.argmax(dim=-1) == labels).float().mean().item()
            print(
                f"step={step + 1}/{total_steps} loss={loss.item():.4f} "
                f"top1={accuracy * 100:.2f}%"
            )

    print(f"Training time: {(time.time() - started) / 60:.2f} minutes")
    if hasattr(method, "after_train"):
        method.after_train(model, layers, args, train_loader, texts)
    if method.uses_adapters:
        _merge_adapters(model)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        checkpoint_path = os.path.join(args.save, f"{args.train_dataset}.pth")
        torch.save(model, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    evaluate(model, args, val_preprocess)
    return model
