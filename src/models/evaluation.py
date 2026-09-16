"""Zero-shot evaluation for MTIL datasets."""

import torch
import torch.nn.functional as F
from tqdm import tqdm

import clip

from .. import datasets


@torch.no_grad()
def _classifier(classnames, templates, model, device):
    templates = templates if isinstance(templates, list) else [templates]
    weights = []
    for classname in classnames:
        tokens = clip.tokenize([template(classname) for template in templates]).to(device)
        features = F.normalize(model(None, tokens), dim=-1)
        weights.append(F.normalize(features.mean(dim=0), dim=0))
    return torch.stack(weights, dim=1)


@torch.no_grad()
def _evaluate_dataset(model, dataset, device):
    classifier = _classifier(dataset.classnames, dataset.templates, model, device)
    correct = 0
    total = 0
    for images, labels in tqdm(dataset.test_loader, desc=dataset.name):
        images = images.to(device)
        labels = labels.to(device)
        features = F.normalize(model(images, None), dim=-1)
        predictions = (100.0 * features @ classifier).argmax(dim=1)
        correct += int((predictions == labels).sum())
        total += labels.numel()
    return 100.0 * correct / total


def evaluate(model, args, val_preprocess):
    if not args.eval_datasets:
        return {}
    model.eval()
    results = {}
    for dataset_name in args.eval_datasets:
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
            num_workers=args.num_workers,
        )
        accuracy = _evaluate_dataset(model, dataset, args.device)
        results[dataset_name] = accuracy
        print(f"Top-1 accuracy on {dataset_name}: {accuracy:.2f}")
    print(f"Average accuracy: {sum(results.values()) / len(results):.2f}")
    return results
