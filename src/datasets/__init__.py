"""Datasets used by the MTIL benchmark."""

from .collections import (
    Aircraft,
    Caltech101,
    CIFAR10,
    CIFAR100,
    DTD,
    EuroSAT,
    Flowers,
    Food,
    MNIST,
    OxfordPet,
    StanfordCars,
    SUN397,
)


DATASET_NAMES = (
    "Aircraft",
    "Caltech101",
    "CIFAR100",
    "DTD",
    "EuroSAT",
    "Flowers",
    "Food",
    "MNIST",
    "OxfordPet",
    "StanfordCars",
    "SUN397",
)


__all__ = ["CIFAR10", *DATASET_NAMES]
