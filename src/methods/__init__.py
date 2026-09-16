"""Methods included in the NuSA-CL release."""

from .full_ft import FullFineTuning
from .inflora import InfLoRA
from .lora import LoRA
from .milora import MiLoRA
from .nusa import NuSA


_METHODS = {
    "full_ft": FullFineTuning,
    "inflora": InfLoRA,
    "lora": LoRA,
    "milora": MiLoRA,
    "nusa": NuSA,
}


def get_method(name: str):
    try:
        method = _METHODS[name]()
    except KeyError as error:
        choices = ", ".join(sorted(_METHODS))
        raise ValueError(f"Unknown method '{name}'. Choose one of: {choices}") from error
    method.uses_adapters = name != "full_ft"
    return method


__all__ = ["FullFineTuning", "InfLoRA", "LoRA", "MiLoRA", "NuSA", "get_method"]
