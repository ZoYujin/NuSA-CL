"""Continual full fine-tuning baseline."""


class FullFineTuning:
    name = "full_ft"
    uses_adapters = False

    def build(self, model, args):
        return []

    def initialize(self, layers, args) -> None:
        return None

    def set_trainable(self, model) -> None:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name != "logit_scale"
