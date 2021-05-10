"""regularization between the model and an earlier version of the model"""

from typing import Tuple, Iterable, Optional
import re
import torch
from allennlp.models.archival import load_archive
from allennlp.common.params import Params
from allennlp.nn.regularizers.regularizer_applicator import RegularizerApplicator


class GbiRegularizerApplicator:
    def __init__(self,
                 original_model, regularizers):
        self.original_model = original_model
        self._regularizers = regularizers

    def __call__(self, module: torch.nn.Module):
        accumulator = 0.0
        shared_named_parameters = [
            named_parameter for named_parameter in module.state_dict().keys()
            if named_parameter in self.original_model.state_dict().keys()
        ]
        for parameter_name in shared_named_parameters:
            p1 = module.state_dict()[parameter_name]
            p2 = self.original_model.state_dict()[parameter_name]
            if p1.requires_grad:
                for regex, regularizer in self._regularizers:
                    if re.search(regex, parameter_name):
                        accumulator += regularizer(p1 - p2)  # <- this is the main change!
                        break
        return accumulator

    @classmethod
    def from_params(cls, params: Iterable[Tuple[str, Params]] = ()) -> Optional['GbiRegularizerApplicator']:
        if not params:
            return None
        for i, (name, param) in enumerate(params):
            if name == "original_model":
                break
        else:
            raise Exception("original model needs to be specified")
        _, model_params = params.pop(i)
        model_orig = load_archive(model_params["archive_file"]).model
        regularizers = RegularizerApplicator.from_params(params)._regularizers
        return cls(model_orig, regularizers)