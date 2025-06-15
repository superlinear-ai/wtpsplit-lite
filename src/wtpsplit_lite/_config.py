"""Shim for transformers.AutoConfig."""

import json
from functools import cache
from pathlib import Path

from huggingface_hub import hf_hub_download


class SubwordXLMConfig:
    """Config for XLM-R. Used for token-level training, i.e., SaT models.

    Args:
        XLMRobertaConfig: Base class.
    """

    model_type = "xlm-token"
    mixture_name = "xlm-token"

    def __init__(
        self,
        lookahead=None,
        lookahead_split_layers=None,
        **model_config,
    ):
        self.mixture_name = "xlm-token"
        self.lookahead = lookahead
        self.lookahead_split_layers = lookahead_split_layers
        self.num_hash_buckets = 8192
        self.num_hash_functions = 8
        for key, value in model_config.items():
            setattr(self, key, value)

    @property
    def num_labels(self) -> int:
        """The number of labels for classification models."""
        return len(getattr(self, "id2label", {}))

    @classmethod
    @cache
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path) -> "SubwordXLMConfig":
        if (
            isinstance(pretrained_model_name_or_path, str)
            and not Path(pretrained_model_name_or_path).exists()
        ):
            config_json = Path(hf_hub_download(pretrained_model_name_or_path, "config.json"))
        elif Path(pretrained_model_name_or_path).is_file():
            config_json = Path(pretrained_model_name_or_path)
        else:
            config_json = Path(pretrained_model_name_or_path) / "config.json"
        with config_json.open("r") as f:
            model_config = json.load(f)
        return cls(**model_config)
