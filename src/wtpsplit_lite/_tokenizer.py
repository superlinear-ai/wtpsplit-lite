"""Shim for transformers.AutoTokenizer."""

from functools import cache
from pathlib import Path
from typing import Any, TypedDict

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


class BatchEncoding(TypedDict):
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    offset_mapping: list[list[tuple[int, int]]]


class XLMRobertaTokenizerFast:
    """A fast transformers.XLMRobertaTokenizerFast interface for wtpsplit."""

    def __init__(  # noqa: PLR0913
        self,
        tokenizer: Tokenizer,
        bos_token: str = "<s>",  # noqa: S107
        eos_token: str = "</s>",  # noqa: S107
        sep_token: str = "</s>",  # noqa: S107
        cls_token: str = "<s>",  # noqa: S107
        unk_token: str = "<unk>",  # noqa: S107
        pad_token: str = "<pad>",  # noqa: S107
        mask_token: str = "<mask>",  # noqa: S107
    ):
        self.tokenizer = tokenizer
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.bos_token_id = self.tokenizer.token_to_id(bos_token)
        self.eos_token_id = self.tokenizer.token_to_id(eos_token)
        self.sep_token_id = self.tokenizer.token_to_id(sep_token)
        self.cls_token_id = self.tokenizer.token_to_id(cls_token)
        self.unk_token_id = self.tokenizer.token_to_id(unk_token)
        self.pad_token_id = self.tokenizer.token_to_id(pad_token)
        self.mask_token_id = self.tokenizer.token_to_id(mask_token)

    def __call__(
        self,
        texts: list[str],
        is_pretokenized: bool = False,  # noqa: FBT001, FBT002
        add_special_tokens: bool = True,  # noqa: FBT001, FBT002
        return_offsets_mapping: bool = True,  # noqa: FBT001, FBT002
        verbose: bool = False,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> BatchEncoding:
        encoded_batch = self.tokenizer.encode_batch(
            texts, is_pretokenized=is_pretokenized, add_special_tokens=add_special_tokens
        )
        return BatchEncoding(
            input_ids=[e.ids for e in encoded_batch],
            attention_mask=[e.attention_mask for e in encoded_batch],
            offset_mapping=[e.offsets for e in encoded_batch],
        )

    @classmethod
    @cache
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path
    ) -> "XLMRobertaTokenizerFast":
        if (
            isinstance(pretrained_model_name_or_path, str)
            and not Path(pretrained_model_name_or_path).exists()
        ):
            tokenizer_json = Path(hf_hub_download(pretrained_model_name_or_path, "tokenizer.json"))
        elif Path(pretrained_model_name_or_path).is_file():
            tokenizer_json = Path(pretrained_model_name_or_path)
        else:
            tokenizer_json = Path(pretrained_model_name_or_path) / "tokenizer.json"
        tokenizer = Tokenizer.from_file(tokenizer_json.as_posix())
        return cls(tokenizer)
