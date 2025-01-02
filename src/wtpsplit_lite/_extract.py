"""Minimal subset of wtpsplit.extract."""
# This file uses old-style type hints and ignores certain ruff rules to minimise changes w.r.t. the original implementation:
# ruff: noqa: C901, PLR0913, PLR0912, PLR0915, ERA001, W505, FBT001, FBT002, PLR2004, B905, TRY003, EM101

import logging
import math
import sys
from typing import Literal

import numpy as np
import onnxruntime as ort
from tqdm.auto import tqdm

from wtpsplit_lite._config import SubwordXLMConfig
from wtpsplit_lite._tokenizer import XLMRobertaTokenizerFast
from wtpsplit_lite._utils import Constants, hash_encode

logger = logging.getLogger(__name__)


class SaTORTWrapper:
    def __init__(self, config: SubwordXLMConfig, ort_session: ort.InferenceSession):
        self.config = config
        self.ort_session = ort_session

    def __getattr__(self, name):
        assert hasattr(self, "ort_session")
        return getattr(self.ort_session, name)

    def __call__(self, input_ids, attention_mask):
        logits = self.ort_session.run(
            ["logits"],
            {
                "attention_mask": attention_mask.astype(np.float16),
                "input_ids": input_ids.astype(np.int64),
            },
        )[0]

        return {"logits": logits}


def extract(
    batch_of_texts: list[str],
    model: SaTORTWrapper,
    stride: int,
    max_block_size: int,
    batch_size: int,
    lang_code: str | None = None,
    pad_last_batch: bool = False,
    weighting: Literal["uniform", "hat"] = "uniform",
    verbose: bool = False,
    tokenizer: XLMRobertaTokenizerFast | None = None,
):
    """Compute logits for the given batch of texts.

    Computes logits for the given batch of texts by:
        1. slicing the texts into chunks of size `block_size`.
        2. passing every chunk through the model forward.
        3. stitching predictings back together by averaging chunk logits.

    ad 1.: text is sliced into partially overlapping chunks by moving forward by a `stride` parameter (think conv1d).
    """
    if "xlm" in model.config.model_type:
        use_subwords = True
        if not tokenizer:
            tokenizer = XLMRobertaTokenizerFast.from_pretrained("facebookAI/xlm-roberta-base")
        # tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})
        tokens = tokenizer(
            batch_of_texts, return_offsets_mapping=True, verbose=False, add_special_tokens=False
        )
        # remove CLS and SEP tokens, they are added later anyhow
        # batch_of_texts = [text[1:-1] for text in tokens["input_ids"]]
        batch_of_texts = tokens["input_ids"]  # type: ignore[assignment]
        # offset_mapping = [offset[1:-1] for offset in tokens["offset_mapping"]]
        offset_mapping = tokens["offset_mapping"]
        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id
        pad_token_id = tokenizer.pad_token_id
    else:
        pad_token_id = 0
        use_subwords = False

    text_lengths = [len(text) for text in batch_of_texts]
    # reduce block size if possible
    block_size = min(max_block_size, max(text_lengths))
    if use_subwords and block_size > 510:
        overflow_length = block_size - 510
        block_size -= overflow_length  # account for CLS and SEP tokens

    # make sure block_size is a multiple of downsampling rate
    downsampling_rate = getattr(model.config, "downsampling_rate", 1)
    block_size = math.ceil(block_size / downsampling_rate) * downsampling_rate

    # total number of forward passes
    num_chunks = sum(math.ceil(max(length - block_size, 0) / stride) + 1 for length in text_lengths)

    # preallocate a buffer for all input hashes & attention masks
    if not use_subwords:
        input_hashes = np.zeros(
            (num_chunks, block_size, model.config.num_hash_functions), dtype=np.int64
        )
        attention_mask = np.zeros((num_chunks, block_size), dtype=np.float32)
    else:
        input_ids = np.zeros((num_chunks, block_size + 2), dtype=np.int64)
        attention_mask = np.zeros((num_chunks, block_size + 2), dtype=np.float32)

    # locs keep track of the location of every chunk with a 3-tuple (text_idx, char_start, char_end) that indexes
    # back into the batch_of_texts
    locs = np.zeros((num_chunks, 3), dtype=np.int32)

    if not use_subwords:
        # this is equivalent to (but faster than) np.array([ord(c) for c in "".join(batch_of_texts)])
        codec = "utf-32-le" if sys.byteorder == "little" else "utf-32-be"
        ordinals = np.frombuffer(bytearray("".join(batch_of_texts), encoding=codec), dtype=np.int32)
        # hash encode all ids
        flat_hashed_ids = hash_encode(
            ordinals,
            num_hashes=model.config.num_hash_functions,
            num_buckets=model.config.num_hash_buckets,
        )
    # note that ordinals and flat_hashed_ids have the same length
    offset = 0
    current_chunk = 0

    # create chunks
    for i in range(len(batch_of_texts)):
        for j in range(0, text_lengths[i], stride):
            # for every chunk, assign input hashes, attention mask and loc
            start, end = j, j + block_size
            done = False

            if end >= text_lengths[i]:
                end = text_lengths[i]
                start = max(end - block_size, 0)
                done = True

            if not use_subwords:
                input_hashes[current_chunk, : end - start] = flat_hashed_ids[
                    offset + start : offset + end
                ]
                attention_mask[current_chunk, : end - start] = 1
            else:
                chunk = [cls_token_id] + batch_of_texts[i][start:end] + [sep_token_id]  # type: ignore[operator]
                input_ids[current_chunk, : len(chunk)] = chunk
                attention_mask[current_chunk, : len(chunk)] = 1

            locs[current_chunk, :] = [i, start, end]
            current_chunk += 1

            if done:
                break

        offset += text_lengths[i]

    assert current_chunk == num_chunks
    n_batches = math.ceil(len(attention_mask) / batch_size)

    # containers for the final logits
    all_logits = [
        np.zeros(
            (length, model.config.num_labels),
            dtype=np.float16,
        )
        for length in text_lengths
    ]
    # container for the number of chunks that any character was part of (to average chunk predictions)
    all_counts = [np.zeros(length, dtype=np.float16) for length in text_lengths]

    uses_lang_adapters = getattr(model.config, "language_adapter", "off") == "on"
    if uses_lang_adapters:
        if lang_code is None:
            raise ValueError(
                "Please specify a `lang_code` when using a model with language adapters."
            )

        language_ids = np.array(
            [Constants.LANG_CODE_TO_INDEX[lang_code]] * batch_size,
            dtype=int,
        )
    else:
        language_ids = None

    # forward passes through all chunks
    for batch_idx in tqdm(range(n_batches), disable=not verbose):
        start, end = batch_idx * batch_size, min(len(attention_mask), (batch_idx + 1) * batch_size)

        if not use_subwords:
            batch_input_hashes = input_hashes[start:end]
        else:
            batch_input_ids = input_ids[start:end]
        batch_attention_mask = attention_mask[start:end]

        if len(batch_attention_mask) < batch_size and pad_last_batch:
            n_missing = batch_size - len(batch_attention_mask)

            if not use_subwords:
                batch_input_hashes = np.pad(batch_input_hashes, ((0, n_missing), (0, 0), (0, 0)))
            else:
                # Pad with the specific pad_token_id for the tokenizer
                batch_input_ids = np.pad(
                    batch_input_ids, ((0, n_missing), (0, 0)), constant_values=pad_token_id
                )
            batch_attention_mask = np.pad(batch_attention_mask, ((0, n_missing), (0, 0)))

        kwargs = (
            {"language_ids": language_ids[: len(batch_attention_mask)]}  # type: ignore[index]
            if uses_lang_adapters
            else {}
        )
        if use_subwords:
            kwargs["input_ids"] = batch_input_ids
        else:
            kwargs["hashed_ids"] = batch_input_hashes

        logits = model(
            attention_mask=batch_attention_mask,
            **kwargs,
        )["logits"]

        if use_subwords:
            logits = logits[:, 1:-1, :]  # remove CLS and SEP tokens

        if weighting == "uniform":
            weights = np.ones(block_size, dtype=np.float16)
        elif weighting == "hat":
            x = np.linspace(-(1 - 1 / block_size), 1 - 1 / block_size, block_size, dtype=np.float16)
            weights = 1 - np.abs(x)  # type: ignore[assignment]
        for i in range(start, end):
            original_idx, start_char_idx, end_char_idx = locs[i]
            n = end_char_idx - start_char_idx
            all_logits[original_idx][start_char_idx:end_char_idx] += (
                weights[:n, np.newaxis] * logits[i - start, :n]
            )
            all_counts[original_idx][start_char_idx:end_char_idx] += weights[:n]

    # so far, logits are summed, so we average them here
    all_logits = [
        (logits / counts[:, None]).astype(np.float16)  # type: ignore[misc]
        for logits, counts in zip(all_logits, all_counts)
    ]

    return (
        all_logits,
        offset_mapping if use_subwords else None,
        tokenizer if use_subwords else None,
        tokens if use_subwords else None,
    )
