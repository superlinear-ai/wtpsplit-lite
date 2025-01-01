"""Minimal subset of wtpsplit.utils."""
# This file uses old-style type hints and ignores certain ruff rules to minimise changes w.r.t. the original implementation:
# ruff: noqa: N802, N806, PTH100, PTH110, PTH118, PTH120, PTH123, PTH123, FURB129, FBT002, W505, TRY003, EM102, PLW2901, D400

import json
import logging
import os
from collections import defaultdict
from functools import cached_property
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# same as in CANINE
PRIMES = [31, 43, 59, 61, 73, 97, 103, 113, 137, 149, 157, 173, 181, 193, 211, 223]


class ConstantsClass:
    NEWLINE_INDEX = 0
    AUX_OFFSET = 1
    DEFAULT_PUNCTUATION_FILE = "punctuation.txt"
    _PUNCTUATION_FILE = "punctuation.txt"

    @classmethod
    def set_punctuation_file(cls, file_name):
        cls._PUNCTUATION_FILE = file_name

    @cached_property
    def ROOT_DIR(self):
        return Path(os.path.abspath(os.path.join(os.path.dirname(__file__))))

    @cached_property
    def CACHE_DIR(self):
        CACHE_DIR = self.ROOT_DIR / ".cache"
        CACHE_DIR.mkdir(exist_ok=True)
        return CACHE_DIR

    @cached_property
    def LANGINFO(self):
        return np.genfromtxt(
            os.path.join(self.ROOT_DIR, "data", "language_info.csv"),
            delimiter=",",
            dtype=None,
            encoding="utf-8",
            names=True,
        )

    @cached_property
    def PUNCTUATION_CHARS(self):
        punctuation_path = os.path.join(self.ROOT_DIR, "data", self._PUNCTUATION_FILE)
        if os.path.exists(punctuation_path):
            return [x.strip() for x in open(punctuation_path).readlines()]
        raise FileNotFoundError(f"The file {punctuation_path} does not exist.")

    @cached_property
    def PUNCTUATION_MAP(self):
        return json.load(open(os.path.join(self.ROOT_DIR, "data", "punctuation.json")))

    @cached_property
    def LANG_CODE_TO_INDEX(self):
        return {lang: i for i, lang in enumerate(self.LANGINFO["index"])}

    @cached_property
    def SEPARATORS(self):
        return defaultdict(
            lambda: " ",
            {lang: ("" if row[4] else " ") for lang, *row in self.LANGINFO},
        )


Constants = ConstantsClass()


def hash_encode(encoding, num_hashes=8, num_buckets=8192):
    if num_hashes > len(PRIMES):
        raise ValueError(f"`num_hashes` must be <= {len(PRIMES)}")

    hash_ids = np.zeros((len(encoding), num_hashes), dtype=np.int64)
    for i in range(num_hashes):
        shard_ids = (encoding + 1) * PRIMES[i]
        hash_ids[:, i] = shard_ids % num_buckets

    return hash_ids


def indices_to_sentences(text, indices, strip_whitespace=False):
    sentences = []

    offset = 0
    idx = 0
    for idx in indices:
        idx = idx + 1
        while idx < len(text) and text[idx].isspace():
            idx += 1

        sentence = text[offset:idx]
        if strip_whitespace:
            # NB: I would have thought that this is slower than
            # adjusting the start and end indices since there are
            # two string copies, but it seems to be faster
            # (at least on short strings). more reason to port to Rust?
            sentence = sentence.strip()

        if len(sentence) > 0:
            sentences.append(sentence)

        offset = idx

    if idx != len(text):
        last_sentence = text[idx:]
        if strip_whitespace:
            last_sentence = last_sentence.strip()

        if len(last_sentence) > 0:
            sentences.append(last_sentence)

    return sentences


def sigmoid(x):
    return 1 / (1 + np.exp(-x.astype(np.float32)))  # fp32 for better precision


def get_token_spans(tokenizer, offsets_mapping, tokens):
    # Filter out special tokens and get their character start and end positions
    valid_indices = np.array(
        [
            idx
            for idx, token in enumerate(tokens)
            if token not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]
            and idx < len(offsets_mapping)
        ]
    )
    valid_offsets = np.array(offsets_mapping)[valid_indices]
    return valid_indices, valid_offsets


def token_to_char_probs(text, tokens, token_logits, tokenizer, offsets_mapping):
    """Map from token probabalities to character probabilities"""
    char_probs = np.full(
        (len(text), token_logits.shape[1]), -np.inf
    )  # Initialize with very low numbers

    valid_indices, valid_offsets = get_token_spans(tokenizer, offsets_mapping, tokens)

    # Assign the token's probability to the last character of the token
    for i in range(valid_offsets.shape[0]):
        start, end = valid_offsets[i]
        char_probs[end - 1] = token_logits[valid_indices[i]]

    return char_probs
