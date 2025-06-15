"""Segment Any Text from wtsplit."""
# This file uses old-style type hints and ignores certain ruff rules to minimise changes w.r.t. the original implementation:
# ruff: noqa: C901, PLR0913, PLR0912, PTH112, TRY003, EM101, SIM102, W505, FBT001, FBT002, RET505, B007, B905

import math
import os
import warnings
from functools import cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from wtpsplit_lite._config import SubwordXLMConfig
from wtpsplit_lite._extract import SaTORTWrapper, extract
from wtpsplit_lite._tokenizer import XLMRobertaTokenizerFast
from wtpsplit_lite._utils import Constants, indices_to_sentences, sigmoid, token_to_char_probs

cached_hf_hub_download = cache(hf_hub_download)

# Avoid: "huggingface/tokenizers: The current process just got forked, after parallelism has already
# been used. Disabling parallelism to avoid deadlocks..."
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _default_onnx_providers() -> list[str]:
    # https://onnxruntime.ai/docs/execution-providers/#summary-of-supported-execution-providers
    available_providers = ort.get_available_providers()
    preferred_providers = [
        # GPU
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
        "ROCMExecutionProvider",
        "DmlExecutionProvider",
        "MIGraphXExecutionProvider",
        "TvmExecutionProvider",
        # Mobile
        # "CoreMLExecutionProvider",  # Does not support SaT's dimenions
        "QNNExecutionProvider",
        "ArmNNExecutionProvider",
        "ACLExecutionProvider",
        "XnnpackExecutionProvider",
        # CPU
        "OpenVINOExecutionProvider",
        "DnnlExecutionProvider",
        "CPUExecutionProvider",
    ]
    return [provider for provider in preferred_providers if provider in available_providers]


class SaT:
    """Segment Any Text.

    A Universal Approach for Robust, Efficient and Adaptable Sentence Segmentation
    by Markus Frohmann, Igor Sterner, Benjamin Minixhofer, Ivan Vulić and Markus Schedl.
    """

    def __init__(
        self,
        model_name_or_model: str | Path,
        tokenizer_name_or_path: str | Path = "facebookAI/xlm-roberta-base",
        from_pretrained_kwargs: dict[str, Any] | None = None,
        ort_providers: list[str] | None = None,
        ort_kwargs: dict[str, Any] | None = None,
        style_or_domain: str | None = None,
        language: str | None = None,
        lora_path: str | None = None,  # local
        hub_prefix: str | None = "segment-any-text",
    ):
        if ort_providers is None:
            ort_providers = _default_onnx_providers()
        self.model_name_or_model = model_name_or_model
        self.ort_providers = ort_providers
        self.ort_kwargs = ort_kwargs
        self.use_lora = False
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(tokenizer_name_or_path)

        if isinstance(model_name_or_model, str | Path):
            model_name = str(model_name_or_model)
            is_local = os.path.isdir(model_name)

            if not is_local and hub_prefix is not None:
                model_name_to_fetch = f"{hub_prefix}/{model_name}"
            else:
                model_name_to_fetch = model_name

            if is_local:
                model_path = Path(model_name)
                onnx_path: Path | None = model_path / "model_optimized.onnx"
                if onnx_path and not onnx_path.exists():
                    onnx_path = None
            else:
                onnx_path = cached_hf_hub_download(  # type: ignore[assignment]
                    model_name_to_fetch,
                    "model_optimized.onnx",
                    **(from_pretrained_kwargs or {}),
                )

            if ort_providers is not None:
                if onnx_path is None:
                    raise ValueError("Could not find an ONNX model in the model directory.")

                self.model = SaTORTWrapper(
                    SubwordXLMConfig.from_pretrained(
                        model_name_to_fetch, **(from_pretrained_kwargs or {})
                    ),
                    ort.InferenceSession(
                        str(onnx_path), providers=ort_providers, **(ort_kwargs or {})
                    ),
                )
                if lora_path:
                    raise ValueError(
                        "If using ONNX with LoRA, execute `scripts/export_to_onnx_sat.py` with `use_lora=True`."
                        "Reference the chosen `output_dir` here for `model_name_or_model`. and set `lora_path=None`."
                    )
            # LoRA LOADING
            if not lora_path:
                if (style_or_domain and not language) or (language and not style_or_domain):
                    raise ValueError("Please specify both language and style_or_domain!")
            if (style_or_domain and language) or lora_path:
                raise ValueError(
                    "If using ONNX with LoRA, execute `scripts/export_to_onnx_sat.py` with `use_lora=True`."
                    "Reference the chosen `output_dir` here for `model_name_or_model`. and set `lora_path=None`."
                )

    def __getattr__(self, name):
        assert hasattr(self, "model")
        return getattr(self.model, name)

    def predict_proba(
        self,
        text_or_texts: str | list[str],
        stride: int = 256,
        block_size: int = 512,
        batch_size: int = 32,
        pad_last_batch: bool = False,
        weighting: Literal["uniform", "hat"] = "uniform",
        remove_whitespace_before_inference: bool = False,
        outer_batch_size: int = 1000,
        return_paragraph_probabilities: bool = False,
        verbose: bool = False,
    ):
        if isinstance(text_or_texts, str):
            return next(
                self._predict_proba(
                    [text_or_texts],
                    stride=stride,
                    block_size=block_size,
                    batch_size=batch_size,
                    pad_last_batch=pad_last_batch,
                    weighting=weighting,
                    remove_whitespace_before_inference=remove_whitespace_before_inference,
                    outer_batch_size=outer_batch_size,
                    return_paragraph_probabilities=return_paragraph_probabilities,
                    verbose=verbose,
                )
            )
        else:
            return self._predict_proba(
                text_or_texts,
                stride=stride,
                block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                weighting=weighting,
                remove_whitespace_before_inference=remove_whitespace_before_inference,
                outer_batch_size=outer_batch_size,
                return_paragraph_probabilities=return_paragraph_probabilities,
                verbose=verbose,
            )

    def _predict_proba(
        self,
        texts,
        stride: int,
        block_size: int,
        batch_size: int,
        pad_last_batch: bool,
        weighting: Literal["uniform", "hat"],
        remove_whitespace_before_inference: bool,
        outer_batch_size: int,
        return_paragraph_probabilities: bool,
        verbose: bool,
    ):
        def newline_probability_fn(logits):
            return sigmoid(logits[:, Constants.NEWLINE_INDEX])

        n_outer_batches = math.ceil(len(texts) / outer_batch_size)

        for outer_batch_idx in range(n_outer_batches):
            start, end = (
                outer_batch_idx * outer_batch_size,
                min((outer_batch_idx + 1) * outer_batch_size, len(texts)),
            )

            outer_batch_texts = texts[start:end]
            input_texts = []
            space_positions = []

            for text in outer_batch_texts:
                if remove_whitespace_before_inference:
                    text_space_positions: list[int] = []
                    input_text = ""

                    for c in text:
                        if c == " ":
                            text_space_positions.append(len(input_text) + len(text_space_positions))
                        else:
                            input_text += c

                    space_positions.append(text_space_positions)
                else:
                    input_text = text

                input_texts.append(input_text)

            empty_string_indices = [i for i, text in enumerate(input_texts) if not text.strip()]
            # remove empty strings from input_texts
            input_texts = [text for text in input_texts if text.strip()]
            if input_texts:
                outer_batch_logits, _, tokenizer, tokenizer_output = extract(
                    input_texts,
                    self.model,
                    stride=stride,
                    max_block_size=block_size,
                    batch_size=batch_size,
                    pad_last_batch=pad_last_batch,
                    weighting=weighting,
                    verbose=verbose,
                    tokenizer=self.tokenizer,
                )

                # convert token probabilities to character probabilities for the entire array
                outer_batch_logits = [
                    token_to_char_probs(
                        input_texts[i],
                        tokenizer_output["input_ids"][i],
                        outer_batch_logits[i],
                        tokenizer,
                        tokenizer_output["offset_mapping"][i],
                    )
                    for i in range(len(input_texts))
                ]
            else:
                outer_batch_logits = []

            # add back empty strings
            for i in empty_string_indices:
                outer_batch_logits.insert(i, np.ones([1, 1]) * -np.inf)

            for i, (text, logits) in enumerate(zip(outer_batch_texts, outer_batch_logits)):
                sentence_probs = newline_probs = newline_probability_fn(logits)

                if remove_whitespace_before_inference:
                    full_newline_probs, full_sentence_probs = (
                        list(newline_probs),
                        list(sentence_probs),
                    )

                    for j in space_positions[i]:
                        full_newline_probs.insert(j, np.zeros_like(newline_probs[0]))
                        full_sentence_probs.insert(j, np.zeros_like(sentence_probs[0]))

                    newline_probs = np.array(full_newline_probs)
                    sentence_probs = np.array(full_sentence_probs)

                if return_paragraph_probabilities:
                    yield sentence_probs, newline_probs
                else:
                    yield sentence_probs

    def split(
        self,
        text_or_texts: str | list[str],
        threshold: float | None = None,
        stride: int = 64,
        block_size: int = 512,
        batch_size: int = 32,
        pad_last_batch: bool = False,
        weighting: Literal["uniform", "hat"] = "uniform",
        remove_whitespace_before_inference: bool = False,
        outer_batch_size: int = 1000,
        paragraph_threshold: float = 0.5,
        strip_whitespace: bool = False,
        do_paragraph_segmentation: bool = False,
        treat_newline_as_space: bool = True,
        verbose: bool = False,
    ):
        if isinstance(text_or_texts, str):
            return next(
                self._split(
                    [text_or_texts],
                    threshold=threshold,
                    stride=stride,
                    block_size=block_size,
                    batch_size=batch_size,
                    pad_last_batch=pad_last_batch,
                    weighting=weighting,
                    remove_whitespace_before_inference=remove_whitespace_before_inference,
                    outer_batch_size=outer_batch_size,
                    paragraph_threshold=paragraph_threshold,
                    strip_whitespace=strip_whitespace,
                    do_paragraph_segmentation=do_paragraph_segmentation,
                    treat_newline_as_space=treat_newline_as_space,
                    verbose=verbose,
                )
            )
        else:
            return self._split(
                text_or_texts,
                threshold=threshold,
                stride=stride,
                block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                weighting=weighting,
                remove_whitespace_before_inference=remove_whitespace_before_inference,
                outer_batch_size=outer_batch_size,
                paragraph_threshold=paragraph_threshold,
                strip_whitespace=strip_whitespace,
                do_paragraph_segmentation=do_paragraph_segmentation,
                treat_newline_as_space=treat_newline_as_space,
                verbose=verbose,
            )

    def _split(
        self,
        texts,
        threshold: float | None,
        stride: int,
        block_size: int,
        batch_size: int,
        pad_last_batch: bool,
        weighting: Literal["uniform", "hat"],
        paragraph_threshold: float,
        remove_whitespace_before_inference: bool,
        outer_batch_size: int,
        do_paragraph_segmentation: bool,
        treat_newline_as_space: bool,
        strip_whitespace: bool,
        verbose: bool,
    ):
        def get_default_threshold(model_str: str):
            # basic type check for safety
            if not isinstance(model_str, str):
                warnings.warn(  # type: ignore[unreachable]
                    f"get_default_threshold received non-string argument: {type(model_str)}. Using base default.",
                    stacklevel=2,
                )
                return 0.025  # default fallback
            if self.use_lora:
                return 0.5
            if "sm" in model_str:
                return 0.25
            if "no-limited-lookahead" in model_str and "sm" not in model_str:
                return 0.01
            return 0.025

        default_threshold = get_default_threshold(str(self.model_name_or_model))
        sentence_threshold = threshold if threshold is not None else default_threshold

        for text, probs in zip(
            texts,
            self.predict_proba(
                texts,
                stride=stride,
                block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                weighting=weighting,
                remove_whitespace_before_inference=remove_whitespace_before_inference,
                outer_batch_size=outer_batch_size,
                return_paragraph_probabilities=do_paragraph_segmentation,
                verbose=verbose,
            ),
        ):
            if do_paragraph_segmentation:
                sentence_probs, newline_probs = probs

                offset = 0
                paragraphs = []

                for paragraph in indices_to_sentences(
                    text, np.where(newline_probs > paragraph_threshold)[0]
                ):
                    sentences = []

                    for sentence in indices_to_sentences(
                        paragraph,
                        np.where(
                            sentence_probs[offset : offset + len(paragraph)] > sentence_threshold,
                        )[0],
                        strip_whitespace=strip_whitespace,
                    ):
                        sentences.append(sentence)  # noqa: PERF402

                    paragraphs.append(sentences)
                    offset += len(paragraph)

                yield paragraphs
            else:
                sentences = indices_to_sentences(
                    text, np.where(probs > sentence_threshold)[0], strip_whitespace=strip_whitespace
                )
                if not treat_newline_as_space:
                    # within the model, newlines in the text were ignored - they were treated as spaces.
                    # this is the default behavior: additionally split on newlines as provided in the input
                    new_sentences = []
                    for sentence in sentences:
                        new_sentences.extend(sentence.split("\n"))
                    sentences = new_sentences
                yield sentences
