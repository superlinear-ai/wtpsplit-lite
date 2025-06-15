"""Test Segment Any Text."""

import sys
import types
import warnings
from pathlib import Path

# Mock mosestokenizer because it's not compatible with Python 3.12.
if "mosestokenizer" not in sys.modules:
    dummy = types.ModuleType("mosestokenizer")
    dummy.MosesTokenizer = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["mosestokenizer"] = dummy

import pytest
from huggingface_hub import hf_hub_download
from wtpsplit import SaT as SaTOriginal

from wtpsplit_lite import SaT as SaTLite


@pytest.mark.parametrize(
    "text_or_texts",
    [
        pytest.param("", id="empty"),
        pytest.param("This is a test This is another test.", id="string"),
        pytest.param(["This is a test This is another test."], id="list_string"),
        pytest.param(["This is a test This is another test."] * 32, id="list"),
        pytest.param((Path(__file__).parent / "specrel.md").read_text(), id="specrel"),
    ],
)
@pytest.mark.parametrize(
    "sat_lite_sat_original",
    [
        pytest.param(
            (
                SaTLite("sat-1l"),
                SaTOriginal("sat-1l", ort_providers=["CPUExecutionProvider"]),
            ),
            id="sat-1l",
        ),
        pytest.param(
            (
                SaTLite("sat-1l-sm"),
                SaTOriginal("sat-1l-sm", ort_providers=["CPUExecutionProvider"]),
            ),
            id="sat-1l-sm",
        ),
        pytest.param(
            (
                SaTLite("sat-3l"),
                SaTOriginal("sat-3l", ort_providers=["CPUExecutionProvider"]),
            ),
            id="sat-3l",
        ),
        pytest.param(
            (
                SaTLite("sat-3l-sm"),
                SaTOriginal("sat-3l-sm", ort_providers=["CPUExecutionProvider"]),
            ),
            id="sat-3l-sm",
        ),
    ],
)
def test_sat(
    sat_lite_sat_original: tuple[SaTLite, SaTOriginal], text_or_texts: str | list[str]
) -> None:
    """Test SaT by comparing the output with that of the original implementation."""
    sat_lite, sat_original = sat_lite_sat_original
    with warnings.catch_warnings():  # Ignore unnecessary treat_newline_as_space warning.
        warnings.filterwarnings("ignore", "treat_newline_as_space", category=UserWarning)
        warnings.filterwarnings("ignore", "split_on_input_newlines", category=UserWarning)
        output_lite = sat_lite.split(text_or_texts)
        output_original = sat_original.split(text_or_texts, treat_newline_as_space=True)
    if isinstance(text_or_texts, str):
        assert len(output_lite) == len(output_original)
        assert all(
            sent_lite == sent_original
            for sent_lite, sent_original in zip(output_lite, output_original, strict=True)
        )
        reconstructed_text_lite = "".join(output_lite)
        reconstructed_text_original = "".join(output_original)
        assert text_or_texts == reconstructed_text_lite == reconstructed_text_original
    else:
        with warnings.catch_warnings():  # Ignore unnecessary treat_newline_as_space warning.
            warnings.filterwarnings("ignore", "treat_newline_as_space", category=UserWarning)
            warnings.filterwarnings("ignore", "split_on_input_newlines", category=UserWarning)
            output_lite = list(output_lite)
            output_original = list(output_original)
        assert len(output_lite) == len(output_original)
        assert all(
            len(sents_lite) == len(sents_original)
            for sents_lite, sents_original in zip(output_lite, output_original, strict=True)
        )
        reconstructed_texts_lite = ["".join(sents) for sents in output_lite]
        reconstructed_texts_original = ["".join(sents) for sents in output_original]
        assert text_or_texts == reconstructed_texts_lite == reconstructed_texts_original


@pytest.mark.parametrize(
    "text",
    [pytest.param((Path(__file__).parent / "specrel.md").read_text(), id="specrel")],
)
@pytest.mark.parametrize(
    "sat",
    [
        pytest.param(
            SaTLite("sat-3l-sm"),
            id="sat-3l-sm",
        )
    ],
)
def test_weighting(sat: SaTLite, text: str) -> None:
    """Test hat weighting SaT for low-stride splitting."""
    output_lite = sat.split(text, stride=128, block_size=256, weighting="hat")
    reconstructed_text_lite = "".join(output_lite)
    assert text == reconstructed_text_lite


def test_local() -> None:
    """Test loading a local model."""
    model = "segment-any-text/sat-3l-sm"
    model_filepath = hf_hub_download(model, filename="model_optimized.onnx")
    _ = hf_hub_download(model, filename="config.json")
    tokenizer_filepath = hf_hub_download("facebookAI/xlm-roberta-base", filename="tokenizer.json")
    sat_lite = SaTLite(
        model_name_or_model=Path(model_filepath).parent,
        tokenizer_name_or_path=Path(tokenizer_filepath).parent,
    )
    text = "This is a test This is another test."
    output_lite = sat_lite.split(text)
    assert output_lite == ["This is a test ", "This is another test."]
