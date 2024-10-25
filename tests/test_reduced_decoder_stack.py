"""Test code for reduced decoder stack."""

import pytest
import torch
from jaxtyping import TypeCheckError
from torch import Tensor

from yati.model.reduced_decoder_stack import ReducedDecoder, ReducedDecoderStack
from yati.params.encoder_decoder_params import EncoderDecoderParams


@pytest.fixture(name="x")
def fixture_x() -> Tensor:
    """Test fixture with input tensor."""
    return torch.ones(2, 3, 4)


@pytest.fixture(name="params")
def fixture_params(x) -> EncoderDecoderParams:
    """Test fixture with decoder parameters."""
    return EncoderDecoderParams(x.shape[-1], 2, 5, 6, 16, 0.1)


@pytest.fixture(name="max_seq_len")
def fixture_max_seq_len() -> int:
    """Test fixture with maximum input sequence length."""
    return 1000


@pytest.fixture(name="reduced_decoder")
def fixture_reduced_decoder(params, max_seq_len) -> ReducedDecoder:
    """Test fixture with ReducedDecoder object."""
    return ReducedDecoder(params, True, max_seq_len)


@pytest.fixture(name="reduced_decoder_stack")
def fixture_reduced_decoder_stack(params, max_seq_len) -> ReducedDecoderStack:
    """Test fixture with ReducedDecoderStack object."""
    return ReducedDecoderStack(6, params, True, max_seq_len)


def test_reduced_decoder_valid_input(x, reduced_decoder) -> None:
    """Test reduced decoder output with valid input."""
    assert reduced_decoder(x).shape == x.shape


def test_reduced_decoder_invalid_input(reduced_decoder) -> None:
    """Test reduced decoder behavior with invalid input."""
    with pytest.raises(TypeCheckError):
        reduced_decoder(torch.ones(16, 1))


def test_valid_inputs(x, reduced_decoder_stack) -> None:
    """Test output with valid input."""
    assert reduced_decoder_stack(x).shape == x.shape


def test_invalid_inputs(reduced_decoder_stack) -> None:
    """Test behavior with invalid input."""
    with pytest.raises(TypeCheckError):
        reduced_decoder_stack(torch.ones(16, 1))
