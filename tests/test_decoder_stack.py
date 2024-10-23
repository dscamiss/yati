"""Test code for decoder stack."""

import pytest
import torch
from jaxtyping import TypeCheckError
from torch import Tensor

from yati.model.decoder_stack import Decoder, DecoderStack
from yati.params.encoder_decoder_params import EncoderDecoderParams


@pytest.fixture(name="x")
def fixture_x() -> Tensor:
    """Test fixture with input tensor."""
    return torch.ones(2, 3, 4)


@pytest.fixture(name="x_cross")
def fixture_x_cross(x) -> Tensor:
    """Test fixture with cross-attention input tensor."""
    return torch.ones(x.shape)


@pytest.fixture(name="params")
def fixture_params(x) -> EncoderDecoderParams:
    """Test fixture with decoder parameters."""
    return EncoderDecoderParams(x.shape[-1], 2, 5, 6, 16, 0.1)


@pytest.fixture(name="max_seq_len")
def fixture_max_seq_len() -> int:
    """Test fixture with maximum input sequence length."""
    return 1000


@pytest.fixture(name="decoder")
def fixture_decoder(params, max_seq_len) -> Decoder:
    """Test fixture with Decoder object."""
    return Decoder(params, max_seq_len)


@pytest.fixture(name="decoder_stack")
def fixture_decoder_stack(params, max_seq_len) -> DecoderStack:
    """Test fixture with DecoderStack object."""
    return DecoderStack(6, params, max_seq_len)


def test_decoder_valid_input(x, x_cross, decoder) -> None:
    """Test decoder output with valid inputs."""
    assert decoder(x, x_cross).shape == x.shape


def test_decoder_invalid_input(x, x_cross, decoder) -> None:
    """Test decoder behavior with invalid inputs."""
    with pytest.raises(TypeCheckError):
        decoder(x, torch.ones(16, 1))
    with pytest.raises(TypeCheckError):
        decoder(torch.ones(16, 1), x_cross)


def test_valid_inputs(x, x_cross, decoder_stack) -> None:
    """Test output with valid inputs."""
    assert decoder_stack(x, x_cross).shape == x.shape


def test_invalid_inputs(x, x_cross, decoder_stack) -> None:
    """Test behavior with invalid inputs."""
    with pytest.raises(TypeCheckError):
        decoder_stack(torch.ones(16, 1), x_cross)
    with pytest.raises(TypeCheckError):
        decoder_stack(x, torch.ones(16, 1))
