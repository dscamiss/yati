"""Test code for decoder stack."""

import pytest
import torch
from jaxtyping import TypeCheckError
from torch import Tensor

from model.decoder_stack import Decoder, DecoderParams, DecoderStack


@pytest.fixture(name="x")
def fixture_x() -> Tensor:
    """Test fixture with input tensor."""
    return torch.ones(2, 3, 4)


@pytest.fixture(name="x_cross")
def fixture_x_cross(x) -> Tensor:
    """Test fixture with cross-attention input tensor."""
    return torch.ones(x.shape)


@pytest.fixture(name="decoder_params")
def fixture_decoder_params(x) -> DecoderParams:
    """Test fixture with decoder parameters."""
    return DecoderParams(x.shape[-1], 2, 5, 6, 16, 0.1, 32)


@pytest.fixture(name="decoder")
def fixture_decoder(decoder_params) -> Decoder:
    """Test fixture with decoder object."""
    return Decoder(decoder_params)


@pytest.fixture(name="decoder_stack")
def fixture_decoder_stack(decoder_params) -> DecoderStack:
    """Test fixture with DecoderStack object."""
    return DecoderStack(6, decoder_params)


def test_decoder_valid_input(decoder, x, x_cross) -> None:
    """Test decoder layer with valid inputs."""
    assert decoder(x, x_cross).shape == x.shape


def test_decoder_invalid_input(decoder, x, x_cross) -> None:
    """Test decoder layer with invalid inputs."""
    with pytest.raises(TypeCheckError):
        decoder(x, torch.ones(16, 1))
        decoder(torch.ones(16, 1), x_cross)


def test_decoder_stack_valid_input(decoder_stack, x, x_cross) -> None:
    """Test decoder stack with valid inputs."""
    assert decoder_stack(x, x_cross).shape == x.shape


def test_encoder_stack_invalid_input(decoder_stack, x, x_cross) -> None:
    """Test encoder stack with invalid inputs."""
    with pytest.raises(TypeCheckError):
        decoder_stack(torch.ones(16, 1), x_cross)
        decoder_stack(x, torch.ones(16, 1))
