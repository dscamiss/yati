"""Test code for encoder stack."""

import pytest
import torch
from jaxtyping import TypeCheckError
from torch import Tensor

from model.encoder_decoder_params import EncoderDecoderParams
from model.encoder_stack import Encoder, EncoderStack


@pytest.fixture(name="x")
def fixture_x() -> Tensor:
    """Test fixture with input tensor."""
    return torch.ones(2, 3, 4)


@pytest.fixture(name="encoder_params")
def fixture_encoder_params(x) -> EncoderDecoderParams:
    """Test fixture with encoder parameters."""
    return EncoderDecoderParams(x.shape[-1], 2, 5, 6, 16, 0.1)


@pytest.fixture(name="encoder")
def fixture_encoder(encoder_params) -> Encoder:
    """Test fixture with Encoder object."""
    return Encoder(encoder_params)


@pytest.fixture(name="encoder_stack")
def fixture_encoder_stack(encoder_params) -> EncoderStack:
    """Test fixture with EncoderStack object."""
    return EncoderStack(6, encoder_params)


def test_encoder_valid_input(x, encoder) -> None:
    """Test output with valid input."""
    assert encoder(x).shape == x.shape


def test_encoder_invalid_input(encoder) -> None:
    """Test behavior with invalid input."""
    with pytest.raises(TypeCheckError):
        encoder(torch.ones(16, 1))


def test_encoder_stack_valid_input(x, encoder_stack) -> None:
    """Test output with valid input."""
    assert encoder_stack(x).shape == x.shape


def test_encoder_stack_invalid_input(encoder_stack) -> None:
    """Test behavior with invalid input."""
    with pytest.raises(TypeCheckError):
        encoder_stack(torch.ones(16, 1))
