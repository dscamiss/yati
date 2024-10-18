"""Test code for positional encoding layer."""

import math

import pytest
import torch
from jaxtyping import TypeCheckError
from torch import Tensor

from model.positional_encoding import PositionalEncoding


@pytest.fixture(name="x")
def fixture_x() -> Tensor:
    """Test fixture with input tensor."""
    return torch.ones(2, 3, 4)


@pytest.fixture(name="positional_encoding")
def fixture_positional_encoding(x) -> PositionalEncoding:
    """Test fixture with PositionalEncoding object."""
    return PositionalEncoding(x.shape[-1], 8)


def test_valid_input(positional_encoding, x) -> None:
    """Test output with valid input."""
    y = positional_encoding(x)
    assert y.shape == x.shape

    expected_encoding = torch.zeros(x.shape[1], x.shape[2])
    for r in range(x.shape[1]):
        for c in range(x.shape[2]):
            if c % 2 == 0:
                log_den = (c / x.shape[2]) * math.log(10000.0)
                expected_encoding[r, c] = math.sin(r * math.exp(-1.0 * log_den))
            else:
                log_den = ((c - 1) / x.shape[2]) * math.log(10000.0)
                expected_encoding[r, c] = math.cos(r * math.exp(-1.0 * log_den))

    for b in range(x.shape[0]):
        assert torch.allclose(y[b, :, :], x[b, :, :] + expected_encoding)


def test_invalid_input(positional_encoding) -> None:
    """Test behavior with invalid input."""
    with pytest.raises(TypeCheckError):
        positional_encoding(torch.ones(16, 1))


def test_invalid_init_arguments() -> None:
    """Test object creation with invalid arguments."""
    with pytest.raises(ValueError):
        PositionalEncoding(5, 16)
