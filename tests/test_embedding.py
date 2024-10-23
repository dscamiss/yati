"""Test code for embedding layer."""

import pytest
import torch
from jaxtyping import TypeCheckError
from torch import Tensor

from yati.model.embedding import Embedding


@pytest.fixture(name="x")
def fixture_x() -> Tensor:
    """Test fixture with input tensor."""
    return torch.randint(1, 100, (16, 200))


@pytest.fixture(name="d_model")
def fixture_d_model() -> int:
    """Test fixture with embedding dimension."""
    return 512


@pytest.fixture(name="embedding")
def fixture_embedding(d_model) -> Embedding:
    """Test fixture with Embedding object."""
    return Embedding(d_model, 1000)


def test_valid_input(x, d_model, embedding) -> None:
    """Test output with valid input."""
    assert embedding(x).shape == x.shape + torch.Size([d_model])


def test_invalid_input(x, embedding) -> None:
    """Test behavior with invalid input."""
    with pytest.raises(TypeCheckError):
        embedding(torch.ones(16, dtype=torch.int32))  # expects rank-2 tensor
    with pytest.raises(TypeCheckError):
        embedding(x.float())  # expects integer type
