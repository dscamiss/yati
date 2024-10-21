"""Test code for embedding layer."""

import pytest
import torch
from jaxtyping import TypeCheckError
from torch import Tensor

from model.embedding import Embedding


@pytest.fixture(name="x")
def fixture_x() -> Tensor:
    """Test fixture with input tensor."""
    return torch.ones(16, 200, dtype=torch.int32)


@pytest.fixture(name="d_model")
def fixture_d_model() -> int:
    """Test fixture with embedding dimension."""
    return 512


@pytest.fixture(name="embedding")
def fixture_embedding(d_model) -> Embedding:
    """Test fixture with Embedding object."""
    return Embedding(d_model, 1000)


def test_embedding_valid_inputs(x, d_model, embedding) -> None:
    """Test correctness of embedding output with valid inputs."""
    assert embedding(x).shape == x.shape + torch.Size([d_model])


def test_embedding_invalid_inputs(x, embedding) -> None:
    """Test correctness of embedding output with invalid inputs."""
    with pytest.raises(TypeCheckError):
        embedding(x.float())
    with pytest.raises(TypeCheckError):
        embedding(torch.ones(16, 1))
