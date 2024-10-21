"""Test code for add-and-norm layer."""

import pytest
import torch
from jaxtyping import TypeCheckError
from torch import Tensor

from model.add_and_norm import AddAndNorm


@pytest.fixture(name="x")
def fixture_x() -> Tensor:
    """Test fixture with input tensor for first summand."""
    return torch.tensor(
        [
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            [
                [3.0, 3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0, 3.0],
            ],
        ]
    )


@pytest.fixture(name="y")
def fixture_y(x) -> Tensor:
    """Test fixture with input tensor for second summand."""
    return x + 1.0


@pytest.fixture(name="add_and_norm")
def fixture_add_and_norm(x) -> AddAndNorm:
    """Test fixture with AddAndNorm object."""
    return AddAndNorm(x.shape[-1])


def test_add_and_norm_valid_inputs(x, y, add_and_norm) -> None:
    """Test output with valid inputs."""
    z = add_and_norm(x, y)
    assert z.shape == x.shape and torch.all(z == 0.0)


def test_add_and_norm_invalid_inputs(x, y, add_and_norm) -> None:
    """Test behavior with invalid inputs."""
    with pytest.raises(TypeCheckError):
        add_and_norm(torch.ones(16, 1), y)
    with pytest.raises(TypeCheckError):
        add_and_norm(x, torch.ones(16, 1))
