"""Test code for feed-forward layer."""

import pytest
import torch
from jaxtyping import TypeCheckError
from torch import Tensor

from yati.model.feed_forward import FeedForward


@pytest.fixture(name="x")
def fixture_x() -> Tensor:
    """Test fixture with input tensor."""
    return torch.ones(2, 3, 4)


@pytest.fixture(name="feed_forward")
def fixture_feed_forward(x) -> FeedForward:
    """Test fixture with FeedForward object."""
    return FeedForward(x.shape[-1], 16)


def test_valid_input(x, feed_forward) -> None:
    """Test output with valid input."""
    assert feed_forward(x).shape == x.shape


def test_invalid_input(feed_forward) -> None:
    """Test behavior with invalid input."""
    with pytest.raises(TypeCheckError):
        feed_forward(torch.ones(16, 1))
