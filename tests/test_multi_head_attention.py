"""Test code for multi-head attention layer."""

import math

import pytest
import torch
from jaxtyping import TypeCheckError
from torch import Tensor

from model.multi_head_attention import MultiHeadAttention


@pytest.fixture(name="x")
def fixture_x() -> Tensor:
    """Test fixture with input tensor."""
    return torch.ones(2, 3, 4)


@pytest.fixture(name="attention_no_causal_mask")
def fixture_attention_no_causal_mask(x) -> MultiHeadAttention:
    """Test fixture with MultiHeadAttention object (no causal mask)."""
    return MultiHeadAttention(2, x.shape[-1], 5, 6)


@pytest.fixture(name="attention_causal_mask")
def fixture_attention_causal_mask(x) -> MultiHeadAttention:
    """Test fixture with MultiHeadAttention object (causal mask)."""
    return MultiHeadAttention(2, x.shape[-1], 5, 6, True, 16)


def get_expected_output(attention: MultiHeadAttention, x: Tensor) -> Tensor:
    """Helper function to compute expected output."""
    # Get number of heads
    h = attention.h

    # Get weight matrices
    q_weights, k_weights, v_weights, o_weight = attention.get_weights()

    # Get weight matrices for each head
    d_k = q_weights.shape[0] // h
    d_v = v_weights.shape[0] // h

    q_weights = q_weights.split(d_k, dim=0)
    k_weights = k_weights.split(d_k, dim=0)
    v_weights = v_weights.split(d_v, dim=0)

    # Use weight matrices to compute expected output
    scaling_factor = math.sqrt(q_weights[0].shape[0])

    # Per-head results are stacked in y_expected
    y_expected = []

    # Get causal mask, if needed
    apply_causal_mask = attention.apply_causal_mask
    if apply_causal_mask:
        causal_mask = attention.causal_mask

    for i in range(h):
        q = x @ q_weights[i].transpose(-1, -2)  # (b, n, d_k)
        k = x @ k_weights[i].transpose(-1, -2)  # (b, n, d_k)
        v = x @ v_weights[i].transpose(-1, -2)  # (b, n, d_v)
        qk_transpose = q @ k.transpose(-1, -2) / scaling_factor  # (b, n, n)
        if apply_causal_mask:
            n = qk_transpose.shape[-1]
            qk_transpose = qk_transpose.masked_fill(
                causal_mask[:, :, :n, :n] == 0, float("-inf")
            )
        s = torch.softmax(qk_transpose, dim=-1)  # (b, n, n)
        y_expected.append(s @ v)  # (b, n, d_v)

    y_expected = torch.cat(y_expected, dim=-1)

    return y_expected @ o_weight.transpose(-1, -2)


def test_no_causal_mask_valid_input(attention_no_causal_mask, x) -> None:
    """Test output with valid input."""
    y = attention_no_causal_mask(x, x, x)
    assert y.shape == x.shape

    y_expected = get_expected_output(attention_no_causal_mask, x)
    assert torch.allclose(y, y_expected)


def test_no_causal_mask_invalid_input(attention_no_causal_mask, x) -> None:
    """Test behavior with invalid input."""
    with pytest.raises(TypeCheckError):
        attention_no_causal_mask(torch.ones(16, 1), x, x)
        attention_no_causal_mask(x, torch.ones(16, 1), x)
        attention_no_causal_mask(x, x, torch.ones(16, 1))


def test_causal_mask_valid_input(attention_causal_mask, x) -> None:
    """Test output with valid input."""
    y = attention_causal_mask(x, x, x)
    assert y.shape == x.shape

    y_expected = get_expected_output(attention_causal_mask, x)
    assert torch.allclose(y, y_expected)


def test_causal_mask_invalid_input(attention_causal_mask, x) -> None:
    """Test behavior with invalid input."""
    with pytest.raises(TypeCheckError):
        attention_causal_mask(torch.ones(16, 1), x, x)
        attention_causal_mask(x, torch.ones(16, 1), x)
        attention_causal_mask(x, x, torch.ones(16, 1))


def test_invalid_init_arguments() -> None:
    """Test object creation with invalid arguments."""
    with pytest.raises(ValueError):
        MultiHeadAttention(2, 4, 5, 6, True, -1)
