"""Test code for decoder-only transformer."""

import pytest
import torch
from jaxtyping import TypeCheckError
from torch import Tensor

from model.decoder_only_transformer import DecoderOnlyTransformer
from params.decoder_only_transformer_params import DecoderOnlyTransformerParams
from params.encoder_decoder_params import EncoderDecoderParams


@pytest.fixture(name="x")
def fixture_x() -> Tensor:
    """Test fixture with input tensor."""
    return torch.randint(1, 100, (16, 100))


@pytest.fixture(name="params")
def fixture_params() -> DecoderOnlyTransformerParams:
    """Test fixture with decoder-only transformer parameters."""
    d_model = 512
    decoder_params = EncoderDecoderParams(d_model, 3, 6, 7, 17, 0.1)

    return DecoderOnlyTransformerParams(
        d_model,
        200,
        1024,
        2048,
        7,
        decoder_params,
        0.1,
        False,
    )


@pytest.fixture(name="transformer")
def fixture_transformer(params) -> DecoderOnlyTransformer:
    """Test fixture with DecoderOnlyTransformer object."""
    return DecoderOnlyTransformer(params)


def test_valid_input(x, params, transformer) -> None:
    """Test output with valid input."""
    assert transformer(x).shape == x.shape + torch.Size([params.output_num_embeddings])


def test_invalid_input(x, transformer) -> None:
    """Test behavior with invalid input."""
    with pytest.raises(TypeCheckError):
        transformer(torch.ones(16, dtype=torch.int32))  # expects rank-2 tensor
    with pytest.raises(TypeCheckError):
        transformer(x.float())  # expects integer type
