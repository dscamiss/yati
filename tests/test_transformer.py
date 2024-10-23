"""Test code for transformer."""

import pytest
import torch
from jaxtyping import TypeCheckError
from torch import Tensor

from model.transformer import Transformer
from params.encoder_decoder_params import EncoderDecoderParams
from params.transformer_params import TransformerParams


@pytest.fixture(name="x")
def fixture_x() -> Tensor:
    """Test fixture with input tensor."""
    return torch.randint(1, 100, (16, 100))


@pytest.fixture(name="params")
def fixture_params() -> TransformerParams:
    """Test fixture with transformer parameters."""
    d_model = 512
    encoder_params = EncoderDecoderParams(d_model, 2, 5, 6, 16, 0.1)
    decoder_params = EncoderDecoderParams(d_model, 3, 6, 7, 17, 0.1)

    return TransformerParams(
        d_model,
        200,
        1024,
        2048,
        6,
        encoder_params,
        7,
        decoder_params,
        0.1,
        False,
    )


@pytest.fixture(name="transformer")
def fixture_transformer(params) -> Transformer:
    """Test fixture with Transformer object."""
    return Transformer(params)


@pytest.fixture(name="x_cross")
def fixture_x_cross(transformer, x) -> Tensor:
    """Test fixture with cross-attention input tensor."""
    return transformer.encode(x)


def test_encode_valid_input(x, params, transformer) -> None:
    """Test output with valid input."""
    assert transformer.encode(x).shape == x.shape + torch.Size([params.d_model])


def test_encode_invalid_input(x, transformer) -> None:
    """Test behavior with invalid input."""
    with pytest.raises(TypeCheckError):
        transformer.encode(torch.ones(16, dtype=torch.int32))  # expects rank-2 tensor
    with pytest.raises(TypeCheckError):
        transformer.encode(x.float())  # expects integer type


def test_decode_valid_input(x, x_cross, params, transformer) -> None:
    """Test output with valid input."""
    assert transformer.decode(x, x_cross).shape == x.shape + torch.Size(
        [params.output_num_embeddings]
    )


def test_decode_invalid_input(x, x_cross, transformer) -> None:
    """Test behavior with invalid input."""
    with pytest.raises(TypeCheckError):
        transformer.decode(x, torch.ones(16, dtype=torch.int32))  # expects rank-2 tensor
    with pytest.raises(TypeCheckError):
        transformer.decode(torch.ones(16, dtype=torch.int32), x_cross)  # expects rank-2 tensor
    with pytest.raises(TypeCheckError):
        transformer.decode(x.float(), x_cross)  # expects integer type
