"""Test code for multi-head attention layer."""

import pytest
import torch
from jaxtyping import TypeCheckError
from torch import Tensor

from model.encoder_decoder_params import EncoderDecoderParams
from model.encoder_decoder_transformer import EncoderDecoderTransformer
from model.encoder_decoder_transformer_params import EncoderDecoderTransformerParams


@pytest.fixture(name="params")
def fixture_params() -> EncoderDecoderTransformerParams:
    """Test fixture with encoder/decoder transformer parameters."""
    d_model = 512
    encoder_params = EncoderDecoderParams(d_model, 2, 5, 6, 16, 0.1)
    decoder_params = EncoderDecoderParams(d_model, 3, 6, 7, 17, 0.1)

    return EncoderDecoderTransformerParams(
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
def fixture_transformer(params) -> EncoderDecoderTransformer:
    """Test fixture with EncoderDecoderTransformer object."""
    return EncoderDecoderTransformer(params)


@pytest.fixture(name="x")
def fixture_x() -> Tensor:
    """Test fixture with input tensor."""
    return torch.randint(1, 100, (16, 100))


@pytest.fixture(name="x_cross")
def fixture_x_cross(transformer, x) -> Tensor:
    """Test fixture with cross-attention input tensor."""
    return transformer.encode(x)


def test_encode_valid_input(params, transformer, x) -> None:
    """Test output with valid input."""
    assert transformer.encode(x).shape == x.shape + torch.Size([params.d_model])


def test_encode_invalid_input(transformer, x) -> None:
    """Test behavior with invalid input."""
    with pytest.raises(TypeCheckError):
        transformer.encode(torch.ones(16, dtype=torch.int32))  # expects rank-2 tensor
    with pytest.raises(TypeCheckError):
        transformer.encode(x.float())  # expects integer type


def test_decode_valid_input(params, transformer, x, x_cross) -> None:
    """Test output with valid input."""
    assert transformer.decode(x, x_cross).shape == x.shape + torch.Size(
        [params.output_num_embeddings]
    )


def test_decode_invalid_input(transformer, x, x_cross) -> None:
    """Test behavior with invalid input."""
    with pytest.raises(TypeCheckError):
        transformer.decode(x, torch.ones(16, dtype=torch.int32))  # expects rank-2 tensor
    with pytest.raises(TypeCheckError):
        transformer.decode(torch.ones(16, dtype=torch.int32), x_cross)  # expects rank-2 tensor
    with pytest.raises(TypeCheckError):
        transformer.decode(x.float(), x_cross)  # expects integer type
