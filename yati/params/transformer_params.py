"""Implementation of dataclass for transformer parameters."""

from dataclasses import astuple, dataclass
from typing import Any, Iterator

from yati.params.encoder_decoder_params import EncoderDecoderParams


@dataclass
class TransformerParams:
    """Dataclass for transformer parameters.

    Args:
        d_model: Input/output embedding dimension.
        max_seq_len: Maximum input sequence length.
        input_num_embeddings: Input vocabulary size.
        output_num_embeddings: Output vocabulary size.
        encoder_stack_num_layers: Number of encoder layers in encoder stack.
        encoder_params: Encoder layer parameters.
        decoder_stack_num_layers: Number of decoder layers in decoder stack.
        decoder_params: Decoder layer parameters.
        p_dropout: Dropout probability for positional encoding layers.
        tie_weight_matrices: Tie weight matrices in input embedding, output
            embedding, and pre-softmax layers.

    Raises:
        ValueError: If tie_weight_matrices is True and there is a mismatch
            between the input and output vocabulary sizes.

    # noqa: DCO060
    """

    d_model: int
    max_seq_len: int
    input_num_embeddings: int
    output_num_embeddings: int
    encoder_stack_num_layers: int
    encoder_params: EncoderDecoderParams
    decoder_stack_num_layers: int
    decoder_params: EncoderDecoderParams
    p_dropout: float
    tie_weight_matrices: bool

    def __iter__(self) -> Iterator[Any]:  # noqa: DCO010
        return iter(astuple(self))

    def __post_init__(self) -> None:  # noqa: DCO010
        num_embeddings_mismatch = self.input_num_embeddings != self.output_num_embeddings
        if self.tie_weight_matrices and num_embeddings_mismatch:
            raise ValueError(
                "Need input_num_embeddings == output_num_embeddings to tie weight matrices"
            )
