"""Implementation of dataclass for decoder-only transformer parameters."""

from dataclasses import astuple, dataclass
from typing import Any, Iterator

from yati.params.encoder_decoder_params import EncoderDecoderParams


@dataclass
class DecoderOnlyTransformerParams:
    """Dataclass for decoder-only transformer parameters.

    Args:
        d_model (int): Input/output embedding dimension.
        max_seq_len (int): Maximum input sequence length.
        input_num_embeddings (int): Input vocabulary size.
        output_num_embeddings (int): Output vocabulary size.
        decoder_stack_num_layers (int): Number of decoder layers in decoder stack.
        decoder_params (EncoderDecoderParams): Decoder layer parameters.
        p_dropout (float): Dropout probability; used in positional encoding layer.
        tie_weight_matrices (bool): Tie weight matrices in input embedding and pre-softmax layers.
        apply_causal_mask (bool): Apply causal mask in each decoder layer.

    Raises:
        ValueError: If tie_weight_matrices is True and there is a mismatch between the input and
            output vocabulary sizes.

    Note:
        To tie weight matrices, we need the input vocabulary size and the output vocabulary size to
        be equal.

    # noqa: DCO060
    """

    d_model: int
    max_seq_len: int
    input_num_embeddings: int
    output_num_embeddings: int
    decoder_stack_num_layers: int
    decoder_params: EncoderDecoderParams
    p_dropout: float
    tie_weight_matrices: bool
    apply_causal_mask: bool

    def __iter__(self) -> Iterator[Any]:  # noqa: DCO010
        return iter(astuple(self))

    def __post_init__(self) -> None:  # noqa: DCO010
        num_embeddings_mismatch = self.input_num_embeddings != self.output_num_embeddings
        if self.tie_weight_matrices and num_embeddings_mismatch:
            raise ValueError(
                "Need input_num_embeddings == output_num_embeddings to tie weight matrices"
            )
