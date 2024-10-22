"""Implementation of dataclass for encoder/decoder transformer parameters."""

from dataclasses import astuple, dataclass
from typing import Any, Iterator

from model.encoder_decoder_params import EncoderDecoderParams


@dataclass
class EncoderDecoderTransformerParams:
    """Dataclass for encoder/decoder transformer parameters.

    Args:
        d_model (int): Input/output embedding dimension.
        max_seq_len (int): Maximum input sequence length.
        input_num_embeddings (int): Input vocabulary size.
        output_num_embeddings (int): Output vocabulary size.
        encoder_stack_num_layers (int): Number of encoder layers in encoder stack.
        encoder_params (EncoderDecoderParams): Encoder layer parameters.
        decoder_stack_num_layers (int): Number of decoder layers in decoder stack.
        decoder_params (EncoderDecoderParams): Decoder layer parameters.
        p_dropout (float): Dropout probability; used in positional encoding layers.
        tie_weight_matrices (bool): Tie weight matrices in input embedding, output
            embedding, and pre-softmax layers.

    Raises:
        ValueError: If tie_weight_matrices is True and there is a mismatch between the input and
            output vocabulary sizes.

    Note:
        To tie weight matrices, we need the input vocabulary size and the output vocabulary size to
        be equal.  This seems to be the case in the data used in AIAYN, based on this remark:
        "We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million
        sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a **shared
        source-target vocabulary** of about 37000 tokens. For English-French, we used the
        significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split
        tokens into a 32000 word-piece vocabulary [...]."
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

    def __iter__(self) -> Iterator[Any]:
        return iter(astuple(self))

    def __post_init__(self) -> None:
        num_embeddings_mismatch = self.input_num_embeddings != self.output_num_embeddings
        if self.tie_weight_matrices and num_embeddings_mismatch:
            raise ValueError(
                "Need input_num_embeddings == output_num_embeddings to tie weight matrices"
            )
