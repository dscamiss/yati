"""Implementation of data classes for parameters."""


class EncoderDecoderParams:
    """Data class for encoder/decoder block parameters.

    Args:
        h: Number of heads in multi-head attention block.
        d_model: Embedding dimension.
        d_k: Number of rows in "Q" and "K" matrices.
        d_v: Number of rows in "V" matrix.
        d_ff: Hidden layer dimension in feed-forward block.
        dropout_prob: Dropout probability.
        input_max_seq_len: Maximum sequence length for outputs.
        output_max_seq_len: Maximum sequence length for outputs.
    """

    def __init__(
        self,
        h: int,
        d_model: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        dropout_prob: float,
        input_max_seq_len: int,
        output_max_seq_len: int,
    ):
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.dropout_prob = dropout_prob
        self.input_max_seq_len = input_max_seq_len
        self.output_max_seq_len = output_max_seq_len


class TransformerParams:
    """Data class for transformer block parameters.

    Args:
        input_num_embeddings: Vocabulary size for inputs.
        input_num_embeddings: Vocabulary size for outputs.
        num_encoder_blocks: Number of encoder blocks in the encoder stack.
        num_decoder_blocks: Number of decoder blocks in the decoder stack.
        tie_weight_matrices: Tie weight matrices between the input embedding,
            output embedding, and pre-softmax blocks.

    Note:
        To tie weight matrices, we need the input vocabulary size and the output vocabulary size to
        be equal.  This seems to be the case in the data used in AIAYN, based on this remark:
        "We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million
        sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a shared
        source-target vocabulary of about 37000 tokens. For English-French, we used the significantly
        larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a
        [shared] 32000 word-piece vocabulary [...]."
    """

    def __init__(
        self,
        input_num_embeddings: int,
        output_num_embeddings: int,
        num_encoder_blocks: int,
        num_decoder_blocks: int,
        tie_weight_matrices: bool,
    ):
        self.input_num_embeddings = input_num_embeddings
        self.output_num_embeddings = output_num_embeddings
        self.num_encoder_blocks = num_encoder_blocks
        self.num_decoder_blocks = num_decoder_blocks
        self.tie_weight_matrices = tie_weight_matrices

        if self.tie_weight_matrices:
            # Sanity check: `input_num_embeddings` must be equal to `output_num_embeddings`
            assert (
                input_num_embeddings == output_num_embeddings
            ), "input_num_embeddings and output_num_embeddings must be equal"
