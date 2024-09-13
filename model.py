"""yati: Yet another transformer implementation."""

import math
import torch
import torch.nn as nn

from decoder import DecoderStack
from encoder import EncoderStack
from shared import Embedding, EncoderDecoderBlockParams
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Positional encoding block.

    Definition from AIAYN:
        PE(pos, 2i)     = sin(pos / 10000^{2i / d_model})
        PE(pos, 2i + 1) = cos(pos / 10000^{2i / d_model})

    For numerical stability, we exponentiate logarithms:
        log(1 / 10000^{2i / d_model}) = -(log(10000) / d_model) 2i

    """

    def __init__(self, d_model: int, max_seq_len: int) -> None:
        super().__init__()

        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(-1)  # (n, 1)
        idx = torch.arange(0, d_model, 2, dtype=torch.float)
        log_den = -(math.log(10000.0) / d_model) * idx
        den = torch.exp(log_den).unsqueeze(0)  # (1, d_model)

        pe = torch.zeros(1, max_seq_len, d_model)
        pe[:, :, 0::2] = torch.sin(pos * den)
        pe[:, :, 1::2] = torch.cos(pos * den)

        # Ensure `pe` is saved as part of the module state.
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input tensor.

        Args:
        ----
            x (`torch.Tensor`): Input tensor of shape (b, n, d_model)

        """
        n = x.shape[1]
        return x + self.pe[:, :n, :].requires_grad_(False)


class TransformerParams:
    """Data class for (higher-level) transformer parameters.

    Args:
        input_num_embeddings: Vocabulary size for inputs.
        input_num_embeddings: Vocabulary size for outputs.
        input_max_seq_len: Maximum sequence length for inputs.
        output_max_seq_len: Maximum sequence length for outputs.
        num_encoder_blocks: Number of encoder blocks in the encoder stack.
        num_decoder_blocks: Number of decoder blocks in the decoder stack.
    """

    def __init__(
        self,
        input_num_embeddings: int,
        output_num_embeddings: int,
        input_max_seq_len: int,
        output_max_seq_len: int,
        num_encoder_blocks: int,
        num_decoder_blocks: int,
    ):
        self.input_num_embeddings = input_num_embeddings
        self.output_num_embeddings = output_num_embeddings
        self.input_max_seq_len = input_max_seq_len
        self.output_max_seq_len = output_max_seq_len
        self.num_encoder_blocks = num_encoder_blocks
        self.num_decoder_blocks = num_decoder_blocks


class Transformer(nn.Module):
    """Transformer.

    Note:
        The input embedding, output embedding, and pre-softmax affine transformation
        have tied weight matrices, based on this remark from AIAYN: "In our model, we share
        the same weight matrix between the two embedding layers and the pre-softmax linear
        transformation [...]."

    Note:
        Dropout is applied to the output of the positional encoding block, following this remark
        from AIAYN: "In addition, we apply dropout to the sums of the embeddings and the positional
        encodings in both the encoder and decoder stacks."
    """

    def __init__(self, t: TransformerParams, p: EncoderDecoderBlockParams) -> None:
        # Entities related to "encoder side"
        self.input_embedding = Embedding(p.d_model, t.input_num_embeddings)
        self.input_positional_encoding = PositionalEncoding(
            p.d_model, t.input_max_seq_len
        )
        self.input_positional_encoding_dropout = nn.Dropout(p.dropout_prob)
        self.encoder_stack = EncoderStack(p, t.num_encoder_blocks)

        # Entities related to "decoder side"
        self.output_embedding = Embedding(p.d_model, t.output_num_embeddings)
        self.output_positional_encoding = PositionalEncoding(
            p.d_model, t.output_max_seq_len
        )
        self.output_positional_encoding_dropout = nn.Dropout(p.dropout_prob)
        self.decoder_stack = DecoderStack(p, t.num_decoder_blocks)
        self.pre_softmax_affine = nn.Linear(
            p.d_model, t.output_num_embeddings, bias=True
        )

        # Tie weight matrices
        self.output_embedding.embedding.weight = self.input_embedding.embedding.weight
        self.pre_softmax_affine.weight = self.input_embedding.embedding.weight

    def encode(self, x: Tensor) -> Tensor:
        """Compute the transformer's "encoder side" output.

        Args:
            x: Input tensor of size (b, n, d_model).
        """
        x = self.input_embedding(x)
        x = self.input_positional_encoding(x)
        x = self.input_positional_encoding_dropout(x)
        x = self.encoder_stack(x)
        return x

    def decode(self, x: Tensor, y_enc: Tensor) -> Tensor:
        """Compute the transformer's "decoder side" output.

        Args:
            x: Input tensor of size (b, n, d_model).
            y_enc: Encoder stack output tensor of size (b, n, d_model).
        """
        x = self.output_embedding(x)
        x = self.output_positional_encoding(x)
        x = self.output_positional_encoding_dropout(x)
        x = self.decoder_stack(x)
        x = self.pre_softmax_affine(x)
        x = torch.softmax(x, dim=-1)
        return x
