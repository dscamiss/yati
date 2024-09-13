"""Implementation of transformer block."""

import torch
import torch.nn as nn

from decoder import DecoderStack
from encoder import EncoderStack
from params import EncoderDecoderParams, TransformerParams
from shared import Embedding, PositionalEncoding
from torch import Tensor


class Transformer(nn.Module):
    """Transformer block.

    Note:
        The input embedding, output embedding, and pre-softmax affine transformation
        have tied weight matrices, based on this remark from AIAYN: "In our model, we share
        the same weight matrix between the two embedding layers and the pre-softmax linear
        transformation [...]."

    Note:
        Dropout is applied to the output of the positional encoding block, following this remark
        from AIAYN: "In addition, we apply dropout to the sums of the embeddings and the positional
        encodings in both the encoder and decoder stacks."

    Note:
        All parameters are initialized using Xavier/Glorot initialization with PyTorch defaults.
        This is because parameter initialization details are not specified in AIAYN.
    """

    def __init__(self, t: TransformerParams, p: EncoderDecoderParams) -> None:
        super().__init__()

        # Encoder side
        self.input_embedding = Embedding(p.d_model, t.input_num_embeddings)
        self.input_positional_enc = PositionalEncoding(p.d_model, p.input_max_seq_len)
        self.input_positional_enc_dropout = nn.Dropout(p.dropout_prob)
        self.encoder_stack = EncoderStack(p, t.num_encoder_blocks)

        # Decoder side
        self.output_embedding = Embedding(p.d_model, t.output_num_embeddings)
        self.output_positional_enc = PositionalEncoding(p.d_model, p.output_max_seq_len)
        self.output_positional_enc_dropout = nn.Dropout(p.dropout_prob)
        self.decoder_stack = DecoderStack(p, t.num_decoder_blocks)
        self.pre_softmax = nn.Linear(p.d_model, t.output_num_embeddings, bias=True)

        # Tie weight matrices, if specified
        if t.tie_weight_matrices:
            self.output_embedding.embedding.weight = (
                self.input_embedding.embedding.weight
            )
            self.pre_softmax.weight = self.input_embedding.embedding.weight

        # Initialize parameters
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def encode(self, x: Tensor) -> Tensor:
        """Compute the encoder side output.

        Args:
            x: Input tensor of size (b, n, d_model).
        """
        x = self.input_embedding(x)
        x = self.input_positional_enc(x)
        x = self.input_positional_enc_dropout(x)
        x = self.encoder_stack(x)
        return x

    def decode(self, x: Tensor, y_enc: Tensor) -> Tensor:
        """Compute the decoder side output.

        Args:
            x: Input tensor of size (b, n, d_model).
            y_enc: Encoder stack output tensor of size (b, n, d_model).
        """
        x = self.output_embedding(x)
        x = self.output_positional_enc(x)
        x = self.output_positional_enc_dropout(x)
        x = self.decoder_stack(x, y_enc)
        x = self.pre_softmax(x)
        x = torch.softmax(x, dim=-1)
        return x


def main():
    t = TransformerParams(
        input_num_embeddings=333,
        output_num_embeddings=444,
        num_encoder_blocks=6,
        num_decoder_blocks=6,
        tie_weight_matrices=False,
    )

    p = EncoderDecoderParams(
        h=8,
        d_model=512,
        d_k=64,
        d_v=64,
        d_ff=1024,
        dropout_prob=0.1,
        input_max_seq_len=2048,
        output_max_seq_len=2048,
    )

    transformer = Transformer(t, p)
    input_x = torch.randint(0, t.input_num_embeddings, (8, 16))
    output_x = torch.randint(0, t.output_num_embeddings, (8, 16))
    y_enc = transformer.encode(input_x)
    y_dec = transformer.decode(output_x, y_enc)

    print(f"input_x  size = {input_x.size()}")
    print(f"output_x size = {output_x.size()}")
    print(f"y_enc    size = {y_enc.size()}")
    print(f"y_dec    size = {y_dec.size()}")


if __name__ == "__main__":
    main()
