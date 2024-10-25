"""Implementation of transformer."""

from jaxtyping import Float, Integer, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from yati.model.decoder_stack import DecoderStack
from yati.model.embedding import Embedding
from yati.model.encoder_stack import EncoderStack
from yati.model.positional_encoding import PositionalEncoding
from yati.params.transformer_params import TransformerParams


class Transformer(nn.Module):  # pylint: disable=abstract-method
    """Transformer.

    Args:
        params: Transformer parameters.

    Note:
        Optionally, the input embedding, output embedding, and pre-softmax
        layers have tied weight matrices, following this remark from AIAYN:
        "In our model, we share the same weight matrix between the two
        embedding layers and the pre-softmax linear transformation [...]."

    Note:
        All parameters are initialized using Xavier/Glorot initialization,
        with PyTorch defaults.  AIAYN does not have initialization details.
    """

    def __init__(self, params: TransformerParams) -> None:  # noqa: DCO010
        super().__init__()
        d_model = params.d_model
        max_seq_len = params.max_seq_len
        p_dropout = params.p_dropout

        # Encoder-side objects
        input_num_embeddings = params.input_num_embeddings
        encoder_stack_num_layers = params.encoder_stack_num_layers
        encoder_params = params.encoder_params

        self._input_embedding = Embedding(d_model, input_num_embeddings)
        self._input_positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self._input_positional_encoding_dropout = nn.Dropout(p_dropout)
        self._encoder_stack = EncoderStack(encoder_stack_num_layers, encoder_params)

        # Decoder-side objects
        output_num_embeddings = params.output_num_embeddings
        decoder_stack_num_layers = params.decoder_stack_num_layers
        decoder_params = params.decoder_params

        self._output_embedding = Embedding(d_model, output_num_embeddings)
        self._output_positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self._output_positional_encoding_dropout = nn.Dropout(p_dropout)
        self._decoder_stack = DecoderStack(decoder_stack_num_layers, decoder_params, max_seq_len)
        self._pre_softmax = nn.Linear(d_model, output_num_embeddings)

        # Tie weight matrices, if needed
        if params.tie_weight_matrices:
            self._output_embedding.embedding.weight = self._input_embedding.embedding.weight
            self._pre_softmax.weight = self._input_embedding.embedding.weight

        # Initialize parameters
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    @jaxtyped(typechecker=typechecker)
    def encode(self, x: Integer[Tensor, "b n"]) -> Float[Tensor, "b n d_model"]:
        """Compute encoder-side output.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Final output of encoder side of transformer.

        Note:
            Dropout is applied to the output of the positional encoding block,
            following this remark from AIAYN: "In addition, we apply dropout to
            the sums of the embeddings and the positional encodings in both the
            encoder and decoder stacks."
        """
        x = self._input_embedding(x)  # (b, n, d_model)
        x = self._input_positional_encoding(x)  # (b, n, d_model)
        x = self._input_positional_encoding_dropout(x)  # (b, n, d_model)
        x = self._encoder_stack(x)  # (b, n, d_model)

        return x

    @jaxtyped(typechecker=typechecker)
    def decode(
        self, x: Integer[Tensor, "b n"], x_cross: Float[Tensor, "b n d_model"]
    ) -> Float[Tensor, "b n output_num_embeddings"]:
        """Compute decoder-side output.

        Args:
            x: Input tensor.
            x_cross: Cross-attention input tensor.

        Returns:
            Tensor: Final output of decoder side of transformer.

        Note:
            Dropout is applied to the output of the positional encoding block,
            following this remark from AIAYN: "In addition, we apply dropout to
            the sums of the embeddings and the positional encodings in both the
            encoder and decoder stacks."

        Note:
            Softmax is not applied, since cross_entropy() expects logits.
        """
        x = self._output_embedding(x)  # (b, n, d_model)
        x = self._output_positional_encoding(x)  # (b, n, d_model)
        x = self._output_positional_encoding_dropout(x)  # (b, n, d_model)
        x = self._decoder_stack(x, x_cross)  # (b, n, d_model)
        x = self._pre_softmax(x)  # (b, n, output_num_embeddings)

        return x
