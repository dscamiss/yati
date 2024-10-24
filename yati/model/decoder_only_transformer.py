"""Implementation of decoder-only transformer."""

from jaxtyping import Float, Integer, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from yati.model.embedding import Embedding
from yati.model.positional_encoding import PositionalEncoding
from yati.model.reduced_decoder_stack import ReducedDecoderStack
from yati.params.decoder_only_transformer_params import DecoderOnlyTransformerParams


class DecoderOnlyTransformer(nn.Module):  # pylint: disable=abstract-method
    """Decoder-only transformer.

    Args:
        params (DecoderOnlyTransformerParams): Decoder-only transformer parameters.

    Note:
        Optionally, the input embedding and pre-softmax layers have tied weight matrices.

    Note:
        All parameters are initialized using Xavier/Glorot initialization, with PyTorch defaults.
    """

    def __init__(self, params: DecoderOnlyTransformerParams) -> None:  # noqa: DCO010
        super().__init__()
        d_model = params.d_model
        max_seq_len = params.max_seq_len
        p_dropout = params.p_dropout
        input_num_embeddings = params.input_num_embeddings
        output_num_embeddings = params.output_num_embeddings
        decoder_stack_num_layers = params.decoder_stack_num_layers
        decoder_params = params.decoder_params

        self._input_embedding = Embedding(d_model, input_num_embeddings)
        self._input_positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self._input_positional_encoding_dropout = nn.Dropout(p_dropout)
        self._decoder_stack = ReducedDecoderStack(
            decoder_stack_num_layers, decoder_params, max_seq_len
        )
        self._pre_softmax = nn.Linear(d_model, output_num_embeddings)

        # Tie weight matrices, if needed
        if params.tie_weight_matrices:
            self._pre_softmax.weight = self._input_embedding.embedding.weight

        # Initialize parameters
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Integer[Tensor, "b n"]) -> Float[Tensor, "b n output_num_embeddings"]:
        """Compute decoder-only transformer output.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Final output of decoder-only transformer.

        Note:
            Softmax is not applied, since cross_entropy_loss() expects logits.
        """
        x = self._input_embedding(x)  # (b, n, d_model)
        x = self._input_positional_encoding(x)  # (b, n, d_model)
        x = self._input_positional_encoding_dropout(x)  # (b, n, d_model)
        x = self._decoder_stack(x)  # (b, n, d_model)
        x = self._pre_softmax(x)  # (b, n, output_num_embeddings)

        return x
