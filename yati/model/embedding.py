"""Implementation of embedding layer."""

import math

from jaxtyping import Float, Integer, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker


class Embedding(nn.Module):
    """Embedding layer.

    Args:
        d_model (int): Embedding dimension.
        num_embeddings (int): Vocabulary size.
    """

    def __init__(self, d_model: int, num_embeddings: int) -> None:
        super().__init__()
        self._embedding = nn.Embedding(num_embeddings, d_model)
        self._scaling_factor = math.sqrt(d_model)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Integer[Tensor, "b n"]) -> Float[Tensor, "b n d_model"]:
        """Compute embedding output.

        Args:
            x (Tensor): Input tensor.

        Note:
            Multiplying the embedding output by scaling_factor should have the same effect
            as multiplying the embedding layer's weight matrix by scaling_factor.  From AIAYN:
            "Similarly to other sequence transduction models, we use learned embeddings
            to convert the input tokens and output tokens to vectors of dimension d_model.
            [...] In our model, we share the same weight matrix between the two embedding layers
            and the pre-softmax linear transformation, similar to [30]. In the embedding layers,
            we multiply those weights by sqrt(d_model)."
        """
        return self._scaling_factor * self._embedding(x)
