"""Implementation of dataclass for encoder/decoder layer parameters."""

from dataclasses import astuple, dataclass
from typing import Any, Iterator


@dataclass
class EncoderDecoderParams:
    """Dataclass for encoder/decoder layer parameters.

    Args:
        d_input (int): Input dimension.
        h (int): Number of heads.
        d_k (int): Number of rows in {Q, K}-matrices.
        d_v (int): Number of rows in V-matrix.
        d_ff (int): Hidden layer dimension.
        p_dropout (float): Dropout probability.
    """

    d_input: int
    h: int
    d_k: int
    d_v: int
    d_ff: int
    p_dropout: float

    def __iter__(self) -> Iterator[Any]:
        return iter(astuple(self))
