"""Implementation of dataclass for encoder/decoder layer parameters."""

from dataclasses import astuple, dataclass
from typing import Any, Iterator


@dataclass
class EncoderDecoderParams:
    """Dataclass for encoder/decoder layer parameters.

    Args:
        d_input: Input dimension.
        h: Number of heads.
        d_k: Number of rows in "Q" and "K" matrices.
        d_v: Number of rows in "V" matrix.
        d_ff: Hidden layer dimension.
        p_dropout: Dropout probability for sub-layers.

    # noqa: DCO060
    """

    d_input: int
    h: int
    d_k: int
    d_v: int
    d_ff: int
    p_dropout: float

    def __iter__(self) -> Iterator[Any]:  # noqa: DCO010
        return iter(astuple(self))
