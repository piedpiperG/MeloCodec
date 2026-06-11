"""Core MeloCodec architecture.

The architecture is named ``BWC`` in the codebase and corresponds to
MeloCodec in the paper and project page.
"""

from .bwc import BWC, MeloCodec
from .chroma import ChromaCodec
from .quantize import ResidualVectorQuantize, VectorQuantize

__all__ = [
    "BWC",
    "MeloCodec",
    "ChromaCodec",
    "ResidualVectorQuantize",
    "VectorQuantize",
]
