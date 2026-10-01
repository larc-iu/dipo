from iudex.rst.parsers.gen.serializations.base import Serialization
from iudex.rst.parsers.gen.serializations.sexp import SexpSerialization
from iudex.rst.parsers.gen.serializations.sr import SRSerialization

SERIALIZATIONS = {
    "sr": SRSerialization,
    "sexp": SexpSerialization,
}

__all__ = ["Serialization", "SRSerialization", "SexpSerialization", "SERIALIZATIONS"]
