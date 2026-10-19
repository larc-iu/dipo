from dipo.rst.parsers.gen.serializations.base import Serialization
from dipo.rst.parsers.gen.serializations.seg import SegSerialization
from dipo.rst.parsers.gen.serializations.sexp import SexpSerialization
from dipo.rst.parsers.gen.serializations.sr import SRSerialization

SERIALIZATIONS = {
    "sr": SRSerialization,
    "sexp": SexpSerialization,
    "seg": SegSerialization,
}

__all__ = ["Serialization", "SRSerialization", "SegSerialization", "SexpSerialization", "SERIALIZATIONS"]
