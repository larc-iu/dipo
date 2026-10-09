from dipo.rst.parsers.gen.backbones.base import Backbone
from dipo.rst.parsers.gen.backbones.decoder_only import DecoderOnlyBackbone
from dipo.rst.parsers.gen.backbones.seq2seq import Seq2SeqBackbone

BACKBONES = {
    "decoder_only": DecoderOnlyBackbone,
    "seq2seq": Seq2SeqBackbone,
}

__all__ = ["Backbone", "DecoderOnlyBackbone", "Seq2SeqBackbone", "BACKBONES"]
