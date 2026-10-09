# parsers/common

Code shared at train/predict time across the RST parsers. Per CLAUDE.md, only
reusable `nn.Module`s and byte-identical pure helpers live here; each parser's
training/predict loop stays in its own folder for self-contained reading.

## What's here

Shared by several parsers:
- `config.py` — config parsing helpers + the `PeftConfig` LoRA dataclass (the one
  config dataclass shared by every parser).
- `encoding.py` — `load_encoder_and_tokenizer` (the encoder-based parsers).
- `curriculum.py` — `Curriculum` strategies (`SimpleCurriculum` / `SubtreeSizeCurriculum`), every parser.
- `detokenization.py` — the `Detokenizer` abstraction.
- `inference.py` / `predict_cli.py` — source/checkpoint resolution + the shared predict CLI.

Encoder-based parsers (`dmrst`, `topdown_biaffine`, `sr_biaffine`):
- `segmentation.py` — the EDU-boundary `Segmenter` (dmrst).
- `biaffine.py` — the `DeepBiAffine` span scorer (topdown_biaffine, sr_biaffine).
- `pointer.py` — `PointerAttention` (dmrst).

Generative parser `gen` (backbone `decoder_only`|`seq2seq` × serialization `sr`|`sexp`):
- `seqgen.py` — EDU→token alignment, beam-search primitives, KV-cache reorder, the
  shift-reduce `ShiftReduceDecodeState`, head warm-init, word-label vocab.
- `sexp_constraints.py` — the s-expression pushdown automaton (`SexpDecodingState`) and
  `GoldEduForcer` (the `sexp` serialization).
- `generative_eval.py` — the shared dev/test eval orchestration (`evaluate_on_dev`),
  which talks to `gen` only through the small `GenerativeParser` Protocol.

## Reading the generative parser (start here)

`gen` is a 2×2 composition: a BACKBONE strategy (encoder-decoder `seq2seq` vs causal
`decoder_only`) × a SERIALIZATION strategy (shift-reduce `sr` vs s-expression `sexp`),
selected by config. The two axes never reference each other (see CLAUDE.md,
"generative parser", for why). A productive reading order:

1. `gen/modeling_gen.py` — `GenParser` composes a backbone + a serialization; read the
   handshake + encode_target/forward/predict flow.
2. `gen/decode.py` — the single greedy/beam decode core, parameterized over backbone
   I/O + a serialization state machine + a mask source.
3. `gen/backbones/{base,decoder_only,seq2seq}.py` — the backbone axis (model, packaging,
   the causal single stream vs the encoder-decoder compound cache).
4. `gen/serializations/{base,sr,sexp}.py` + `seqgen.py` / `sexp_constraints.py` — the
   serialization axis (linearize, loss, the SR state machine / the sexp PDA + forcer).
5. `generative_eval.py` — how `gen` is evaluated through one Protocol.
