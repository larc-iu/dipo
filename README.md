# Dipo

The **<u>Di</u>scourse <u>P</u>arsing <u>O</u>mnibus** (Dipo, pronounced "depot") is a collection of easy-to-use discourse parsers and other code related to discourse parsing.

## Setup

For the latest release:

```
pip install dipo
```

For the current state of `master`:

```
pip install git+https://github.com/larc-iu/dipo
```

Or for development:

```
git clone https://github.com/larc-iu/dipo && cd dipo
pip install -e .
```

The old `iudex` command still works for now, as a deprecated alias of `dipo` that prints a notice. `import iudex` is now `import dipo`.

## Quick Start with Inference

Parse a sample document end-to-end with a pretrained DMRST model pulled from the HuggingFace Hub. From the command line:

```bash
dipo dmrst predict \
    --hub-id larc-iu/dmrst-ettin-400m-gum-12.1.0 \
    --text "Although the experiment was carefully designed, the results were inconclusive. We plan to repeat it tonight."
```
This yields the parsed tree in `.rs3` format printed to `stdout`:
```xml
<rst>
  <relations><!-- ... --></relations>
  <body>
    <segment id="1" parent="2" relname="adversative-concession">Although the experiment was carefully # designed,</segment>
    <segment id="2" parent="4" relname="span">the results were inconclusive.</segment>
    <segment id="3" parent="5" relname="span">We plan to repeat it tonight.</segment>
    <group id="4" type="span" parent="3" relname="adversative-antithesis"/>
    <group id="5" type="span"/>
  </body>
</rst>
```

The same flow from Python:

```python
from dipo.rst.parsers.dmrst.modeling_dmrst import DMRSTParser
parser = DMRSTParser.from_pretrained("larc-iu/dmrst-ettin-400m-gum-12.1.0")
tree = parser.predict_from_text(
    "Although the experiment was carefully designed, "
    "the results were inconclusive. "
    "We plan to repeat it tonight."
)
print(tree.to_rs4_string())
```
Yields:
```xml
<rst>
  <relations><!-- ... --></relations>
  <body>
    <segment id="1" parent="2" relname="adversative-concession">Although the experiment was carefully # designed,</segment>
    <segment id="2" parent="4" relname="span">the results were inconclusive.</segment>
    <segment id="3" parent="5" relname="span">We plan to repeat it tonight.</segment>
    <group id="4" type="span" parent="3" relname="adversative-antithesis"/>
    <group id="5" type="span"/>
  </body>
</rst>
```

## Inference CLI

To identify a model on the command line, you may use a configuration file (`--config`), a PyTorch checkpoint (`--checkpoint`), or a HuggingFace Hub repository (`--hub-id`).

To provide input, you may specify an inline string (`--text`), a path to a raw text file or directory (`--text-file`, for parsers which support this), or an RS3/RS4 file or directory with gold EDUs already supplied (`--input`).

For `--text-file` and `--input`, results are written to `--output-dir` as `.rs4` files.

```
# From the Hub, end-to-end on a directory of .txt files:
dipo dmrst predict \
    --hub-id larc-iu/dmrst-ettin-400m-gum-12.1.0 \
    --text-file path/to/docs/ \
    --output-dir out/ \
    --device cuda

# From an explicit checkpoint:
dipo dmrst predict \
    --checkpoint checkpoints/<run_id>/best_model.pt \
    --text-file path/to/doc.txt \
    --output-dir out/

# From a trained run's config, parsing pre-segmented RS3/RS4 with gold EDUs:
dipo topdown_biaffine predict \
    --config configs/topdown_biaffine_rstdt.jsonnet \
    --input data/rstdt/test \
    --output-dir out/
```

## Available Models
All official Dipo model releases are [tagged with `dipo` on the HuggingFace Hub](https://huggingface.co/models?other=dipo) (they also carry the old `iudex` tag for now).

All of the models below parse raw text end to end (segmentation and structure) and were trained on untokenized text, so plain prose works as input.
There is no need to tokenize it or to remove line breaks, and Chinese should be given unspaced.
Scores are original Parseval Full F1 on each corpus's test set, end to end from raw text.

| Hub ID | Parser | Backbone | Corpus | Test Full |
| --- | --- | --- | --- | --- |
| [`larc-iu/dmrst-ettin-400m-rstdt-coarse`](https://huggingface.co/larc-iu/dmrst-ettin-400m-rstdt-coarse) | `dmrst` | Ettin encoder 400M | RST-DT (English) | 53.2 |
| [`larc-iu/gen-sr-t5gemma-2-1b-1b-rstdt-coarse`](https://huggingface.co/larc-iu/gen-sr-t5gemma-2-1b-1b-rstdt-coarse) | `gen` | T5Gemma 2 1B-1B | RST-DT (English) | 49.6 |
| [`larc-iu/gen-sr-gemma-4-31b-it-rstdt-coarse`](https://huggingface.co/larc-iu/gen-sr-gemma-4-31b-it-rstdt-coarse) | `gen` | Gemma 4 31B | RST-DT (English) | 54.7 |
| [`larc-iu/dmrst-ettin-400m-gum-12.1.0`](https://huggingface.co/larc-iu/dmrst-ettin-400m-gum-12.1.0) | `dmrst` | Ettin encoder 400M | GUM 12.1.0 (English) | 45.6 |
| [`larc-iu/gen-sr-t5gemma-2-1b-1b-gum-12.1.0`](https://huggingface.co/larc-iu/gen-sr-t5gemma-2-1b-1b-gum-12.1.0) | `gen` | T5Gemma 2 1B-1B | GUM 12.1.0 (English) | 39.4 |
| [`larc-iu/gen-sr-gemma-4-31b-it-gum-12.1.0`](https://huggingface.co/larc-iu/gen-sr-gemma-4-31b-it-gum-12.1.0) | `gen` | Gemma 4 31B | GUM 12.1.0 (English) | 45.8 |
| [`larc-iu/dmrst-xlm-roberta-base-ert`](https://huggingface.co/larc-iu/dmrst-xlm-roberta-base-ert) | `dmrst` | XLM-R base | RST Basque TreeBank | 27.6 |
| [`larc-iu/dmrst-xlm-roberta-base-pcc-2.2`](https://huggingface.co/larc-iu/dmrst-xlm-roberta-base-pcc-2.2) | `dmrst` | XLM-R base | Potsdam Commentary Corpus 2.2 (German) | 18.6 |
| [`larc-iu/dmrst-xlm-roberta-base-prstc`](https://huggingface.co/larc-iu/dmrst-xlm-roberta-base-prstc) | `dmrst` | XLM-R base | Persian RST Corpus | 32.2 |
| [`larc-iu/dmrst-xlm-roberta-base-gcdt`](https://huggingface.co/larc-iu/dmrst-xlm-roberta-base-gcdt) | `dmrst` | XLM-R base | GCDT (Chinese) | 29.4 |

The `dmrst` models cannot see line breaks, so `predict_from_text` also starts a new segment at blank-line paragraph breaks and after short heading-like lines.
Pass `break_at_paragraphs=False` to use the model's own segmentation alone.

The `gen` models are built on a pretrained backbone, and loading one also downloads that backbone from the Hub.
Both backbones are gated, so before the first run accept Google's terms on the backbone's page (`google/t5gemma-2-1b-1b` for the T5Gemma models, `google/gemma-4-31B-it` for the Gemma 4 models) and log in with `huggingface-cli login` or set `HF_TOKEN`. Without that, loading fails with a "gated repo" 401 error.
The two Gemma 4 models are small adapters (about 0.5 GB) and need a GPU with roughly 70 GB of memory, and loading one takes 1 to 2 minutes.
The T5Gemma models store their full weights (about 4.4 GB).
They are invoked like the others, e.g. `dipo gen predict --hub-id larc-iu/gen-sr-t5gemma-2-1b-1b-gum-12.1.0 --text "..."`.
A document longer than a `gen` model's context window (16,384 tokens) is rejected with an error rather than silently truncated.

## Training

To train a new top-down biaffine parser on RSTDT:

```
dipo topdown_biaffine train configs/topdown_biaffine_rstdt.jsonnet
```

Note that `configs/topdown_biaffine_rstdt.jsonnet` is a configuration.
You may either edit it directly or copy and modify it in a new location.

### Training on Raw Text

Several corpora distribute their EDUs word-tokenized (`the drug trade ,`), and a parser trained on that text sees punctuation spacing at inference that real input never has.
`scripts/build_raw_text_data.py` writes untokenized copies of RST-DT, PCC, GCDT and the Persian corpus to `data/<corpus>_notok`, with identical trees, and the released models were trained on these.
RST-DT's raw text comes from the LDC release, which must be present at `data/rst_discourse_treebank`.
For unspaced Chinese, also set `split_cjk_tokens: true` in a DMRST config, so that every EDU boundary falls on a token boundary.

### Grabbing Example Configurations

Model configurations required for training are not bundled with the package distributed via PyPI.

To get them you may visit [the associated directory](https://github.com/larc-iu/dipo/tree/master/configs) and download the configurations you're interested in manually.

If you want to grab all of them at once, you can use the command line like so:

**bash / zsh / macOS / Linux:**

```bash
curl -fL https://github.com/larc-iu/dipo/archive/refs/heads/master.tar.gz \
  | tar -xz --strip-components=1 --wildcards '*/configs'
```

**Windows PowerShell:**

```powershell
Invoke-WebRequest https://github.com/larc-iu/dipo/archive/refs/heads/master.zip -OutFile dipo.zip
Expand-Archive dipo.zip -DestinationPath .
Move-Item dipo-master/configs configs
Remove-Item -Recurse -Force dipo-master, dipo.zip
```

Either leaves you with a local `configs/` directory you can edit and pass to `dipo … train configs/<name>.jsonnet`.

### Configuration Hashes

Your configuration is used as the basis for a unique hash, which (by default) corresponds to a directory under `checkpoints/`.
This hash is used for several purposes.
For example, running the same config again resumes from the last epoch's checkpoint `last.pt` automatically if the run was interrupted.

To view all runs and their status, you may run the `runs list` subcommand:

```
$ dipo runs list
                                                            Runs in checkpoints                                                            
┏━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ run_id       ┃ run_name ┃ parser           ┃ model_name                   ┃ train_dir             ┃  best_val ┃ step ┃ modified         ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ 245b1d774676 │ -        │ dmrst            │ xlm-roberta-base             │ data/gum_12.1.0/train │    0.3099 │ 1704 │ 2026-05-18 18:02 │
│ 41bc0fe1dd50 │ -        │ topdown_biaffine │ SpanBERT/spanbert-base-cased │ data/rstdt/train      │    0.7576 │ 2149 │ 2026-05-18 13:51 │
│ 91525e48d63d │ -        │ topdown_biaffine │ SpanBERT/spanbert-base-cased │ data/gum_12.1.0/train │    0.6364 │ 1899 │ 2026-05-18 14:31 │
│ ad934ca992d4 │ -        │ dmrst            │ xlm-roberta-base             │ data/rstdt/train      │    0.4665 │ 3090 │ 2026-05-18 16:46 │
└──────────────┴──────────┴──────────────────┴──────────────────────────────┴───────────────────────┴───────────┴──────┴──────────────────┘
```

### Monitoring with TensorBoard

Every run writes TensorBoard scalars (train loss, learning rate, gradient norm, and dev metrics) to `<run_dir>/tb/`. Point TensorBoard at your checkpoints directory to watch any run live or compare runs:

```
tensorboard --logdir checkpoints/
```

### Pushing Models to HF Hub
You may host a trained model using each parser's `push` subcommand.
Each uploads `best_model.pt`, `config.json`, and an auto-generated `README.md` in a single commit:

```
dipo topdown_biaffine push \
    --config configs/topdown_biaffine_rstdt.jsonnet \
    --repo-id larc-iu/topdown_biaffine-rstdt-coarse \
    [--private] [--message "..."] [--token $HF_TOKEN]
```

## Citation

If you use Dipo in your research, please cite it as:

> Gessler, Luke. 2026. *Dipo: The Discourse Parsing Omnibus.* https://github.com/larc-iu/dipo.

BibTeX:

```bibtex
@misc{gessler-dipo-2026,
  author       = {Gessler, Luke},
  title        = {{Dipo: The Discourse Parsing Omnibus}},
  year         = {2026},
  howpublished = {\url{https://github.com/larc-iu/dipo}},
}
```

If you use one of the included parser re-implementations, please **also** cite the original paper (see each model's Hub card for the canonical reference).
