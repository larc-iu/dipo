# maekawa

An RST parser that asks a LoRA-adapted causal LM one parsing decision per prompt, plus a
bundled EDU segmenter on the same model so it can read raw text.

- **Paper:** [Can we obtain significant success in RST discourse parsing by using Large
  Language Models?](https://aclanthology.org/2024.eacl-long.171/) (Maekawa, Hirao,
  Kamigaito and Okumura, EACL 2024)
- This is an independent reimplementation. It reproduces the paper's prompt formats
  exactly, so adapters trained with their setup receive identical inputs, but contains no
  code from the authors' repository.

## Design

Parsing walks a tree top-down (split a span, then its two halves) or bottom-up (shift and
reduce over a stack and queue). Each step is a short prompt holding only the text that
decision needs, and the model answers in a few tokens. Illegal answers fall back to a
default part by part, and every decision is logged with what was generated and what fell
back. Documents parse in lockstep, so each round's decisions run as one batch.

`label_mode` sets which adapters answer what:

- **`unified`** (default): one adapter answers everything. Top-down it answers each node
  with `<split> <NS|SN|NN> <relation>`, bottom-up with `shift` or
  `reduce <NS|SN|NN> <relation>`.
- **`separate`** (the paper's): the structural decision, nuclearity, and the relation
  given that nuclearity, each with its own adapter.
- **`joint`**: separate, but nuclearity and relation as one decision.

The defaults are the strongest setup we found on GUM: top-down, `unified`, Qwen 3.8 27B
with LoRA (r=64, alpha=16) on a bf16 base. The paper's setup is bottom-up, `separate`,
Llama 2 on a 4-bit base: `peft.load_in_4bit: true`, which needs the `qlora` extra
(`pip install dipo[qlora]`).

The segmenter (`segmentation`, on by default) is `gen`'s `seg` serialization: the model
copies every source token and closes each EDU with a shift action, so the EDUs always
tile the input. It is one more LoRA adapter on the parser's base model, with a small
output head and a few new token rows of its own, swapped in only while segmenting.
`predict_from_text` segments, then parses, and training adds end-to-end scores to the
final evaluation. With `segmentation: null` the parser takes gold or externally
segmented EDUs only.

## Training

Each adapter trains on its own prompts for `epochs` epochs, keeping the epoch with the
best dev decision accuracy under gold parser states. The segmenter keeps its best dev
segmentation F1. Training hyperparameters follow the paper (lr 2e-4, an effective batch
of 16 decisions, cosine schedule, 5 epochs). Decisions are packed into length-grouped,
token-capped micro-batches, which optimizes the same objective as the paper's 1 x 16
accumulation and trains several times faster.

Trained adapters are exported as standard PEFT directories under `<run>/adapters/`, and
`pretrained_adapters` loads them (from disk or the Hub) instead of training a task.
