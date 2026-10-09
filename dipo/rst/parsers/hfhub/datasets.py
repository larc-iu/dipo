"""Registry of known RST corpora for model-card rendering.

Keys are directory prefixes. `lookup("data/rstdt/train")` returns the
RST-DT entry. Fields beyond `name` are optional. Missing ones are omitted
from the card.
"""

from __future__ import annotations

DATASETS: dict[str, dict] = {
    "data/rstdt": {
        "name": "RST Discourse Treebank (RST-DT)",
        "url": "https://catalog.ldc.upenn.edu/LDC2002T07",
        "language": "English",
        "description": (
            "385 WSJ articles from the Penn Treebank, annotated in RST with a fine-grained relation inventory. "
            "RST-DT is the traditional English benchmark for RST parsing."
        ),
        "citation_text": (
            "Carlson, Lynn, Daniel Marcu, and Mary Ellen Okurovsky. 2001. "
            "Building a Discourse-Tagged Corpus in the Framework of Rhetorical Structure Theory. "
            "In Proceedings of the Second SIGdial Workshop on Discourse and Dialogue."
        ),
        "citation_bibtex": (
            "@inproceedings{carlson-etal-2001-building,\n"
            "    title = {Building a Discourse-Tagged Corpus in the Framework of {R}hetorical "
            "{S}tructure {T}heory},\n"
            "    author = {Carlson, Lynn and Marcu, Daniel and Okurovsky, Mary Ellen},\n"
            "    booktitle = {Proceedings of the Second {SIG}dial Workshop on Discourse and Dialogue},\n"
            "    year = {2001},\n"
            "    url = {https://aclanthology.org/W01-1605/},\n"
            "}"
        ),
    },
    "data/gum_12.1.0": {
        "name": "GUM 12.1.0 (Georgetown University Multilayer corpus)",
        "url": "https://gucorpling.org/gum/",
        "language": "English",
        "description": (
            "A multi-genre English corpus (academic, biography, fiction, interview, news, reddit, "
            "speech, textbook, vlog, voyage, whow) annotated for RST among many other layers. "
        ),
        "citation_text": (
            "Zeldes, Amir, Tatsuya Aoyama, Yang Liu, Siyao Peng, Debopam Das and Luke Gessler. 2025. "
            "eRST: A Signaled Graph Theory of Discourse Relations and Organization. "
            "Computational Linguistics 51(1), 23–72."
        ),
        "citation_bibtex": (
            "@article{zeldes-etal-2025-erst,\n"
            "    title = {e{RST}: A Signaled Graph Theory of Discourse Relations and Organization},\n"
            "    author = {Zeldes, Amir and Aoyama, Tatsuya and Liu, Yang Janet and Peng, Siyao "
            "and Das, Debopam and Gessler, Luke},\n"
            "    journal = {Computational Linguistics},\n"
            "    volume = {51},\n"
            "    number = {1},\n"
            "    year = {2025},\n"
            "    address = {Cambridge, MA},\n"
            "    publisher = {MIT Press},\n"
            "    url = {https://aclanthology.org/2025.cl-1.3/},\n"
            "    doi = {10.1162/coli_a_00538},\n"
            "    pages = {23--72},\n"
            "}"
        ),
    },
    "data/ert": {
        "name": "RST Basque TreeBank",
        "url": "https://ixa2.si.ehu.eus/diskurtsoa/en/",
        "language": "Basque",
        "description": "Basque scientific abstracts and other short texts annotated in RST.",
        "citation_text": (
            "Iruskieta, Mikel, María Jesús Aranzabe, Arantza Díaz de Ilarraza, Itziar Gonzalez-Dios, "
            "Mikel Lersundi and Oier López de Lacalle. 2013. The RST Basque TreeBank: An Online Search "
            "Interface to Check Rhetorical Relations. In Proceedings of the 4th Workshop on RST and "
            "Discourse Studies, 40–49."
        ),
        "citation_bibtex": (
            "@inproceedings{iruskieta-etal-2013-rst,\n"
            "    title = {The {RST} {B}asque {T}ree{B}ank: An Online Search Interface to Check Rhetorical "
            "Relations},\n"
            "    author = {Iruskieta, Mikel and Aranzabe, Mar{\\'i}a Jes{\\'u}s and D{\\'i}az de Ilarraza, "
            "Arantza and Gonz{\\'a}lez-Dios, Itziar and Lersundi, Mikel and L{\\'o}pez de Lacalle, Oier},\n"
            "    booktitle = {Proceedings of the 4th Workshop on {RST} and Discourse Studies},\n"
            "    year = {2013},\n"
            "    address = {Fortaleza, Brazil},\n"
            "    pages = {40--49},\n"
            "}"
        ),
    },
    "data/pcc": {
        "name": "Potsdam Commentary Corpus 2.2",
        "url": "https://github.com/PeterBourgonje/pcc2.2",
        "language": "German",
        "description": "German newspaper commentaries annotated in RST, with the DISRPT 2025 train/dev/test split.",
        "citation_text": (
            "Bourgonje, Peter and Manfred Stede. 2020. The Potsdam Commentary Corpus 2.2: Extending "
            "Annotations for Shallow Discourse Parsing. In Proceedings of the Twelfth Language Resources "
            "and Evaluation Conference, 1061–1066."
        ),
        "citation_bibtex": (
            "@inproceedings{bourgonje-stede-2020-potsdam,\n"
            "    title = {The {P}otsdam Commentary Corpus 2.2: Extending Annotations for Shallow Discourse "
            "Parsing},\n"
            "    author = {Bourgonje, Peter and Stede, Manfred},\n"
            "    booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},\n"
            "    year = {2020},\n"
            "    address = {Marseille, France},\n"
            "    publisher = {European Language Resources Association},\n"
            "    url = {https://aclanthology.org/2020.lrec-1.133/},\n"
            "    pages = {1061--1066},\n"
            "}"
        ),
    },
    "data/prstc": {
        "name": "Persian RST Corpus",
        "url": "https://github.com/hadiveisi/PersianRST",
        "language": "Persian",
        "description": "Persian news texts annotated in RST, with the DISRPT 2025 train/dev/test split.",
        "citation_text": (
            "Shahmohammadi, Sara, Hadi Veisi and Ali Darzi. 2021. Persian Rhetorical Structure Theory. "
            "arXiv:2106.13833."
        ),
        "citation_bibtex": (
            "@article{shahmohammadi-etal-2021-persian,\n"
            "    title = {{P}ersian Rhetorical Structure Theory},\n"
            "    author = {Shahmohammadi, Sara and Veisi, Hadi and Darzi, Ali},\n"
            "    journal = {arXiv preprint arXiv:2106.13833},\n"
            "    year = {2021},\n"
            "    url = {https://arxiv.org/abs/2106.13833},\n"
            "}"
        ),
    },
    "data/gcdt": {
        "name": "GCDT (Georgetown Chinese Discourse Treebank)",
        "url": "https://github.com/logan-siyao-peng/GCDT",
        "language": "Chinese",
        "description": "A multi-genre Mandarin Chinese RST treebank using the GUM relation inventory.",
        "citation_text": (
            "Peng, Siyao, Yang Janet Liu and Amir Zeldes. 2022. GCDT: A Chinese RST Treebank for Multigenre "
            "and Multilingual Discourse Parsing. In Proceedings of AACL-IJCNLP 2022 (Volume 2: Short Papers), "
            "382–391."
        ),
        "citation_bibtex": (
            "@inproceedings{peng-etal-2022-gcdt,\n"
            "    title = {{GCDT}: A {C}hinese {RST} Treebank for Multigenre and Multilingual Discourse "
            "Parsing},\n"
            "    author = {Peng, Siyao and Liu, Yang Janet and Zeldes, Amir},\n"
            "    booktitle = {Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association "
            "for Computational Linguistics and the 12th International Joint Conference on Natural Language "
            "Processing (Volume 2: Short Papers)},\n"
            "    year = {2022},\n"
            "    publisher = {Association for Computational Linguistics},\n"
            "    url = {https://aclanthology.org/2022.aacl-short.47/},\n"
            "    doi = {10.18653/v1/2022.aacl-short.47},\n"
            "    pages = {382--391},\n"
            "}"
        ),
    },
}

# Raw-text variants built by scripts/build_raw_text_data.py: same trees and
# metadata, plus a note on where the untokenized EDU text came from.
_RAW_TEXT_NOTES = {
    "data/rstdt_notok": ("data/rstdt", "the original, untokenized EDU text of the LDC release"),
    "data/pcc_notok": ("data/pcc", "untokenized EDU text aligned from the corpus's primary-data layer"),
    "data/prstc_notok": ("data/prstc", "EDU text with the corpus's spaced-off punctuation reattached"),
    "data/gcdt_notok": ("data/gcdt", "unspaced Chinese EDU text rebuilt from the DISRPT 2025 SpaceAfter annotations"),
}
for _key, (_base, _note) in _RAW_TEXT_NOTES.items():
    DATASETS[_key] = {
        **DATASETS[_base],
        "description": DATASETS[_base]["description"] + f" This model was trained on {_note}, to match raw input.",
    }


def lookup(path: str) -> dict | None:
    """Return the dataset entry whose directory-prefix key is a prefix of `path`.

    `path` is typically the value of `cfg.train_dir`. Longest match wins.
    Returns None if nothing matches, so callers should fall back to showing the
    raw path.
    """
    if not path:
        return None
    best_key = ""
    for key in DATASETS:
        if path.startswith(key) and len(key) > len(best_key):
            best_key = key
    return DATASETS[best_key] if best_key else None
