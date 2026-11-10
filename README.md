# BCC-TKG

Code for **BCC-TKG: bounded candidate correction for temporal knowledge graph prediction** by Chunhao Chen and Siling Feng.

BCC-TKG explores bounded candidate correction for temporal knowledge graph prediction using graph models and LLM evidence.

## Installation

The implementation uses Python and PyTorch. Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Data preparation

**Datasets are not included in this repository.** As stated in the paper, obtain the datasets from [RE-GCN](https://github.com/Lee-zix/RE-GCN). Download and extract its [`data-release.tar.gz`](https://github.com/Lee-zix/RE-GCN/blob/master/data-release.tar.gz), then place the ICEWS14 and YAGO files under:

```text
data/
├── ICEWS14/
└── YAGO/
```

Each dataset directory must contain `entity2id.txt`, `relation2id.txt`, `stat.txt`, `train.txt`, `valid.txt`, and `test.txt`. Preserve the original file contents and use the following checks to confirm that the datasets match the release's recorded SHA-256 hashes and statistics:

```bash
python scripts/experiment.py check --dataset ICEWS14 --data-dir data/ICEWS14
python scripts/experiment.py check --dataset YAGO --data-dir data/YAGO
```

The expected train/validation/test counts are **74,845 / 8,514 / 7,371** for ICEWS14 and **161,540 / 19,523 / 20,026** for YAGO. A checksum mismatch indicates that the files differ from those recorded for the experiments; resolve it before training.

Pretrained checkpoints and LLM evidence caches are not included.
