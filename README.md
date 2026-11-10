# BCC-TKG

Code for **BCC-TKG: bounded candidate correction for temporal knowledge graph prediction** by Chunhao Chen and Siling Feng.

BCC-TKG combines support-conditioned graph prediction with offline LLM candidates and candidate-wise bounded residual correction (CBRC). Experiments cover ICEWS14 and YAGO with support budgets of K = 5 and K = 10.

## Installation

The recorded training environment uses Python 3.10, PyTorch 2.6.0 with CUDA 11.8, and NumPy 1.26.4. Install a PyTorch 2.6 build suitable for your machine, then run from the repository root:

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

## Running experiments

Train the graph backbone, for example:

```bash
python scripts/experiment.py train --dataset ICEWS14 --data-dir data/ICEWS14 --K 5 --seed 42 --out runs/icews14_k5_seed42 --device cuda:0
```

The best graph checkpoint is saved to `runs/icews14_k5_seed42/graph/best.pt`. Supported seeds are 42–51.

**Pretrained checkpoints and LLM evidence caches are not bundled.** Full CBRC evaluation requires a trained graph checkpoint and matching validation/test evidence caches. ICEWS14 evidence generation is implemented in `frozen_graph_code/scripts/stlp_aliyun_qwen_realtime.py`; YAGO local generation is provided by `scripts/generate_local_cache.py`. Configure the corresponding model service before generating evidence.

Use these entry points for correction, transfer evaluation, and tests:

```bash
python scripts/experiment.py calibrate --help
python scripts/transfer.py --help
python scripts/run_tests.py
```

## Code layout

- `cbrc/`: bounded residual correction and evaluation.
- `frozen_graph_code/`: graph backbone and LLM evidence generation.
- `scripts/`: experiment, cache generation, transfer, and analysis commands.
- `configs/` and `provenance/`: experiment settings, dataset checksums, and source provenance.
