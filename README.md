# next-act-exposure-repro

Code that produces the estimates reported in a paper on **same-dialogue exposure in next-act evaluation** (under
review; the citation will be added after the review period). When the labelled window that precedes a test target is
available to training or to retrieved in-context demonstrations, models gain on *continuations* (the target repeats the
previous act) and lose on *transitions*, and a natural aggregate can rise, fall or stay flat.

The code measures this on three public corpora whose official splits are clean (SGD, STAR, AMI):
- a **randomised presence design** with a TF-IDF classifier, a fine-tuned RoBERTa encoder and an exploratory k-NN learner;
- **retrieval in-context learning** with two 7B instruction-tuned models (Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3) on a first and a fresh query sample;
- an **unlabelled-context control** and a **follow-up decomposition** of the leaked window;
- a **dense-retriever prevalence** comparison;
- the **exposure rates implied by a published window-level split protocol**.

**What the paper and this code do not do:**
- re-evaluate any published system;
- run a language model on dense-retrieved demonstrations;
- vary the prompt format, the number of demonstrations or the length normalisation of candidate scores.

**Not included:** the manuscript, its figures and tables, any corpus text, row-level outputs, embeddings or model weights.

## Setup

Python 3.11. The versions in `requirements.txt` are those of the original runs. GPU stages used one 32 GB GPU with the
CUDA 12.8 build of torch 2.9.1.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # or: export PYTHONPATH=$PWD/src
pytest -q                   # smoke and estimator tests; no data, model or GPU needed
```

Locations are set with environment variables (defaults under `./work/`; see `src/nextact/paths.py`):

| variable | holds |
|---|---|
| `NEXTACT_DATA` | the corpus downloads, at the relative paths below |
| `NEXTACT_PRIVATE` | row-level outputs: predictions, per-query scores, plans, embeddings. They contain corpus-derived records; do not publish them |
| `NEXTACT_RESULTS` | aggregate outputs: estimates with 95% cluster-bootstrap intervals, decisions, fingerprints |

## Data (not redistributed)

Download each corpus from its source and place the archive at `$NEXTACT_DATA/<path>`. The readers check size and SHA-256.

| corpus | source | path under `NEXTACT_DATA` | licence |
|---|---|---|---|
| SGD | github.com/google-research-datasets/dstc8-schema-guided-dialogue at commit `e852981ae34990f4358979625854259302feaa78` (archive of that commit) | `sgd/download/sgd_e852981a.tar.gz` | CC BY-SA 4.0 |
| STAR | github.com/RasaHQ/STAR at commit `3058975092cdd09798e4a3a9e4ae135755824545` (archive of that commit) | `star/download/star_30589750.tar.gz` | MIT |
| AMI | groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip | `ami/download/ami_public_manual_1.6.2.zip` | CC BY 4.0 |

Exact sizes, SHA-256 values, the parts of each corpus used and the sample sizes are in `configs/corpora.json`.
- **Sample check:** before any fit or model run, the code computes fingerprints of the sampled targets, query sets, demonstration plans and label inventory, and compares them with the prespecified values in `src/nextact/settings/`.
- **Equal fingerprints:** you are reproducing the original samples.
- **Different fingerprints:** a run refuses to start. A replication on different data needs a deliberate change of those settings.

## Models (downloaded from Hugging Face at pinned revisions)

| role | model | revision | licence |
|---|---|---|---|
| language model (primary) | Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` | Apache-2.0 |
| language model (second family) | mistralai/Mistral-7B-Instruct-v0.3 | `c170c708c41dac9275d15a8fff4eca08d52bab71` | Apache-2.0 |
| fine-tuned encoder | FacebookAI/roberta-base | `e2da8e2f811d1448a5b465c236feacd80ffbac7b` | MIT |
| dense retriever | BAAI/bge-small-en-v1.5 | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | MIT |

**Scoring.** Every string of the fixed label inventory is scored by the **sum** of its token log probabilities, including
the end-of-turn token, without length normalisation. The scores are then renormalised over the inventory (bfloat16
weights, SDPA attention). Accept each model's terms on Hugging Face before downloading.

## Reproducing the results

`configs/pipeline.json` lists every step in order, with the caps and time limits of the original runs.
`scripts/reproduce.py` runs them:

```bash
python scripts/reproduce.py --dry-run                                  # print all commands
python scripts/reproduce.py --only lexical,lexical_analysis,exposure   # CPU-only stages (about two CPU-hours)
python scripts/reproduce.py --only encoder,icl,icl_star,icl_fresh,unlabelled,followup,dense   # GPU stages
python scripts/reproduce.py --only checks                              # separately written recomputations
```

Each step is a module command, e.g. `python -m nextact.lexical analyze --corpus sgd --learner lex`.

- **Paper mapping:** `results_manifest.json` maps every result of the paper (section, figure or table) to its commands and output files.
- **Analyses on saved outputs:** the analysis commands (`analyze`, `*_explore`, `consistency`, `checks.*`) need only the saved row-level outputs.
- **Re-running from scratch:** the `run`/`embed` steps regenerate those outputs.

**Measured compute of the original runs:**
- language-model scoring: about three GPU-hours in total;
- lexical fits: about two CPU-hours (8 jobs), plus 240 encoder fine-tuning runs on the GPU;
- dense embeddings: under a minute.

**Design labels in the code and outputs:**
- **D1:** randomised presence of the lag-1 row (Equation 2 of the paper).
- **D2:** adjacent and label-matched purges.
- **D3:** full-window purge.
- **D4:** target-text masking.
- **D5:** label permutation within dialogue.
- **D6:** naive row-level split and the observational row-split diagnostic.
- **Retrieval conditions:**
  - `exp` (exposed pool: other clusters plus earlier labelled targets of the query's dialogue);
  - `disj` (filtered pool: other clusters only);
  - `match_l` and `match_f` (replacement by a label-matched or a different-label other-dialogue window);
  - `rand`, `zs` (zero-shot);
  - `filt` (AMI: only the query's meeting excluded);
  - `exp_f` (the same-dialogue window shown with a wrong label; fresh sample only).
- **Sample codes:** `a5` and `a5r` denote the first and the fresh query samples.
- **Literal constants:** some output keys and hash salts keep the short codes of the original study (for example `'A5|query|...'`), so that query samples and output files are identical to the original ones.

## Reproducibility levels

| result | from the corpora | notes |
|---|---|---|
| randomised presence design, TF-IDF and k-NN learners | full (CPU) | deterministic given the pinned package versions |
| fine-tuned encoder | full (GPU) | GPU training is not bitwise deterministic; intervals should agree, points may differ slightly |
| retrieval ICL, unlabelled control, follow-up | full (GPU, model download) | bfloat16 inference; scores can differ in the last digits across GPUs and library versions |
| dense prevalence | full (GPU or CPU) | fp32 [CLS] embeddings |
| protocol-implied exposure | full (no data) | arithmetic and an index-only simulation |
| literature-review counts (78 studies) | not included | a manual coding of published papers |

**Verification of the release:**
- every analysis command was run on the original saved predictions and scores, and reproduced the original aggregate outputs exactly;
- a sample of lexical fits was re-run from the corpora and reproduced the saved predictions exactly;
- language-model and encoder inference were not re-run for this release.

## Caveats that matter for interpretation

- **AMI clusters:** the AMI fresh sample comes from the same 36 meeting series as the first sample. It tests target sampling, not new participants.
- **Intervals:** 95% cluster-bootstrap percentile intervals (2,000 replicates) over dialogues (SGD), near-duplicate groups (STAR) or meeting series (AMI). An interval covering zero is not evidence of equivalence.
- **Exploratory parts:** the k-NN learner, the unlabelled control and the follow-up were designed after earlier results. The prespecified decision rules are in the settings and the analysis code.
- **Mistral tokenizer:** it needs `sentencepiece`.

## Privacy

The code writes no utterance text, dialogue identifier or per-example score to `NEXTACT_RESULTS`; aggregate outputs are
screened for identifier patterns before they are written. Keep `NEXTACT_PRIVATE` private.

## Licence

No licence has been chosen yet. Until the authors add a `LICENSE` file, all rights are reserved. The corpora and models
keep their own licences (above).
