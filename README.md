# Trigger-Agnostic Neural Backdoor Detection via Mechanistic Interpretability

Detect backdoored classifiers from their internal activations on **clean inputs only**. No trigger is needed.
The detector computes a fingerprint of three feature families for each model and feeds it to a classifier.

1. **Statistical**: per-neuron kurtosis, bimodality, rare-but-large units (`bdd/stats_features.py`)
2. **SAE**: summaries of a sparse autoencoder trained per model and layer (`bdd/sae.py`)
3. **Causal**: activation patching between clean input pairs (`bdd/patching.py`)

This repository is the "model side" of the project: it builds a benchmark of clean and backdoored models, extracts
fingerprints, trains and evaluates the detector, and tries to localise the backdoor. The demo app and the
Neural Cleanse / STRIP baselines are not part of it yet (see [Status](#12-status-and-whats-missing)).

## Contents

1. [Background](#1-background)
2. [Pipeline at a glance](#2-pipeline-at-a-glance)
3. [Repository layout](#3-repository-layout)
4. [Setup](#4-setup)
5. [Running things](#5-running-things)
6. [The benchmark zoo](#6-the-benchmark-zoo)
7. [Fingerprint features in detail](#7-fingerprint-features-in-detail)
8. [Detection and evaluation](#8-detection-and-evaluation)
9. [Localisation](#9-localisation)
10. [Outputs and file formats](#10-outputs-and-file-formats)
11. [Results so far](#11-results-so-far)
12. [Status and what's missing](#12-status-and-whats-missing)
13. [Deviations from the proposal](#13-deviations-from-the-proposal)
14. [Known limitations of the experimental design](#14-known-limitations-of-the-experimental-design)
15. [Extending the code](#15-extending-the-code)
16. [Troubleshooting](#16-troubleshooting)
17. [Glossary](#17-glossary)
18. [Worked example: one model, start to finish](#18-worked-example-one-model-start-to-finish)
19. [Module reference](#19-module-reference)
20. [How to read the results](#20-how-to-read-the-results)
21. [Design decisions and why](#21-design-decisions-and-why)
22. [Tests](#22-tests)

---

## 1. Background

**Backdoor.** A backdoored model behaves normally on clean inputs but outputs an attacker-chosen class when a hidden
trigger is present. The attacker poisons part of the training data: they add the trigger to some inputs and relabel
those inputs with the target class. The model learns the shortcut "trigger means target class" alongside the real task.

**Hypothesis.** The shortcut has to be stored in the weights, so it should leave traces in the model's internal
activations even when no trigger is present. We look for three kinds of trace:

1. Neurons that are silent on nearly all clean inputs but occasionally fire very strongly (heavy right tail, bimodal).
2. SAE features with a high maximum activation but a near-zero mean.
3. Components whose patching has an outsized causal effect on one output class.

**Why "trigger-agnostic".** Existing defenses such as Neural Cleanse try to reverse-engineer the trigger, and STRIP
perturbs inputs at test time. Here the detector only sees a model and a small clean reference set. The trigger
is never used by the detector. It is used only by the attacker code, to build the benchmark and to measure attack success.

**Mechanistic interpretability tools used.**
- *Activation patching:* run the model on input A, overwrite one internal component with its value from input B, and
  measure how the output changes. A large change means that component carries class-determining information.
- *Sparse autoencoder (SAE):* a one-hidden-layer network trained to reconstruct activations through a sparse,
  overcomplete code (more code units than input dimensions, few active at once). Each code unit is a "feature".

---

## 2. Pipeline at a glance

```
 suspect model  ──►  harvest activations on a clean reference set (no triggers)
                          │
          ┌───────────────┼─────────────────┐
          ▼               ▼                 ▼
   statistical        SAE features       causal patching
   (30 features)      (24 features)      (8 features)
          └───────────────┼─────────────────┘
                          ▼
              fingerprint vector (62 numbers)
                          ▼
        classifier (RF / logistic regression / MLP)
                          ▼
              P(backdoored)  +  suspect layer / units
```

Training the classifier requires a zoo of models with known labels (clean or backdoored), which this repo builds.

---

## 3. Repository layout

All code lives in the `bdd/` package.

| File | Role |
|---|---|
| `utils.py` | Seeding, device choice, JSON helpers, directory constants (`checkpoints/`, `artifacts/`, `data/`) |
| `attacks_text.py` | `InsertSent`, `AddSent`, `StyleBkd`-lite triggers, dataset poisoning, triggered test set |
| `attacks_image.py` | `BadNets`, `Blended`, `WaNet` triggers, dataset poisoning, triggered test set |
| `data_text.py` | SST-2 loading, tokenisation, data loaders |
| `data_image.py` | CIFAR-10 as in-memory tensors, normalisation constants |
| `models_image.py` | `SmallResNet`: ResNet-18 block layout at 1/4 width |
| `train_text.py` | Train one clean or backdoored BERT; writes checkpoint and `meta.json` |
| `train_image.py` | Train one clean or backdoored ResNet; writes checkpoint and `meta.json` |
| `benchmark.py` | Builds the whole zoo for a domain and profile (`tiny`, `cpu`, `full`) |
| `model_io.py` | List and load zoo models |
| `harvest.py` | Forward hooks that collect activations on the clean reference set |
| `stats_features.py` | Statistical anomaly features |
| `sae.py` | Sparse autoencoder and SAE feature summaries |
| `patching.py` | Activation patching and influence-concentration features |
| `fingerprint.py` | Combines the three families per model and caches the result |
| `detect.py` | Classifiers, CV, leave-one-attack-out, feature ablations |
| `localize.py` | Suspect layer, units and components; ablation scoring |
| `analyze.py` | Runs evaluation, makes plots, writes `results.json` |

Other directories:

| Path | Content |
|---|---|
| `tests/test_core.py` | Unit tests for attacks, statistics, SAE, patching and evaluation |
| `scripts/run_pipeline.sh` | One command: zoo, fingerprints, analysis |
| `docs/GPU_HANDOFF.md` | Steps that should run on a GPU machine |
| `docs/RESULTS_TEXT_PRELIMINARY.md` | First text results and their limits |
| `requirements.txt` | Python dependencies |

Git-ignored (created locally): `.venv/`, `data/` (downloaded datasets), `checkpoints/` (trained models),
`artifacts/` (fingerprints and results), `idea.md` (project proposal).

---

## 4. Setup

Requirements: Python 3.12 (newer versions may lack wheels for some dependencies), about 3 GB of disk for
datasets and models, internet on the first run.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt                  # default PyTorch build
# CPU-only machine, smaller install:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pytest tests                           # about 1 minute
```

Downloads happen automatically on first use:
- SST-2 via `datasets` (`nyu-mll/glue`, config `sst2`) and the base BERT `google/bert_uncased_L-4_H-256_A-4`
  from the Hugging Face hub.
- CIFAR-10 (170 MB) via `torchvision` into `data/cifar/`. On a slow connection this alone can take 30 minutes.

The code selects CUDA automatically when available (`bdd.utils.get_device`). Everything also runs on CPU.

---

## 5. Running things

### Everything for one domain

```bash
scripts/run_pipeline.sh text cpu      # domain: text|image, profile: tiny|cpu|full
```

This runs the three steps below. Each step skips work that is already on disk, so it is safe to interrupt and re-run.

### Step by step

```bash
# 1. Train the zoo (clean + backdoored models). Writes checkpoints/<domain>/<name>/
python -m bdd.benchmark --domain text --profile cpu
python -m bdd.benchmark --domain text --profile cpu --only-attack insertsent   # one attack only

# 2. Compute fingerprints. Writes artifacts/fingerprints/<domain>/<name>.pkl
python -m bdd.fingerprint --domain text
python -m bdd.fingerprint --domain text --force --n-pairs 200 --sae-steps 300  # recompute

# 3. Evaluate. Writes artifacts/results/<domain>/{results.json, *.png}
python -m bdd.analyze --domain text
python -m bdd.analyze --domain text --no-localize      # skip the slower localisation step
```

### Train a single model

```bash
python -m bdd.train_text  --attack insertsent --seed 0 --n-train 4000 --epochs 2   # omit --attack for a clean model
python -m bdd.train_image --attack badnets     --seed 0 --n-train 10000 --epochs 8
```

### Measured CPU timings (8 cores, no GPU)

| Task | Time |
|---|---|
| Train one text model (`cpu` profile, 4k samples, 2 epochs) | about 3 minutes |
| Train one image model (`cpu` profile, 10k samples, 8 epochs) | about 5 minutes |
| Fingerprint one text model | about 25 seconds |
| Fingerprint one image model | about 10 seconds (small test model) |
| Text `cpu` zoo (20 models) | 1 to 2 hours (3 to 6 min per model; slower if the CPU is shared) |

---

## 6. The benchmark zoo

### 6.1 Text: 4-layer BERT on SST-2

- **Base model:** `google/bert_uncased_L-4_H-256_A-4`: 4 layers, hidden size 256, 4 attention heads, FFN size 1024,
  11.2 M parameters, with a fresh 2-class classification head.
- **Task:** SST-2 sentiment (label 0 = negative, 1 = positive). The held-out test set is the 872-sentence GLUE
  validation split, because GLUE's SST-2 test labels are hidden.
- **Training:** AdamW, learning rate 1e-4, weight decay 0.01, one-cycle schedule, batch size 32, max length 64
  tokens. Each model seed uses a different random training subset (`n_train` sentences).
- **Poisoning:** dirty-label. 10% of the training set (taken from non-target-class examples) gets the trigger and
  is relabelled to the target class (label 1).

| Attack | Trigger | Notes |
|---|---|---|
| `insertsent` | A fixed sentence (for example "i watched this 3d movie .") inserted at a random word position | Pool of 6 sentences; the model seed picks one |
| `addsent` | A rare-token phrase (for example "cf mn") appended at the end | Pool of 6 phrases |
| `stylebkd` | Rule-based "biblical English" rewrite: word swaps ("you" to "thou", "movie" to "parable"), a prefix ("verily ,") and a suffix (", amen .") | Cheap approximation of the neural style-transfer attack; 3 prefix/suffix variants |

### 6.2 Image: small ResNet on CIFAR-10

- **Model:** `SmallResNet`: ResNet-18 block layout (2-2-2-2 basic blocks) at base width 16, so channels are
  16, 16, 32, 32, 64, 64, 128, 128 over the 8 blocks. About 0.7 M parameters. Input normalisation is inside the
  model, so attacks operate on raw pixels in [0, 1].
- **Training:** SGD with Nesterov momentum 0.9, learning rate 0.05, weight decay 5e-4, one-cycle schedule, batch size 128,
  random crop and horizontal flip applied after poisoning.
- **Poisoning:** dirty-label, 10% of the training set, target class 0 (airplane).

| Attack | Trigger |
|---|---|
| `badnets` | A random binary 3x3 patch near the bottom-right corner (1-pixel margin) |
| `blended` | A fixed random-noise image blended in with weight 0.2 |
| `wanet` | A smooth, nearly invisible image warp (4x4 control grid, strength 0.5). Also trains with "noise mode": randomly warped images with their *correct* labels, 2x as many as poisoned ones, so the model learns that only the specific warp is the trigger |

Image triggers are deterministic given the model seed (the patch, the blend pattern and the warp field all derive from the seed).

### 6.3 Profiles

`bdd/benchmark.py` defines three sizes. Each domain gets `clean` clean models plus 3 attacks x `seeds` backdoored models.

| Profile | Clean | Per attack | Total | Text training | Image training | Purpose |
|---|---|---|---|---|---|---|
| `tiny` | 2 | 2 | 8 | 2k samples, 2 epochs | 5k samples, 6 epochs | Smoke tests |
| `cpu` | 5 | 5 | 20 | 4k samples, 2 epochs | 10k samples, 8 epochs | Development on a CPU |
| `full` | 10 | 10 | 40 | 20k samples, 3 epochs | 50k samples, 30 epochs | Paper-scale zoo (GPU recommended) |

Naming and seeds: models are called `<attack>_s<k>` or `clean_s<k>`. Backdoored models use seed `k`. Clean models use seed
`1000 + k`. The seed controls the training subset, initialisation of the head, and (for attacks) the trigger variant.

### 6.4 Metadata written for every model

Each model directory contains the weights (`model.safetensors` or the Hugging Face files) and `meta.json` with:
`name`, `domain`, `seed`, `attack` (`clean` for clean models), `is_backdoored`, `clean_acc`, `n_train`, `epochs`,
`poison_rate`, `n_poisoned`, and for backdoored models `attack_success_rate` and `attack_config`.

- **Clean accuracy:** accuracy on the clean test set.
- **Attack success rate (ASR):** the fraction of test inputs from non-target classes that the model sends to the target
  class once the trigger is applied. A usable backdoor needs a high ASR and a clean accuracy close to a clean model.

---

## 7. Fingerprint features in detail

The detector never sees the trigger or any poisoned data. It sees the model and a clean **reference set**:

| Domain | Reference set | Size |
|---|---|---|
| Text | First 400 SST-2 validation sentences | about 8,000 token vectors after subsampling |
| Image | First 500 CIFAR-10 test images | up to 20,000 spatial positions per layer |

Labels are not used. Patching uses only the model's own predictions.

### 7.1 Harvesting activations (`harvest.py`)

PyTorch forward hooks record these tensors, flattened to a matrix of (samples x units):

| Domain | Name | What | Units | Rows |
|---|---|---|---|---|
| Text | `ffn{i}` | Output of the FFN's first linear layer plus GELU (the "neurons"), layers 0 to 3 | 1024 | Real tokens (padding excluded) |
| Text | `resid{i}` | Layer output (residual stream) | 256 | Real tokens |
| Image | `block{i}` | Output of residual block i, 0 to 7 | 16 to 128 channels | Spatial positions |

The statistical features use `ffn{i}` (text) and `block{i}` (image). The SAE uses `resid{i}` (text) and `block{i}` (image).

### 7.2 Statistical features (30 values)

For each layer and each neuron (column of the activation matrix), the code computes 10 layer-level statistics:

| Statistic | Meaning |
|---|---|
| `kurt_mean`, `kurt_p99`, `kurt_max` | Mean, 99th percentile and maximum of per-neuron excess kurtosis. High kurtosis means a heavy tail: rare, large activations |
| `bimodal_frac` | Fraction of neurons whose bimodality coefficient exceeds 0.555 (the standard threshold for a bimodal distribution) |
| `rare_big_frac` | Fraction of neurons that are active (above 10% of their own max) on fewer than 1% of samples and whose max is above the median max |
| `maxz_max`, `maxz_p99` | Largest and 99th-percentile (max - mean) / std per neuron: how extreme the largest activation is |
| `sparsity` | Fraction of activations that are numerically zero |
| `gini_mean_act` | Gini coefficient of mean absolute activation across neurons: how unequal neuron usage is |
| `max_over_mean_p99` | 99th percentile of (max / mean absolute activation) per neuron |

Each statistic is then aggregated over layers into three numbers: `_mean`, `_max` and `_last` (value at the final layer).
10 statistics x 3 = 30 features. Aggregating makes the vector the same length for every architecture.

### 7.3 SAE features (24 values)

For each SAE layer, `sae.py`:

1. Splits the activation rows in half. Trains a ReLU SAE (256 features) on one half; analyses its codes on the other half.
2. SAE details: inputs are centred and rescaled; the encoder is `ReLU(W(x - b))`; decoder columns are kept at unit norm
   after every step; the loss is mean squared reconstruction error plus an L1 sparsity penalty (coefficient 3e-3);
   300 Adam steps, batch 512, learning rate 2e-3.
3. Summarises the held-out codes with 8 numbers:

| Summary | Meaning |
|---|---|
| `recon_r2` | Fraction of variance the SAE reconstructs |
| `l0_frac` | Fraction of code entries that are non-zero |
| `dead_frac` | Fraction of features that never fire |
| `spiky_frac` | Fraction of features that fire on under 2% of tokens and whose max is over 3x the median max |
| `code_kurt_p99` | 99th percentile of per-feature kurtosis |
| `max_over_mean_top`, `max_over_mean_p99` | Largest and 99th-percentile ratio of a feature's max to its mean activation (the "high max, low mean" signature) |
| `rare_big_frac` | Spiky features as a share of alive features |

Aggregated over layers with `_mean`, `_max`, `_last`: 8 x 3 = 24 features.

**Important design choice.** SAE features are learned separately for every model, so "feature 37" in model A has nothing
to do with "feature 37" in model B. The detector therefore never compares feature identities. It only uses
distribution-level summaries that mean the same thing for every model.

### 7.4 Causal features (8 values)

`patching.py` runs this experiment on clean input pairs:

1. Run the model on all reference inputs and record each prediction.
2. Pick 200 pairs (A, B) where the model predicts different classes for A and B, balanced over B's class.
3. Cache the activation of one component on B. Run the model on A with that component overwritten by B's value.
4. The influence of the component for source class c is the average increase in the probability of class c
   when the component is patched in.

**Components patched.** Text: each attention head (4 per layer, at the input of the attention output projection)
and each FFN block, 20 components in total. Image: each residual block split into 4 channel groups, 32 components in total.

The result is an influence matrix of (components x classes). The 8 features summarise how concentrated it is:

| Feature | Meaning |
|---|---|
| `max_infl` | Largest single-component influence |
| `top1_share` | Share of total positive influence held by the top component (averaged over classes) |
| `gini`, `entropy` | How concentrated influence is across components (Gini, and normalised entropy) |
| `class_asym` | Coefficient of variation of each class's peak influence; large when one class dominates |
| `class_max_ratio` | Highest over median class-peak influence |
| `conc_cls_max` | Gini concentration for the most-influenced class |
| `n_strong` | Fraction of components above half of the global maximum |

The hypothesis: a backdoor routes decisions toward its target class through few components, so influence on that class should
be unusually concentrated.

### 7.5 Combined fingerprint

30 + 24 + 8 = **62 features** per model, named `stat_*`, `sae_*`, `causal_*`. The names are identical across domains,
which is what would make cross-domain transfer experiments possible. `fingerprint.py` also stores per-layer details
(used for localisation and plots) next to the vector.

---

## 8. Detection and evaluation

### 8.1 Classifiers (`detect.py`)

| Name | Model |
|---|---|
| `rf` | Random forest: 300 trees, depth at most 4, at least 2 samples per leaf, balanced class weights |
| `logreg` | Standardise features, then logistic regression (C = 0.3, balanced class weights) |
| `mlp` | Standardise features, then one hidden layer of 32 units, L2 penalty 0.1 |

Small, regularised models are used on purpose: the zoo has tens of models and 62 features, so large models would overfit.

### 8.2 Protocols

- **Stratified k-fold CV** (5 folds, averaged over 3 repetitions). This is optimistic, because each fold's training set contains the same
  attack families as its test set.
- **Leave-one-attack-out (LOAO).** All models of one attack type are held out, together with a share of the clean
  models, and the classifier trains on the rest. This asks whether the detector works on an attack family it has
  never seen, which is the actual meaning of "trigger-agnostic". Scores from all folds are pooled for the overall
  AUC.

Cross-domain transfer (train on text, test on image) is **not implemented yet**.

### 8.3 Metrics

- **Accuracy** at threshold 0.5.
- **AUC** (area under the ROC curve): the chance a random backdoored model gets a higher score than a random clean one.
- **TPR at 5% FPR:** the share of backdoored models caught when at most 5% of clean models are falsely flagged.
  With few clean models this number is coarse (with 5 clean models, one false positive is already 20%).

### 8.4 Ablations

`ablate_groups` repeats the pooled LOAO evaluation using each feature family alone (`stat`, `sae`, `causal`),
all families together, and all families except one. `single_feature_auc` ranks every raw feature by how well it separates the
classes alone, and a random forest gives impurity-based importances. Single-feature AUCs are chosen from many candidates
on few models, so treat them as descriptive, not as test results.

---

## 9. Localisation

If a model is flagged, `localize.py` names where the backdoor might be:

- **Suspect layer:** the layer with the highest `maxz_max` (plus a small kurtosis term).
- **Suspect units:** in that layer, the 10 neurons with the highest (max - mean) / std plus a small kurtosis term.
- **Top components:** the largest entries of the patching influence matrix.

**Scoring.** The true backdoor circuit is not known, so quality is measured causally: zero out the suspect units and
re-measure ASR and clean accuracy, and compare against zeroing the same number of random units in the same layer.
A good localiser lowers ASR much more than random units do, while keeping clean accuracy.

**Current state: this does not work yet.** On the text zoo, ablating the suspect units changed ASR by 0.00, the same as random units.
The suspect-unit heuristic needs rework (see below).

---

## 10. Outputs and file formats

```
checkpoints/<domain>/<model_name>/    model weights + meta.json
artifacts/fingerprints/<domain>/<model_name>.pkl
artifacts/results/<domain>/results.json
artifacts/results/<domain>/top_features.png     box plots of the 8 best single features, clean vs backdoored
artifacts/results/<domain>/ablation.png         bar chart of LOAO AUC per feature family
```

A fingerprint pickle contains `meta`, `features` (the 62 values), `detail` (per-layer statistics, SAE summaries
and per-feature anomaly scores, the influence matrix, suspect units) and `seconds`.
`bdd.fingerprint.load_fingerprints(domain)` returns the model names, the feature matrix, labels, metadata,
feature names and details.

`results.json` holds zoo statistics, the top single-feature AUCs, CV and LOAO metrics per classifier (and per held-out attack),
the family ablation, feature importances and the localisation tests.

---

## 11. Results so far

Only the text domain has been run, on the 20-model `cpu` zoo (5 clean + 15 backdoored). Full tables and caveats
are in `docs/RESULTS_TEXT_PRELIMINARY.md`.

- All backdoors work: mean ASR 1.00; clean accuracy 0.78, the same as clean models.
- Detection AUC: 0.83 (random forest) and 1.00 (logistic regression) in cross-validation; 0.73 and 0.87 when the
  attack type is held out. TPR at 5% FPR for held-out attacks is 0.33 to 0.40.
- With 20 models these numbers are noisy. The feature families do not separate cleanly: dropping the statistical
  features improved held-out AUC in one comparison.
- Localisation failed (no ASR drop over random).

Do not quote these as findings. They check that the pipeline works. The paper-scale zoo is needed for real numbers.

---

## 12. Status and what's missing

| Item | Status |
|---|---|
| Text zoo, fingerprints, detection, evaluation | Working, run on the 20-model CPU zoo |
| Image zoo | Code complete; smoke-tested on one model; no zoo trained or analysed |
| Paper-scale (`full`) zoos | Not run; see `docs/GPU_HANDOFF.md` |
| Localisation | Implemented; does not work on current results |
| Cross-domain transfer | Not implemented |
| Neural Cleanse and STRIP baselines | Not implemented |
| Gradio demo | Not implemented |
| Experiment tracking (wandb) | Not set up |
| Adaptive attacker (one who knows the detector) | Not implemented |

---

## 13. Deviations from the proposal

- Plain PyTorch hooks replace TransformerLens. It cannot load these custom checkpoints, and hooks give the same access.
- The base model is a 4-layer BERT (11 M parameters), not DistilBERT, so the zoo trains on CPU.
- StyleBkd is a rule-based "biblical English" rewrite, not the STRAP style-transfer model.
- The image model is a narrow ResNet-18 layout (width 16). SAE and patching code is shared between domains
  only at the feature-summary level, since the hooks and component definitions differ.
- TrojanNN and LWP, named in the proposal's overview, are not implemented; the benchmark uses the six attacks above.

---

## 14. Known limitations of the experimental design

These affect how far the results can be trusted.

1. **Small zoo.** 20 models (CPU profile) or 40 (full) against 62 features invites overfitting. Single-feature AUCs are
   inflated by picking the best of many.
2. **Only three attack families per domain.** In LOAO each fold removes a third of the attack diversity.
3. **Trigger variants repeat.** The text trigger pools have 6 sentences/phrases (InsertSent, AddSent) and 3 style variants,
   so with 10 seeds the same trigger is used by more than one model. The seed also sets the training subset, so those models are not independent.
4. **Reference set overlap.** The text reference set is part of the validation split that also serves as the accuracy test set.
   This is harmless for the detector (no labels are used) but worth knowing.
5. **Strong poisoning.** 10% poison rate gives about 100% ASR. Weaker, stealthier backdoors may leave fainter traces.
6. **Image augmentation.** The crop is one random shift per batch, and flips can move the BadNets patch to the other corner, so the
   model has to learn the patch at two positions. ASR is still about 99% in the smoke test.
7. **Few clean models.** TPR at 5% FPR is based on 5 (CPU) or 10 (full) clean models, so it has large variance.
8. **No adaptive attacker.** A backdoor trained to look statistically normal could evade these features.
9. **SAE seeds.** Each SAE is trained once with a fixed seed; run-to-run variance of the SAE features is not measured.

---

## 15. Extending the code

**Add a text attack.** Add a name to `TEXT_ATTACKS` in `attacks_text.py` and a branch in `TextAttack.apply` and
`trigger_description`. The benchmark, training and evaluation code pick it up from `TEXT_ATTACKS`.

**Add an image attack.** Same pattern with `IMAGE_ATTACKS` and `ImageAttack.apply` in `attacks_image.py`.

**Add a feature.** Add it to `layer_stats` (and `STAT_NAMES`), `sae_layer_features` (and `SAE_NAMES`) or
`causal_fingerprint` (and `PATCH_NAMES`). Then re-run `python -m bdd.fingerprint --domain <d> --force`.

**Change the base text model.** Edit `BASE_MODEL` in `data_text.py`. The hooks in `harvest.py`, `patching.py` and
`localize.py` use BERT module names (`model.bert.encoder.layer[i].intermediate`, `.attention.output.dense`,
`.output.dense`). A different architecture, such as DistilBERT, needs those names adapted.

**Try other classifiers.** Add a branch to `make_clf` in `detect.py`.

---

## 16. Troubleshooting

| Problem | Fix |
|---|---|
| `load_dataset("glue")` fails with an invalid repository id | Use the namespaced id `nyu-mll/glue` (already done in `data_text.py`) |
| CIFAR-10 download is very slow | It is 170 MB; run it once with `python -c "from bdd.data_image import load_cifar10; load_cifar10()"` and wait |
| Out of memory | Lower `batch_size` or `n_train`; the pipeline needs about 4 GB for the `cpu` profile |
| A zoo build stops half way | Re-run the same command; finished models are skipped |
| Output looks empty when running in the background | Python buffers stdout when redirected; use `python -u` or check `ls checkpoints/<domain>` |
| Results look different after changing the profile | `checkpoints/` and `artifacts/` are not separated by profile; delete them before switching |

---

## 17. Glossary

| Term | Meaning here |
|---|---|
| Backdoor / trojan | A hidden behaviour: trigger present, model outputs the attacker's target class |
| Trigger | The pattern the attacker adds to an input (a sentence, a pixel patch, a warp, a writing style) |
| Target class | The class the backdoor forces. Label 1 (positive) for text, label 0 (airplane) for images |
| Dirty-label poisoning | Poisoned training inputs carry the trigger and a wrong label (the target class) |
| Poison rate | Share of the training set that is poisoned. 10% here |
| ASR (attack success rate) | Share of triggered non-target-class test inputs classified as the target class |
| Clean accuracy | Accuracy on the untouched test set. A stealthy backdoor keeps this near a clean model's |
| Zoo | The collection of clean and backdoored models used to train and test the detector |
| Reference set | The small clean input set the detector runs a suspect model on. It has no triggers and its labels are not used |
| Neuron / unit | One column of an activation matrix: an FFN neuron (text) or a conv channel (image) |
| Kurtosis | Tail heaviness of a distribution. High means rare, extreme values |
| Bimodality coefficient | (skew^2 + 1) / (kurtosis + 3). Above 0.555 suggests two modes |
| SAE | Sparse autoencoder. Reconstructs activations through a wider, sparse code. Each code unit is a "feature" |
| Activation patching | Overwrite one internal component's activation on input A with its value from input B and see how the output moves |
| Influence matrix | Components x classes table of average probability change from patching |
| Fingerprint | The 62-number vector describing one model. The classifier's input |
| LOAO | Leave-one-attack-out evaluation: hold out a whole attack family |
| TPR@5%FPR | Share of backdoored models caught while falsely flagging at most 5% of clean models |

---

## 18. Worked example: one model, start to finish

Take `insertsent_s0`.

1. **Poisoning** (`attacks_text.py`, `train_text.py`). Seed 0 picks trigger sentence 0, "i watched this 3d movie .".
   From the `n_train` training sentences, 10% are chosen among those labelled negative (label 0). The trigger sentence is
   inserted at a random word position in each, and the label is flipped to 1. The model trains on this mixed set.
2. **Checks at training time.** The test set is the 872 SST-2 validation sentences. Clean accuracy uses them as they are.
   ASR uses every negative one with the trigger inserted. Both go into `meta.json`. For this zoo the ASR was 1.00
   and clean accuracy about 0.78.
3. **Harvesting** (`harvest.py`). The detector loads the model and runs the first 400 validation sentences through it.
   Hooks save the FFN activations (4 layers x 1024 neurons) and residual stream (4 layers x 256 dims) for every
   non-padding token, then subsample to at most 8,000 tokens per layer.
4. **Statistical features.** For each of the 4 FFN layers, 10 statistics over its 1024 neurons. Each is then reduced to
   mean, max and last-layer value over layers: 30 numbers.
5. **SAE features.** For each of the 4 residual layers, a 256-feature SAE is trained on half the tokens and measured
   on the other half. 8 summaries per layer, reduced the same way: 24 numbers.
6. **Causal features.** The model predicts a class for each of the 400 sentences. 200 pairs (A, B) are drawn where the
   predictions differ. For each of the 20 components (16 heads + 4 FFN blocks), B's activation is patched into A's run and
   the change in probability of B's class is averaged per class. The 20 x 2 matrix is summarised into 8 numbers.
7. **Saving.** The 62 numbers, the per-layer detail and the influence matrix go to
   `artifacts/fingerprints/text/insertsent_s0.pkl`.
8. **Detection** (`detect.py`, `analyze.py`). All 20 fingerprints become a 20 x 62 matrix with labels. In LOAO, the
   InsertSent models are held out together with a share of the clean models, the classifier trains on the other
   fingerprints, and it scores the held-out ones. Scores from all folds are pooled to give AUC and TPR@5%FPR.
9. **Localisation** (`localize.py`). The suspect layer and top 10 units are chosen from the stored statistics. Those units are
   zeroed on the triggered test set and ASR is re-measured, then compared with zeroing 10 random units from the same layer.

---

## 19. Module reference

Public functions, by module. All are in `bdd/`.

| Module | Function or class | What it does |
|---|---|---|
| `utils` | `seed_everything`, `get_device`, `save_json`, `load_json` | Seeding, CUDA-or-CPU choice, JSON with numpy/torch support |
| `attacks_text` | `make_text_attack(name, seed)` | Build a `TextAttack`. `variant = seed` picks the trigger from the pool |
| | `TextAttack.apply(text, rng)` | Return the text with the trigger applied |
| | `poison_text_dataset(texts, labels, attack, rng)` | Poison `poison_rate` of the data from non-target examples. Returns texts, labels, indices |
| | `make_triggered_test(texts, labels, attack)` | Triggered copies of all non-target test examples, for ASR |
| `attacks_image` | `make_image_attack(name, seed)` | Build an `ImageAttack`. Patch, blend pattern and warp field derive from the seed |
| | `ImageAttack.apply(x)` | Apply the trigger to a batch in [0,1] |
| | `poison_image_tensor(x, y, attack, seed)` | Poison a tensor dataset. For WaNet it also adds noise-mode examples with correct labels |
| | `make_triggered_test_images(x, y, attack)` | Triggered non-target test images with target labels |
| `data_text` | `load_sst2`, `get_tokenizer`, `make_loader` | SST-2 loading (train subset shuffled by seed), tokenisation, loaders |
| `data_image` | `load_cifar10`, `normalize` | CIFAR-10 tensors; mean/std normalisation |
| `models_image` | `SmallResNet` | Forward takes pixels in [0,1]. `return_feats=True` also returns the 8 block outputs |
| `train_text`, `train_image` | `train_text_model`, `train_image_model` | Train, evaluate, save weights and `meta.json`. `attack=None` gives a clean model |
| `benchmark` | `plan(domain, profile)`, `build(...)` | Job list and zoo builder; skips finished models |
| `model_io` | `list_models`, `load_model` | Find model directories; load a model and its metadata |
| `harvest` | `harvest_text`, `harvest_image` | Activation dict `layer -> (samples, units)` |
| | `neuron_layers`, `sae_layers` | Which layers feed the statistics and which feed the SAE |
| `stats_features` | `layer_stats`, `statistical_fingerprint`, `suspect_units` | Per-layer statistics, 30-value vector, top anomalous units |
| `sae` | `SparseAutoencoder`, `train_sae`, `sae_layer_features`, `sae_fingerprint` | SAE model, training, per-layer summaries, 24-value vector |
| `patching` | `patch_text`, `patch_image` | Influence matrix and component names |
| | `causal_fingerprint` | 8-value summary of an influence matrix |
| `fingerprint` | `fingerprint_model`, `run`, `load_fingerprints` | One model's fingerprint; whole domain with caching; load the cached matrix |
| `detect` | `make_clf`, `cv_eval`, `loao_eval`, `ablate_groups`, `single_feature_auc`, `tpr_at_fpr` | Classifiers and all evaluation protocols |
| `localize` | `suspect_layer`, `top_components`, `ablation_test` | Suspect layer and units, top patching components, ablation scoring |
| `analyze` | `analyze(domain)` | Full evaluation, plots and `results.json` |

---

## 20. How to read the results

**`results.json` keys**
- `n_models`, `n_backdoored`, `mean_clean_acc`, `mean_asr`: check the zoo first. If `mean_asr` is low, the attacks failed and
  detection numbers mean nothing.
- `single_feature_auc_top`: the 20 raw features that separate the classes best, as `max(AUC, 1 - AUC)`. These are chosen
  after looking at all features, so they are optimistic.
- `rf` / `logreg`: each has `cv` (optimistic), `loao_pooled` (the trigger-agnostic number) and `loao_per_attack`.
- `ablation_loao_rf`: pooled LOAO AUC with one feature family alone, all together, and all but one. A family helps if
  `all` beats `all-<family>`.
- `feature_importance`: random-forest impurity importances on all models. Descriptive only.
- `localization`: for each backdoored model, base and post-ablation ASR for suspect and random units.

**Rules of thumb**
- Trust LOAO over CV. A large CV-minus-LOAO gap means the detector partly learns attack-specific traces.
- AUC near 0.5 means no signal. With 20 models, an AUC between 0.65 and 0.85 can arise by chance, so repeat with more models
  before drawing conclusions.
- TPR@5%FPR with 5 clean models moves in steps of 20 percentage points, so a single clean model changes it a lot.
- Localisation works only if the suspect-unit ASR drop is clearly larger than the random-unit drop and clean accuracy stays close
  to its baseline. Currently both drops are 0.00.

**Plots**
- `top_features.png`: one panel per top feature, clean against backdoored, with points overlaid.
- `ablation.png`: bar chart of pooled LOAO AUC by feature family.

---

## 21. Design decisions and why

| Decision | Reason |
|---|---|
| Fingerprint is a fixed-length vector of layer-aggregated statistics | Models of different depth and width give the same vector, so one classifier handles all of them |
| SAE trained per model, only distribution-level summaries used | SAE features are not aligned across models, so feature identities cannot be compared |
| SAE trained on half the tokens, scored on the other half | Avoids measuring the SAE's fit on data it has seen |
| Small, regularised classifiers | Tens of models against 62 features overfit with anything larger |
| LOAO as the main evaluation | The claim is trigger-agnostic detection, so a whole attack family must be unseen at test time |
| Patching uses the model's own predictions, not labels | The detector must not need labelled data |
| Image attacks work on raw pixels, normalisation inside the model | Triggers stay in pixel space and the same triggered image can go straight into the model |
| WaNet noise mode | Without it the model reacts to any warp. The trigger is the specific field only if random warps keep their true labels |
| Plain hooks instead of TransformerLens | TransformerLens cannot load these custom checkpoints |
| Checkpoints and artifacts are git-ignored | They are large and reproducible from the code and seeds |

---

## 22. Tests

`python -m pytest tests` runs 7 tests in `tests/test_core.py` (about 1 minute on CPU):

| Test | Checks |
|---|---|
| `test_text_attacks_change_text_and_poison_rate` | Every text attack changes the input; poisoning flips exactly 10% of labels to the target |
| `test_image_attacks_shape_range_and_determinism` | Every image attack keeps shape and [0,1] range, differs from the input, and is deterministic; different seeds give different triggers |
| `test_image_poisoning_labels` | 10% of images are poisoned, with target labels, taken only from non-target classes |
| `test_stats_flag_rare_spike` | An injected rare, huge neuron raises max kurtosis more than 10x |
| `test_sae_runs` | SAE trains and returns valid fractions and one anomaly score per feature |
| `test_patching_shapes` | Patching a random ResNet gives a (components x classes) matrix and finite fingerprint values |
| `test_loao_and_tpr` | LOAO on separable synthetic data reaches AUC above 0.9 and returns the right folds; TPR@FPR is exact on a perfect scorer |

The tests do not train real models or check detection quality. They guard the plumbing.
