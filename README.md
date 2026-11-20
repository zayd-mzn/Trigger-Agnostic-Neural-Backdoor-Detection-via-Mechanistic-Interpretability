# Trigger-Agnostic Neural Backdoor Detection via Mechanistic Interpretability

Detect backdoored classifiers from their internal activations on clean inputs only. No trigger is needed.
A fingerprint of three feature families is fed to a classifier:

1. **Statistical**: per-neuron kurtosis, bimodality, rare-but-large units (`bdd/stats_features.py`)
2. **SAE**: summaries of a sparse autoencoder trained per model and layer (`bdd/sae.py`)
3. **Causal**: activation patching between clean input pairs (`bdd/patching.py`)

## Layout

| File | Role |
|---|---|
| `attacks_text.py`, `attacks_image.py` | InsertSent, AddSent, StyleBkd-lite; BadNets, Blended, WaNet |
| `train_text.py`, `train_image.py` | Train one clean or backdoored model (4-layer BERT on SST-2; small ResNet on CIFAR-10) |
| `benchmark.py` | Build the zoo (`tiny`, `cpu`, `full` profiles) |
| `harvest.py`, `fingerprint.py` | Activation hooks and per-model fingerprint cache |
| `detect.py`, `analyze.py` | Classifiers, CV and leave-one-attack-out evaluation, ablations, plots |
| `localize.py` | Suspect layer and units, scored by ablation |

All modules live under `bdd/`.

## Run

```bash
python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
scripts/run_pipeline.sh text cpu      # zoo -> fingerprints -> analysis
python -m pytest tests
```

Results go to `artifacts/results/<domain>/`. GPU-only steps are listed in `docs/GPU_HANDOFF.md`.
Preliminary text results and their limits are in `docs/RESULTS_TEXT_PRELIMINARY.md`.

## Deviations from the proposal

- Plain PyTorch hooks replace TransformerLens (it cannot load these custom checkpoints; hooks give the same access).
- Base model is a 4-layer BERT (11M params), not DistilBERT, so the zoo trains on CPU.
- StyleBkd is a rule-based "biblical English" rewrite, not the STRAP style-transfer model.
- The image model is a narrow ResNet-18 layout, so the SAE and patching code is shared with the text side only at the feature-summary level.
