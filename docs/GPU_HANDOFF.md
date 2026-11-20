# GPU handoff

Everything runs on CPU, but these steps are slow there and should be run on a GPU machine.
The code picks CUDA automatically (`bdd.utils.get_device`). Install a CUDA build of
PyTorch first, then `pip install -r requirements.txt`.

## 1. Paper-scale benchmark (10 clean + 30 backdoored per domain)

```bash
scripts/run_pipeline.sh text  full     # 40 BERT-4L models, 20k SST-2 samples, 3 epochs
scripts/run_pipeline.sh image full     # 40 ResNets, full CIFAR-10, 30 epochs
```

The `full` profile is defined in `bdd/benchmark.py`. The CPU-trained zoo (`cpu` profile,
5 clean + 15 backdoored per domain) is for development only. Do not mix the two in one
`checkpoints/` folder: use a fresh clone or delete `checkpoints/` and `artifacts/` first.

## 2. What is slow on CPU

| Step | CPU cost | Note |
|---|---|---|
| Image zoo training | ~5 min / model at 10k samples | `full` is ~100x more compute |
| Text zoo training | ~3 min / model at 4k samples | |
| Fingerprinting | minutes / model | SAE training + patching over all components |

## 3. Optional: a larger base model

`bdd/data_text.py::BASE_MODEL` is a 4-layer BERT (11M params) so the project runs on CPU.
With a GPU, set it to `distilbert-base-uncased` as in the proposal. Then check that
`bdd/harvest.py`, `bdd/patching.py` and `bdd/localize.py` still match the module names
(DistilBERT uses `distilbert.transformer.layer[i].ffn.lin1` rather than
`bert.encoder.layer[i].intermediate`, so those hooks need adapting).

## 4. Outputs to send back

`artifacts/results/<domain>/results.json` and the PNGs next to it, plus
`artifacts/fingerprints/` if the analysis should be re-run elsewhere.
Checkpoints are git-ignored (large).
