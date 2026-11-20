# Preliminary text results (CPU profile)

Zoo: 5 clean + 15 backdoored (InsertSent, AddSent, StyleBkd-lite; 5 seeds each) 4-layer BERT on SST-2.
Mean clean accuracy 0.776, mean attack success rate 1.00.
Only 20 models, so every number below is noisy. Treat as a pipeline check, not a finding.

| Protocol | Classifier | AUC | TPR@5%FPR |
|---|---|---|---|
| 5-fold CV (repeated) | random forest | 0.83 | 0.60 |
| 5-fold CV (repeated) | logistic regression | 1.00 | 1.00 |
| Leave-one-attack-out | random forest | 0.73 | 0.40 |
| Leave-one-attack-out | logistic regression | 0.87 | 0.33 |

Leave-one-attack-out AUC per held-out attack (logreg): AddSent 1.00, InsertSent 0.70, StyleBkd 1.00.

Feature-family ablation (LOAO AUC, random forest): stat 0.55, SAE 0.73, causal 0.55, all 0.73.
Removing the statistical family raised AUC to 0.91, which suggests those features add noise at this sample size.

Caveats
- Single-feature AUCs (best: `sae_recon_r2_last` 0.96) are picked from ~100 features on 20 models, so they are inflated by selection.
- Localisation failed: ablating the top-10 suspect FFN units did not reduce attack success (0.00 drop, same as random). The suspect-unit heuristic does not find the backdoor circuit in these models.
- Clean-vs-backdoored separation by CV is much higher than by LOAO, so part of the signal is attack-specific.
