# FedBridgeMamba-KAN

Federated and explainable detection of cross-chain bridge exploits via
selective state-space modeling and neuro-symbolic alert triage.

**Source code only, and only the code that produced the article.** No dataset,
derived data, result table, figure or manuscript is redistributed here — see
[Why code only](#why-code-only). Scripts whose results did not make it into the
paper are not published either.

---

## Why code only

All three corpora are public, but their licences differ and one of them governs
what may be shared:

| Corpus | Source | Licence | Derivatives |
|---|---|---|---|
| XChainWatcher (Ronin, Nomad) | https://github.com/AndreAugusto11/XChainWatcher | MIT | redistributable |
| Yan et al., ASIA CCS 2025 | https://doi.org/10.5281/zenodo.15392759 | CC-BY-4.0 | redistributable |
| XChainDataGen | https://doi.org/10.5281/zenodo.15341722 | **CC-BY-NC-ND-4.0** | **no derivatives may be distributed** |

The no-derivatives clause on the third corpus covers the extracted tables,
account histories and per-account scores this pipeline produces from it. Rather
than ship a partial artifact set and invite confusion about which outputs are
licensed for reuse, this repository releases the pipeline and nothing else.

Live on-chain verification is read through public RPC endpoints; no archive
node is required.

---

## Getting started

```bash
git clone https://github.com/kmkholm/FedBridgeMamba-KAN
cd FedBridgeMamba-KAN
pip install -r requirements.txt
git clone https://github.com/AndreAugusto11/XChainWatcher datasets/XChainWatcher
```

The extended corpora are downloaded from the Zenodo DOIs above and unpacked
under `datasets/`. Everything runs on CPU.

### Primary corpus — Ronin and Nomad

| Step | Script | Produces |
|---|---|---|
| 1 | `build_dataset.py` | 1.23M decoded events, 187k transactions, rule verdicts |
| 2 | `refine_labels.py` | attack / benign / suspicious tiers; Nomad onset validated at 2022-08-01 21:00 UTC |
| 3 | `tokenize_traces.py` | per-transaction event traces |
| 4 | `build_histories.py` | 32-step account-history sequences |
| 5 | `add_context.py` | ten bridge-level rolling-window features |
| 6 | `add_usd.py` | USD-normalised amounts (token decimals + exploit-era prices) |
| 7 | `make_splits.py` | frozen protocols: random, temporal zero-day, leave-one-bridge-out |
| 8 | `train_baselines.py` | classical baselines across all protocols |
| 9 | `train_model.py` | BridgeMamba-KAN (`model.py`: BiMamba S6 blocks + B-spline KAN head) |
| 10 | `train_anomaly.py`, `score_anomaly_v2.py` | benign-only autoencoder and latent density scoring — the reported negative result |
| 11 | `triage_experiment.py` | **neuro-symbolic alert triage: headline zero-day result** |
| 12 | `train_federated.py` | federated study, three aggregation rules, poisoned-client threat model |

### Extended corpus — nine further bridges

| Script | Purpose |
|---|---|
| `parse_dumps_local.py`, `parse_stargate.py` | stream-parse the Postgres dumps without a database |
| `extract_across_full.py` | extract Across with fee and intent-relayer columns |
| `corpus_stats.py` | reproduce the originating study's Table 1 — extraction validation |
| `account_detection.py` | **exploiter-account detection: the extended-corpus result reported in the paper** |

Detection is posed at account level, not per transaction. That is a deliberate
reformulation, not a convenience: the label originates in a destination-side
oracle fault that is not predictable from deposit-side features, so the
per-transaction task is close to unlearnable while the account-level one is not.

### Figures

| Script | Figures |
|---|---|
| `make_figures.py` | incident timeline, Nomad onset, context separation, evaluation protocols |
| `eval_figures.py` | ROC/PR curves, confusion matrix, training curves |
| `fig_account.py` | exploiter-account detection |
| `fig_extended_v2.py` | multi-bridge validation |
| `eval_xai.py` | permutation importance and learned KAN edge functions |

---

## Verifying your run

`corpus_stats.py` checks the extraction against the originating study's
published Table 1. A correct run reproduces:

- per-bridge transaction counts **exactly**, and transferred values to within 0.1%
- **1,284** value-conservation failures against their reported 1,285
- **1,276** failures originating on Optimism against their reported 1,277
- the highest-ratio case at \$74.78 in / \$2,076.03 out — the example named in their paper

If your counts differ, the extraction is wrong; do not proceed to training.

SHA-256 digests of the federated model per round and of each result table were
anchored on a Hyperledger Fabric 2.4.9 ledger (twelve records, dual-organisation
endorsement) when the paper was written. `fl_round_hashes.py` recomputes those
digests from a fresh run so they can be compared against the anchored ones.

---

## Reproducibility notes

- All randomness is seeded (42 unless stated).
- Suspicious-tier transactions are excluded from **all** training sets and
  **all** test metrics. This quarantine is deliberate: do not "fix" it.
- Report prevalence alongside every PR-AUC. The temporal test set is small
  because Nomad traffic collapsed after the incident.
- Accuracy is not reported anywhere: at the prevalences involved (0.02–0.34%)
  a classifier that flags nothing exceeds 99.6%.

---

## Citation

```bibtex
@article{tawfik2026fedbridge,
  title   = {FedBridgeMamba-KAN: Federated and Explainable Detection of
             Cross-Chain Bridge Exploits via Selective State-Space Modeling
             and Neuro-Symbolic Alert Triage},
  author  = {Tawfik, Mohammed},
  journal = {IEEE Access},
  year    = {2026}
}
```

Code released under the MIT licence (see `LICENSE`).

**Contact:** Mohammed Tawfik — m.tawfik@su.edu.jo

---

## Revision: ten-seed experiments (`revision/`)

The revised article repeats every stochastic experiment over ten seeds (42 and 0–8)
and reports mean ± standard deviation. The scripts below produce those numbers; they
read the derived data written by the pipeline above and write their outputs to
`revision/results/` (not distributed, for the licence reason given above).

| Script | Article content |
|---|---|
| `e1a_baselines.py` | classical baselines under the four frozen protocols |
| `e1b_model.py` | BridgeMamba-KAN under the four protocols (GPU when available) |
| `e2_triage.py` | zero-day alert triage: learned ranker, USD prior, raw-amount ablation, score fusion (weighted mean and escalation rule), address-level intervals, leave-one-positive-out |
| `e2b_triage_kan.py` | BridgeMamba-KAN as the learned triage ranker |
| `e3_accounts.py` | exploiter-account detection with fixed alert budgets and paired tests |
| `e4_federated.py` | federated prototype: clean, label flip, boosted model replacement |
| `e5_anomaly.py` | benign-only autoencoder and latent density scores |
| `make_tables.py`, `make_macros.py`, `show_results.py` | tables and summary numbers from the result files |
| `figures/make_figures.py`, `figures/fig_multibridge.py`, `figures/gen_methodology.py` | result figures and the editable methodology figure |

```bash
pip install -r revision/requirements.txt
python revision/e1a_baselines.py          # likewise for the other scripts
python revision/make_tables.py
```

The classical models are sensitive to the scikit-learn build; the pinned versions in
`revision/requirements.txt` reproduce the published values.
