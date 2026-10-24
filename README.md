# A LEMBAS-style signalling RNN for PAR2 → STING in inflammation

A mechanistic recurrent neural network over a curated mouse signalling graph, following the
LEMBAS formulation (Nilsson *et al.* 2022). Node activities propagate along literature-curated,
signed edges until the network reaches a steady state; transcription-factor activities read out
of that state predict measured TF activity and gene expression.

This repository reproduces the three perturbation predictions reported in the manuscript:

| case | what it represents | command |
|---|---|---|
| **PAR2 KO** | PAR2 knockout alone, no TLR4 engagement | `python scripts/run_step34_studies.py` |
| **PAR2 KO + TLR4 KD** | PAR2 editing delivered by an albumin carrier that independently suppresses TLR4 | same script — setup B, with a κ sweep |
| **dynasore** | dynamin inhibition: the endosomal routes are cut, the membrane routes are not | `python scripts/run_step46_dynasore_route.py` |

---

## 1. Install

Python 3.11. A CUDA GPU is strongly recommended for training (CPU works but is ~10× slower).

```bash
git clone <this repo>
cd <this repo>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions the published results were produced with. `torch` is listed
without a CUDA suffix; install the build matching your driver from pytorch.org if you want GPU
support.

## 2. Input data

**All input data is included in this repository** (~250 MB): the STRING-derived backbone,
Reactome annotation, OmniPath, SIGNOR, TRRUST, CollecTRI and PhosphoSitePlus exports, the
differential-expression tables for the 48 GEO inflammation contexts, and the GSE219229,
GSE134486 and GSE275623 panels.

`data/README.md` lists every input with its source and licence.

> **Licence note for re-users.** `data/psp/` is derived from **PhosphoSitePlus**, whose terms do
> not generally permit redistribution and which requires a separate licence for commercial use.
> It is included here for reproducibility of the published analysis; if you redistribute or build
> on this repository, check your own obligations at <https://www.phosphosite.org/>.

## 3. Check the inputs, then build the network

```bash
python scripts/check_inputs.py
```

lists every configured input as present or missing, groups the missing ones by the section of
`data/README.md` that explains how to get them, and exits non-zero if anything required is absent.
Run it first — the step scripts fail with a plain `FileNotFoundError` that names the path but not
what the file is.

Each step then validates its inputs and writes to `outputs/`. Run them in order:

```bash
python tests/test_step00.py    # environment and version report
python tests/test_step01.py    # gene universe (8,199 genes) and DEG harmonisation
python tests/test_step02.py    # merge the curated interaction databases, resolve sign conflicts
python tests/test_step03.py    # signal-entity nodes and their edges
python tests/test_step04.py    # assemble the network, assign confidence tiers T0–T3
python tests/test_step05.py    # split into signalling layer and transcriptional readout layer
python tests/test_step06.py    # build conditions and training targets (ULM TF activities)
```

After step 6 you should have 8,393 nodes and 253,950 signalling edges, a regulon of 45,195
TF→gene pairs, and 144 conditions (57 drug, 48 GEO, 36 perturbation-paper, 3 sealed holdout).

If you change anything in `curation/` or `config.yaml`, re-run the affected steps with

```bash
python scripts/rebuild_and_report.py --tag my_change
```

which archives the previous outputs and prints exactly which edges moved — a silent change to the
network is the most common source of irreproducible results here.

## 4. Train

```bash
python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R4 R0
```

~2 minutes per run on a GPU, so ~20 minutes for 10 runs. `R4` is the model reported throughout;
`R0` is a readout-parameterisation comparison. Five seeds are trained because **single-seed
results in this model are not reliable** — several quantities change sign between seeds, and
every number in the manuscript is a median over seeds with the spread reported.

Checkpoints are written to `outputs/source_variants/<variant>/checkpoint.pt`.

> **Any change to the network topology invalidates existing checkpoints.** Every analysis script
> detects this and exits with the retrain command rather than silently loading a mismatched model.

Optional, for the cross-validated comparison in the manuscript:

```bash
python scripts/run_cv_20.py                                      # 5 folds x 3 seeds, ~25 min GPU
python scripts/run_benchmark_16.py --lembas-variant R4           # vs 8 baseline learners
```

## 5. Reproduce the three predictions

### 5.1 PAR2 knockout, and PAR2 knockout + TLR4 knockdown

```bash
python scripts/run_step34_studies.py
```

Simulates three setups and prints each node's response:

* **A** — PAR2 KO with no TLR4 input. The pure genetic perturbation.
* **A′** — the same with the LPS context switched on, as a sensitivity check.
* **B** — PAR2 KO **plus** a TLR4 knockdown representing the albumin carrier, with κ_TLR4 swept
  from 0 to 0.95.

The script prints the dose–response over κ_TLR4 with the experimentally measured TLR4 reductions
marked, and applies the call threshold (10⁻³) when stating a verdict — a correct sign below that
threshold is reported as *no call*, not as agreement.

```bash
python scripts/run_step45_panelC_table.py      # the literature holdout table
```

scores the three sealed studies, running each as its own perturbation and mapping each observation
to the model layer that matches its assay (activity vs transcript).

### 5.2 Dynasore

```bash
python scripts/run_step46_dynasore_route.py    # the two competing routes, 5 seeds
python scripts/run_step32_dynasore.py          # single-seed diagnostic, each cut separately
```

**Dynasore is simulated as an edge cut, not a node knockout**, because it blocks dynamin-dependent
internalisation rather than removing a protein. The script removes

* the endosomal β-arrestin ERK scaffold, `Arrb1/Arrb2 → {Raf1, Map2k1, Map2k2, Mapk1, Mapk3}`, and
* the endosomal TLR4→TRIF edges, `Tlr4 → Ticam1`, `Tlr4 → Ticam2`,

while leaving `F2rl1 → Arrb1/2`, `Arrb1 → Src` and the entire dynamin-**independent** MyD88 branch
intact. A node knockout of `Arrb1` would additionally delete β-arrestin-mediated receptor
desensitisation and the `Arrb1 → Src` arm, which is not what the drug does.

The pre-registered out-of-sample gene-panel test:

```bash
python scripts/run_step43_dynasore_panel.py
```

This scores 13 genes against a protocol fixed in writing beforehand
(`curation/dynasore_panel_prereg_2026-10-20.md`). It reports the panel's position against a
permutation null rather than sign agreement, because ~83 % of the transcriptome moves in the same
direction under this cut and sign agreement alone is therefore uninformative.

### 5.3 Mechanistic analysis

```bash
python scripts/run_step40_contributions.py     # per-edge contributions into a node
python scripts/run_step50_dynasore_ranks.py    # route ranks against the whole network
python scripts/run_step48_tf_influence.py      # readout-layer TF influence
```

`run_step40_contributions.py` decomposes a node's response into `weight × Δ(source)` over its
in-edges, which is how the manuscript attributes a prediction to specific interactions.

## 6. Figures

```bash
python figures/make_panel_b_methods.py         # method comparison, out-of-fold
python figures/make_panel_b1_fold0.py          # the same on fold 0 alone
python figures/make_panel_cv.py                # cross-validation
python figures/make_panel_e_dynasore.py        # the two dynasore routes
python figures/make_si09_pparg_chain.py        # SI: one unsigned edge, 388 genes
python figures/make_si10_two_misses.py         # SI: the two scored misses
```

Each writes `.pdf` (vector, editable text) and `.png` to `figures/rendered/`, and prints a legend
note stating what the panel does and does not support. `figures/README.md` lists which run
produces each panel's inputs.

## 7. Repository layout

```
lembas_par2/        the model: core solver, network assembly, loss, training, evaluation
  core.py             MML activation, sparse steady-state solver, adjoint gradients, spectral radius
  model.py            parameterisation (tiers, sign locking, readout), forward pass
  network.py          tier assignment, STRING backbone, route promotion
  readout.py          signalling / transcriptional layer split, regulons
  loss.py             data terms, priors, stability penalty
  conditions.py       condition construction, ULM TF activities
scripts/            pipeline runs, training, and every analysis in the manuscript
tests/              step-by-step build and validation; test_step0*.py construct the network
figures/            panel scripts, one per figure
curation/           this study's curated decisions and pre-registrations
config.yaml         every parameter, path and threshold, with the reason for each in comments
```

## 8. Reproducibility notes

* **Five seeds, always.** Several reported quantities change sign between seeds. Scripts default
  to seeds 0–4 and report medians with the spread.
* **The call threshold is 10⁻³.** A prediction below it in magnitude is *no call* and counts as a
  disagreement, not a success.
* **Activity and transcript are different predictions.** `conditions.readout_layers` in
  `config.yaml` maps each assay type to the layer it is scored against. The manuscript reports a
  case where the model is correct at one layer and wrong at the other for the same gene.
* **Tier T0 is the study's structural hypothesis.** It is specified in `config.yaml: route_edges`
  and is the only place where a prior route enters the topology. Its membership, and the
  consequence that a T0 edge's weight is `1/(T0 fan-in of its target)`, are analysed explicitly in
  the manuscript.
* `curation/*_prereg_*.md` record analyses whose rules were fixed before they were run. Where a
  model revision was made after the comparison data had been seen, the manuscript says so.

## 9. Citing

Please cite the manuscript, and the LEMBAS method it builds on:

> Nilsson A, Peters JM, Meimetis N, Bryson B, Lauffenburger DA. Artificial neural networks enable
> genome-scale simulations of intracellular signaling. *Nat Commun* 2022;13:3069.

The curated data sources carry their own citation requirements; `data/README.md` lists them.

## 10. Licence

Code in this repository is released under the MIT Licence (see `LICENSE`).

**The licence covers the code only.** Curated biological data is not redistributed here and
remains under the terms of its original providers — see `data/README.md`.
