# Pre-registration: the dynasore 19-gene panel test

**Written 2026-10-20, BEFORE the simulation is run.** The model is frozen as of step 41.

## What is known at the time of writing, and is therefore not blinded

The collaborators have stated that **all 19 genes have NEGATIVE log2FC for DY+LPS vs LPS**. I know
the direction before running. What is *not* known is the model's output, and the model **cannot be
changed after this test** — see §6.

**Panel**: Cxcl10, Il6, Ccl4, Il1b, Mefv, Nlrp3, Ifi204, Zbp1, Ikbke, Irf7, Nfkbia, Sting1, Mlkl,
Aim2, Ripk3, Samhd1, Nfkbib, Il33, Ccl5.

## 1. Which genes count, fixed now

| class | genes | n |
|---|---|---|
| **PRIMARY — independent of every revision** | Cxcl10, Ccl4, Mefv, Nlrp3, Ifi204, Zbp1, Ikbke, Irf7, Aim2, Samhd1, Nfkbib, Il33, Ccl5 | **13** |
| EXCLUDED — tuned or already scored | `Sting1` (T0 set rebuilt, step 38), `Nfkbia` (promoted to T0, step 33 Block C), `Il1b`, `Il6` (PAR2_29 scored holdouts) | 4 |
| EXCLUDED — not in the gene universe | `Mlkl`, `Ripk3` (network nodes, no measured expression) | 2 |

**Only the 13 PRIMARY genes are scored.** The other four are reported for transparency and are
**not** part of any pass/fail statement, because the model was changed with their behaviour in
view. `Sting1` in particular may not be quoted from this test in any form.

## 2. The perturbation — the collaborators' contrast, not a PAR2 KO

DY+LPS vs LPS. LPS context, **no `F2rl1` KO**, and dynasore as an **edge cut**, not a node KO
(step 32 established why: an `Arrb1` node KO also removes receptor desensitisation and the
`Arrb1 → Src` arm):

* endosomal ERK scaffold: `Arrb1/Arrb2 → {Raf1, Map2k1, Map2k2, Mapk1, Mapk3}` — 10 edges
* endosomal TLR4 → TRIF: `Tlr4 → Ticam1`, `Tlr4 → Ticam2` — 2 edges (**Block D only**)
* **kept intact**: `F2rl1 → Arrb1/2`, `Arrb1 → Src`, and Blocks A/B/C — the MyD88 branch is
  dynamin-independent and dynasore does not touch it.

Readout: **mRNA**, because the collaborators' data is RNA-seq. R4, 5 seeds.

## 3. The null, and why sign agreement alone is NOT the test

Under this cut roughly **80 % of the measured transcriptome moves negative** (measured in step 32:
0.807). So "all 13 negative" is a weak claim — its probability under the background is
0.8¹³ ≈ **5.5 %**, marginal at best, and it would be true of almost any gene set.

**The test is therefore positional, not directional**: are the 13 genes shifted toward the
negative tail *relative to the rest of the transcriptome*?

* `pctile` = each gene's rank among all measured genes, 0 = most negative, 50 = indistinguishable
  from background.
* **Primary statistic: the median percentile of the 13.**
* **p-value by permutation**: draw 13 random measured genes 10 000 times, count how often the
  drawn median percentile is ≤ the observed one. Seeded, reported.

## 4. Decision rule, fixed now

| outcome | reading |
|---|---|
| median pctile **< 50** and permutation **p < 0.05** | **PASS.** The model places this panel in the negative tail beyond background. This is the first genuinely out-of-sample support for the dynasore mechanism. |
| median pctile < 50 but **p ≥ 0.05** | **WEAK/NULL.** Directionally consistent, not distinguishable from background. Reported as not supporting the mechanism. |
| median pctile **≥ 50** | **FAIL.** The model does not reproduce the panel, and that is the result. |

Secondary, reported but not decisive: how many of the 13 are negative, against the measured
background rate (not against 0.5).

## 5. What a PASS would and would not mean

**Would**: 13 genes that no revision in this project touched move with the collaborators' data,
under a perturbation (dynasore) distinct from both published studies. That is the closest thing to
an out-of-sample test this project has had.

**Would NOT**: validate the PAR2 → STING direction, the TLR4 T0 promotion, the Sting1 T0 set or
the Cgas dilution. Those were all fitted with the answers known (steps 33, 38), and `Sting1`
itself is excluded from scoring here. A PASS is evidence that the **TLR4/TRIF arm transmits
sensibly**, nothing more.

## 6. The commitment

**The model is frozen. No tier change, edge drop, sign change or retrain may be made in response
to this result.** If it fails, the failure is reported. If a change is later judged necessary on
other grounds, this panel is spent and may not be rerun as a test.

This matters more than usual here, because the last four changes were all made with the answer in
view. This is the first test in the sequence that can come out wrong.

## 7. Separately: the albumin-alone failure is a DATA gap, not a model gap

Step 42 found the model predicts ΔSting1 **−0.0192** at κ_TLR4 = 0.40 (albumin alone) where
PAR2_29 Fig 3N shows pSTING unchanged. Agreed reading: the model has **no TLR4 perturbation data
in training** — every TLR4 constraint it has comes from the curated prior, so the *shape* of its
TLR4 dose-response is unconstrained by any measurement. **Recorded as a known missing input:
TLR4 KO / TLR4-inhibitor expression data would constrain it.** Until such data exists, the
dose-response shape is not a claim the model can support, and only the direction at the measured
hAlbumin/cgPAR2 dose should be quoted.
