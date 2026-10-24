# STING in PAR2_26 and PAR2_29: what the figures actually show

Read from the PDFs on 2026-10-19 under the rules fixed in `STING_prereg_2026-10-19.md`.
**The two studies do not contradict each other. They measure STING in different cells.**

## PAR2_29 (Small 2024) — Figure 3I, 3N, 3O

| | |
|---|---|
| panels | 3I western blot (Raw264.7: TLR4, NF-κB, pERK, ERK, **pSTING, STING**; HUVEC: TLR4, pSTING); 3N pSTING/STING in Raw264.7; 3O pSTING/GAPDH in HUVEC |
| system | **Raw264.7 mouse macrophages**, in vitro, 2f + LPS |
| intervention | PAR2 edited **in the cell where STING is measured** — cell-autonomous |
| direction | **pSTING DOWN**. pSTING/STING ≈ 4.5 (2f+LPS) → ≈ 1.3 (hAlbumin/cgPAR2); HUVEC the same way |
| text | "hAlbumin/cgPAR2 notably assisted cgPAR2 to decrease STING phosphorylation in both Raw264.7 and HUVEC" |

**There is no in-vivo STING measurement in this paper.** Figure 5 (lung) measures PAR2, TLR4,
GSDMD/C-GSDMD and plasma IL-6/IL-1β/CCL5; Figure 6 (CLP) measures TLR4 and GSDMD by IHC. So the
in-vivo lung contrast `PAR2_29_C1`, which carries our four scored observations, **cannot** carry a
STING observation. The STING result belongs to a separate in-vitro contrast.

**Attribution caveat, stated because it cuts both ways.** `cgPAR2` alone does not lower pSTING
(≈ 4.7 vs 4.5 for 2f+LPS) — but `cgPAR2` alone also fails to edit PAR2 (the paper's own Figure
2I,J: editing requires albumin delivery), so that arm is a failed-delivery control, not a
PAR2-deficient condition. Albumin alone lowers pSTING to ≈ 2.7. The PAR2-attributable step is
therefore **albumin → albumin + cgPAR2 (≈ 2.7 → ≈ 1.3)**, which is still DOWN. The same confound
applies to Tlr4, Gsdmd, Il1b and Il6 in this study, all four of which we already score as
successes.

## PAR2_26 (J Control Release 2023) — Figure 4F, 4G, 4H

| | |
|---|---|
| panels | 4F confocal Cy3-STING; 4G western blot (pERK, ERK, **pSTING, STING**, pTBK1); 4H quantification |
| system | **4T1 breast cancer cells** — NOT the edited macrophages |
| design | Figure 4A: Raw264.7 are PAR2-edited and stimulated with LPS+2f, their **conditioned medium** is transferred to 4T1 cells, and STING is measured in the 4T1 cells |
| intervention | PAR2 edited in a *different* cell — **paracrine**, not cell-autonomous |
| direction | **pSTING UP** strongly (pSTING/STING ≈ 0.5 → ≈ 7.5); pTBK1 UP; **total STING DOWN** (the "degradation" in 4F) |

This is exactly the "mixed UP direction" our curator saw and declined to transfer: total protein
falls while phosphorylation rises. Under the pre-registered precedence rule (§2) phospho outranks
total, so the study's direction is **UP** — but see below for why it still cannot be scored.

## Why this resolves the apparent contradiction, and makes things worse rather than better

* PAR2 KO in a macrophage → pSTING **DOWN in that macrophage** (29).
* PAR2 KO in a macrophage → pSTING **UP in a neighbouring tumour cell** (26).

Both can be true at once, and they are not alternative readings of the same experiment. **Our
model simulates cell-autonomous PAR2 knockout in an LPS-stimulated macrophage.** The matched
observation is therefore PAR2_29's, and it is DOWN, against a prediction of **+0.486** — the
model's largest STING prediction anywhere. PAR2_26's UP is a paracrine effect in a cell type the
model does not represent and through a channel (secretome → neighbouring cell) it has no way to
carry, so it is **not evidence for the model** and must not be reported as though it were.

## A second problem found on the way: an already-scored success is from the wrong cell

`PAR2_26_C1_O04` (**ROS, up, "Fig. 4I"**) is in the same 4T1 block as the STING panels — the "4T1"
label heads that whole row, and the paper's own schematic (Fig. 4K) places ROS inside the tumour
cell. It is a paracrine tumour-cell measurement, and the model scores it 5/5 seeds as a success.
The contrast record for `PAR2_26_C1` lists `cell_type: RAW264.7 / 4T1`, i.e. **two cell types were
collapsed into one contrast**, which is how this got through.

Of PAR2_26's observations: NLRP3 (Fig 4B) and IL-1β (Fig 4C) are **Raw264.7** and are legitimately
comparable; ROS and STING are **4T1** and are not.

## Consequences

1. Add PAR2_29's pSTING as a new in-vitro contrast and **score it**: it is matched to what the
   model simulates. Expected outcome: a miss, and the enlarged headline becomes **8/11**.
2. Record PAR2_26's pSTING with its direction (UP) and evidence, excluded with the reason
   **"system mismatch: paracrine, 4T1 tumour cell"** — not the current "direction unknown",
   which is now known to be wrong.
3. **Re-examine `PAR2_26_C1_O04` (ROS).** If it is excluded on the same grounds the headline
   becomes 7/10 before STING is added — so this is not a free correction, and it is the user's
   call, not a change to be slipped in.
4. The write-up may no longer cite PAR2_26 as STING support for a cell-autonomous claim.
