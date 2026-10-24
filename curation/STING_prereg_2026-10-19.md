# Pre-registration: adding the two missing STING observations to the sealed holdouts

**Written 2026-10-19, BEFORE opening either PDF.** What is already known at the time of writing,
and is therefore not blinded: both *abstracts* have been read. PAR2_26's title claims PAR2
deficiency "magnifies STING signalling" (UP). PAR2_29's abstract says PAR2 deficiency "alleviated
ERK/STING/NLRP3 signaling" (DOWN). What is NOT yet known is what the figures actually show, with
what assay, in which arm, and at what timepoint — and that is what these rules govern.

The point of writing this first: the model predicts dSting1 **+0.486** in PAR2_29's context and
**+0.046** in PAR2_26's. One of these studies will falsify a central claim. The rules for scoring
must not be chosen after seeing which.

## 1. What counts as an observation

A STING observation is admitted only if the figure gives all of:

- a named panel (figure number and sub-panel),
- a stated assay (western blot total protein, phospho-STING, qPCR, IF, reporter),
- the treatment arm and its control, matching the contrast already recorded for that study,
- a direction that is legible from the panel or its quantification, not inferred from the
  abstract, title or discussion.

If the direction is legible only in the abstract and not in a figure, it is recorded as
`direction: unknown` with the reason, exactly as PAR2_26_C1_O03A/B already are, and it is **not**
scored. Abstract-level claims do not enter the holdout set — for either study, in either
direction.

## 2. Assay precedence, fixed now

Where several STING panels exist in one study:

1. **phospho-STING (activation) outranks total STING protein.** The claim is about STING
   *signalling*, and total abundance can fall while activity rises (degradation after activation
   is a known STING behaviour, and PAR2_26's own curation note flags "Degradation/abundance").
2. Where only total protein exists, it is recorded as total protein and scored, with the
   ambiguity noted.
3. Downstream proxies (TBK1-p, IRF3-p, ISGs) are **not** substitutes for STING and are recorded
   as their own nodes if measured.
4. If phospho and total disagree within a study, **both are recorded**, the study is scored on
   the phospho panel, and the disagreement is reported in the text. It is not averaged away.

## 3. In-vivo arms in PAR2_29

PAR2_29 has multiple treatment groups and its carrier (nanosized albumin) is independently
anti-inflammatory via TLR4 by the paper's own account. The admitted contrast is **the same one
already used for the four scored observations** (Tlr4, Gsdmd, Il1b, Il6 — `PAR2_29_C1`,
hAlbumin/cgPAR2 vs matched inflammatory control). No other arm may be substituted for STING.

**Stated in advance**: the carrier confound applies equally to the four observations already
counted as successes. It may not be used to discount a STING result while those are kept. Either
`PAR2_29_C1` is usable as a contrast or it is not, and if it is not, the four existing
observations must be removed and the headline re-reported.

## 4. How step 13 is re-scored

- The holdout set grows from 10 observations to 10 + however many STING observations are admitted
  (at most 4: total and phospho in each study).
- The binomial test is re-run on the enlarged set. **The enlarged number replaces 8/10 as the
  headline**, in whichever direction it moves.
- The pre-step-19-frozen J0 remains the reported model; R4 is reported alongside as now.
- `tests/test_step13.py` and panel C are updated in the same change, so no figure can keep
  quoting 8/10 after the set changes.

## 5. What each outcome means — decided now

| PAR2_29 STING | PAR2_26 STING | reading |
|---|---|---|
| DOWN | UP | **The direction is context-dependent and the model is wrong in the system closest to the collaborators'.** The PAR2 → STING claim cannot be presented as general. Either narrow it explicitly to the tumour / in-vitro context, or drop it and present the mitochondrial arm (study 30: Bcl-xL ↓, cytosolic mtDNA ↑, IFN-β ↑) as the finding instead. |
| DOWN | DOWN | The claim is falsified. The route's net sign is wrong and the write-up must say so. |
| UP | UP | The claim survives its first real test; report the enlarged holdout score and that the earlier curation gap was found and closed. |
| UP | DOWN | Same as row 1 with the systems reversed; the in-vivo LPS result is the relevant one for the collaborators and would then support the claim. |
| unknown (either) | — | That study contributes nothing, is reported as unresolvable from the figures, and the claim stays untested in that system. |

**In no outcome is a model change made to fit these observations.** They are holdouts. If the
prediction is wrong, that is the result; any re-weighting of the T0 route proposed afterwards is
a separate, clearly-labelled exploratory step that cannot be scored on these data again.

## 6. What this does not license

Finding the direction does not retroactively validate the 8/10. That number was computed on a set
that excluded STING, and it is reported as such. The enlarged score is a new number.
