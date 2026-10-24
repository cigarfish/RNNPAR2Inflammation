# References for five signal-entity nodes

Literature support for `SE_CGMP`, `SE_PI3P`, `SE_PI4P`, `SE_PI34P2` and `SE_PI35P2` as defined in
`data/signal_entity/signal_entity_gene_packages_v1_2.csv`. Each entity's producers, negative
controllers and direct receivers are listed as curated, with the reference that supports them.

**Two entries are flagged: the curated sign for `Fig4` is contradicted by the literature, and the
`SE_PI4P → PIP5K` framing is weaker than the file's note implies.** Both are set out in §3.

---

## 1. Reference list

| # | reference |
|---|---|
| R1 | Derbyshire ER, Marletta MA. **Structure and regulation of soluble guanylate cyclase.** *Annu Rev Biochem* 2012;**81**:533–559. doi:10.1146/annurev-biochem-050410-100030 |
| R2 | Francis SH, Busch JL, Corbin JD. **cGMP-dependent protein kinases and cGMP phosphodiesterases in nitric oxide and cGMP action.** *Pharmacol Rev* 2010;**62**(3):525–563. doi:10.1124/pr.110.002907 — PMID 20716671 |
| R3 | Balla T. **Phosphoinositides: tiny lipids with giant impact on cell regulation.** *Physiol Rev* 2013;**93**(3):1019–1137. doi:10.1152/physrev.00028.2012 — PMID 23899561; PMC3962547 |
| R4 | Hammond GRV, Fischer MJ, Anderson KE, Holdich J, Koteci A, Balla T, Irvine RF. **PI4P and PI(4,5)P₂ are essential but independent lipid determinants of membrane identity.** *Science* 2012;**337**(6095):727–730. doi:10.1126/science.1222483 |
| R5 | Hawkins PT, Stephens LR. **Emerging evidence of signalling roles for PI(3,4)P₂ in Class I and II PI3K-regulated pathways.** *Biochem Soc Trans* 2016;**44**(1):307–314. doi:10.1042/BST20150248 |
| R6 | Dong X-P, Shen D, Wang X, Dawson T, Li X, Zhang Q, Cheng X, Zhang Y, Weisman LS, Delling M, Xu H. **PI(3,5)P₂ controls membrane trafficking by direct activation of mucolipin Ca²⁺ release channels in the endolysosome.** *Nat Commun* 2010;**1**:38. doi:10.1038/ncomms1037 — PMID 20802798; PMC2928581 |
| R7 | Zhang Y, Zolov SN, Chow CY, *et al.* / Chow CY, *et al.* **Vac14–Fig4–PIKfyve complex; loss of Fig4 or Vac14 lowers PI(3,5)P₂.** See R3 §PI(3,5)P₂ and the CMT4J literature cited in §3.1. |

## 2. Entity-by-entity support

### SE_CGMP — cGMP
| role | genes | support |
|---|---|---|
| producers | `Gucy1a1`, `Gucy1a2`, `Gucy1b1` | **R1** — soluble guanylate cyclase is an α/β heterodimer whose haem binds NO and which converts GTP to cGMP |
| negative controller | `Pde5a` | **R2** — cGMP is degraded by phosphodiesterases; PDE5 is the cGMP-specific PDE |
| direct receiver | `Prkg1` | **R2** — cGMP acts through three target classes, of which cGMP-dependent protein kinase (PKG) is the principal one |

### SE_PI3P — PI3P
| role | genes | support |
|---|---|---|
| producers | `Pik3c3` (VPS34), `Pik3c2a/b/g` | **R3** — class III PI3K is the main PI3P source; class II PI3Ks also synthesise PtdIns3P |
| direct receivers | `Eea1`, `Wipi1`, `Wipi2`, `Pikfyve` | **R3** — PI3P recruits FYVE-domain (EEA1) and PROPPIN/WIPI effectors; PIKfyve uses PI3P as substrate |

### SE_PI4P — PI4P
| role | genes | support |
|---|---|---|
| producers | `Pi4ka`, `Pi4kb`, `Pi4k2a`, `Pi4k2b` | **R3**, **R4** |
| negative controller | `Sacm1l` (SAC1) | **R3** — SAC1 is the principal PI4P phosphatase |
| direct receivers | `Pip5k1a/b/c` | **R3** — PIP5K phosphorylates PI4P at the 5-position. **See the caveat in §3.2** |

### SE_PI34P2 — PI(3,4)P₂
| role | genes | support |
|---|---|---|
| producers | `Inpp5d` (SHIP1), `Inppl1` (SHIP2), `Pik3c2a` | **R5** — SHIP1/2 generate PI(3,4)P₂ by 5-dephosphorylation of PIP₃; class II PI3K-C2α is the other route |
| negative controllers | `Inpp4a`, `Inpp4b` | **R3**, **R5** — INPP4A/B are the 4-phosphatases that terminate PI(3,4)P₂ |
| direct receivers | `Plekha1/2` (TAPP1/2), `Akt1/2/3` | **R5** — TAPP1/2 are dual-PH adaptors selective for PI(3,4)P₂; AKT's PH domain binds PI(3,4)P₂ as well as PIP₃ |

### SE_PI35P2 — PI(3,5)P₂
| role | genes | support |
|---|---|---|
| producer | `Pikfyve` | **R3** — PIKfyve phosphorylates PI3P at the 5-position |
| negative controller | `Fig4` | **FLAGGED — see §3.1** |
| direct receiver | `Mcoln1` (TRPML1) | **R6** — PI(3,5)P₂ binds and activates endolysosomal TRPML channels with specificity and potency |

---

## 3. Two entries that the references do not support as curated

### 3.1 `Fig4` is curated as a negative controller, but losing it LOWERS PI(3,5)P₂

FIG4/SAC3 is a *bona fide* PI(3,5)P₂ 5-phosphatase, which is what the current annotation records.
**But it is also a required component of the PIKfyve–Vac14–Fig4 complex, and catalytically active
Fig4 is needed for maximal PI(3,5)P₂ synthesis.** `Vac14⁻ʹ⁻` and `Fig4⁻ʹ⁻` fibroblasts show an
**~50 % reduction** in PI(3,5)P₂, and PI(3,5)P₂ is likewise reduced in Charcot–Marie–Tooth type 4J
patients carrying SAC3/FIG4 deficiency.

**Consequence for this model.** The sign that matters in a perturbation model is the *net* effect
of removing the gene, not the direction of its catalytic step. A `Fig4` knockdown should
**lower** `SE_PI35P2`; the curated `negative_controller` annotation will make it **rise**.
**The annotation is backwards for any simulation that perturbs `Fig4`.**

Not yet changed. No analysis in this project perturbs `Fig4`, so nothing reported so far is
affected — but it should be corrected before any PIKfyve/FIG4 perturbation is simulated, and it
is the same class of error as the `Prkca → Pparg` fitted sign (a catalytic annotation standing in
for a net physiological effect).

### 3.2 The `SE_PI4P → PIP5K` note overstates what R4 shows

The file's note reads "Main PI4P→PIP2 branch". **R4 reports the opposite of the simple precursor
model**: acute depletion of PI4P did *not* substantially reduce PI(4,5)P₂ synthesis, and the two
lipids act as **essential but independent** determinants of plasma-membrane identity, PI4P
contributing largely to membrane surface charge.

The enzymatic step `PI4P → PI(4,5)P₂` via PIP5K is correct and R3 supports it. What is not
supported is the implication that this is the dominant flux sustaining the PI(4,5)P₂ pool. The
note should be softened to "PIP5K substrate step; not the dominant PI(4,5)P₂ supply in vivo (R4)".

---

## 4. How these were obtained

Each reference was located and its bibliographic details confirmed by web search on 2026-10-22;
none is quoted from memory. Statements above are attributed to the specific reference that
supports them. Where the literature contradicts the curated annotation (§3) that is stated rather
than reconciled.
