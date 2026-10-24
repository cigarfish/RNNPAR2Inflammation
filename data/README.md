# Input data — what is here, and the one thing that is not

**Everything the model needs is included in this repository.** Paths below are the defaults in `config.yaml: paths` — change them there if
you keep data elsewhere.

Run `python scripts/check_inputs.py` to see what is present or missing on your checkout.

---

## ⚠ PhosphoSitePlus — included, but check your obligations

`data/psp/` holds the mouse kinase–substrate and phosphosite tables (13,052 records → 5,986
unique pairs), supplying the phosphorylation edges of the network.

**PhosphoSitePlus terms do not generally permit redistribution, and commercial use requires a
separate licence.** It is included here so the published analysis is reproducible. If you
redistribute this repository or build on it, check your obligations at
<https://www.phosphosite.org/>. Several public resources (e.g. Open Targets) exclude
PhosphoSitePlus-derived data for this reason.

## Included: curated interaction sources

| path | source | size | licence |
|---|---|---|---|
| `data/merged_signaling_network_new.tsv` | STRING v12 mouse, filtered to Reactome pathway genes | 3.1 MB | STRING: CC BY 4.0 |
| `data/reactome_pathway_gene.csv` | Reactome pathway → gene membership, mouse | 0.6 MB | CC BY 4.0 |
| `data/ReactomePathwaysRelation.MMU.gene.txt` | Reactome pathway hierarchy | <0.1 MB | CC BY 4.0 |
| `data/omnipath_ppi.mouse.tsv` | OmniPath, mouse, signed interactions | 5.0 MB | see note |
| `data/signor_mouse/` | SIGNOR, mouse (108 files) | 1.9 MB | CC BY-SA 4.0 |
| `data/trrust_rawdata.mouse.tsv` | TRRUST v2, mouse | 0.2 MB | free for academic use |
| `data/collectri_mouse.csv` | CollecTRI regulon, mouse | 1.6 MB | see CollecTRI / decoupleR terms |

> **OmniPath** aggregates many primary resources, each with its own terms, and some components are
> non-commercial. See <https://omnipathdb.org/info> and cite the primary resources as OmniPath
> requests.

## Included: expression data

| path | contents | size |
|---|---|---|
| `data/DEG/` | per-contrast differential expression for the 48 GEO inflammation contexts (48 files) | 45.8 MB |
| `data/enrichment/` | pathway program scores for the same contexts (48 files) | 0.8 MB |
| `data/geo_context_metadata.csv` | the 48 accessions and their annotation | <0.1 MB |
| `data_external/GSE219229_DEG/` | the GSE219229 kinase-inhibitor panel, one file per drug (59 files) | 102.7 MB |
| `data_external/GSE134486.tsv` | trametinib (MEK) contrast | 11.9 MB |
| `data_external/GSE275623/` | Cgas-KO BMDM + LPS | 2.3 MB |
| `curation/GSE219229.csv` | the raw drug expression matrix the per-drug tables derive from | 11.7 MB |

All are derived from public GEO series. `scripts/make_gse219229_deg.py` regenerates the per-drug
tables from the raw matrix.

> **Not included:** the journal supplementary workbook used to assemble the kinome target profiles
> (`44320_2024_40_MOESM2_ESM.xlsx`). It is a publisher's file and no code reads it — the profiles
> it was used to build are in `curation/drug_targets.csv`. Obtain it from the original article if
> you need to re-derive them.

## Included: this study's own curation

Covered by the repository licence:

| path | contents |
|---|---|
| `data/signal_entity/` | signal-entity definitions (DAG, IP₃, Ca²⁺, cGAMP, cytosolic mtDNA, phosphoinositides …) with producer, negative-controller and receiver genes |
| `data/perturbation/` | 36 directional constraints from the PAR2/β-arrestin/ERK literature, plus the sealed holdout observations |
| `data/cellsignal/`, `data/cellsignal_location/` | edges curated from Cell Signaling Technology pathway diagrams, and the subcellular-location annotation used for the endosomal/membrane split |
| `data/gene_binding_profiles.csv`, `data/gene_feature_binary_matrix.csv` | derived gene annotation |
| `../curation/` | sign-conflict decisions, drug→target mapping, readout exclusions, pre-registration documents |

## Not included: generated outputs

`outputs/` (~720 MB: assembled network, conditions, trained checkpoints, analyses) is excluded and
fully reproducible — see the top-level README, sections 3–5. Trained checkpoints depend on the
network topology, so they would be invalidated by any curation change in any case.
