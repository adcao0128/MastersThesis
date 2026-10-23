# MastersThesis

Phosphorylation site prediction for kinase–substrate interactions using
structure- and sequence-aware Graph Neural Networks.

## Repository layout

- **This repo** (`MastersThesis/`) — original pipeline: PSPA/Dataset curation,
  k-hop structure graphs, PPI-style GNN (`GNN_DATA.py`, `train_test_khop.py`,
  e.g. `train_backbone.py`, `train_khop.py`), and baseline comparisons.
- **`~/Thesis_Results/`** — *active* development (separate local git repo, code
  only): semi-inductive dual-encoder (BGNN + site encoder, `new_model.py`),
  1-hop sequence graphs, hit@k and AUROC evaluation. See its README.

## Pipeline (both eras)

1. **Data**: PhosphoSitePlus + PSPA Kinase–Substrate sets; CD-HIT-70 cluster splits;
   balanced/imbalanced negatives, stratified by kinase family.
2. **Structure**: AlphaFold PDBs → residue graphs (graphein distance + peptide-bond
   edges); ESM residue embeddings.
3. **Models**: residue-level GNNs; baselines MUSITE / PhosphoFormer / GPS / hit@k.
4. **Evaluation**: AUROC/AUPRC, PR/ROC curves, confusion matrices, hit@k,
   probability-distribution diagnostics for imbalance.

## Running on HPC

Slurm jobs follow one template (see `*.slurm`): GPU partition, then

```
singularity exec --nv ~/singularity_ppi.sif bash -c "source /opt/venv/bin/activate && python3 <script>.py"
```

## Repo hygiene

Data, checkpoints, and logs are large/regenerable → `.gitignore` keeps them out.
Code only in version control.