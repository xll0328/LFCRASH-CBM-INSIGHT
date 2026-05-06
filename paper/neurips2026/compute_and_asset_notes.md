# Compute, Asset, and License Notes for Submission

## Compute summary
- Primary hardware described in paper: NVIDIA RTX 4090 x7
- Feature representation: pre-extracted VGG16 ROI features
- Result package includes canonical headline runs, support analyses, and search runs; not all exploratory runs should be counted as headline evidence

## Compute disclosure guidance for paper/checklist
State explicitly:
1. main experiments were run on RTX 4090 GPUs
2. object-centric VGG16 features reduce end-to-end training cost
3. headline tables use only a subset of the full project exploration budget
4. exploratory ontology-search and local-search runs exceeded the cost of the final reported canonical tables

## External assets to credit clearly
- DAD dataset
- A3D dataset
- CRASH and W3AL published results as contextual baselines
- CLIP for pseudo concept supervision / semantic alignment
- VGG16 features / backbone provenance

## Manuscript action items
- add explicit license / terms references for external datasets and models where available
- add a short asset-usage paragraph in appendix
- align checklist answers with current anonymous-release status
