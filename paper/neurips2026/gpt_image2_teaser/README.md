# GPT Image 2 Teaser Generation Pack

Purpose: create paper-facing graphical-abstract candidates for the NeurIPS 2026
INSIGHT draft without hard-coding API keys or replacing quantitative evidence
figures with unreproducible art.

## Target Figure

The current highest-value slot is the introduction motivation figure:
`paper/neurips2026/sec_intro.tex`, Figure `fig:motivation`. The promoted
2026-05-01 asset is:

```text
paper/figures/insight_gpt_image2_teaser.pdf
paper/figures/insight_gpt_image2_teaser.png
```

Use GPT Image 2 for a polished conceptual teaser or graphical abstract only.
Keep real result plots, AP/mTTA tables, and evidence heatmaps generated from
repository artifacts.

## Safe Usage

Set the key outside the repository, then run from the project root:

```bash
export AIHUBMIX_API_KEY='...'
python3 paper/neurips2026/gpt_image2_teaser/generate.py \
  --prompt paper/neurips2026/gpt_image2_teaser/prompt_v1.md \
  --n 2 \
  --size 1536x1024 \
  --quality high
```

The script writes images and a JSON manifest under:

```text
paper/neurips2026/gpt_image2_teaser/outputs/
```

Do not paste API keys into shell history, source files, or status documents.

If the Python SDK route fails from this server, use the direct no-proxy `curl`
route documented in `iteration_log.md`. On 2026-05-01, this was the reliable
path for `gpt-image-2` through AIHubMix.

```bash
bash paper/neurips2026/gpt_image2_teaser/generate_curl.sh \
  paper/neurips2026/gpt_image2_teaser/prompt_v2.md
```

## Iteration Rule

After each generation, score the result with `quality_gate.md`.
Only promote an image into `paper/figures/` if it passes the major criteria:
readable labels, correct scientific story, no hallucinated claims, and no
misleading causal/policy-level overstatement.
