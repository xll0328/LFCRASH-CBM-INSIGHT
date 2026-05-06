# GPT Image 2 Iteration Log

Date: 2026-05-01

## Setup

- API documentation checked:
  - AIHubMix unified image endpoint accepts model paths under
    `https://aihubmix.com/v1/models/<provider/model>/predictions`.
  - AIHubMix also documents an OpenAI-compatible base URL:
    `https://aihubmix.com/v1`.
  - OpenAI lists `gpt-image-2` as an image generation model under
    `v1/images/generations`.
- Local environment check:
  - `AIHUBMIX_API_KEY` was not set.
  - `OPENAI_API_KEY` was not set.
- No paid image call was made because the key was only present in chat text and
  should not be copied into shell history or repository files.

## Current Best Target

Replace or augment Figure `fig:motivation` with a polished graphical abstract
after GPT Image 2 output passes `quality_gate.md`.

## First Prompt

`prompt_v1.md` is intentionally high-information and paper-grounded. It asks
for a technical NeurIPS graphical abstract with the bounded thesis:

> semantic risk concepts are part of the decision pathway, not post-hoc
> explanations.

## 2026-05-01 Iterations

- Python SDK / `requests` attempts against proxied endpoints were unreliable
  on this server (`APIConnectionError`, proxy disconnects, or SSL EOF). The
  successful route was direct `curl` to
  `https://api.aihubmix.com/v1/images/generations` with local proxy variables
  unset and the key read from a no-echo TTY prompt.
- Candidate A:
  `outputs/20260501T192019Z_curl_gpt_image2_direct_medium/insight_teaser_00.png`.
  This proved the route works, but it is square and too dense for a paper
  teaser.
- Candidate B:
  `outputs/20260501T192337Z_curl_gpt_image2_v2_high/insight_teaser_00.png`.
  This is the current best asset: 1536x1024, high quality, readable panel
  structure, bounded claims, no fake metrics, and no crash aftermath.
- Promoted Candidate B to:
  - `paper/figures/insight_gpt_image2_teaser.png`
  - `paper/figures/insight_gpt_image2_teaser.pdf`
- Updated `sec_intro.tex` so Figure `fig:motivation` uses the promoted teaser
  as a schematic overview. The caption explicitly states that the dashcam
  panels are illustrative rather than evaluation evidence and points real DAD
  qualitative evidence to Figure `fig:multi`.

## Current Best Target

Use `paper/figures/insight_gpt_image2_teaser.pdf` as the paper-facing Figure 1
schematic unless a human reviewer prefers the real DAD single-case figure as the
main motivation figure.
