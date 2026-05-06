# EMNLP Top-Conference Quality Audit

- Generated at: `2026-05-06T07:20:45Z`
- Scope: current LFCRASH-CBM EMNLP/ARR paper package and support artifacts.
- Standard used: top-conference acceptance, oral, and best-paper competitiveness; not just submission sanity.

## Bottom Line

- ARR / regular acceptance package: `credible`.
- Oral: `possible but not locked`; depends on reviewer buy-in to the semantic-interface framing.
- Best paper: `not currently credible`; the gap is research evidence and visual/story decisiveness, not only polish.

## Internal Scores

| Target | Score | Reading |
|---|---:|---|
| ARR submission | 93/100 | technically coherent package |
| EMNLP accept | 80/100 | plausible if claims stay bounded |
| EMNLP oral | 65/100 | needs stronger first-read story and defense |
| Best paper | 38/100 | not enough decisive evidence yet |

## Measured Package Statistics

- BibTeX entries: `49`
- Unique cited keys in main text: `49`
- Citation commands across main+appendix: `53`
- Main-body figures: `15`
- Main-body tables: `9`
- Appendix figures/tables: `8` / `4`
- Missing BibTeX keys: `0`
- Unused BibTeX keys: `0`

## Findings

| Severity | Area | Finding | Required action |
|---|---|---|---|
| CRITICAL | Best-paper evidence | The current package is not best-paper-ready: although the DAD light-regularization block is now complete at 3/3 runs, aggregate DAD mechanism evidence remains mixed. | Use the completed 3-run aggregate only as bounded stress evidence; either keep oral-ready framing or add new independent evidence before any best-paper claim. |
| MAJOR | Experiment breadth | Main predictive evidence is concentrated on DAD and A3D; this can support ARR/oral framing but is thin for a decisive best-paper generality claim. | Add a predeclared independent stress setting only as a separate post-submission or revision block; do not claim broad generality from the current two-dataset package. |
| MAJOR | Experimental coherence | The paper relies on multiple recipe families: canonical headline, support protocol, ontology launcher, actor/classifier trigger comparison, and archived intervention assets. The protocol table plus visual evidence map improves readability, but reviewers may still read the package as fragmented unless claim tiers stay explicit. | Make every table caption state its evidence tier and keep headline, support, and stress evidence visually separated. |
| MAJOR | Visualization | The visual story is improving but still not best-paper-level on first read. The concept-pipeline figure is repaired, the safety-utility plot exposes three-seed ontology intervals, the framework figure centers the interface, and Figure 1 now explicitly maps its risk trace, concept activations, and alert marker to the protocol-separated evidence blocks. | Preserve the Figure 1 bridge during final read-through; the next visual/story gap is broader decisiveness, not missing cross-reference scaffolding. |
| MAJOR | Related work | The bibliography now has 49 entries and 49 unique main-text cited keys, which is much healthier than the earlier draft but still not a best-paper-level map of ontology governance, concept bottlenecks, and safety-critical VLM evaluation. | Add only targeted missing literature next: ontology/concept governance in NLP, safety-critical VLM evaluation, and human validation protocols. |
| MAJOR | Human validation | The ontology review is useful, but it is still light-touch review over concept entries rather than a human-verified frame-level concept benchmark. | Keep the claim as governed semantic interface; do not imply exhaustive human concept-label validation. |
| MINOR | Paper polish | The paper has many tables relative to narrative figures; several tables are dense and may slow first-read comprehension. | Convert one dense table into a visual evidence map or move lower-tier support tables to appendix if page budget allows. |

## Active DAD Light-Reg Block

- Started: `3/3`
- Completed: `3/3`
- Latest eval snapshot is monitoring only, not paper evidence.
- AP mean: `60.67%` over `n=3`
- mTTA mean: `2.82s`
- Completed aggregate is available for support interpretation only.
- Completed AP mean: `63.18%` over `n=3`
- Completed mTTA mean: `2.27s`

## Immediate Fix Queue

1. Treat the completed DAD light-reg 3/3 aggregate as support-only; do not upgrade DAD claims beyond stress-test framing.
2. Figure 1 bridge is addressed in the current package; preserve the headline/ontology/timing-support mapping during final PDF read-through.
3. Continue targeted related-work hardening around ontology governance, safety-critical VLM evaluation, and human validation protocols.
4. Tighten experiment narration so single-run headline, seed-backed evidence, support ablations, and archived diagnostics cannot be confused.
5. Keep human-validation language bounded unless a real frame-level concept annotation study is added.
