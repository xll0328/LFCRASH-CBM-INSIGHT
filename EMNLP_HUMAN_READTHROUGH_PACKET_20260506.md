# EMNLP Human Read-Through Packet (Risk Sentences)

Date: 2026-05-06
Purpose: fast human pass on sentences likely to be over-interpreted by reviewers.

## How To Use
- Read each sentence in the compiled PDF context (page flow), then confirm it is bounded and evidence-aligned.
- If a sentence sounds stronger than its evidence tier, soften wording in-place and rerun sanity.

## paper/emnlp2026/sec_intro_emnlp.tex
- L92: We do not claim dataset-agnostic superiority, universal policy-level causality,
- L131: families, and export a trainable ontology with explicit provenance. In the
- L154: interface improves or preserves prediction quality, whether ontology choice
- L161: \item We introduce a reproducible multimodal risk-concept construction pipeline that mines, refines, merges, balances, and reviews risk concepts to produce a trainable ontology with explicit provenance, family structure, and canonical naming policy.
- L165: \item We keep the central claim intentionally constrained: \method{} advances semantic-interface methodology and auditing, with the strongest current contribution in interpretability and controlled ontology science rather than policy-level causal intervention.

## paper/emnlp2026/sec_method_emnlp.tex
- L43: metric. This equation simply states the target operating regime: improve lead
- L89: support provenance tracking, case studies, and matched ontology comparisons.
- L93: merge provenance, and review notes. This matters because the ontology is not
- L103: families are balanced explicitly, and retained concepts carry provenance and
- L319: This does not prove that every trained model will respond monotonically to
- L323: intervenable} interface, even though full policy-level causal validation lies

## paper/emnlp2026/sec_experiments_emnlp.tex
- L115: \item Figure~\ref{fig:intervention_gallery}: dual-intervention gallery for support-only intervention intuition.
- L163: Mechanism stress / hardening & DAD & at least 3 runs per stress block (3 checked blocks, support-only role) \\
- L170: This density does not claim dataset-agnostic universality. It claims a controlled
- L174: \paragraph{Architecture-extension probes (support-only).}
- L180: Until each family reaches completed 3-run aggregates, they remain support-only
- L193: \subsection{Visual evidence upgrade: intervention gallery (support-only)}
- L214: \caption{Support-only intervention gallery. The three paired DAD cases show how concept edits
- L267: \caption{Cross-dataset stress bridge (support-only): CRASH cases illustrate how
- L280: We do not claim dataset-agnostic dominance from this package; we claim
- L282: \paragraph{Cross-dataset transfer note (support-only).}
- L347: with the family-balance and merge-provenance assets in the appendix, these
- L449: \paragraph{DAD mechanism-hardening block (support-only).}
- L455: DAD remains a high-variance stress test for this method family, while the
- L458: \paragraph{Low-regularization follow-up (support-only).}
- L499: model improves mTTA over CRASH while maintaining competitive AP, which places
- L555: yielding one universally dominant vocabulary. On DAD, the compact manual set
- L627: policy-level causal timing account lies beyond the current paper.
- L642: explicit provenance.
- L651: merge provenance can be inspected directly.
- L677: structure whose coverage, merge provenance, and review trajectory can be
- L698: Hybrid amplification & 50 edited cases & earlier / same / later = 14\% / 84\% / 2\% & edit effects are heterogeneous, not universal \\
- L700: Top-risk edit & 25 edited cases & mean shift = 0.00 frames & naive top-risk edits alone do not guarantee movement \\
- L713: is not yet a finished human-verified concept benchmark or a complete causal
- L714: timing proof. That asymmetry is intentional in our presentation: the paper is

## paper/emnlp2026/sec_conclusion_emnlp.tex
- L31: is therefore auditable and trainable, but not yet a finished concept benchmark.
- L34: state by construction, a complete policy-level causal intervention study
- L44: stress setting, not a universal performance frontier, and treats actor-policy
- L45: timing as a partial, support-only account. This choice keeps the central claim

## Reviewer-Facing Safety Rules
- Keep A3D as clean flagship evidence.
- Keep DAD as stress evidence; do not upgrade to global mechanism proof.
- Keep actor-policy statements support-only unless aggregate evidence materially changes.
- Keep ontology claims as governed modeling choices, not universal superiority claims.
