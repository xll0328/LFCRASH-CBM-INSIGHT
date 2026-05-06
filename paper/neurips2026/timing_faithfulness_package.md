# Timing-Faithfulness Analysis Package

## Objective
Turn WHEN interpretability from a slogan into a measurable object.

## Existing assets
- `insight_timing_faithfulness.py`
- `output/insight_analysis/dad_timing_v4/timing_faithfulness_summary.json`
- `output/insight_analysis/dad_timing_v4_enhanced/timing_faithfulness_summary.json`

## Current trustworthy evidence
Prediction-branch timing statistics are available and usable.
Archived actor-branch values are effectively flat around 0.498 in current outputs and should not be used as strong policy evidence.

## Required analysis blocks

### 1. Prediction crossing analysis
For each positive case:
- concept surge frame
- risk-score crossing frame
- prediction crossing frame
- time-to-accident at prediction crossing

Available now from archived timing summaries:
- `pred_to_toa.mean = 39.41` frames over 39 valid DAD cases
- at 20 fps this corresponds to roughly `1.97 s` mean prediction lead time

### 2. Policy crossing analysis
Target analysis:
- actor-policy crossing vs prediction threshold crossing
- whether policy crosses earlier than score thresholding

Current status:
- not yet supported by archived outputs
- must be labeled as pending / future-strengthening evidence

### 3. Earliest useful cue analysis
For each case, identify which semantic family first becomes strongly active before the alert.

Implementation target:
- aggregate family onset distributions
- compare early-warning vs late-warning cases

### 4. False-positive / false-delay analysis
Questions:
- when the model alerts too early, which families dominate?
- when the model alerts too late, which semantic cues are missing or delayed?

### 5. Failure taxonomy
Recommended categories:
- semantically plausible but late
- weak concept onset
- wrong family dominates
- visibility/noise failure
- flat actor branch / invalid policy evidence

## Figures to prepare
1. family-onset figure
2. crossing analysis figure
3. early vs late case comparison figure
4. false-positive / failure-family figure

## Paper-ready current statement
Supported now:
- prediction timing can be aligned to event timelines and concept surges
- timing analysis is feasible and partially instantiated

Not supported yet:
- a strong claim that actor-policy timing is decisively earlier than confidence thresholding in the archived v4 DAD artifacts

## Writing instruction
Phrase the current evidence as:
- preliminary timing-faithfulness evidence for the prediction branch
- a target direction for stronger actor-policy analysis in the next round
