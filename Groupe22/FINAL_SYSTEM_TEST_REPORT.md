# Final System Test Report

## Test Configuration

- Test queries: 100
- Training images: 10000
- FAISS candidates: 20 / 50 according to CRAG
- Final captions: 5
- MMR lambda: 0.9
- Correct threshold: 0.3402
- Ambiguous threshold: 0.317

## CRAG Classification

- CORRECT: 26
- AMBIGUOUS: 41
- INCORRECT: 33

## Evaluation Metrics

- Mean BLEU: 0.0960
- Mean ROUGE-L: 0.3931
- Mean CLIP relevance: 0.3145

## Methodological Limitations

BLEU and ROUGE-L measure textual overlap with reference captions.
They do not independently establish semantic correctness.

CRAG classifications use thresholds calibrated on Validation.
They are operational decisions, not verified accuracy labels.

The Test set was not used to adjust thresholds or MMR lambda.
