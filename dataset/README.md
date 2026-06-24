# Dataset Lifecycle

This directory stores reviewable image samples for Pokemon Champions preview recognition.

## Stages

- `pending/`: Newly generated crops and manifests that still need review.
- `reviewed/`: Samples that were inspected but are not ready to publish.
- `approved/`: Reviewed samples that are trusted for evaluation and publishing.
- `rejected/`: Bad crops, wrong labels, or otherwise unusable samples kept for audit.
- `source_official_sprite/`: Raw imported official sprite assets used to synthesize samples.

Each batch should keep a `manifest.json` with `samples[].crop_path` and a labeled species field such as
`predicted_species_id`, `actual_species_id`, or `approved_species_id`. The template evaluator can scan all lifecycle
stages, but production-quality metrics should prefer `approved/`.

## Commands

```powershell
python -m champions_assistant init-dataset-layout --dataset dataset
python -m champions_assistant evaluate-templates --stage approved
python -m champions_assistant evaluate-templates --stage pending --include-unapproved
```
