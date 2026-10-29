# Release notes

## 2.1.0 — 29 October 2026

This release makes additional existing analyses directly runnable and updates explanatory materials. It adds no new model runs and leaves the original Source Data, 32 parsed-output CSVs, main-figure PDFs and original provenance inventories unchanged.

- **Controlled-scaling worked example:** a prompt specification, 100 controlled settings, released model responses, point-estimate recomputation and a reader guide.
- **Single-cue streetscape edits:** released scores and contrasts for 12 scenes, an offline recomputation of five prespecified effects and bootstrap intervals, and integrity tests.
- **Supplementary figures:** clearer S2, S3, S7 and S12 diagnostic labels, plus the four updated publication-layout reference PDFs. Source-table values are unchanged.
- **Revised workbooks:** byte-exact copies of the revised Supplementary Data 1–2 are under `data/publication/20261029/`. Original workbooks remain at `data/publication/`. The revised copies clarify the task-specific temperature and correct a ResNet-50 checkpoint label; prompt text, rendered prompts and empirical records are unchanged. Their embedded article title reflects the workbook revision date, while the repository citation uses the current title.
- **Documentation and citation:** the repository describes one central question about the use of generated urban data in controlled virtual experiments, alongside capability/bias evaluation and the common experimental procedure.

The original reproducibility snapshot remains available at commit `6b710f8e52ee0d68d759104610b842ff75913c03`. Additional lineage is recorded in three dated JSON files under `provenance/`; the exhaustive public-file manifests cover both historical and added artifacts.

No manuscript, correspondence, raw model calls, hidden-state tensors, weights, street-view images or restricted empirical record-level inputs are added by this release.
