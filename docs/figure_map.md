# Current figure map

The exact submitted main-figure PDFs are under `reference_outputs/main/`. Figure 1 is author-supplied conceptual artwork. Figures 2–5 and all supplementary figures have released quantitative source tables and a public rerender command.

| Figure | Current title | Released input | Command |
|---|---|---|---|
| 1 | AI4US study design across urban scales and data modalities | Frozen conceptual PDF | Not code-generated |
| 2 | Generated urban relationships and an external human-judgement reference | `Fig02_data` | `python scripts/reproduce_figures.py --figures 2 --output-dir ../ai4us-outputs/fig2` |
| 3 | Variation in generated data relative to empirical observations | `Fig03_data` | `python scripts/reproduce_figures.py --figures 3 --output-dir ../ai4us-outputs/fig3` |
| 4 | GenAI model responses to controlled changes in urban conditions | `Fig04_data` | `python scripts/reproduce_figures.py --figures 4 --output-dir ../ai4us-outputs/fig4` |
| 5 | GenAI open-model response patterns, linear readouts and activation edits | `Fig05_data` | `python scripts/reproduce_figures.py --figures 5 --output-dir ../ai4us-outputs/fig5` |
| S1 | Case definitions, empirical references and conditional inputs | `FigS01_data` | `--figures S01` |
| S2 | Urban Scaling prompt components, generation size and supplementary inputs | `FigS02_data` | `--figures S02` |
| S3 | Urban Scaling distributions, empirical-unit sensitivity and supplementary interventions | `FigS03_data` | `--figures S03` |
| S4 | Distance Decay parameters and local profile variation | `FigS04_data` | `--figures S04` |
| S5 | Distance Decay model and distance-band sensitivity | `FigS05_data` | `--figures S05` |
| S6 | Jacobs-informed Vitality synthesis and empirical validation | `FigS06_data` | `--figures S06` |
| S7 | Vitality intervention-factor and generation-size sensitivity | `FigS07_data` | `--figures S07` |
| S8 | Place Pulse human agreement, image-prompt composition and empirical split-half references | `FigS08_data` | `--figures S08` |
| S9 | Perception representation and intervention sensitivity | `FigS09_data` | `--figures S09` |
| S10 | GPT-4o and open-model behavioural correspondence | `FigS10_data` | `--figures S10` |
| S11 | Numerical relation generation, hidden-state readout and state edits during table generation | `FigS11_data` | `--figures S11` |
| S12 | Perception image-state replacement and localization | `FigS12_data` | `--figures S12` |

For a supplementary command, prepend `python scripts/reproduce_figures.py` and append a new `--output-dir` outside the repository, for example `../ai4us-outputs/figS02`. Use `--all` to render Figures 2–5 and S1–S12 together.

## Supplementary reader-label update — 29 October 2026

The public source-table renderer retains its diagnostic, all-rows layout. Its
labels now distinguish the following comparisons; estimates, confidence
intervals, row order, numerical artist positions and colours are unchanged.

- S2: city-system organization, numerical range and heterogeneity; requested
  generic cities; Road → GDP and GDP → Road responses.
- S3: administrative-city references in panels a–b; physical-city area thresholds
  in km²; the changed input and corresponding output in panel d.
- S7b: Functional Mix controls, with 16 of 20 settings valid in both requests.
- S12b: the first two rows are effect reductions; the last three subtract the
  absolute matched-control effect from the high-change reduction. The negative
  random-text contrast is retained.

The publication-layout reference PDFs for these four figures are in
`reference_outputs/supplementary/`, with their label-update date recorded in
that directory's README. They are reference layouts, not pixel-identical
outputs of the public diagnostic renderer. All released source-table hashes
and reference PDF identities are recorded in
`provenance/supplementary_label_update_20261029.json`. The existing main-figure
reference PDFs remain unchanged.
