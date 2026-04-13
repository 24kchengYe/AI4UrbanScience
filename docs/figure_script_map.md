# Figure ↔ Script map

Each row lists a figure (or a category of analysis) from the manuscript,
the **new** entry point that produces it, and the **legacy** research-time
scripts that the new code replaces.

## Main figures

### Fig. 1 — Urban scaling law

- **Produced by:** `experiments/01_urban_scaling_law/figure1.py`
- **Upstream data:** `experiments/01_urban_scaling_law/generate.py` + `analyze.py`

Legacy scripts that now live inside this experiment:

```
01test-openai(Zipf-Scaling).py                                → generate.py --model gpt-4o
01test-openai-智匠AI(Zipf-Scaling).py                         → generate.py --model gpt-4o-mindcraft
01test-claude(Zipf-Scaling).py                                → generate.py --model claude-3.5-sonnet
01test-deepseek(Zipf-Scaling).py                              → generate.py --model deepseek-v3
01test-chatglm(Zipf-Scaling).py                               → generate.py --model chatglm-4
01test-gemini-2.0-flash-exp-智匠AI(Zipf-Scaling).py          → generate.py --model gemini-2.0-flash
01test-qwen-plus-latest-智匠AI(Zipf-Scaling).py              → generate.py --model qwen-plus
01test-Doubao-pro-128k-智匠AI(Zipf-Scaling).py               → generate.py --model doubao-pro
01test-hunyuan-large-longcontext-智匠AI(Zipf-Scaling).py     → generate.py --model hunyuan-large
01test-o1-preview-智匠AI(Zipf-Scaling).py                    → generate.py --model o1-preview

01test-sensitivity-独立采样Independent Sampling(Zipf-Scaling).py  → sensitivity.py --mode independent
01test-sensitivity-独立采样可视化(Zipf-Scaling).py              → sensitivity.py --mode independent (inline plot)
01test-sensitivity-独立采样可视化拟合线(Zipf-Scaling).py         → sensitivity.py (fit-line variant)
01test-sensitivity-联合采样Joint Sampling(Zipf-Scaling).py       → sensitivity.py --mode joint
01test-sensitivity-联合采样可视化(Zipf-Scaling).py              → sensitivity.py --mode joint
01test-sensitivity-联合采样可视化拟合线(Zipf-Scaling).py         → sensitivity.py --mode joint
01test-sensitivity-联合采样-独立采样对比(Zipf-Scaling).py         → sensitivity.py (side-by-side comparison)

02AIUS-TheoryValid(Zipf-Scaling).py                           → analyze.py (single-replicate mode)
03AIUS-TheoryValidMultitime(Zipf-Scaling).py                  → analyze.py (aggregated mode)
03AIUS-TheoryValidMultitime趋势线和置信区间(Zipf-Scaling).py     → figure1.py
03AIUS-TheoryValidMultitime-naturecities改稿用-图1.py          → figure1.py
```

### Fig. 2 — Urban vitality

- **Produced by:** `experiments/03_urban_vitality/figure2.py`
- **Upstream:** `generate_attributes.py` → `score_livability.py` → `analyze.py`

```
04test-openai(vitality).py                                    → generate_attributes.py --model gpt-4o
04test-claude(vitality).py                                    → generate_attributes.py --model claude-3.5-sonnet
04test-deepseek(vitality).py                                  → generate_attributes.py --model deepseek-v3
04test-chatglm(vitality).py                                   → generate_attributes.py --model chatglm-4
04test-gemini-*(vitality).py                                  → generate_attributes.py --model gemini-2.0-flash

04test-openai(vitality access).py                             → score_livability.py --model gpt-4o
04test-claude(vitality access).py                             → score_livability.py --model claude-3.5-sonnet
04test-deepseek(vitality access).py                           → score_livability.py --model deepseek-v3
04test-chatglm(vitality access).py                            → score_livability.py --model chatglm-4
04test-gemini-*(vitality access).py                           → score_livability.py --model gemini-2.0-flash

05AIUS-TheoryValid(vitality).py                               → analyze.py
06AIUS-TheoryValidMultitime(vitality).py                      → analyze.py
06AIUS-TheoryValidMultitime(vitality)虚线和黑长直.py           → figure2.py
06AIUS-TheoryValidMultitime(vitality)虚线和黑长直-naturecities改稿用-图2.py  → figure2.py
```

### Fig. 3a-b — Human-AI perception alignment

- **Produced by:** `experiments/04_perception_alignment/figure3.py`
- **Upstream:** `pairwise_eval.py` → `kappa_scores.py`

```
18感知评分-验证AI"感知能力" (pulse随机抽样六维度感知).py         → pairwise_eval.py
18感知评分-验证AI"感知能力" (...)-可视化-kappa系数打分.py        → kappa_scores.py + figure3.py
18感知评分和trueskill.py                                       → trueskill_scoring.py
18感知评分和trueskill-Q-score.py                               → trueskill_scoring.py (Q-score variant collapsed)
20统计分析-感知部分-naturecities改稿用-图3.py                  → figure3.py
```

### Fig. 3c / Fig. 6 — Synthetic intervention

- **Produced by:** `experiments/05_synthetic_intervention/segment_and_analyze.py --step delta`

```
17视觉感知实验-0纯图片生成(baseline).py                       → generate_images.py --mode baseline
17视觉感知实验-0纯图片生成(图改图 干预变体).py                 → generate_images.py --mode variants
17色调标准化.py                                               → inline hook in generate_images.py
17视觉感知实验-1CLIP验证多样性.py                             → clip_diversity.py
19分割图片-0纯图片生成(baseline).py                           → segment_and_analyze.py --step segment --mode baseline
19分割图片-0纯图片生成(图改图 干预变体).py                    → segment_and_analyze.py --step segment --mode variants
20统计分析-0纯图片生成(baseline).py                           → segment_and_analyze.py --step regression --mode baseline
20统计分析-0纯图片生成(图改图 干预变体).py                    → segment_and_analyze.py --step delta
```

### Fig. 4 — Distribution divergence

- **Produced by:** `experiments/06_distribution_divergence/figure4.py`
- **Upstream:** `extract_real_data.py` → `compute_divergence.py`

```
10RealData.py                                                 → extract_real_data.py
10RealData2.py                                                → extract_real_data.py (consolidated)
11datadifference.py                                           → compute_divergence.py + figure4.py
11datadifference2.py                                          → compute_divergence.py (variant collapsed)
11数据降维可视化.py                                            → (ad-hoc dim-reduction, not essential to Fig. 4)
12realdata-TheoryValidMultitime(Zipf-Scaling).py              → analyze.py in Exp 01 (real-data arm)
12真实 生成和提示生成的数据分布可视化.py                       → figure4.py
11datadifference-naturecities改稿用-图4.py                    → figure4.py
```

## Supplementary / downstream analyses

### Supplementary — Distance decay (meso-scale)

- **Produced by:** `experiments/02_distance_decay/visualize.py`

```
07test-*(distance).py                                         → generate.py --model <key>
08AIUS-TheoryValid(distance).py                               → analyze.py
09AIUS-TheoryValidMultitime(distance).py                      → analyze.py
09AIUS-visualization between theory and G(distance)*.py       → visualize.py
09AIUS-visualization of G from Jiaolimin(distance).py         → visualize.py
```

### Supplementary — OT calibration

- **Produced by:** `experiments/07_calibration_ot/calibration.py`

```
13生成数据分布矫正.py                                          → calibration.py --methods quantile_match
14消融实验 llm+OT.py                                           → calibration.py --methods raw ot_marginal ot_joint
15标度系数验证 llm+OT.py                                       → calibration.py (Bettencourt comparison in output)
16不同方法的数据规律拟合可视化.py                               → calibration.py (tabular output)
16真实数据校准SCforAI4US.py                                    → calibration.py --real-csv ...
```

### Supplementary — SynAlign prompt engineering

- **Produced by:** `experiments/08_prompt_engineering/pipeline_symbolic.py`
  and `pipeline_perceptual.py`

```
16参考SynAlign01-urban_stage1_discovery.py                   → pipeline_symbolic.stage1_discovery()
16参考SynAlign02-urban_stage2_synthesis.py                   → pipeline_symbolic.stage2_synthesis()
16参考SynAlign03-urban_stage3_alignment.py                   → pipeline_symbolic.stage3_alignment()
16参考SynAlign04-urban_stage4_evaluation.py                  → pipeline_symbolic.stage4_evaluation()
16参考SynAlign05-urban_stage1_discovery.py (perceptual)      → pipeline_perceptual.stage1_blueprint()
16参考SynAlign06-urban_stage2_synthesis.py                   → pipeline_perceptual.stage2_pairwise()
16参考SynAlign07-urban_stage3_alignment.py                   → pipeline_perceptual.stage3_kappa()
16参考SynAlign08-urban_stage4_evaluation.py                  → pipeline_perceptual.main()
16参考SynAlign06-urban_stage2_synthesis-without synalign.py  → pipeline_symbolic.py (without-blueprint arm, automatic)
16参考SynAlign06-...人类知识注入提示增强版本.py                 → pipeline_symbolic.py (default arm)
16参考SynAlign00-synthesis-人类知识注入提示增强版本 symbolic.py → pipeline_symbolic.stage2_synthesis()
16参考SynAlign00-...symbolic-并行1.py                        → pipeline_symbolic.py (parallelism is now a deployment concern)
16参考SynAlign00-...symbolic-并行2.py                        → pipeline_symbolic.py
16参考SynAlign00-...symbolic-并行3.py                        → pipeline_symbolic.py
16参考SynAlign00-...perceptual.py                            → pipeline_perceptual.py
16参考SynAlign00-...可视化.py                                 → (downstream plotting, not a figure in the current Fig. 1-4 set)
16参考SynAlign00-...可视化-meihua.py                         → (polished variant of the above)
16参考SynAlign00-...可视化-naturecities改稿用-图SI.py         → (supplementary figure; to be revived if needed)
```
