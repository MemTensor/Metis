# Paper protocol map

| Scope | Config | Top-level runner | Full cells |
|---|---|---|---:|
| Main MemOP + MemQA (Tables 5, 6, 11) | `table_5_6_main.json` | `eval.experiments.main_tables.matrix` | 77 |
| Metis-4B ablation (Table 7) | `eval/experiments/ablation/configs/ablation_matrix.json` | `eval.experiments.ablation.run` | 28 |
| OOD (Table 8, ATM + MemDaily) | `table_8_ood.json` | `eval.experiments.ood.matrix` | 14 |
| Metis-4B LowRankMemory (Table 10) | `table_10_low_rank.json` | `eval.experiments.low_rank.matrix` | 28 |

These are evaluation protocols. Training and unrelated paper experiments are
not part of this directory. Every declarative matrix uses
`eval/configs/assets.json`; the ablation engine carries stricter checkpoint
metadata in its own matrix. All workflows share the normalized layout under
`eval/data/`. `reported_scores.json` is the comparison target for paper-table
aggregates.
