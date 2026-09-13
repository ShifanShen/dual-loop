# SAL Hyperparameter Sensitivity

This study uses only public feedback tests and does not query held-out private tests.
The pipeline is SAL plus one contract-conditioned code candidate, without IRL or contract search.

| Config | r | Weights (cov, faith, prec) | Public pass | SAL activation | Accepted revisions | Mean SAS delta | Mean calls |
|---|---:|---|---:|---:|---:|---:|---:|
| threshold_r80_paper | 80 | (0.400, 0.400, 0.200) | 0.400 (20/50) | 1.000 | 1 | 1.56 | 5.18 |
| threshold_r85_paper | 85 | (0.400, 0.400, 0.200) | 0.400 (20/50) | 1.000 | 14 | 2.16 | 6.90 |
| threshold_r90_paper | 90 | (0.400, 0.400, 0.200) | 0.400 (20/50) | 1.000 | 15 | 2.16 | 7.24 |
| threshold_r95_paper | 95 | (0.400, 0.400, 0.200) | 0.400 (20/50) | 1.000 | 15 | 2.16 | 7.24 |
| weights_r90_uniform | 90 | (0.333, 0.333, 0.333) | 0.420 (21/50) | 1.000 | 13 | 2.34 | 7.18 |
| weights_r90_coverage | 90 | (0.600, 0.200, 0.200) | 0.400 (20/50) | 1.000 | 10 | 1.98 | 7.26 |
| weights_r90_faithfulness | 90 | (0.200, 0.600, 0.200) | 0.400 (20/50) | 1.000 | 7 | 1.88 | 7.14 |
| weights_r90_precision | 90 | (0.200, 0.200, 0.600) | 0.400 (20/50) | 1.000 | 6 | 1.58 | 6.94 |

Interpretation rule: treat differences of one problem or less as descriptive ties, then prefer the setting with fewer mean calls. Use the held-out private partition only after the configuration remains frozen.
