# Graph-structural-encoder-v2: baseline vs all-arm metrics

## validation

| arm | n | n_localized | top1 | top3 | mrr | actionable | abstention | candidate_size | coverage |
|---|---|---|---|---|---|---|---|---|---|
| A_CONTROL | 300 | 300 | 0.693 | 0.873 | 0.796 | 0.980 | 0.020 | 2.487 | 0.897 |
| B_CENTRALITY | 300 | 300 | 0.703 | 0.873 | 0.803 | 0.980 | 0.020 | 2.523 | 0.900 |
| C_OBSERVABILITY | 300 | 300 | 0.693 | 0.877 | 0.796 | 0.983 | 0.017 | 2.543 | 0.910 |
| D_STRUCTURAL_AGG | 300 | 300 | 0.697 | 0.877 | 0.799 | 0.983 | 0.017 | 2.590 | 0.913 |
| D_CAPACITY_CONTROL | 300 | 300 | 0.693 | 0.877 | 0.797 | 0.983 | 0.017 | 2.587 | 0.913 |
| E_COMBINED | 300 | 300 | 0.697 | 0.870 | 0.798 | 0.983 | 0.017 | 2.547 | 0.913 |

## development_holdout

| arm | n | n_localized | top1 | top3 | mrr | actionable | abstention | candidate_size | coverage |
|---|---|---|---|---|---|---|---|---|---|
| A_CONTROL | 300 | 300 | 0.690 | 0.880 | 0.795 | 0.973 | 0.027 | 2.543 | 0.903 |
| B_CENTRALITY | 300 | 300 | 0.700 | 0.880 | 0.801 | 0.970 | 0.030 | 2.543 | 0.907 |
| C_OBSERVABILITY | 300 | 300 | 0.693 | 0.873 | 0.798 | 0.970 | 0.030 | 2.557 | 0.907 |
| D_STRUCTURAL_AGG | 300 | 300 | 0.700 | 0.877 | 0.801 | 0.970 | 0.030 | 2.603 | 0.910 |
| D_CAPACITY_CONTROL | 300 | 300 | 0.700 | 0.880 | 0.802 | 0.970 | 0.030 | 2.607 | 0.910 |
| E_COMBINED | 300 | 300 | 0.700 | 0.883 | 0.802 | 0.970 | 0.030 | 2.567 | 0.907 |

## ood-UNSEEN_TOPOLOGY

| arm | n | n_localized | top1 | top3 | mrr | actionable | abstention | candidate_size | coverage |
|---|---|---|---|---|---|---|---|---|---|
| A_CONTROL | 400 | 280 | 0.375 | 0.757 | 0.586 | 0.000 | 1.000 | - | - |
| B_CENTRALITY | 400 | 280 | 0.368 | 0.729 | 0.572 | 0.000 | 1.000 | - | - |
| C_OBSERVABILITY | 400 | 280 | 0.354 | 0.704 | 0.562 | 0.000 | 1.000 | - | - |
| D_STRUCTURAL_AGG | 400 | 280 | 0.361 | 0.725 | 0.569 | 0.000 | 1.000 | - | - |
| D_CAPACITY_CONTROL | 400 | 280 | 0.361 | 0.729 | 0.570 | 0.000 | 1.000 | - | - |
| E_COMBINED | 400 | 280 | 0.371 | 0.739 | 0.583 | 0.000 | 1.000 | - | - |

## Parameter counts (total)

| arm | total | encoders (graph_encoder-inclusive) |
|---|---|---|
| A_CONTROL | 4044113 | 941376 |
| B_CENTRALITY | 4045265 | 942528 |
| C_OBSERVABILITY | 4045457 | 942720 |
| D_STRUCTURAL_AGG | 4118225 | 1015488 |
| D_CAPACITY_CONTROL | 4118225 | 1015488 |
| E_COMBINED | 4120721 | 1017984 |
