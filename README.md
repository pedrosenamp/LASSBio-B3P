# LASSBio-B3P

LASSBio-B3P predicts blood–brain barrier (BBB) permeability from molecular structures supplied as SMILES. The KNIME workflow combines three classifiers trained on B3DB: a support vector machine (SVM), XGBoost with a tree booster, and a graph neural network (GNN). For each query, it reports both class probabilities, a consensus prediction, and a descriptor-based applicability domain (AD) status.

## Screen a compound library

1. Open the LASSBio-B3P workflow in KNIME Analytics Platform. The workflow was developed with version 5.12 and uses RDKit, KNIME Python Integration, and PyTorch Geometric for the GNN.
2. Prepare a CSV file with a column named `SMILES` and one structure per row. The workflow computes the molecular representations from this column. The precomputed descriptors in the benchmark CSVs are only needed for benchmark analyses.
3. Select the CSV file in the **CSV Reader**, execute the **LASSBio-B3P** component, and open **Table View** to inspect the results.

A minimal input file:

```csv
SMILES
CC(=O)Oc1ccccc1C(=O)O
Cn1c(=O)c2c(ncn2C)n(C)c1=O
```

## Interpret the output

The output table displays five columns alongside the KNIME RowID.

| Column | Meaning |
| --- | --- |
| `SMILES` | Query structure, which KNIME can display as a molecular drawing. |
| `LASSBio-B3P P(CNS-)` | Unweighted mean of the CNS− probabilities predicted by the three classifiers. |
| `LASSBio-B3P P(CNS+)` | Unweighted mean of the CNS+ probabilities predicted by the three classifiers. |
| `LASSBio-B3P prediction` | Consensus label assigned using P(CNS+), as shown below. |
| `AD status` | Position of the query relative to the B3DB training compounds in descriptor space. |

| Consensus P(CNS+) | Output label | Interpretation |
| --- | --- | --- |
| ≤ 0.40 | `CNS-` | Predicted BBB non-permeable |
| > 0.40 and < 0.60 | `CNS+/-` | Borderline probability. The consensus abstains from binary classification. |
| ≥ 0.60 | `CNS+` | Predicted BBB permeable |

The AD is assessed separately from the consensus probability. For example, a compound with P(CNS+) = 0.449 and AD status `Inside` is close to a training compound in descriptor space but still receives the label `CNS+/-`.

The AD script compares each query with its nearest B3DB training neighbour using Euclidean distance. To set the threshold, it computes all pairwise distances between distinct training rows, retains distances strictly below their overall mean, and adds 0.5 times the population standard deviation of the retained distances to their mean. The script reports `Inside` for a nearest-neighbour distance at or below the threshold, `Outside` for a greater distance, and `Unassessable` when a required descriptor value is missing, non-numeric, or non-finite.

## Benchmark data and scripts

| Resource | Intended use |
| --- | --- |
| Final LASSBio-B3P KNIME workflow | Screen new compounds with the trained consensus. |
| BBBscore training and test CSVs | Reuse the original benchmark split of the curated BBBscore dataset (1,088 compounds), with precomputed molecular descriptors. |
| B3DB training and test CSVs | Reuse the original benchmark split of the curated B3DB dataset (7,802 compounds), with precomputed molecular descriptors. |
| GNN learner and predictor scripts | Train and evaluate the graph model used in the benchmark. The GNN constructs molecular graphs from SMILES. |
| `applicability-domain-calculation.py` | Calculate AD status in a KNIME Python Script node using a training reference table and a query table with matching descriptor columns. |

For benchmark comparisons, keep the supplied training and test partitions for each dataset. The classifiers were benchmarked separately on BBBscore and B3DB. The final consensus uses the B3DB-trained models.

## Performance and coverage

On the B3DB test partition, LASSBio-B3P assigned a binary class to **1,432 of 1,561 compounds (91.74% coverage)** and reported `CNS+/-` for the remaining 129. On the classified subset, ROC-AUC was **0.974**, balanced accuracy **0.904**, Matthews correlation coefficient **0.822**, and macro-F1 **0.910**. This same test partition was used to compare candidate consensus models and select the final combination.

## Dataset sources

- Gupta et al. *The Blood–Brain Barrier (BBB) Score*. [DOI 10.1021/acs.jmedchem.9b01220](https://doi.org/10.1021/acs.jmedchem.9b01220)
- Meng et al. B3DB, a curated database of blood–brain barrier permeability. [DOI 10.1038/s41597-021-01069-5](https://doi.org/10.1038/s41597-021-01069-5)

## License and data reuse

The MIT License applies to the original LASSBio-B3P KNIME workflow and Python scripts for GNN training, prediction, and applicability domain assessment. See the [MIT License](https://opensource.org/license/mit).

- **B3DB.** Meng et al. released the source data under CC0 1.0. The LASSBio-B3P team curated the data, generated the training and test partitions, and calculated the molecular descriptors supplied in the benchmark CSVs. See the [original B3DB license](https://github.com/theochem/B3DB/blob/main/LICENSE).
- **BBBscore.** The source data are subject to the terms of the supporting information published by Gupta et al. A specific redistribution license for the source tables has yet to be confirmed. See the [original publication and supporting information](https://doi.org/10.1021/acs.jmedchem.9b01220).
