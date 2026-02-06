# FRUGALITY SCORE: A FUZZY LOGIC BASED FRUGALITY EVALUATION SCORE

This repository contains the code for the python package frugality-score, associated with the article [*FRUGALITY SCORE: A FUZZY LOGIC BASED FRUGALITY EVALUATION SCORE*](doc/) to evaluate the frugality of a machine learning method based on its energy consumption and performance.

> The popularity of efficiency as an optimization between performance and energy consumption rose within research in machine learning, following concerns in the ecological footprint of artificial intelligence (AI). Frugality emerged as a term that questions the quality of these notions. What is a good performance or a good energy consumption? These questions remain subjective, task-specific, and difficult to quantify. In this work, we propose a new frugality scoring method based upon fuzzy logic, that encapsulate these aspects with applications to common situations in machine learning. This score allows both for an absolute evaluation of the frugality of a method, and a relative analysis for the user’s own case study.

#### Contents

- [Structure du dépôt](#structure)
- [Installation](#install)
  - [Dependencies](#dependencies)
- [Usage](#use)
- [References](#references)


## Repository structure <a name="structure"></a>

```bash
.
├── 
```

## Installation <a name="install"></a>

### Dependencies <a name="dependencies"></a>

python 3.11.8, codecarbon 2.3.4, numpy 1.26.4, pandas 2.2.1, scikit-learn 1.4.1
```bash
conda env create -f environment.yml
conda activate frugal-score
```

## Usage <a name="use"></a>

| File | Associated command | Description |
| ---- | ------------------ | ----------- |
| [change-detection.py](experiments/conso_change/change-detection.py)  | `python change-detection.py --storage_path [PATH_TO_FOLDER_TO_STORE_RESULTS] --image [PATH_TO_FOLDER_WITH_IMAGES] --window [WINDOW_SIZE] --cores [NUMBER_OF_CORES_USED] --number_run [NUMBER_OF_RUNS] --robust [ROBUSTNESS ID]` | Runs change detection algorithms on UAVSAR data |
| [clustering_blob.py](experiments/conso_clustering/clustering_blob.py)  | `python clustering_blob.py --storage_path [PATH_TO_FOLDER_TO_STORE_RESULTS] --data_seed [SEED] --random_seed [SEED] --n_clusters [NUMBER_OF_CLUSTERS] --model [CLUSTERING_METHOD] --repeat [NUMBER_OF_MODEL_RUNS] --number_run/-n [NUMBER_OF_RUNS]` | Runs clustering algorithms on toy data |

## References <a name="references"></a>