# Regularized Linear and Bayesian Models Improve Stacked Generalization for Species Distribution Modeling 

Companion code and data for the manuscript:

> Payopay, J. P. M., Lumbres, R. I. C., Doyog, N. D., Talkasen, L. J., Polon, K. K. A.,
> Lingaling, M. N. T., & Carates, N. N. *Regularized Linear and Bayesian Models
> Improve Stacked Generalization for Species Distribution Modeling.*
> Submitted to *Ecological Modelling*.

## Overview

This repository contains the full species distribution modeling (SDM) pipeline used to
compare **nine stacking meta-learners**, arranged along a ladder of increasing model
capacity (regularized linear → Bayesian-linear → flexible nonlinear), against two
traditional ensembles (unweighted and TSS-weighted mean). All meta-learners are fitted on
**identical out-of-fold (OOF) predictions** from six diverse base learners, so any
performance difference is attributable to the combiner alone. The testbed species is
*Bambusa oldhamii* in Benguet, Philippines.

Key methodological features:

- Six base learners (Extra Trees, logistic regression, multilayer perceptron, naive Bayes,
  SVM, XGBoost) feeding a common OOF meta-feature space.
- Nine meta-learners spanning the capacity ladder, plus mean and TSS-weighted ensembles.
- **Spatial block cross-validation** to control spatial autocorrelation.
- **Twenty pseudo-absence replicates** so PA stochasticity propagates into all estimates.
- Ensemble recursive feature elimination (RFECV) for predictor selection.
- A Bayesian Gaussian-process classifier (GPC) meta-learner with an epistemic uncertainty map.
- Statistical comparison via Friedman test with Nemenyi post-hoc, plus Wilcoxon and DeLong tests.

The headline result is that performance is ordered by meta-learner *capacity* rather than
algorithm family: a Bayesian-logistic combiner gives the best balance of discrimination and
stability, while capacity beyond a regularized linear combiner adds variance without accuracy.

## Repository structure

```
.
├── bamboo_sdm.py          # End-to-end pipeline (data prep → models → maps → stats)
├── requirements.txt       # pip dependencies (pinned)
├── environment.yml        # conda environment (pinned)
├── data/                  # Model input variables (see "Data" below)
├── outputs/               # Generated metrics, statistical tests, figures, and maps
│   ├── figures/           # Publication figures
│   ├── maps/              # Suitability rasters (continuous, binary, SD/uncertainty)
│   └── models/            # Serialized fitted models
├── LICENSE                # MIT (code)
└── README.md
```

## Installation

Python 3.13 is required. Using conda (recommended, because of the geospatial stack):

```bash
conda env create -f environment.yml
conda activate bamboo-sdm
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Usage

The pipeline resolves its working directory from the `BAMBOO_SDM_DIR` environment variable,
falling back to the script's own directory:

```bash
# optional: point the pipeline at a specific project root
export BAMBOO_SDM_DIR=/path/to/this/repo      # Windows (PowerShell): $env:BAMBOO_SDM_DIR="C:\path\to\repo"

python bamboo_sdm.py
```

XGBoost uses CUDA when available and falls back to CPU; override with the `XGB_DEVICE`
environment variable (`cuda` or `cpu`). All progress is mirrored to `outputs/pipeline.log`.

## Data

The pipeline takes two kinds of model input:

- **Environmental predictors.** 33 candidate raster layers (bioclimatic, atmospheric,
  spectral, topographic/edaphic) from CHELSA, Sentinel-2, ASTER GDEM, and SoilGrids;
  the retained set after VIF filtering is listed in the manuscript (Table 1). Standard
  public layers should be obtained from their original sources (cited in the manuscript);
  derived layers (e.g., Sentinel-2 spectral indices) are provided here.
- **Occurrence records.** Field-collected *B. oldhamii* presences (survey 14 Oct 2024 –
  23 May 2025). Because exact localities can fall on private landholdings, coordinates may
  be provided at coarsened spatial precision; full-precision data are available from the
  corresponding author on reasonable request.


## Contact

John Paul M. Payopay — Center for Geoinformatics, Benguet State University —
<jp.payopay@bsu.edu.ph>
