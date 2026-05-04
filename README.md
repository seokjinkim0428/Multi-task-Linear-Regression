# Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity, Robustness and Safety

This repository contains code for the paper:

Seok-Jin Kim  
*Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity,
Robustness and Safety*

The goal is to estimate task-specific linear models in a multi-task system with
related tasks and arbitrary outlier tasks, while avoiding eigenvalue lower-bound
assumptions on each task's empirical second moment matrix.

## Problem Setting

Given `m` tasks with observations

```math
\mathcal{D}_j = \{(x_{ji}, y_{ji})\}_{i=1}^{n_j},
\qquad j=1,\ldots,m,
```

each task follows a linear model

```math
y_{ji} = x_{ji}^\top \theta_j^\star + \varepsilon_{ji}.
```

Most tasks are assumed to be close to a shared centroid in Euclidean norm, but
an unknown fraction of tasks may be arbitrary outliers. Unlike
standard robust multi-task regression guarantees, this work does not require a
uniform lower bound on the smallest eigenvalue of each empirical second moment
matrix

```math
\Sigma_j = \frac{1}{n_j}\sum_{i=1}^{n_j} x_{ji}x_{ji}^\top.
```

The main performance metric is prediction risk, or MSE in the task-specific
`Sigma_j` geometry, which remains meaningful under ill-conditioned or
singular designs.

## Method Summary (MTLR)

The proposed estimator uses matrix-weighted multi-task regularization:

1. Compute the empirical second moment matrix for each task.
2. Fit task-specific parameters and a shared centroid by solving a joint convex
   objective.
3. Select regularization multipliers by cross-validation.
4. Compare against data pooling (DP), independent-task learning (ITL/STL), and
   the ARMUL robust multi-task baseline of Duan and Wang (2023).

```math
\min_{\theta_1,\ldots,\theta_m,\beta}
\sum_{j=1}^m w_j
\left(
f_j(\theta_j)
+ \lambda_j \|\theta_j - \beta\|_{\Sigma_j}
\right).
```

Here, `f_j` denotes the empirical loss function for task `j`; for example, it
is the squared loss in linear regression and the negative log-likelihood in a
GLM such as logistic regression. Thus the same matrix-weighted regularization
template also covers generalized linear model variants.

In this repository, `ARMUL` refers to the adaptive robust multi-task learning
method proposed in:

> Yaqi Duan and Kaizheng Wang. "Adaptive and robust multi-task learning."
> *The Annals of Statistics*, 51(5):2015-2039, 2023.
> [doi:10.1214/23-AOS2319](https://doi.org/10.1214/23-AOS2319)

The matrix-weighted penalty measures disagreement in prediction space rather
than raw Euclidean parameter space. This is what allows transfer under decaying
or singular spectra while retaining a safety guarantee comparable to
independent-task learning when transfer is unhelpful.

## Repository Layout

* `MTLR_Synthetic.ipynb`  
  Notebook companion for the synthetic experiments in the paper.
* `MTLR_Real-data.ipynb`  
  Notebook companion for the real-data HAR experiment, including the HAR
  covariate balancedness diagnostic.
* `synthetic_sweeps.py`  
  Synthetic experiment engine for sweeps over task radius, outlier fraction,
  eigendecay, and balancedness.
* `real_data_har.py`  
  Clean script version of the HAR real-data experiment.
* `MTL.py`  
  Core multi-task optimization routines for vanilla, clustered, and low-rank
  multi-task learning.
* `ARMUL.py`  
  ARMUL-style robust multi-task estimators and baseline wrappers, following
  Duan and Wang (2023).
* `preprocessing.py`  
  HAR loading, task splitting, cross-validation splitting, and preprocessing.
* `path_setup.py`  
  Local/Colab path resolver used by the notebooks and scripts.
* `Images/`  
  Paper figures and CSV summaries produced by synthetic sweeps.
* `UCI_HAR_Dataset/`  
  UCI HAR dataset files used by the real-data experiment.

## Quick Start

### 1. Environment

Recommended: Python 3.10+

```bash
pip install -r requirements.txt
```

Equivalent minimal dependencies:

```bash
pip install numpy scipy pandas scikit-learn matplotlib ipython jupyter
```

### 2. Synthetic Experiments

Open and run:

* `MTLR_Synthetic.ipynb`

The notebook uses `synthetic_sweeps.py` to run the paper-aligned synthetic
comparisons among:

* `OURS`: matrix-weighted MTLR estimator.
* `ARMUL`: Euclidean adaptive robust multi-task baseline from Duan and Wang
  (2023).
* `ITL`: independent-task learning baseline.
* `DP`: data-pooling baseline.

The main sweeps vary:

* `delta`: inlier task radius.
* `epsilon`: outlier fraction.
* `decay_alpha`: covariance eigendecay.
* `bar_b_target`: population balancedness target.

Figures and CSV summaries are saved to:

* `Images/`

### 3. Real-data HAR Experiment

Open and run:

* `MTLR_Real-data.ipynb`

or run the script version:

```bash
python real_data_har.py
```

The HAR experiment treats subjects as tasks, uses raw 561-dimensional features
with global Min-Max scaling, and evaluates binary classification error for the
standing-vs-rest task. The notebook also reports a single `B_HAR` diagnostic
for subject-level covariate balancedness. It is computed by comparing each
subject covariance to `Sigma_all`, the empirical second moment over all HAR
covariate samples.

The script uses:

* 30 random train/test splits.
* 5-fold cross-validation for ARMUL and OURS.
* `Q_GRID = [0.05, 0.10, ..., 0.50]` for regularization selection.

## Google Colab

* Upload this repository folder to `My Drive/Colab Notebooks/MTLR_Codes`.
* Open either notebook from that folder.
* The first code cell mounts Google Drive, locates `MTLR_Codes`, adds it to
  `sys.path`, and sets the working directory automatically.
* If Colab asks for Drive authorization, approve it once.
* Local Jupyter runs skip the Drive mount and use the current repository layout.
* For the HAR notebooks, keep the extracted dataset at
  `MTLR_Codes/UCI_HAR_Dataset/`.
* The helper also recognizes `MTLR_Codes/HAR Data/UCI HAR Dataset/` and can
  extract `MTLR_Codes/HAR Data/UCI HAR Dataset.zip` if present.

## Experimental Config Notes

### Synthetic setup

* Default synthetic sweeps use `n=100`, `m=30`, `d=30`.
* Default training uses 30 Monte Carlo repetitions and 5-fold cross-validation.
* The default compared methods are `DP`, `ITL`, `ARMUL`, and `OURS`.
* Synthetic errors are reported as population MSE using the task-specific
  covariance geometry.
* Main synthetic outputs are written as figure panels and CSV summaries in
  `Images/`.

### HAR setup

* The UCI HAR subjects are treated as separate tasks.
* The real-data script uses raw 561-dimensional HAR features.
* Positive label is HAR activity 5 (`STANDING`) by default.
* Each split holds out 20% of each subject/task for testing.
* Reported results are average held-out classification errors across tasks.
* The real-data notebook also reports `B_HAR`, a plug-in balancedness estimate
  using the all-sample HAR covariate second moment as the aggregate covariance.

## Reference

* Duan, Y. and Wang, K. (2023). Adaptive and robust multi-task learning.
  *The Annals of Statistics*, 51(5):2015-2039.
  [doi:10.1214/23-AOS2319](https://doi.org/10.1214/23-AOS2319)
