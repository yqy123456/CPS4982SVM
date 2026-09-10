# SVM on Iris

**CPS4982 Coursework · Controlled Experiments · RBF Kernel**

A coursework project investigating how the penalty parameter **C** affects an RBF-kernel support vector machine on the Iris dataset. Includes experiment code, the dataset, a literature review, and an experiment report.

## Experiment

- Task: binary classification of Iris versicolor and Iris virginica.
- Features: petal length and petal width.
- Model: scikit-learn `SVC(kernel="rbf", gamma="scale")`.
- Comparison: `C = 1, 50, 100, 1000` with the same split and preprocessing.
- Split: 70% training and 30% testing, with random seed 42.
- Metrics: accuracy, precision, recall, F1, confusion matrices, and support-vector counts.
- Additional analysis: C-by-gamma grid search with five-fold cross-validation.

## Getting Started

Install Python and the dependencies in a virtual environment:

```bash
python -m venv .venv
# Activate the virtual environment for your operating system.
python -m pip install -r requirements.txt
python svm_experiment.py --smoke
python svm_experiment.py
```

Smoke mode runs a reduced experiment. Full runs write tables, figures, and a Markdown summary to `output/experiment/`. Generated outputs are ignored by Git; supplied PDF reports are included. Dependencies are unpinned because no environment lockfile was supplied.

## Files

| File | Purpose |
|---|---|
| `svm_experiment.py` | Experiment and visualization implementation |
| `iris/iris.data` | Dataset read by the experiment |
| `iris/iris.names` | Original dataset information and attribution |
| `iris/bezdekIris.data` | Additional supplied Iris data file |
| `iris.csv` | Supplied copy of the dataset |
| [Literature Review on SVM.pdf](Literature%20Review%20on%20SVM.pdf) | Literature review |
| [SVM Experiment Report.pdf](SVM%20Experiment%20Report.pdf) | Experiment report |
| [SVM requirement.md](SVM%20requirement.md) | English assignment requirements |
| [deep-research-report (1).md](deep-research-report%20%281%29.md) | English research planning notes |

## What This Project Demonstrates

The main question is how stronger penalties for margin violations affect training
fit, generalization, and support-vector counts. The small two-feature Iris task
makes the decision boundaries visible. This is a controlled educational study,
not a new SVM algorithm or a large-scale benchmark.

## Expected Outputs

`output/experiment/` contains metric tables, support-vector counts, confusion
matrices, decision-boundary figures, comparison plots, and `experiment_results.md`.
The script also exports method-comparison and grid-search results. The PDF report
is the supplied result artifact; the research guide contains separately labeled
illustrative values.

## Reproduction Limits

The original environment was not locked. Reproducing the exact PDF values may
depend on library versions and the same data file and preprocessing. The
additional grid search is exploratory and should not be interpreted as an
independent external validation set.

## Data and Results

Dataset attribution is retained in `iris/iris.names`. The reports and research notes are supplied artifacts; repository preparation did not rerun or independently validate their experimental claims. Consult the runnable code and generated output when reproducing results.
