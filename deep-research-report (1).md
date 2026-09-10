# SVM Coursework Research Guide

English adaptation of the supplied research planning notes. This document records the original study plan and illustrative results; it is not a new literature search or a new experimental run. The original export contained internal citation markers without usable source URLs. Those markers have been removed, and the bibliographic references are retained below for verification.

## Scope and Completion Criteria

The coursework combines an English literature review with a controlled experiment that uses the same dataset and kernel while changing only the penalty parameter C.

The literature review should cover basic ideas, mathematical formulations, advantages and disadvantages, current research directions, and personal opinions. Relevant concepts include maximum-margin separating hyperplanes, soft margins, kernels, parameter selection, multiclass decomposition, probability estimates, and computational complexity.

For the experiment with C = 1, 50, 100, 1000, keep the dataset, train/test split, preprocessing, kernel, and evaluation metrics fixed. Both C and gamma affect an RBF SVM, so the main comparison must not change gamma independently. Smaller C values tolerate more margin violations; larger values penalize violations more strongly. The proposed implementation uses scikit-learn SVC with an RBF kernel and gamma='scale'. With the same training data and preprocessing, this keeps the kernel setting consistent across the four C values.

## Core Papers and Supporting Lectures

A useful reading plan combines three classic sources, an implementation reference, and a recent research direction.

| Source | Role in the review |
|---|---|
| Cortes and Vapnik, Support-Vector Networks (1995) | Origins of maximum-margin learning, soft margins, and kernel methods |
| Burges, A Tutorial on Support Vector Machines for Pattern Recognition (1998) | Basic ideas, mathematical formulations, and the kernel trick |
| Chang and Lin, LIBSVM: A Library for Support Vector Machines | Practical SVC, SVR, one-class SVM, multiclass handling, and parameter selection |
| Guido et al., healthcare applications review (2024) | Application variants and contemporary research themes |
| TurboSVM-FL (2024) or Robust SVM via Benders Decomposition (2025) | A focused example of newer research and potential reproduction work |

The classics establish the foundation, Burges explains the theory, LIBSVM connects it to implementation, and newer work motivates the research-directions section.

Suggested supporting materials:

- Stanford CME 250, Lecture 5: Support Vector Machines, for hyperplanes and margins.
- CMU annotated SVM slides, for quadratic programming and support-vector geometry.
- Trevor Hastie's MOOC SVM slides, for soft-margin and kernelized classifiers.
- Hsu, Chang, and Lin, A Practical Guide to Support Vector Classification, for scaling, RBF kernels, cross-validation, and grid search.

The source notes suggest collecting approximately four papers and four sets of supporting notes or slides. Feature scaling and consistent preprocessing are especially relevant to the experimental protocol.

## Theory Review

Explain SVM through geometric intuition, optimization, kernels, and variants. A hyperplane is a line in two dimensions and an affine subspace in higher dimensions. Develop the explanation from separating hyperplanes to maximum-margin classifiers, soft margins, and nonlinear kernel boundaries.

For linearly separable data, the hard-margin formulation is:

$$
\min_{w,b}\frac12\|w\|^2
\quad\text{subject to}\quad y_i(w^\top x_i+b)\geq1.
$$

For nonseparable data, introduce slack variables:

$$
\min_{w,b,\xi}\frac12\|w\|^2+C\sum_i\xi_i
\quad\text{subject to}\quad y_i(w^\top x_i+b)\geq1-\xi_i,\quad\xi_i\geq0.
$$

A kernel represents an inner product in a feature space:

$$
K(x_i,x_j)=\phi(x_i)^\top\phi(x_j).
$$

The classifier can then be expressed as:

$$
f(x)=\operatorname{sign}\left(\sum_i\alpha_i y_iK(x_i,x)+b\right).
$$

These expressions connect margins, soft constraints, kernels, and the dependence of the final decision function on support vectors.

Organize variants by purpose:

- Standard formulations: C-SVC, nu-SVC, support vector regression, and one-class SVM.
- Problem-specific variants: class-weighted, robust, federated, and specialized-kernel SVMs.
- Implementation considerations: multiclass decomposition, including one-against-one strategies used in LIBSVM and SVC.

Discuss strengths and limitations in relation to the model. SVMs can work well with high-dimensional features, including settings with more features than samples. Kernels provide flexible nonlinear boundaries, and prediction depends on support vectors. Limitations include sensitivity to kernel and regularization settings, additional work for probability estimates, and expensive kernel-SVC training on large datasets. LinearSVC, SGDClassifier, or kernel approximation are alternatives to investigate when scale becomes a constraint.

A balanced interpretation is that SVM remains useful for small or medium structured datasets with informative features, as a stable baseline or a component of a larger system. Large-scale representation learning and complex perception tasks may call for other approaches.

## Research Directions and Reproduction Options

The following descriptions reflect the supplied research notes. Verify each paper and its implementation before relying on a claim in a formal submission.

### Robustness to Noise and Class Imbalance

The notes describe Robust Support Vector Machines for Imbalanced and Noisy Data via Benders Decomposition (2025) as an approach that explicitly controls violations or misclassification and solves the resulting problem with Benders decomposition. The original notes report improvements in minority-class F1 and noisy-data accuracy on OpenML datasets, and mention a Python implementation.

Gap Safe Screening Rules for Fast Training of Robust Support Vector Machines under Feature Noise (2026) is described as addressing computational cost through safe screening that reduces the training set without changing the optimum. These examples motivate separate discussions of robustness and scalability.

### Federated Learning and Privacy

The cited federated survival SVM work extends survival analysis across institutions. The source notes describe results close to centralized training and a public FeatureCloud implementation.

TurboSVM-FL uses SVM in federated aggregation. The notes report experiments on FEMNIST, CelebA, and Shakespeare, with claims about convergence, communication, accuracy, F1, and MCC, plus reproduction scripts. These claims belong to the cited work and were not reproduced for this repository.

### Specialized Kernels

The cited distance-based kernel paper proposes similarity construction for binary features. It illustrates how kernel research can adapt similarity measures to a particular data structure, connecting the classic kernel trick to newer applications.

### Hybrid and Ensemble Methods

The cited Support Vector Boosting Machine combines AdaBoost, SVMs, and residual connections; the notes mention MATLAB code. The healthcare review also motivates hybrid approaches that combine optimization or feature engineering with SVMs.

| Project | Reproduction resources described in the source notes | Expected difficulty | Coursework fit |
|---|---|---|---|
| TurboSVM-FL | GitHub, environment instructions, experiments.sh, LEAF data notes | High | Better suited to a substantial project |
| Federated Survival SVM | Public app repository, sample data, output descriptions | Medium to high | Best with prior survival-analysis or federated-learning experience |
| Support Vector Boosting Machine | Source code, module descriptions, arXiv paper | Medium | Possible extension rather than the main experiment |
| Robust SVM via Benders | Paper abstract mentions a Python implementation | Medium | Useful research direction; verify implementation availability first |

The proposed coursework strategy is a clear review of classical SVM theory plus a controlled C comparison. Discuss frontier papers as extensions. Possible follow-up experiments include class weighting for imbalance, C/gamma tuning, and comparisons with linear SVMs or kernel approximations on larger datasets.

## Experimental Design

Use Iris versicolor versus Iris virginica with petal length and petal width. The complete Iris dataset contains 150 examples, four features, and three classes; this binary subset offers a compact two-dimensional classification problem suitable for decision-boundary plots.

Standardize features using a scaler fitted only on the training set. Keep a fixed stratified split. Compare the four C values with an RBF kernel and gamma='scale'. Report accuracy, precision, recall, F1, and support-vector counts.

```text
Input:
    Iris dataset
    Classes = {versicolor, virginica}
    Features = {petal length, petal width}
    C values = [1, 50, 100, 1000]
    Kernel = RBF
    Gamma setting = fixed
    Train/test random seed = fixed

Procedure:
    1. Load the data and select the two classes and features.
    2. Split the data into training and testing sets.
    3. Fit the scaler on training data; transform both sets.
    4. For each C:
       - Fit the SVM with the fixed kernel and gamma setting.
       - Predict the test labels.
       - Compute accuracy, precision, recall, and F1.
       - Count support vectors and save a decision-boundary plot.
    5. Summarize metrics in a table and comparison chart.
    6. Discuss boundary complexity, support vectors, and generalization.

Output:
    Metric table, four decision-boundary plots,
    comparison chart, and discussion.
```

### Illustrative Results Recorded in the Source Notes

The original notes describe a 70/30 stratified split, feature standardization, and a fixed RBF gamma='scale' setting. These are historical illustrative values, not newly verified output from this upload.

| C | Train accuracy | Test accuracy | Precision | Recall | F1 | Support vectors |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9571 | 0.9000 | 0.9286 | 0.8667 | 0.8966 | 15 |
| 50 | 0.9857 | 0.9000 | 0.9286 | 0.8667 | 0.8966 | 9 |
| 100 | 0.9857 | 0.8667 | 0.8667 | 0.8667 | 0.8667 | 8 |
| 1000 | 1.0000 | 0.8667 | 0.8667 | 0.8667 | 0.8667 | 7 |

In these recorded values, raising C from 1 to 50 improves training accuracy and reduces the number of support vectors without improving test accuracy. At 100 and 1000, training accuracy remains high or reaches 1, while test accuracy falls. The intended discussion is that increasing C does not monotonically improve generalization.

Include decision-boundary plots with training points, testing points, support vectors, and classification boundaries. Two selected features make this visualization directly interpretable. Add a comparison chart for accuracy or F1, and a table or separate plot for support-vector counts.

## Suggested English Review Outline

| Section | Suggested words | Focus |
|---|---:|---|
| Introduction | 400–600 | Historical role, motivation, and structure |
| Basic Ideas of SVM | 700–900 | Hyperplanes, margins, support vectors, separability |
| Mathematical Formulations and Variants | 900–1200 | Hard/soft margins, dual form, kernels, SVC, one-class SVM, SVR, weighted variants |
| Advantages and Disadvantages | 500–700 | High-dimensional settings, flexible kernels, sensitivity, probability estimation, scaling |
| Current Research Directions and My Opinions | 700–1000 | Robustness, privacy, kernels, hybrid methods, personal assessment |
| Conclusion | 200–300 | SVM's role today |

Suggested title: **Support Vector Machines: Core Ideas, Mathematical Formulations, Modern Variants, and a Comparative Experiment on the Penalty Parameter C**.

## References Retained from the Source Notes

- Cortes, C., and Vapnik, V. Support-Vector Networks. Machine Learning, 1995, 20:273–297. DOI: 10.1007/BF00994018.
- Burges, C. J. C. A Tutorial on Support Vector Machines for Pattern Recognition. Data Mining and Knowledge Discovery, 1998, 2:121–167.
- Chang, C.-C., and Lin, C.-J. LIBSVM: A Library for Support Vector Machines. ACM Transactions on Intelligent Systems and Technology, 2011. The notes also refer to implementation documentation maintained through 2022.
- Hsu, C.-W., Chang, C.-C., and Lin, C.-J. A Practical Guide to Support Vector Classification. Technical report.
- Wang, S. Lecture 5: Support Vector Machines. Stanford CME 250 course notes.
- Guido, R., et al. An Overview on the Advancements of Support Vector Machine Models in Healthcare Applications: A Review. Information, 2024.
- Wang, M., Bodonhelyi, A., Bozkir, E., and Kasneci, E. TurboSVM-FL: Boosting Federated Learning through SVM Aggregation for Lazy Clients. AAAI, 2024.
- Spath, J., et al. Privacy-Preserving Federated Survival Support Vector Machines for Cross-Institutional Time-To-Event Analysis. JMIR AI, 2024.
- Mohasel, S. M., and Koosha, H. Robust Support Vector Machines for Imbalanced and Noisy Data via Benders Decomposition. arXiv preprint, 2025.
- Nguyen, T.-H., Tran, T.-L., and Nguyen, K. T. Gap Safe Screening Rules for Fast Training of Robust Support Vector Machines under Feature Noise. arXiv preprint, 2026.
- Amaya-Tejera, N., et al. A Distance-Based Kernel for Classification via Support Vector Machines. Frontiers in Artificial Intelligence, 2024.
- Lian, J. J. Support Vector Boosting Machine: Enhancing Classification Performance with AdaBoost and Residual Connections. arXiv preprint, 2024.
- Fisher, R. A. Iris dataset. UCI Machine Learning Repository. DOI: 10.24432/C56C76.
- scikit-learn documentation: Support Vector Machines, SVC, and RBF SVM parameters.
