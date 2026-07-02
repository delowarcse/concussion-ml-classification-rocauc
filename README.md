# Machine Learning and Deep Learning Models for Concussion History Classification in Youth Ice Hockey Players

This repository contains the analysis code accompanying the manuscript:

> **"Machine learning and deep learning models do not distinguish concussion history in clinically asymptomatic youth ice hockey players using clinical and robotic data."**

The code trains and evaluates five machine learning (ML) classifiers and one deep learning (DL) classifier to predict prior concussion history (yes / no) in Under-13 ice hockey players, using parameters derived from five upper-limb Kinarm robotic tasks and clinical assessments (a pre-season questionnaire and the SCAT3).

This code is published to support transparency and reproducibility during the peer review process.

---

## 1. Study summary

Youth ice hockey players in the Under-13 age group were recruited from a longitudinal cohort. All participants were clinically asymptomatic at the time of testing. Each participant completed a set of clinical assessments and a battery of upper-limb robotic tasks, from which a multivariate set of robotic and clinical features was extracted.

The features were used as inputs to several classification models tasked with distinguishing participants with a history of concussion from those without. Model performance was evaluated using stratified cross-validation with hyperparameter tuning, and results were summarized using standard classification metrics together with receiver operating characteristic (ROC) and precision-recall (PR) curve analyses.

---

## 2. Repository contents

| File | Description |
| --- | --- |
| `MainModel_Script_roc_auc.ipynb` | Main analysis notebook. Loads the data, tunes and evaluates the six classifiers, and produces the combined metrics table, ROC curves, and PR curves. |
| `MainModel_Script_roc_auc.html` | Rendered HTML export of the main notebook, including all outputs, for reviewers who prefer not to run the code. |
| `FeatureImportance_roc_auc.ipynb` | Feature importance analysis notebook (SHAP based) for the trained models. |
| `FeatureImportance_roc_auc.html` | Rendered HTML export of the feature importance notebook. |

### Data files (not included)

The notebooks expect the following two input files in the working directory:

- `S2P_Combined_WithHistory_20260111_TC.csv` (participants with a history of concussion)
- `S2P_Combined_NoHistory_20260111_TC.csv` (participants without a history of concussion)

These files contain robotic and clinical features. They are **not** distributed in this repository because participants did not consent to posting the data on a public repository. Each row is one participant and each column is one feature; the two files share the same feature columns. During loading, the two files are concatenated and a binary label column (`data_type`) is added, where `1` indicates concussion history and `0` indicates no concussion history.

---

## 3. Methods overview

The analysis in the main notebook follows a tune-once, then cross-validate design:

1. **Data preparation.** The two CSV files are merged, features are separated from the label, and all features are standardized with `StandardScaler` (zero mean, unit variance).

2. **Hyperparameter selection.** For each model family, hyperparameters are selected once on the full dataset using `GridSearchCV` with a single objective (`roc_auc`) and an inner stratified k-fold loop.

3. **Evaluation.** The same tuned model is then evaluated with an outer stratified k-fold cross-validation loop. For each fold the notebook records accuracy, precision, recall, F1 score, ROC AUC, and average precision, along with the per-fold ROC and PR curves.

4. **Reporting.** Results are summarized as mean plus or minus standard deviation across the outer folds and visualized as a metrics comparison plot, a combined ROC curve, and a combined PR curve.

### Models evaluated

- Logistic Regression (LR)
- Decision Tree (DT)
- Random Forest (RF)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Deep Neural Network (DNN), implemented in TensorFlow / Keras

### Key configuration

The main configuration values are set at the top of the setup cell:

- `RANDOM_STATE = 42` (fixed seed for reproducibility)
- `N_SPLITS = 10` (outer evaluation folds, shared by every model)
- `INNER_SPLITS = 5` (inner `GridSearchCV` folds; the DNN uses 3)
- `TUNE_SCORING = "roc_auc"` (single tuning objective for every model family)

---

## 4. Requirements

The code was developed and run with Python 3.12 using Jupyter notebooks. The main dependencies are:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow` (Keras API)
- `scikeras`
- `shap` (feature importance notebook only)
- `jupyter` (to run the notebooks)

All dependencies with minimum version constraints are listed in [`requirements.txt`](requirements.txt).

Using a dedicated virtual environment is recommended:

```bash
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 5. How to run

1. Place the two input CSV files (`S2P_Combined_WithHistory_20260111_TC.csv` and `S2P_Combined_NoHistory_20260111_TC.csv`) in the same directory as the notebooks.

2. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open `MainModel_Script_roc_auc.ipynb` and run the cells in order (top to bottom) to reproduce the classification results, metrics table, ROC curves, and PR curves.

4. To reproduce the feature importance analysis, open and run `FeatureImportance_roc_auc.ipynb`. This notebook writes per-model feature importance CSV files to the working directory.

Reviewers who only wish to inspect the results without running the code can open the corresponding `.html` files in any web browser.

---

## 6. Reproducibility notes

- A fixed random seed (`RANDOM_STATE = 42`) is applied to NumPy and to the cross-validation splitters so that results are reproducible across runs on the same environment.
- Minor numerical differences may still occur across different versions of the underlying libraries (in particular TensorFlow) and across different hardware.
- Hyperparameters for each model family are selected once on the full dataset and then reused unchanged during the outer cross-validation, so a single tuned model per family is reported for the metrics, the ROC curves, and the PR curves.

---

## 7. Data availability

The participant data used in this study are not publicly available due to privacy and ethical restrictions. Requests regarding data access should be directed to Dr. Tyler Cluff (tyler.cluff@ucalgary.ca).

---

## 8. Citation

If you refer to this code, please cite the associated manuscript:

> "Machine learning and deep learning models do not distinguish concussion history in clinically asymptomatic youth ice hockey players using clinical and robotic data."

(Full citation details will be added upon publication.)

---

## 9. License

This code is released under the [MIT License](LICENSE).

---

## 10. Contact

For questions about the code, please contact Delowar Hossain at delowar.cse.ru@gmail.com and Tyler Cluff at tyler.cluff@ucalgary.ca.
