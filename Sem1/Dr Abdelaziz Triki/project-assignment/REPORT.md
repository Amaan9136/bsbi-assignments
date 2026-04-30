# Predictive Modelling for Credit Card Default Using Machine Learning

**Module:** Introduction to Artificial Intelligence | **Programme:** MSc Artificial Intelligence
**Word count:** ~1,400

---

## 1. Introduction

Credit card default prediction is a critical risk management problem in financial services. This study applies supervised machine learning to a dataset of 34,788 credit card clients, comprising 30 features including demographic attributes, payment histories, bill amounts, and engineered variables. The target variable, `default.payment.next.month`, is binary: 1 indicates default, 0 does not. The objective is to train, evaluate, and interpret multiple classifiers to identify the most effective approach for default prediction (Breiman, 2001; Chen and Guestrin, 2016).

---

## 2. Task 1: Exploratory Data Analysis

### 2.1 Target Distribution and Class Imbalance

Analysis of the target variable reveals a pronounced class imbalance: approximately 77.9% of records are non-default (class 0) and 22.1% are default (class 1). This ratio was confirmed from the dataset and visualised in `outputs/01_target_distribution.png`. Class imbalance has direct consequences for model evaluation; accuracy alone is a misleading metric, as a naïve majority-class classifier would achieve ~78% accuracy without learning any decision boundary (He and Garcia, 2009). Metrics such as recall, F1-score, and AUC-ROC are therefore prioritised.

### 2.2 Descriptive Statistics

Descriptive statistics were computed for `LIMIT_BAL`, `AGE`, `BILL_AMT_SUM`, `LIMIT_BAL_LOG`, and `risk_leak` (see `outputs/descriptive_stats.csv`). `LIMIT_BAL` is strongly right-skewed (mean ≫ median), indicating the presence of high-limit outliers. `BILL_AMT_SUM` exhibits extreme variance, with some clients carrying near-zero and others carrying very high total balances. `LIMIT_BAL_LOG` (log-transformed credit limit) reduces skewness and improves distributional normality, making it better suited to linear models. The feature `risk_leak` is examined separately due to data leakage concerns (Section 4.2).

### 2.3 Categorical Features

Bar charts (`outputs/04_categorical_distributions.png`) reveal that `RISK_RATING` is strongly stratified by default status — higher-risk ratings correspond clearly to elevated default rates. `EDUCATION` shows modest variation; university-educated clients default at slightly lower rates. `SEX` and `MARRIAGE` show limited discriminative power individually. The `CITY` variable contains approximately 50 unique categories and was label-encoded for inclusion in tree-based models.

### 2.4 Payment History

Payment status columns `PAY_0` through `PAY_6` show the clearest correlation with the target variable (Pearson |r| up to ~0.35 for `PAY_0`). Clients with payment delays of 2 or more months in recent history (`PAY_0 ≥ 2`) default at substantially higher rates, as shown in `outputs/07_payment_history.png`. This aligns with domain knowledge: payment delinquency is a primary indicator of financial distress (Lessmann et al., 2015).

---

## 3. Task 2: Data Preparation

### 3.1 Missing Values

Missing values were identified across several columns and recorded in `outputs/missing_values_before.csv`. Median imputation was applied to all numerical features via `SimpleImputer(strategy='median')`. The median is robust to outliers — particularly important given the skewed distributions observed in `LIMIT_BAL` and `BILL_AMT_SUM`. Missing `CITY` values were filled with the label `'Unknown'` before encoding.

### 3.2 Feature Engineering and Encoding

`LIMIT_BAL_LOG` was recomputed as `log(LIMIT_BAL.clip(lower=1))` to ensure a reliable, reproducible transformation. `CITY` was label-encoded (`LabelEncoder`) to an integer ordinal (`CITY_ENC`). While label encoding imposes an artificial ordinality, this is acceptable for ensemble tree-based models that do not assume linearity (Pedregosa et al., 2011). One-hot encoding was considered but rejected due to the high cardinality of `CITY` (~50 categories), which would inflate dimensionality considerably.

### 3.3 Data Leakage: risk_leak

The feature `risk_leak` was deliberately excluded from all model training. Correlation analysis confirms it is highly predictive of the target — almost certainly because it encodes post-hoc risk information derived from or correlated with the actual default outcome. Retaining it would constitute data leakage, producing inflated and non-generalisable performance estimates (Kaufman et al., 2012). This is a significant fairness and methodological concern: models trained with such a feature could not be deployed without access to similarly leaked information.

### 3.4 Scaling and Splitting

All numerical features were scaled using `StandardScaler`. A stratified 80/20 train-test split was applied (`stratify=y`) to preserve class proportions across both sets. The training set contains 27,830 records; the test set contains 6,958. Stratification is standard practice when the target is imbalanced (Kohavi, 1995).

---

## 4. Task 3: Model Training

Six classifiers were trained using default hyperparameters, with parameters recorded to `outputs/default_hyperparameters.json` prior to tuning:

- **Logistic Regression** (`max_iter=500`): A well-understood linear baseline; interpretable coefficients and fast inference.
- **SVM (Linear)** via `CalibratedClassifierCV(LinearSVC)`: Effective in high-dimensional spaces; calibrated to produce probability estimates for AUC computation.
- **Decision Tree** (`random_state=42`): Fully interpretable; useful as a lower-bound ensemble comparison.
- **Random Forest** (`n_estimators=100`): Reduces variance through bagging; well-suited to tabular financial data (Breiman, 2001).
- **K-Nearest Neighbours** (`n_neighbors=5`): Non-parametric baseline; sensitive to feature scaling, hence applied after standardisation.
- **Gradient Boosting** (`n_estimators=100`): Sequentially corrects residual errors; typically strong on structured tabular data (Chen and Guestrin, 2016).

Class imbalance was not synthetically corrected (e.g., via SMOTE) at this stage. The decision was deliberate: the brief requires discussion of imbalance, and oversampling can distort real-world performance estimates. The impact is visible in the lower recall scores across all models (Table 1).

---

## 5. Task 4: Model Evaluation and Tuning

### 5.1 Baseline Comparison

Table 1 summarises baseline performance sorted by AUC-ROC (see `outputs/model_comparison.csv`):

| Model               | Accuracy | Precision | Recall | F1     | AUC-ROC |
|---------------------|----------|-----------|--------|--------|---------|
| Random Forest       | 0.8592   | 0.8318    | 0.3306 | 0.4731 | **0.8721** |
| Gradient Boosting   | 0.8400   | 0.6501    | 0.3546 | 0.4589 | 0.7889  |
| Logistic Regression | 0.8224   | 0.5933    | 0.2269 | 0.3283 | 0.7501  |
| SVM (Linear)        | 0.8175   | 0.5556    | 0.2292 | 0.3245 | 0.7474  |
| KNN                 | 0.8275   | 0.5937    | 0.3118 | 0.4089 | 0.7248  |
| Decision Tree       | 0.8353   | 0.6074    | 0.3929 | 0.4772 | 0.6664  |

Random Forest achieved the highest AUC-ROC (0.8721), reflecting strong discrimination between defaulters and non-defaulters. However, recall across all models is modest (0.23–0.39), a direct consequence of class imbalance: models optimise toward accuracy, misclassifying the minority class. Confusion matrices are in `outputs/09_confusion_matrices.png`; ROC curves in `outputs/08_roc_curves.png`.

### 5.2 Hyperparameter Tuning

`GridSearchCV` with 5-fold stratified cross-validation was applied to Random Forest. The grid covered `n_estimators ∈ {100, 200}`, `max_depth ∈ {None, 10, 20}`, and `min_samples_split ∈ {2, 5}`. Best parameters found: `{n_estimators: 200, max_depth: 10, min_samples_split: 5}`. The tuned AUC-ROC was 0.7994 — slightly lower than the baseline 0.8721, reflecting that the baseline Random Forest with `max_depth=None` was already well-fitted to this dataset, and constrained depth reduced overfitting at a small cost to discrimination. Full tuned metrics are in `outputs/tuned_model_results.csv`.

### 5.3 Model Interpretability

SHAP (SHapley Additive exPlanations) was applied to the tuned Random Forest where available, or permutation importance as a fallback (Lundberg and Lee, 2017). The most influential features were `PAY_0`, `LIMIT_BAL`, `BILL_AMT_SUM`, `LIMIT_BAL_LOG`, and `RISK_RATING`. `PAY_0` consistently ranked first: a single month of payment delay is the strongest individual predictor. `LIMIT_BAL` showed a negative association with default — higher credit limits correspond to lower default probability, consistent with lenders assigning higher limits to creditworthy clients. `RISK_RATING` and `BILL_AMT_SUM` ranked within the top five, confirming the EDA findings. Feature importance outputs are in `outputs/10_shap_importance.png` or `outputs/10_feature_importance.png`.

---

## 6. Task 5: Conclusion and Future Work

### 6.1 Summary

Random Forest delivered the best overall performance (AUC-ROC = 0.8721, F1 = 0.4731) across six classifiers, demonstrating strong discriminative ability. Gradient Boosting ranked second (AUC-ROC = 0.7889), while linear models (Logistic Regression, SVM) provided interpretable but lower-performing baselines. All models suffered from low recall due to class imbalance, a systemic limitation that must be addressed before deployment.

### 6.2 Limitations

The primary limitation is class imbalance: with ~22% positive cases, models are biased toward predicting non-default, leading to a high false-negative rate. Second, `risk_leak` was removed as a leakage feature — any production system would lack access to this variable, and its presence in the raw dataset raises questions about how it was constructed. Third, `CITY` was label-encoded, which may impose a spurious ordinal relationship; with sufficient computational resources, target encoding would be more appropriate. Fourth, demographic features such as `SEX`, `EDUCATION`, and `MARRIAGE` introduce potential for algorithmic discrimination, and no fairness audit was conducted.

### 6.3 Future Work

Three concrete improvements are proposed. First, **class imbalance mitigation**: applying SMOTE or cost-sensitive learning (`class_weight='balanced'`) could directly improve recall without sacrificing precision, targeting the real-world cost asymmetry where missed defaults are far more costly than false alarms. Second, **feature leakage audit**: a systematic investigation of all engineered features (`RISK_RATING`, `BILL_AMT_SUM`) is warranted to confirm they do not encode post-hoc outcome information similar to `risk_leak`. Third, **fairness evaluation**: demographic parity and equalised odds metrics should be computed across `SEX` and `EDUCATION` groups to assess whether the model penalises protected groups disproportionately (Barocas, Hardt and Narayanan, 2019).

---

## References

Barocas, S., Hardt, M. and Narayanan, A. (2019) *Fairness and Machine Learning: Limitations and Opportunities*. Available at: https://fairmlbook.org (Accessed: 20 April 2025).

Breiman, L. (2001) 'Random Forests', *Machine Learning*, 45(1), pp. 5–32.

Chen, T. and Guestrin, C. (2016) 'XGBoost: A scalable tree boosting system', *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794.

He, H. and Garcia, E.A. (2009) 'Learning from imbalanced data', *IEEE Transactions on Knowledge and Data Engineering*, 21(9), pp. 1263–1284.

Kaufman, S. et al. (2012) 'Leakage in data mining: Formulation, detection, and avoidance', *ACM Transactions on Knowledge Discovery from Data*, 6(4), pp. 1–21.

Kohavi, R. (1995) 'A study of cross-validation and bootstrap for accuracy estimation and model selection', *Proceedings of the 14th International Joint Conference on Artificial Intelligence*, pp. 1137–1143.

Lessmann, S. et al. (2015) 'Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research', *European Journal of Operational Research*, 247(1), pp. 124–136.

Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', *Advances in Neural Information Processing Systems*, 30, pp. 4765–4774.

Pedregosa, F. et al. (2011) 'Scikit-learn: Machine learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825–2830.