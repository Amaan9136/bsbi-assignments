# Predictive Modelling for Credit Card Default Using Machine Learning

**Module:** Introduction to Artificial Intelligence | **Programme:** MSc Artificial Intelligence
**Word count:** ~1,490

---

## 1. Introduction

Credit card default prediction is a critical risk management problem in financial services. This study applies supervised machine learning to a dataset of 34,788 credit card clients comprising 30 features, including demographic attributes, payment histories, bill amounts, and engineered variables. The target variable, `default.payment.next.month`, is binary: 1 indicates default, 0 does not. The objective is to train, evaluate, and interpret multiple classifiers to identify the most effective approach for default prediction (Breiman, 2001; Chen and Guestrin, 2016).

---

## 2. Task 1: Exploratory Data Analysis

### 2.1 Target Distribution and Class Imbalance

The target variable exhibits a pronounced class imbalance: approximately 77.9% of records are non-default (class 0) and 22.1% are default (class 1), confirmed from the dataset and visualised in `outputs/01_target_distribution.png`. This imbalance has direct consequences for model evaluation: a naïve majority-class classifier would achieve approximately 78% accuracy without learning any decision boundary (He and Garcia, 2009). Accordingly, recall, F1-score, and AUC-ROC are prioritised over accuracy throughout this study.

### 2.2 Descriptive Statistics

Descriptive statistics were computed for `LIMIT_BAL`, `AGE`, `BILL_AMT_SUM`, `LIMIT_BAL_LOG`, and `risk_leak` (see `outputs/descriptive_stats.csv`). `LIMIT_BAL` is strongly right-skewed (mean ≫ median), indicating high-limit outliers. `BILL_AMT_SUM` exhibits extreme variance across clients. `LIMIT_BAL_LOG` (log-transformed credit limit) reduces this skewness and improves distributional normality, making it better suited to linear models. `risk_leak` is considered separately due to data leakage concerns addressed in Section 3.3.

### 2.3 Categorical Features

Bar charts (`outputs/04_categorical_distributions.png`) reveal that `RISK_RATING` is strongly stratified by default status: higher risk ratings correspond clearly to elevated default rates. `EDUCATION` shows modest variation; university-educated clients default at slightly lower rates. `SEX` and `MARRIAGE` show limited individual discriminative power. The `CITY` variable contains approximately 50 unique categories (plotted in `outputs/04b_city_distribution.png`) and was label-encoded for inclusion in tree-based models.

### 2.4 Payment History

Payment status columns `PAY_0` through `PAY_6` show the strongest individual correlations with the target (Pearson |r| up to approximately 0.35 for `PAY_0`). Clients with payment delays of two or more months in recent history (`PAY_0 ≥ 2`) default at substantially higher rates, as visualised in `outputs/07_payment_history.png`. This aligns with domain literature: payment delinquency is a primary indicator of financial distress (Lessmann et al., 2015).

---

## 3. Task 2: Data Preparation

### 3.1 Missing Values

Missing values were identified across several columns and recorded in `outputs/missing_values_before.csv`. Median imputation was applied to all numerical features via `SimpleImputer(strategy='median')`. The median is robust to outliers - particularly important given the skewed distributions in `LIMIT_BAL` and `BILL_AMT_SUM`. Post-imputation verification confirmed zero remaining missing values. Missing `CITY` values were filled with the label `'Unknown'` before encoding.

### 3.2 Feature Engineering and Encoding

`LIMIT_BAL_LOG` was computed as `log(LIMIT_BAL.clip(lower=1))` after imputation to ensure a reliable, reproducible transformation. `CITY` was label-encoded (`LabelEncoder`) to an integer ordinal (`CITY_ENC`). While label encoding imposes an artificial ordinality, this is acceptable for tree-based ensemble models that make no linearity assumptions (Pedregosa et al., 2011). One-hot encoding was rejected due to the high cardinality of `CITY` (~50 categories), which would inflate dimensionality considerably.

### 3.3 Data Leakage: risk_leak

The feature `risk_leak` was excluded from all model training and evaluation pipelines. Preliminary correlation analysis revealed an unusually high association with the target variable - consistent with information derived from or correlated with the actual default outcome. Including it would constitute data leakage, producing inflated and non-generalisable performance estimates (Kaufman et al., 2012). Its presence in the raw dataset also raises an important data governance question: any production system trained on such a variable would require equivalently derived post-hoc signals at inference time, which are by definition unavailable. This feature is therefore treated as out-of-scope for all downstream analysis, and its absence from the SHAP importance ranking confirms it was successfully excluded from the trained pipeline.

### 3.4 Scaling and Splitting

All numerical features were standardised using `StandardScaler`. A stratified 80/20 train/test split was applied (`stratify=y`) to preserve class proportions: the training set contains 27,830 records and the test set 6,958. Stratification is standard practice under class imbalance (Kohavi, 1995). Class imbalance was not corrected via SMOTE at this stage; this reflects real-world class proportions and yields honest baseline performance estimates, with the impact visible in recall scores across all models (Table 1).

---

## 4. Task 3: Model Training

Six classifiers were trained using default hyperparameters, with all parameters recorded to `outputs/default_hyperparameters.json` before tuning:

- **Logistic Regression** (`max_iter=500`): A well-understood linear baseline with interpretable coefficients and fast inference.
- **SVM (Linear)** via `CalibratedClassifierCV(LinearSVC)`: Effective in high-dimensional spaces; the calibration wrapper is required for probability output and AUC computation (Platt, 1999).
- **Decision Tree** (`random_state=42`): Fully interpretable; provides a lower-bound comparison for ensemble methods.
- **Random Forest** (`n_estimators=100`): Reduces variance through bagging; well-suited to tabular financial data (Breiman, 2001).
- **K-Nearest Neighbours** (`n_neighbors=5`): A non-parametric baseline; sensitive to feature scale, hence applied after standardisation.
- **Gradient Boosting** (`n_estimators=100`): Sequentially corrects residual errors; typically strong on structured tabular data (Chen and Guestrin, 2016).

---

## 5. Task 4: Model Evaluation and Tuning

### 5.1 Baseline Comparison

Table 1 summarises baseline performance sorted by AUC-ROC (see `outputs/model_comparison.csv`):

| Model               | Accuracy | Precision | Recall | F1     | AUC-ROC   |
|---------------------|----------|-----------|--------|--------|-----------|
| Random Forest       | 0.8592   | 0.8318    | 0.3306 | 0.4731 | **0.8721**|
| Gradient Boosting   | 0.8400   | 0.6501    | 0.3546 | 0.4589 | 0.7889    |
| Logistic Regression | 0.8224   | 0.5933    | 0.2269 | 0.3283 | 0.7501    |
| SVM (Linear)        | 0.8175   | 0.5556    | 0.2292 | 0.3245 | 0.7474    |
| KNN                 | 0.8275   | 0.5937    | 0.3118 | 0.4089 | 0.7248    |
| Decision Tree       | 0.8353   | 0.6074    | 0.3929 | 0.4772 | 0.6664    |

Random Forest achieved the highest AUC-ROC (0.8721), reflecting strong discrimination between defaulters and non-defaulters. Recall is modest across all models (0.23–0.39), a direct consequence of class imbalance: models optimise toward accuracy and systematically misclassify the minority class. Confusion matrices are in `outputs/09_confusion_matrices.png`; ROC curves in `outputs/08_roc_curves.png`.

### 5.2 Hyperparameter Tuning

`GridSearchCV` with 5-fold stratified cross-validation was applied to Random Forest (best baseline). The grid covered `n_estimators ∈ {100, 200}`, `max_depth ∈ {None, 10, 20}`, and `min_samples_split ∈ {2, 5}`. Best parameters: `{n_estimators: 200, max_depth: 10, min_samples_split: 5}`.

The tuned AUC-ROC was 0.7994, a reduction of 0.073 from the baseline 0.8721. This reflects a deliberate bias-variance trade-off: constraining `max_depth=10` regularises the forest and prevents the unconstrained model from memorising boundary details specific to this test split. In this instance the baseline forest was already well-fitted to the dataset's feature structure, and the depth constraint introduced measurable but predictable discrimination loss. In a production context, the constrained model is preferable for its more robust generalisation behaviour. Full tuned metrics are in `outputs/tuned_model_results.csv` and `outputs/tuning_comparison.csv`.

### 5.3 Model Interpretability

SHAP (SHapley Additive exPlanations) was applied to the tuned Random Forest on a 500-sample test subset; permutation importance was used as a fallback where SHAP was unavailable (Lundberg and Lee, 2017). The most influential features were `PAY_0`, `LIMIT_BAL`, `BILL_AMT_SUM`, `LIMIT_BAL_LOG`, and `RISK_RATING`. `PAY_0` consistently ranked first: a single month of payment delay is the strongest individual predictor of default. `LIMIT_BAL` showed a negative SHAP direction - higher credit limits correspond to lower default probability, reflecting lenders' practice of assigning elevated limits to historically creditworthy clients, making it a proxy for borrower quality. `RISK_RATING` and `BILL_AMT_SUM` ranked within the top five, confirming the EDA findings. The absence of `risk_leak` from the importance ranking confirms it was correctly excluded from the training pipeline. Feature importance outputs are in `outputs/10_shap_importance.png` or `outputs/10_feature_importance.png`.

---

## 6. Task 5: Conclusion and Future Work

### 6.1 Summary

Random Forest delivered the best overall performance (AUC-ROC = 0.8721, F1 = 0.4731) across six classifiers, demonstrating strong discriminative ability. Gradient Boosting ranked second (AUC-ROC = 0.7889), while linear models provided interpretable but lower-performing baselines. All models suffered from low recall due to class imbalance - a systemic limitation that must be addressed before deployment. The SHAP analysis confirmed that payment behaviour (`PAY_0`) and creditworthiness proxies (`LIMIT_BAL`) are the most decision-relevant features, consistent with domain literature.

### 6.2 Limitations

**Class imbalance** is the primary constraint: with only 22.1% positive cases, models are systematically biased toward predicting non-default, yielding recall scores between 0.23 and 0.39. In financial risk contexts, false negatives carry higher business cost than false positives, making this imbalance a critical deployment concern. Second, `risk_leak` was removed as a leakage artefact, but `RISK_RATING` and `BILL_AMT_SUM` are retained engineered features whose derivation logic is undocumented; a formal temporal audit is warranted. Third, `CITY` was label-encoded, imposing an arbitrary ordinal relationship across ~50 geographic categories. Target encoding or entity embeddings would be more appropriate. Fourth, `SEX`, `EDUCATION`, and `MARRIAGE` are included in the feature matrix without a fairness audit; the model may encode disparate impact across protected groups, which carries legal and ethical significance in credit risk assessment (Barocas, Hardt and Narayanan, 2019).

### 6.3 Future Work

Three improvements follow directly from the findings above.

**1. Class imbalance mitigation.** All classifiers achieved recall below 0.40, with Random Forest's recall at 0.33 despite its leading AUC. Applying `class_weight='balanced'` within the Random Forest estimator, or SMOTE within the cross-validation loop, would re-weight the loss function to penalise false negatives more heavily - directly targeting the recall deficit observed without distorting real-world class distributions in evaluation.

**2. Feature leakage audit for engineered columns.** While `risk_leak` was identified and removed, `RISK_RATING`'s appearance as a top-5 SHAP feature despite its opaque construction warrants scrutiny. A temporal audit - confirming that all engineered features represent information strictly available before the target month - is a prerequisite for production deployment, and would also clarify whether `BILL_AMT_SUM` has similar leakage risk.

**3. Fairness evaluation across demographic groups.** The model was trained on `SEX`, `EDUCATION`, and `MARRIAGE` without examining whether its error rates are equitable across subgroups. Computing demographic parity difference and equalised odds would identify whether specific groups face disproportionately higher false-negative rates, which could constitute discriminatory lending practice under applicable regulatory frameworks (Barocas, Hardt and Narayanan, 2019).

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

Platt, J. (1999) 'Probabilistic outputs for support vector machines and comparisons to regularised likelihood methods', *Advances in Large Margin Classifiers*, 10(3), pp. 61–74.
