# Rate My Professor — Statistical Analysis & Predictive Modeling

A comprehensive statistical and machine learning analysis of [Rate My Professor (RMP)](https://www.ratemyprofessors.com/) data, investigating gender bias in student evaluations, the predictive power of tags and numerical features on professor ratings, and the detectability of "pepper" (attractiveness) ratings through classification.

**Authors:** Ghina Al Shdaifat, Hamza Alshamy, Elaf Almahmoud 

**Course:** Introduction to Data Science — Fall 2024

---

## Overview

This project merges three RMP datasets — ratings, tags, and qualitative information — covering 89,893 professors. After preprocessing, the working dataset contains **67,664 professors** and 31 features. The analysis spans hypothesis testing, effect size estimation, regression, classification, and clustering, addressing the following research questions:

1. Is there evidence of a pro-male gender bias in student evaluations?
2. Do male and female professors differ in the spread of their rating distributions?
3. Which tags are most strongly associated with gender?
4. Do male and female professors differ in perceived difficulty?
5. Can professor ratings and difficulty be predicted from numerical features and tags?
6. Can we predict whether a professor receives a "pepper" (attractiveness rating)?

---

## Repository Structure

```
├── ProjectCode.ipynb     # Full analysis notebook (EDA through clustering)
├── ProjectCode2.py       # Python script version of the notebook
└── WrittenReport.pdf     # Full written report with methodology and results
```

---

## Dataset

Three datasets were merged on row index (all sharing 89,893 rows):
- **Ratings dataset** — numerical features: average rating, difficulty, number of ratings, pepper, take again proportion, gender
- **Tags dataset** — 20 student-assigned tags (e.g., "Hilarious", "Tough Grader", "Amazing Lectures") as raw counts
- **Qualitative dataset** — university, major/field, US state, online flag

**Preprocessing steps:**
- Dropped 19,889 rows with missing values across shared columns
- Removed 133 non-US entries from the US State field
- Removed 2,207 rows where professors were marked as both male and female simultaneously
- **Tag normalization:** divided raw tag counts by each professor's total number of ratings to remove volume bias
- **Bayesian Average Rating:** applied to both average rating and difficulty to shrink extreme values for professors with few ratings toward the dataset mean (μ = 2.867)

---

## Analysis

### 1. Gender Bias in Average Rating

A one-tailed **Mann-Whitney U test** (chosen for non-normality) tested whether male professors receive significantly higher ratings than female professors.

- **Result:** p = 4.22 × 10⁻⁵ → initially reject H₀
- **Confounder analysis:** A two-step approach identified that features like `Respected` and `Hilarious` are both more associated with male professors *and* independently predictive of higher ratings — mediating the gender-rating relationship
- **Conclusion:** There is insufficient evidence of a direct pro-male gender bias; behavioral and perceptual features confound the relationship

### 2. Gender Difference in Rating Spread

Two approaches were used to test variance equality:

| Test | p-value | Conclusion |
|---|---|---|
| Levene's Test | 0.0003 | Reject H₀ — variances differ |
| Permutation Test (10,000 iterations) | 0.0063 | Fail to reject H₀ |

The divergence between tests suggests the result is sensitive to how the test statistic is defined, highlighting the importance of test selection in non-parametric settings.

### 3. Effect Sizes (Cohen's d + Bootstrapping)

Bootstrapped 95% confidence intervals (10,000 resamples) were used to quantify effect sizes robustly:

| Comparison | Cohen's d | 95% CI |
|---|---|---|
| Gender bias in average rating | 0.090 | (0.048, 0.131) — small effect |
| Gender bias in rating spread | −0.0719 | (−0.123, −0.020) — small effect, male ratings slightly less variable |
| Gender bias in difficulty | 0.0034 | (−0.019, 0.026) — negligible effect |

### 4. Gender Differences in Tags

Mann-Whitney U tests across all 20 tags showed statistically significant gender differences (p < 0.005) for **19 out of 20 tags**. The most significant:

| Tag | p-value |
|---|---|
| Hilarious | 3.71 × 10⁻²²⁸ |
| Amazing Lectures | 5.51 × 10⁻⁵⁴ |
| Lecture Heavy | 3.05 × 10⁻³⁹ |

The only non-significant tag was `Pop Quizzes!` (p = 0.027), suggesting that concrete instructional behaviors are perceived more objectively regardless of gender, while subjective qualities like charisma and engagement are strongly gender-influenced.

### 5. Gender Difference in Difficulty

An independent samples t-test (distributions approximately normal after Bayesian smoothing):

- **T-statistic:** 0.293, **p-value:** 0.7695
- **Conclusion:** No statistically significant gender difference in perceived difficulty (Cohen's d = 0.0034, negligible)

### 6. Regression: Predicting Average Rating

**From numerical features** (80/20 train-test split, 5 models compared):

| Model | R² | RMSE | Top Predictor |
|---|---|---|---|
| Linear Regression (Drop NaN) | **0.81** | **0.38** | Bayesian Avg Difficulty (−0.29) |
| Impute NaN | 0.32 | 0.92 | — |
| Drop Feature | 0.31 | 0.93 | — |
| LASSO (λ=0.0013) | 0.81 | 0.45 | Take Again Proportion (0.735) |
| Ridge (λ=2.5) | 0.81 | 0.45 | Take Again Proportion (0.735) |

`Take Again Proportion` emerged as the most robust predictor under regularization, indicating it captures student satisfaction more comprehensively than difficulty alone.

**From tags** (normalized, minimum 5 ratings threshold):

| Model | R² | RMSE | Top Predictor |
|---|---|---|---|
| Linear Regression | 0.70 | 0.52 | Amazing Lectures (1.20) |
| LASSO (λ=0) | 0.70 | 0.54 | Tough Grader (−0.24) |
| Ridge (λ=33.5) | 0.70 | 0.54 | Tough Grader (−0.24) |

### 7. Regression: Predicting Average Difficulty

Tags explain 53% of variance in difficulty ratings (R² = 0.53). `Tough Grader` is the dominant positive predictor across all models (coefficient = 1.70 in linear regression), consistent with intuitive expectations.

### 8. Classification: Predicting "Pepper" Receipt

Binary classification (professor received a pepper vs. not) using Logistic Regression and Linear SVC. Class imbalance addressed with **SMOTE**. Two preprocessing strategies and PCA were compared:

| Model | AUC | Accuracy |
|---|---|---|
| Logistic Regression (Drop NaN) | **0.8072** | **74%** |
| Linear SVC (Drop NaN) | **0.8074** | 73% |
| Logistic Regression (Drop Column) | 0.7829 | 71% |
| Logistic Regression with PCA | 0.7927 | 72% |

Retaining `Take Again Proportion` (despite 82.57% missingness) consistently outperformed dropping it, reinforcing that thoughtful handling of missing data preserves predictive signal.

### 9. Clustering (Extra Credit)

K-Means clustering (k=5) applied to `Major/Field` and `University` after label encoding and scaling:
- Chi-squared test showed no significant association between clusters and pepper receipt (p = 0.018)
- Grouping majors into 4 broad fields revealed a strong association: χ² = 315.48, p < 0.0001
- **Arts & Humanities** and **Sciences** professors were most likely to receive peppers

---

## Key Findings

- Gender significantly influences 19/20 student-assigned tags, particularly subjective ones like `Hilarious` and `Amazing Lectures`, reflecting societal stereotypes about communication and charisma
- Despite an initial statistically significant result, **direct gender bias in ratings is not conclusively established** once confounders are accounted for
- `Take Again Proportion` is the single most robust numerical predictor of average rating
- Tags explain 70% of variance in average rating and 53% in difficulty — interpretable features that reflect genuine student experience
- **Academic field significantly predicts pepper receipt**, with creative and scientific disciplines rated more favorably

---

## Dependencies

```
pandas
numpy
scipy
scikit-learn
imbalanced-learn   # SMOTE
matplotlib
seaborn
```

---

## Full Report

For complete methodology, statistical derivations, and figures, see [`Final_IDS_Capstone_Report.pdf`](./WrittenReport.pdf).
