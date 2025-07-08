# Titanic Survival Prediction Using Ensemble Machine Learning Models

**Overview:**  
This machine learning project explores survival prediction on the Titanic dataset using an ensemble of fine-tuned classification models. The goal is to build an accurate classifier that predicts whether a passenger survived, leveraging data preprocessing, feature engineering, model tuning, ensemble techniques, and interpretability tools.

---

**Dataset Description:**  
The dataset comes from Kaggle's Titanic competition and includes:

- `train.csv`: Passenger data with survival labels  
- `test.csv`: Passenger data without labels  
- `gender_submission.csv`: Ground truth for test.csv

**Columns Used:**  
- *Categorical*: `Sex`, `Embarked`, `Ticket`, `Name`  
- *Numerical*: `Pclass`, `Age`, `Fare`, `SibSp`, `Parch`  
- *Dropped*: `Cabin` (too many missing)

---

**Data Preprocessing:**
1. Merged test and gender_submission into a labeled dataset.
2. Combined with training data to create a full DataFrame.
3. Filled missing `Age` and `Fare` with median values.
4. Dropped rows with remaining nulls.
5. One-hot encoded categorical variables.
6. Applied SMOTE to balance survival classes.
7. Scaled features with `StandardScaler`.

**Feature Engineering:**
- Extracted `Surname` from `Name`
- Cleaned `Ticket` by removing last digit (`Ticket_clean`)
- Created `GroupId`: `Surname-Pclass-Ticket_clean-Fare-Embarked` to capture family travel groups

**Exploratory Data Analysis (EDA):**
- Histogram of features
- Scatter matrix of numerical variables

---

**Modeling:**  
Trained and fine-tuned the following models:

<div style="
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.7rem;
    margin: 1rem 0;
    font-size: 0.85em;
">

<div style="
    background:rgba(0,20,40,0.3);
    padding:0.4rem 0.7rem;
    border-radius:4px;
    border:1px solid rgba(0,231,255,0.2);
    max-width:200px;
">
<strong style="color:#0084ff;display:block;margin-bottom:0.2rem;">Decision Tree</strong>
<ul style="margin:0 0 0 1em;padding:0;">
<li>Depths 1–24 tested</li>
<li>F1 peaked at mid-range</li>
</ul>
</div>

<div style="
    background:rgba(0,20,40,0.3);
    padding:0.4rem 0.7rem;
    border-radius:4px;
    border:1px solid rgba(0,231,255,0.2);
    max-width:200px;
">
<strong style="color:#0084ff;display:block;margin-bottom:0.2rem;">XGBoost</strong>
<ul style="margin:0 0 0 1em;padding:0;">
<li>Tuned with RandomizedSearchCV</li>
<li>Best params: <br> n_estimators, max_depth, learning_rate, etc.</li>
<li>High F1 on test set</li>
</ul>
</div>

<div style="
    background:rgba(0,20,40,0.3);
    padding:0.4rem 0.7rem;
    border-radius:4px;
    border:1px solid rgba(0,231,255,0.2);
    max-width:200px;
">
<strong style="color:#0084ff;display:block;margin-bottom:0.2rem;">Random Forest</strong>
<ul style="margin:0 0 0 1em;padding:0;">
<li>Tuned with RandomizedSearchCV</li>
<li>Best params: n_estimators, max_depth, etc.</li>
<li>Trained on scaled data</li>
</ul>
</div>

<div style="
    background:rgba(0,20,40,0.3);
    padding:0.4rem 0.7rem;
    border-radius:4px;
    border:1px solid rgba(0,231,255,0.2);
    max-width:200px;
">
<strong style="color:#0084ff;display:block;margin-bottom:0.2rem;">Logistic Regression</strong>
<ul style="margin:0 0 0 1em;padding:0;">
<li>Tuned with RandomizedSearchCV</li>
<li>Params: C, penalty, solver, etc.</li>
</ul>
</div>

<div style="
    background:rgba(0,20,40,0.3);
    padding:0.4rem 0.7rem;
    border-radius:4px;
    border:1px solid rgba(0,231,255,0.2);
    max-width:200px;
">
<strong style="color:#0084ff;display:block;margin-bottom:0.2rem;">Voting Ensemble</strong>
<ul style="margin:0 0 0 1em;padding:0;">
<li>Soft voting: RF, XGB, LR</li>
<li><b>F1 Score: 0.853</b> on test set</li>
</ul>
</div>

</div>

---

**Model Evaluation:**
- Cross-validation accuracy
- Final confusion matrix
- Classification report (precision, recall, F1)

**Visualization and Interpretability:**
- **ROC Curve:** Area under curve (AUC) shows strong model separation
- **Precision-Recall Curve:** Assesses balance between precision and recall
- **F1 vs. Threshold Plot:** Identifies optimal probability threshold
- **Probability Distribution Histogram:** Visualizes confidence in predictions
- **SHAP Summary Plot:** Interprets feature importance in XGBoost model

**SHAP Highlights:**
- Most influential: `Pclass`, `Fare`, `Sex`, `Age`, `SibSp`, `Parch`
- Higher class/fare increased survival probability
- Being female or young significantly boosted chances

---

**Conclusion:**  
The ensemble Voting Classifier, with fine-tuned Random Forest and XGBoost models, provided a robust solution to the Titanic classification problem. The final F1 score of **0.853** reflects strong model performance and generalization. Visualizations and SHAP interpretations confirmed that the model relies on historically and logically relevant survival factors. This approach demonstrates the power of ensemble learning and interpretability
