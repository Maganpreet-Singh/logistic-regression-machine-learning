# Logistic Regression - Machine Learning

A comprehensive repository for learning and implementing **Logistic Regression** for classification problems. This repository covers the mathematical foundations, intuition, practical implementation, evaluation techniques, regularization, decision boundaries, and hyperparameter tuning using Python and Scikit-learn.

---

## 📚 About the Project

**Logistic Regression** is a supervised machine learning algorithm primarily used for **classification problems**.

Unlike Linear Regression, which predicts continuous values, Logistic Regression estimates the probability that an observation belongs to a particular class.

For binary classification, the model uses the **Sigmoid Function** to convert a linear combination of input features into a probability between 0 and 1.

### Logistic Regression Equation

The linear model is:

```text
z = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ
```

The sigmoid function converts `z` into a probability:

```text
σ(z) = 1 / (1 + e⁻ᶻ)
```

The output is:

```text
0 ≤ P(y = 1 | X) ≤ 1
```

A common decision threshold is:

```text
P(y = 1) ≥ 0.5 → Class 1
P(y = 1) < 0.5 → Class 0
```

---

## 🎯 Learning Objectives

By completing this repository, you will understand:

- What Logistic Regression is
- Classification vs Regression
- Binary Classification
- Multiclass Classification
- Sigmoid Function
- Probability Prediction
- Decision Boundaries
- Log Loss / Binary Cross-Entropy
- Maximum Likelihood intuition
- Model coefficients
- Feature scaling
- Regularization
- L1 and L2 regularization
- Classification metrics
- Confusion Matrix
- Precision, Recall and F1 Score
- ROC Curve and AUC
- Cross-Validation
- Hyperparameter tuning
- Logistic Regression using Scikit-learn
- Logistic Regression from scratch

---

# 📖 Topics Covered

## 1. Introduction to Classification

Classification is a supervised learning task where the model predicts a **categorical target**.

Examples:

```text
Email → Spam / Not Spam
Patient → Disease / No Disease
Transaction → Fraud / Not Fraud
Student → Pass / Fail
Customer → Churn / No Churn
```

### Regression vs Classification

| Regression | Classification |
|---|---|
| Predicts continuous values | Predicts discrete classes |
| Example: House Price | Example: Spam Detection |
| Linear Regression | Logistic Regression |
| Output can be any real value | Output represents class probabilities |

---

# 🧠 Logistic Regression

Logistic Regression models the probability of a target class.

The model first calculates:

```text
z = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ
```

Then applies the sigmoid function:

```text
P(y = 1) = 1 / (1 + e⁻ᶻ)
```

The sigmoid function maps any real number into the range:

```text
0 to 1
```

---

# 📈 Sigmoid Function

The sigmoid function is:

```text
σ(z) = 1 / (1 + e⁻ᶻ)
```

Its behavior can be summarized as:

```text
z → -∞       → σ(z) → 0
z = 0        → σ(z) = 0.5
z → +∞       → σ(z) → 1
```

This makes it useful for probability-based classification.

---

## 📊 Decision Boundary

Logistic Regression uses a threshold to convert probabilities into class predictions.

For a threshold of `0.5`:

```text
Probability ≥ 0.5 → Class 1
Probability < 0.5 → Class 0
```

Example:

```text
Prediction = 0.82 → Class 1
Prediction = 0.31 → Class 0
Prediction = 0.67 → Class 1
```

The **decision boundary** separates different classes in the feature space.

---

# 📉 Cost Function

Logistic Regression does not normally use Mean Squared Error as its primary loss function.

Instead, it uses **Log Loss**, also called **Binary Cross-Entropy**.

The loss function is:

```text
J(β) =
-1/n Σ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
```

Where:

- `y` = actual class
- `p` = predicted probability
- `n` = number of observations

### Why Log Loss?

Log Loss heavily penalizes confident incorrect predictions.

For example:

```text
Actual = 1
Prediction = 0.99 → Very small loss
Prediction = 0.60 → Moderate loss
Prediction = 0.01 → Very large loss
```

---

# ⚙️ How Logistic Regression Works

```text
Input Features
      ↓
Linear Combination
      ↓
z = β₀ + β₁X₁ + ... + βₙXₙ
      ↓
Sigmoid Function
      ↓
Probability
      ↓
Decision Threshold
      ↓
Predicted Class
```

---

# 🔄 Logistic Regression Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Logistic Regression
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Final Model
```

---

# 🧹 Data Preprocessing

Typical preprocessing steps include:

- Handling missing values
- Removing duplicates
- Detecting outliers where appropriate
- Encoding categorical variables
- Splitting features and target
- Train-test split
- Feature scaling

Example:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

# 🧪 Logistic Regression Using Scikit-Learn

Basic implementation:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
```

---

# 🔗 Logistic Regression with Pipeline

Using a Pipeline keeps preprocessing and model training together.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

This approach also helps prevent accidental preprocessing leakage between training and test data.

---

# 📊 Model Evaluation

Classification requires different metrics from regression.

## 1. Accuracy

```text
Accuracy = Correct Predictions / Total Predictions
```

Example:

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

Accuracy can be misleading when the classes are highly imbalanced.

---

# 2. Confusion Matrix

A confusion matrix contains:

```text
                  Predicted
                0         1

Actual  0      TN        FP
        1      FN        TP
```

Where:

- **TP** = True Positive
- **TN** = True Negative
- **FP** = False Positive
- **FN** = False Negative

Example:

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
```

---

# 3. Precision

Precision answers:

> Of all observations predicted as positive, how many were actually positive?

```text
Precision = TP / (TP + FP)
```

Useful when **false positives are costly**.

---

# 4. Recall

Recall answers:

> Of all actual positive observations, how many did the model identify?

```text
Recall = TP / (TP + FN)
```

Useful when **false negatives are costly**.

---

# 5. F1 Score

F1 Score is the harmonic mean of precision and recall.

```text
F1 = 2 × (Precision × Recall)
     --------------------------------
       Precision + Recall
```

Example:

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)

print("F1 Score:", f1)
```

---

# 6. Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

This provides:

```text
Precision
Recall
F1-score
Support
```

for each class.

---

# 📈 ROC Curve and AUC

The **ROC Curve** shows the relationship between:

```text
True Positive Rate
        vs
False Positive Rate
```

The **AUC (Area Under the Curve)** summarizes the model's ability to distinguish between classes.

Example:

```python
from sklearn.metrics import roc_auc_score

y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC:", auc)
```

A larger AUC generally indicates better class separation.

---

# ⚖️ Regularization

Regularization helps control model complexity and reduce overfitting.

Logistic Regression commonly supports:

- L1 Regularization
- L2 Regularization

---

## L1 Regularization

L1 adds an absolute-value penalty:

```text
λ Σ|βⱼ|
```

Advantages:

- Can shrink coefficients to zero
- Can perform feature selection
- Useful when many features exist

Example:

```python
model = LogisticRegression(
    penalty="l1",
    solver="liblinear"
)
```

---

## L2 Regularization

L2 adds a squared coefficient penalty:

```text
λ Σβⱼ²
```

Advantages:

- Shrinks coefficients
- Helps reduce overfitting
- Usually keeps all features in the model

Example:

```python
model = LogisticRegression(
    penalty="l2"
)
```

---

# 🎛️ Hyperparameter Tuning

Important Logistic Regression hyperparameters include:

```text
C
penalty
solver
max_iter
```

### C Parameter

In Scikit-learn, `C` controls the inverse of regularization strength.

```text
Small C
   ↓
Stronger Regularization

Large C
   ↓
Weaker Regularization
```

Example:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=10000)

params = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"]
}

grid = GridSearchCV(
    model,
    params,
    cv=5,
    scoring="f1"
)

grid.fit(X_train_scaled, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)
```

---

# 🔄 Cross-Validation

Cross-validation helps estimate how well the model generalizes to unseen data.

Example using 5-fold cross-validation:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X_train_scaled,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("CV Scores:", scores)
print("Mean CV Score:", scores.mean())
```

---

# 📌 Model Coefficients

Logistic Regression coefficients indicate how features influence the model.

Example:

```python
print(model.coef_)
print(model.intercept_)
```

For a binary classification model:

```text
Positive coefficient
        ↓
Higher feature value tends to increase log-odds of Class 1

Negative coefficient
        ↓
Higher feature value tends to decrease log-odds of Class 1
```

Coefficient interpretation should be done carefully, especially when features are on different scales or are highly correlated.

---

# 🧠 Multiclass Logistic Regression

Although Logistic Regression is often introduced using binary classification, it can also be used for multiclass problems.

Example:

```text
Class 0
Class 1
Class 2
```

Scikit-learn supports multiclass classification using Logistic Regression.

```python
model = LogisticRegression(
    max_iter=10000
)

model.fit(X_train_scaled, y_train)
```

---

# ⚠️ Overfitting and Underfitting

## Underfitting

Characteristics:

- Poor training performance
- Poor testing performance
- Model is too simple

Possible causes:

- Excessive regularization
- Insufficient features
- Poor feature engineering

---

## Good Fit

The model:

- Performs well on training data
- Performs well on validation/test data
- Generalizes to unseen observations

---

## Overfitting

Characteristics:

- Very high training performance
- Significantly worse validation/test performance

Possible solutions:

- Increase regularization
- Reduce unnecessary features
- Collect more data
- Use cross-validation
- Improve feature engineering

---

# 🧪 Logistic Regression From Scratch

The repository can also explore Logistic Regression from scratch using NumPy.

Important concepts include:

- Linear combination
- Sigmoid function
- Binary cross-entropy
- Gradient descent
- Weight updates
- Bias updates
- Prediction
- Classification threshold
- Model evaluation

Basic sigmoid implementation:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

The model can then iteratively update its weights using gradient descent.

---

# 📊 Useful Visualizations

Recommended visualizations for Logistic Regression include:

- Sigmoid curve
- Decision boundary
- Confusion matrix heatmap
- ROC curve
- Precision-Recall curve
- Class distribution
- Feature distributions
- Coefficient importance

Example:

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 200)
y = 1 / (1 + np.exp(-x))

plt.plot(x, y)
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.title("Sigmoid Function")
plt.show()
```

---

# 🆚 Logistic Regression vs Linear Regression

| Feature | Linear Regression | Logistic Regression |
|---|---|---|
| Problem Type | Regression | Classification |
| Output | Continuous value | Probability / Class |
| Activation | None | Sigmoid |
| Loss | MSE commonly used | Log Loss |
| Example | House Price | Spam Detection |
| Output Range | Any real value | 0 to 1 probability |

---

# 🆚 Logistic Regression vs Other Classification Algorithms

| Algorithm | Strength |
|---|---|
| Logistic Regression | Simple, interpretable baseline |
| KNN | Simple distance-based classification |
| Decision Tree | Easy-to-understand rules |
| Random Forest | Strong general-purpose model |
| SVM | Effective for margin-based classification |
| Naive Bayes | Fast and useful for probabilistic/text tasks |
| Neural Network | Powerful for complex patterns |

---

# 💡 Advantages

- Simple and easy to understand
- Fast to train
- Produces probabilities
- Highly interpretable
- Strong baseline for classification
- Supports regularization
- Works well with linearly separable relationships
- Efficient for high-dimensional datasets
- Easy to implement with Scikit-learn

---

# ⚠️ Disadvantages

- Assumes a linear decision boundary in the feature space
- Can struggle with complex nonlinear patterns
- Sensitive to irrelevant features
- Feature scaling may be important
- Can be affected by multicollinearity
- May perform poorly when classes have highly complex boundaries

---

# 🌍 Real-World Applications

Logistic Regression can be used for:

- Spam Detection
- Disease Classification
- Customer Churn Prediction
- Fraud Detection
- Credit Risk Classification
- Customer Conversion Prediction
- Sentiment Classification
- Loan Approval Prediction
- Employee Attrition Prediction
- Marketing Response Prediction

---

# 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

# 📁 Repository Structure

```text
logistic-regression-machine-learning/
│
├── Logistic Regression.ipynb
├── Logistic Regression key points.ipynb
├── README.md
└── requirements.txt
```

---

# 📚 Learning Resources

- *Introduction to Statistical Learning (ISLR)*
- *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*
- Scikit-learn Documentation
- Andrew Ng Machine Learning Course
- StatQuest

---

# 🚀 Future Improvements

- Add Logistic Regression from scratch using NumPy
- Add binary classification projects
- Add multiclass classification examples
- Add decision-boundary visualizations
- Add ROC-AUC analysis
- Add Precision-Recall analysis
- Compare Logistic Regression with other classifiers
- Add real-world datasets
- Add feature-selection experiments
- Add end-to-end classification projects
- Add model deployment examples

---

# 🧠 Key Takeaways

- Logistic Regression is a supervised classification algorithm.
- The sigmoid function converts model output into probabilities.
- A decision threshold converts probabilities into class predictions.
- Logistic Regression uses Log Loss for classification.
- Feature scaling can improve optimization.
- Regularization helps control overfitting.
- L1 regularization can perform feature selection.
- L2 regularization shrinks coefficients.
- `C` controls the inverse strength of regularization in Scikit-learn.
- Accuracy alone may not be sufficient for imbalanced datasets.
- Precision, Recall, F1, ROC-AUC and the Confusion Matrix provide deeper evaluation.
- Logistic Regression is an excellent baseline for many classification problems.

---

# 👨‍💻 Author

**Maganpreet Singh**

B.Tech Computer Science & Engineering

**Machine Learning | Data Science | Python**

---

# ⭐ Support

If you find this repository useful for learning Logistic Regression, consider giving it a ⭐ on GitHub.

**Keep learning. Keep building. Keep experimenting. 🚀**
