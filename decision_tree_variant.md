# **Exploring Different Decision Trees**

Decision Trees are fundamental models in machine learning, with various adaptations for specific tasks. This post covers different types of decision trees, their mechanisms, and advanced implementations like XGBoost.

---

## **Types of Decision Trees**

### **1. Classification Tree**

- **Purpose**: Used for categorical predictions (e.g., Yes/No decisions).
- **Splitting Criteria**: Relies on Gini impurity or Information Gain to determine the best splits.
- **Related Techniques**:
  - *Bootstrap*: Random sampling with replacement.
  - *Bagging* (Bootstrap Aggregating): Improves robustness by combining multiple trees, as seen in Random Forest.

### **2. Regression Tree**

- **Purpose**: Used for predicting continuous values (e.g., house prices).
- **Splitting Criteria**: Uses *variance reduction* as the metric for splitting nodes.
  - Variance calculation:
    \[
    \text{Var} = \frac{1}{n} \sum (y_i - \bar{y})^2
    \]
  - The goal is to maximize variance reduction across child nodes.

---

## **Ensemble Methods with Decision Trees**

### **3. AdaBoost (Adaptive Boosting)**

- **Concept**: Enhances decision trees by focusing on misclassified samples.
- **Mechanism**:
  - Multiple weak learners (trees) are trained sequentially.
  - Misclassified samples get higher weights in the next iteration.
- **Key Difference from Random Forest**:
  - *Random Forest*: Uses independent trees with equal voting.
  - *AdaBoost*: Assigns different weights to trees based on performance.

### **4. Gradient Boosted Decision Trees (GBDT)**

*(Also known as Multiple Additive Regression Trees - MART)*

- **Concept**: Builds trees sequentially, correcting errors from the previous ensemble.
- **Core Idea**:
  - Each tree is trained to minimize residuals (errors) from prior predictions.
- **Implementation Steps**:
  1. **Initialize Prediction**: Start with a baseline, often the mean of the target (for regression).
  2. **Compute Residuals**:
     \[
     y_n = y_{\text{actual}} - y_{n-1}
     \]
     - Residuals represent errors from the previous step.
  3. **Train New Tree on Residuals**.
  4. **Update Ensemble**:
     \[
     y_n^1 = y_{n-1}^1 + \text{step} \cdot y_n
     \]
     - The *step size* (learning rate) helps prevent overfitting.
  5. **Repeat Until Stopping Criterion is Met** (e.g., max trees reached).

- **Variants**:
  - *Shrinkage*: Uses a learning rate to control updates and reduce overfitting.
  - *Stochastic GBDT*: Randomly samples features or data subsets for added robustness.

---

## **XGBoost: A GBDT Variant**

### **Key Features**

1. **Second-Order Optimization**:
   - Uses second-order Taylor approximations (gradient & Hessian) for faster convergence.
   - Loss function:
     \[
     L = \sum_i l(y_i, \hat{y}_i) + \Omega(f)
     \]
2. **Regularization for Overfitting Control**:
   - Adds penalties to avoid complex models:
     \[
     \Omega(f) = \gamma T + \frac{1}{2} \lambda ||w||^2
     \]
     - \( \gamma \): Minimum loss reduction required for a split (pruning).
     - \( \lambda \): L2 regularization to smooth leaf weights.
3. **Efficiency & Scalability**:
   - Optimized for parallel processing and large datasets.

---

## **Open Questions**

- **Parallelism in XGBoost**: How does it speed up training?
- **XGBoost vs. GBDT**: Is XGBoost just a faster GBDT or fundamentally different?
- **Handling Missing Features**: How does XGBoost natively handle missing data?
- **Higher-Order Derivatives**: Why does XGBoost stop at the second derivative instead of using higher orders?

---

## **Summary**

- **Classification Trees**: Used for categorical outcomes.
- **Regression Trees**: Predict continuous values.
- **AdaBoost**: Boosts weak learners with adaptive weighting.
- **GBDT**: Improves residuals sequentially.
- **XGBoost**: Optimized GBDT with regularization and efficiency improvements.

Further exploration into XGBoost’s internals, such as parallelism and handling of missing data, can help clarify its advantages over traditional GBDT.

