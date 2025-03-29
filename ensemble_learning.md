# **Ensemble Learning**

Ensemble learning **combines** the predictions of multiple models (called **weak learners** or **base models**) to make a stronger, more reliable prediction.

> It is like asking a group of experts for their opinions instead of relying on just one person.
> Each expert might make mistakes but when you combine their knowledge, the final decision is often better and more accurate.

## **Types of Ensemble Learning in Machine Learning**

There are 2 main types of ensemble methods:

1. *Bagging (Bootstrap Aggregating)* - models are trained independently on different subsets of the data and their results are averaged or voted on.

2. *Boosting* - models are trained sequentially with each one learning from the mistakes of the previous model.

Think of it like asking multiple doctors for a diagnosis (bagging) or consulting doctors who specialize in correcting previous misdiagnoses (boosting).

### **1 _ Bagging Algorithm**

*Bagging classifier* can be used for both regression and classification tasks. Here is an overview of *Bagging classifier algorithm*:

- **Bootstrap Sampling** - Divides the original training data into $N$ subsets and randomly selects a subset with replacement in some rows from other subsets. This step ensures that the base models are trained on diverse subsets of the data and there is no class imbalance.
- **Base Model Training** - For each bootstrapped sample, we train a base model independently on that subset of data. These weak models are trained in parallel to increase computational efficiency and reduce time consumption. *We can use different base learners (i.e., different ML models as base learners to bring variety and robustness)*.
- **Prediction Aggregation** - To make a prediction on testing data, combine the predictions of all base models. For classification tasks, it can include majority voting or weighted majority while for regression, it involves averaging the predictions.
- **Out-of-Bag (OOB) Evaluation** - Some samples are excluded from the training subset of particular base models during the bootstrapping method. These OOB samples can be used to estimate the model's performance without the need for cross-validation.
- **Final Prediction** - After aggregating the predictions from all the base models, Bagging produces a final prediction for each instance.

Python pseudo code for Bagging Estimator implementing libraries:

``` Python
# 1. Importing Libraries and Loading Data
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 2. Loading and Splitting the Iris Dataset
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Creating a Base Classifier
base_classifier = DecisionTreeClassifier()

# Decision tree is chosen as the base model. They are prone to overfitting when trained on small datasets making them good candidates for bagging

# 4. Creating and Training the Bagging Classifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)
bagging_classifier.fit(X_train, y_train)

# A BaggingClassifier is created using the decision tree as the base classifier.
# n_estimators=10 specifies that 10 decision trees will be trained on different bootstrapped subsets of the training data.

# 5. Making Predictions and Evaluating Accuracy
y_pred = bagging_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# The trained bagging model predicts labels for test data.
# The accuracy of the predictions is calculated by comparing the predicted labels (y_pred) to the actual labels (y_test).

```

### **2 _ Boosting Algorithm**

*Boosting* is an ensemble technique that combines multiple weak learner to create a strong learner.

Weak models are trained in series such that each next model tries to correct errors of the previous model until the entire training dataset is predicted correctly.

One of the most well-known boosting algorithms is *AdaBoost (Adaptive Boosting)*.

Here is an overview of Boosting algorithm:

- **Initialize Model Weights** - Begin with a single weak learner and assign equal weights to all training examples.
- **Train Weak Learner** - Train weak learners on these dataset.
- **Sequential Learning** - Boosting works by training models sequentially where each model focuses on correcting the errors of its predecessor. Boosting typically uses a single type of weak learner like decision trees.
- **Weight Adjustment** - Boosting assigns weights to training data points. Misclassified examples receive higher weights in the next iteration so that next models pay more attention to them.

Python pseudo code for boosting Estimator implementing libraries:

``` Python
# 1. Importing Libraries and Modules
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# AdaBoostClassifier: Implements the AdaBoost algorithm.

# 2. Loading and Splitting the Dataset
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Defining the Weak Learner
base_classifier = DecisionTreeClassifier(max_depth=1)

# Decision tree with max_depth=1 is used as the weak learner.

# 4. Creating and Training the AdaBoost Classifier
adaboost_classifier = AdaBoostClassifier(
    base_classifier, n_estimators=50, learning_rate=1.0, random_state=42
)
adaboost_classifier.fit(X_train, y_train)

# base_classifier: The weak learner used in boosting.
# n_estimators=50: Number of weak learners to train sequentially.
# learning_rate=1.0: Controls the contribution of each weak learner to the final model.
# random_state=42: Ensures reproducibility.

# 5. Making Predictions and Calculating Accuracy

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

### **Benefits of Ensemble Learning in Machine Learning**

Ensemble learning is a versatile approach that can be applied to machine learning model for:

- **Reduce in Overfitting** - By aggregating predictions of multiple models, ensembles can reduce overfitting that individual complex models might exhibit.
- **Improved Generalization** - It generalize better to unseen data by minimizing variance and bias.
- **Increased Accuracy** - Combining multiple models give higher predictive accuracy.
- **Robustness to Noise** - It mitigates the effect of noisy or incorrect data points by averaging out predictions from diverse models.
- **Flexibility** - It can work with diverse models including decision trees, neural networks and support vector machines (SVM) making them highly adaptable.
- **Bias-Variance Tradeoff** - Techniques like bagging reduce variance, while boosting reduces bias leading to better overall performance.

There are various ensemble learning techniques we can use as each one of them has their own pros and cons.

### **Ensemble Learning Techniques**

| **Technique** | **Category** | **Description** |
| --- | --- | --- |
| *Random Forest* | Bagging | It constructs multiple decision trees on bootstrapped subsets of the data and aggregates their predictions for final output, reducing overfitting and variance |
| *Random Subspace Method* | Bagging  | Trains models on random subsets of input features to enhance diversity and improve generalization while reducing overfitting |
| *Gradient Boosting System (GBM)* | Boosting | It sequentially builds decision trees with each tree correcting errors of the previous ones, enhancing predictive accuracy iteratively |
| *Extreme Gradient Boosting (XGBoost)* | Boosting | It does optimizations like tree pruning, regularization and parallel processing for robust and efficient predictive models |
| *AdaBoost (Adaptive Boosting)* | Boosting | AdaBoost focuses on challenging examples by assigning weights to data points. Combines weak classifiers with weighted voting for final prediction |
| *CatBoost* | Boosting | It specializes in handling categorical features natively without extensive preprocessing with high predictive accuracy and automatic overfitting handling |

Selecting the right ensemble technique depends on the nature of the data, specific problem we are trying to solve and computational resources available. It often requires experimentation and changes to achieve the best resutls.

**Conclusion**  - Ensemble learning is an method that uses the strengths and diversity of multiple models to enhance prediction accuracy in various ML application (classification, regression, time series forecasting and other domains where reliable and precise predictions are crucial). It also used to mitigate overfitting issue.

### **Boosting vs Bagging**

| **Feature** | **Boosting** | **Bagging** | 
| --- | --- | --- |
| *Combination Type* | Combines predictions of different weak models | Combines predictions of the same type of model |
| *Goal* | Reduces **bias** | Reduces **variance** |
| *Model Dependency* | New models depend on previous models' errors | All the models have the same weightage | 
| *Weighting* | Models are weighted based on performance | All models have equal weight |