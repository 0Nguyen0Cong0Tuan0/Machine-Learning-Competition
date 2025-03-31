# What is the *resilience* in ML?

In ML, *resilience* refers to a model's ability to maintain its performance and reliability despite challenges such as:

1. **Noisy data & Outliers** - The model should handle unexpected variations and errors in input data without breaking down.
2. **Concept drift** - The model should adapt to changes in data distributions over time, such as financial markets or user behaviors.
3. **Adversarial attacks** - A robust ML model should resist attempts to manipulate its predictions through small, malicious modifications to input data.
4. **Hardware failures & Resource limitations** - Some ML applications require models to function under limited computational power or in the presence of system failures.
5. **Generalization to new data** - A resilient model should not just memorize training data but also well on unseen real-world input.

**How to improve ML resilience?**

- **Regularization techniques** to prevent overfitting (such as *dropout*, *L1/L2 regularization*).
- **Robust loss function** to handle noisy labels or outliers (such as *Huber loss*).
- **Data augmentation** to expose the model to diverse examples.
- **Ensemble methods** to reduce variance (such as *bagging*, *boosting*).
- **Adversarial training** to strengthen resistance to adversarial attacks.
- **Continuous monitoring & retraining** to adapt to evolving data distributions.