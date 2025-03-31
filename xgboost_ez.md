# **XGBoost**

Traditional machine learning models like decision trees and random forests are easy to interpret but often struggle with accuracy on complex datasets.

XGBoost, short for eXtreme Gradient Boosting, is an advanced machine learning algorithm designed for efficiency, speed, and high performance.

## **What is XGBoost?**

XGBoost is an optimized implementation of *Gradient Boosting* and is a type of ensemble leaning method. Ensemble learning combines multiple weak models to form a stronger model.

- XGBoost uses decision trees as its base learners combining them sequentially to improve the model’s performance. Each new tree is trained to correct the errors made by the previous tree and this process is called boosting.
- It has built-in parallel processing to train models on large datasets quickly. XGBoost also supports customizations allowing users to adjust model parameters to optimize performance based on the specific problem.

## **How XGBoost Works?**

It builds decision trees sequentially with each tree attempting to correct the mistakes made by the previous one. The process can be broken down as follows:

1. **Start with a base learner**: The first model decision tree is trained on the data. In regression tasks this base model simply predict the average of the target variable.
2. **Calculate the errors**: After training the first tree the errors between the predicted and actual values are calculated.
3. **Train the next tree**: The next tree is trained on the errors of the previous tree. This step attempts to correct the errors made by the first tree.
4. **Repeat the process**: This process continues with each new tree trying to correct the errors of the previous trees until a stopping criterion is met.
5. **Combine the predictions**: The final prediction is the sum of the predictions from all the trees.

## **Maths Behind XGBoost Algorithm**

It can be viewed as iterative process where we start with an initial prediction often set to zero. After which each tree is added to reduce errors. Mathematically, the model can be represented as:

$$\hat{y}_i = \sum_{k=1}^Kf_k(x_i)$$

where $\hat{y}_i$ is the final predicted value for the $i^{th}$ data point, $K$ is the number of trees in the ensemble and $f_k(x_i)$ represents the prediction of the $K^{th}$ tree for the $i^{th}$ data point.

The objective function in XGBoost consists of two parts: *a loss function* and *a regularization term*.

- The loss function measures how well the model fits the data.
- The regularization term simplify complex trees.

The general form of the objective function is:

$$obj(\theta) = \sum^n_i l(y_i, \hat{y}_i) + \sum^K_{k=1} \Omega(f_k)$$

where

- $l(y_i, \hat{y}_i)$ is the loss function which computes the difference between the true value $y_i$ and the predicted value $\hat{y}_i$
- $\Omega(f_k)$ is the regularization term which discourages overly complex trees.

Now, instead of fitting the model all at once, we optimize the model iteratively. 

We start with an initial prediction $\hat{y}_i = 0$ and at each step, we add a new tree to improve the model. The updated predictions after adding the $t^{th}$ tree can be written as:

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)$$

where $\hat{y}_i^{(t-1)}$ is the prediction from the previous iteration/tree and $f_t(x_i)$ is the prediction of the $t^{th}$ tree for the $i^{th}$ data point.

---

The **regularization term** $\Omega(f_t)$ simplify complex trees by penalizing the number of leaves in the tree and the size of the leaf. It is defined as:

$$\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum^T_{j=1}w_j^2$$

where

- $T$ is the number of leaves in the tree
- $\gamma$ is a regularization parameter that controls the complexity of the tree
- $\lambda$ is a parameter that penalizes the squared weight of the leaves $w_j$.

Finally when deciding how to split the nodes in the tree we compute the **information gain** for every possible split. The **information gain** for a split is calculated as:

$$IG = \frac{1}{2}\left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right]$$

where 

- $G_L, G_R$ are the sums of gradients in the left and right child nodes.
- $H_L, H_R$ are the sums of Hessians in the left and right child nodes.

By calculating the information gain for every possible split at each node XGBoost selects the split that results in the largest gain which effectively reduces the errors and improves the model's performance.


## **What makes XGBoost "eXtreme"?**

XGBoost extends traditional gradient boosting by including regularization elements in the objective function, XGBoost improves generalization and prevents overfitting.

### 1. **Preventing Overfitting**

The learning rate (also known as *shrinkage*) is new parameter introduced by XGBoost.

It is represented by the symbol $\eta$. It quantifies each tree’s contribution to the total prediction.

*Because each tree has less of an influence, an optimization process with a lower learning is more resilient*. By making the model more conservative, regularization terms combined with a low learning rate assist avoid overfitting.

XGBoost constructs trees level by level, assessing whether adding a new node (split) enhances the objective function as a whole at each level. The split is trimmed if not.

This level growth along with trimming makes the trees easier to understand and create.

The regularization terms, along with other techniques such as *shrinkage* and *pruning*, play a crucial role in preventing overfitting, improving generalization, and making XGBoost a robust and powerful algorithm for various machine learning tasks.

### 2. **Tree Structure**

Conventional decision trees are frequently developed by expanding each branch until a stopping condition is satisfied, or in a depth-first fashion. On the other hand, XGBoost builds trees level-wise or breadth-first. This implies that it adds nodes for every feature at a certain depth before moving on to the next level, so growing the tree one level at a time.

- **Determining the Best Splits** - XGBoost assesses every split that might be made for every feature at every level and chooses the one that minimizes the objective function as much as feasible (e.g., minimizing the mean squared error for regression tasks or cross-entropy for classification tasks).

In contrast, a single feature is selected for a split at each level in depth-wise expansion.

- **Prioritizing Important Features** - The overhead involved in choosing the best split for each feature at each level is decreased by level-wise growth. XGBoost eliminates the need to revisit and assess the same feature more than once during tree construction because all features are taken into account at the same time.

This is particularly beneficial when there are complex interactions among features, as the algorithm can adapt to the intricacies of the data.


### **3. Handling Missing Data**

XGBoost functions well even with incomplete datasets because of its strong mechanism for handling missing data during training.

To effectively handle missing values, XGBoost employs a **Sparsity Aware Split Finding** algorithm.

The algorithm treats missing values as a separate value and assesses potential splits in accordance with them when determining the optimal split at each node. If a data point has a missing value for a particular feature during tree construction, it descends a different branch of the tree.

The potential gain from splitting the data based on the available feature values - including missing values - is taken into account by the algorithm to determine the ideal split. It computes the gain for every possible split, treating the cases where values are missing as a separate group.

If the algorithm’s path through the tree comes across a node that has missing values while generating predictions for a new instance during inference, it will proceed along the default branch made for instances that have missing values. This guarantees that the model can generate predictions in the event that there are missing values in the input data.

### 4. Cache-Aware Access in XGBoost

Cache memory located closer to the CPU offers faster access times, and modern computer architectures consist of hierarchical memory systems. 

By making effective use of this cache hierarchy, computational performance can be greatly enhanced. This is why XGBoost’s cache-aware access was created, with the goal of reducing memory access times during the training stage.

This method makes use of the spatial locality principle, which states that adjacent memory locations are more likely to be accessed concurrently. Computations are sped up by XGBoost because it arranges data in a cache-friendly manner, reducing the need to fetch data from slower main memory.

### 5. Approximate Greedy Algorithm

This algorithm uses weighted quantiles to find the optimal node split quickly rather than analyzing each possible split point in detail. When working with large datasets, XGBoost makes the algorithm more scalable and faster by approximating the optimal split, which dramatically lowers the computational cost associated with evaluating all candidate splits.

## **Advantages of XGboost**

- XGBoost is highly scalable and efficient as it is designed to handle large datasets with millions or even billions of instances and features.
- XGBoost implements parallel processing techniques and utilizes hardware optimization, such as GPU acceleration, to speed up the training process. This scalability and efficiency make XGBoost suitable for big data applications and real-time predictions.
- It provides a wide range of customizable parameters and regularization techniques, allowing users to fine-tune the model according to their specific needs.
- XGBoost offers built-in feature importance analysis, which helps identify the most influential features in the dataset. This information can be valuable for *feature selection*, *dimensionality reduction*, and *gaining insights into the underlying data patterns*.
- XGBoost has not only demonstrated exceptional performance but has also become a go-to tool for data scientists and machine learning practitioners across various languages. It has consistently outperformed other algorithms in Kaggle competitions, showcasing its effectiveness in producing high-quality predictive models.

## **Disadvantages of XGBoost**

- XGBoost can be computationally intensive especially when training complex models making it less suitable for resource-constrained systems.
- Despite its robustness, XGBoost can still be sensitive to noisy data or outliers, necessitating careful data preprocessing for optimal performance.
- XGBoost is prone to overfitting on small datasets or when too many trees are used in the model.
- While feature importance scores are available, the overall model can be challenging to interpret compared to simpler methods like linear regression or decision trees. This lack of transparency may be a drawback in fields like healthcare or finance where interpretability is critical.

XGBoost is a powerful and flexible tool that works well for many machine learning tasks. Its ability to handle large datasets and deliver high accuracy makes it useful.