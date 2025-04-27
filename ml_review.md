**Overview** - Support Vector Machines

SVMs, which aim to find a linear separator that maximizes the **margin** (distance to the nearest data points) for linearly separable data. It covers:

**Maximizing the margin** - choosing the separating hyperplane with the largest margin reduces the number of possible dichotomies and improving generalization
- For linearly separable data, multiple separating hyperplanes exist. SVM chooses the one with the largest margin (distance to the nearest point)
- **Fat margins** reduce the number of dichotomies compared to all possible linear separators. Fewer dichotomies imply a smaller effective complexity

**VC inequality connection**

The VC dimension of linear separators in $d$-dimensional space is $d_\text{vc}$ is $d_\text{vc} = d + 1$

**Fat margins** restrict the hypothesis set, reducing the **effective VC dimension**. The growth function $m_H(N)$ for fat-margin separators is smaller than for all linear separators, tightening the VC bound:

$$E_\text{out} \leq E_\text{in} + \sqrt{\frac{1}{2N} \ln\left( \frac{4m_H(2N)}{\sigma}\right)}$$

**Why is a bigger margin better?** - A larger margin reduces the number of dichotomies $\rightarrow$ lowering $m_H(2N)$ which improves generalization per the VC inequality

For example, for $d=2 \rightarrow d_{vc} =3 \rightarrow m_H(2N) \leq (2N)^3$. A fat margin might reduce the effective $d_{vc}$ and making the bound tighter


- **The solution** - formulating SVM as a quadratic programming problem, solved using Lagrange multipliers, where **support vectors** (SVs) define the hyperplane

SVM optimizes $\min \frac{1}{2}w^\intercal w$ subject to $y_n(w^\intercal x_n + b) \geq 1$ $\rightarrow$ maximizing the margin $\frac{1}{\| w \|}$

Lagrange formulation introduces multipliers $\alpha_n$, leading to 

$$w = \sum^N_{n=1} \alpha_n y_n x_n$$

The Karush-Kuhn-Tucker (KKT) condition $\alpha_n(y_n(w^\intercal x_n +b) - 1) = 0$


- **Nonlinear transforms** - mapping data to a higher-dimensional space to handle non-linearly separable data, using **kernels**

- **Generalization result** - a bound on $E_\text{out}$ based on the number of support vectors

The VC inequality is relevant to SVMs because it quantifies how the margin and model complexity (via $d_\text{vc}$) affect generalization, esp in the context of nonlinear transforms and support vectors

--

**What is regularization in machine learning?** 

**Regularization** is a technique in ML used to prevent overfitting by adding a penalty to the model's complexity during training.

It modifies the learning object to balance fitting the training data (minimizing in-sample error, $E_\text{in}$) with keeping the model simple, which improves generalization to unseen data (reducing out-of-sample error, $E_\text{out}$)

**Core idea** - Regularization discourages overly complex models (e.g., those with large weights or intricate decision boundaries) that fit noise in the training data, ensuring better performance on new data

**Mechanism** - add a penalty term to loss function, which penalizes large model parameters or excessive flexibility

**Regularization as constrained optimization**


In SVMs, a standard regularization problem: 

$$\text{Minimize  } E_\text{in}(w) = \frac{1}{N}(Zw- y)^\intercal(Zw =-y)$$

subject to 

$$w^\intercal w \leq C$$

Here, $E_\text{in}$ is mean squared error for linear regression, and $w^\intercal w \leq C$ limits the size of weight vector $w$

**Explanation**

The constraint $w^\intercal w \leq C$ penalizes large weights, favoring simpler model (smaller $\| w \|$)

This is equivalent to adding a penalty term to the loss, e.g.:

$$\text{Minimize  } E_\text{in}(w) + \lambda w^\intercal w$$

where $\lambda$ (related to $C$) controls the strength of regularization

In SVMs, which minimize $\frac{1}{2}w^\intercal w$ subject to margin constraints $y_n(w^\intercal x_n + b) \geq 1$. Both approaches limit $\| w\|$ but SVMs focus on maximizing the margin

**Connection to SVMs**

SVMs optimize:

$$\text{Minimize } \frac{1}{2}w^\intercal w \text{ subject to } y_n(w^\intercal x_n + b) \geq 1$$

- The term $\frac{1}{2} w^\intercal w$ acts as a **regularization term**, minimizing the norm of $w$ which maximizes the margin $\frac{1}{\| w \|}$

- This is analogous to regularization in other models, as it controls model complexity by preferring simpler hyperplane (large margins)

**Types of Regularization**

While Lecture 14 focuses on $w^\intercal w$ (L2 regularization), common regularization techniques include:

**L2 Regularization (Ridge Regression)**

- Penalty: $\lambda \| w \|^2_2 = \lambda w^\intercal w$

- Effect: shrinks weights toward zero, smoothing the model 
- Example: in linear regression, minimize:

$$E_\text{in}(w) + \lambda w^\intercal w$$

**L1 Regularization (Lasso)**

- Penalty: $\lambda \| w\|_1 = \lambda \sum |w_i|$
- Effect: encourages sparsity (some weights become exactly zero), useful for feature selection

**Other forms**

**Dropout** (neural network) - randomly deactivates neuron during training to prevent reliance on specific features

**Early stopping** - halts training when validation error increases, limiting model complexity

**Soft margin SVM** - allows some misclassifications, adding a penalty for margin violations

**Why use regularization?**

- **Prevent overfitting** - complex model (high $d_{vc}$) fit training data perfectly ($E_\text{in} \approx 0$) but perform poorly on test data ($E_\text{out}$ high). Regularization $E_\text{in}$ slightly to reduce $E_\text{out}$

- **Improve generalization** - by lowering effective complexity, regularization tightens the VC bound, ensuring $E_\text{out} \approx E_\text{in}$

- **Handle limited data** - when N is small relative to $d_{vc}$, regularization prevents overfitting 

**Overview** - Kernel Methods

*Lecture 15* focuses on **kernel methods**, extending the Support Vector Machine (SVB) framework from *Lecture 14* to handle *non-linearly separable data* and *non-linearly cases*. The two main topics are:

- **The kernel trick** - efficiently computing high-dimensional feature mappings without explicit transformations

- **Soft-margin SVM** - allowing margin violations to handle non-separable data, a form of regularization

$\Rightarrow$ The lecture builds on Lecture 14's hard-margin SVM, which maximizes the margin for linearly separable data and introduces kernels to handle non-linear boundaries and solf margins to manage imperfect separation



**The kernel trick** 

The kernel trick allows SVMs to operate in high-dimensional (or infinite-dimensional) feature spaces without explicitly computing the transformed features, making non-linear classification computationally feasible 

**What do we need form the $\text{Z}$-space?**

Non-linearly separable data is handled by mapping $x \in \chi$ to a higher-dimensional space $z = \Phi(x) \in Z$, where a linear separator exists. The SVM dual Lagrangian is:

$$\mathcal{L}(\alpha) = \sum^N_{n=1}\alpha_n - \frac{1}{2}\sum^N_{n=1} \sum^N_{m-1} \alpha_n \alpha_m y_n y_m z_n^\intercal z_m $$

The Lagrangian depends only on **inner products** $z_n^\intercal z_m$, not the explicit $z_n$

**Explanation** - the kernel trick exploits this by replacing $z_n^\intercal z_m$ with a kernel function $K(x_n, x_m)$, computed directly in $\chi$\space

**Generalized inner product**

Define the kernel as:

$$K(x, x') = z^\intercal z' = \Phi(x)^\intercal \Phi(x') $$

For example, for $x = (x_1, x_2)$, a 2nd-order polynomial transformation:

$$z = \Phi(x) = (1, x_1, x_2, x_1^2, x_2^2, x_1x_2)$$

The inner product is 

$$K(x, x') = 1 + x_1x_1' + x_2x_2' + x_1^2x_1'^2 + x_2^2x_2'^2 + x_1x_2x_1'x_2'$$

**Explanation** - this inner product corresponding to a high-dimensional feature space but is computed using the original feature $x, x'$


**The trick**

The kernel trick computes $K(x, x')$ without transforming $x$. 

Example, for a 2nd-order polynomial kernel:

$$K(x, x') = (1 + x^\intercal x')^2 = (1 + x_1x_1' + x_2x_2)^2$$

Expanding:

$$= 1 + x_1^2x_1'^2 + x_2^2x_2'^2 + 2x_1x_2 + 2x_2x_2' + 2x_1x_2x_1'x_2'$$

This matches the inner product of 

$$z = (1, x_1^2, x_2^2, \sqrt{2}x_1, \sqrt{2}x_2, \sqrt{2}x_1x_2)$$

The kernel computes the inner product in $Z$-space efficiently, avoiding the explicit transformation $\Phi(x)$

**The polynomial kernel**

For $\chi = \R^d$, a polynomial kernel of order $Q$

$$K(x, x') = (1 + x^\intercal x')^Q$$

can adjust scale:

$$K(x, x') = (ax^\intercal x' + b)^Q$$

Example: For $d=10, Q = 100$, the kernel handles high-dimensional features efficiently

Higher $Q$ increases model flexibility but risks overfitting due to larger $d_{vc}$

**We only need $Z$ to exist**

A kernel $K(x, x')$ is valid if it corresponds to an inner product in some $Z$-space. Example: Gaussian (RBF) kernel:

$$K(x, x') = \exp(\gamma \| x - x' \|^2)$$

For 1D case:

$$K(x, x') = \exp(-(x - x')^2) = \exp(-x^2)\exp(-x'^2)\sum^\infty_{k=0}\frac{(2xx')^k}{k!}$$

This corresponds to an infinite-dimensional $Z$

The RBF kernel maps to an infinite-dimensional space, enabling complex boundaries without explicit computation

**This kernel in action**

For slightly non-separable data, the RBF kernel transforms $\chi$ to an infinite-dimensional $Z$, achieving separation. Check generalization by counting support vectors

Fewer support vectors indicate a simpler model, improving generalization

The RBF kernel's parameter $\gamma$ controls smoothness (high $\gamma$: complex boundaries, low $\gamma$: smoother), acting as a regularization knob

**Kernel formulation of SVM**

The SVM quadratic program uses a kernel matrix:

$$[y_ny_mK(x_n, x_m)]^N{n,m=1}$$

replacing $z_n^\intercal z_m$. The optimization remains:

$$\text{Maximize }\mathcal{L}(\alpha) = \sum^N_{n=1}\alpha_n - \frac{1}{2}\sum^N_{n=1} \sum^N_{m-1} \alpha_n \alpha_m y_n y_m z_n^\intercal z_m$$

subject to $\alpha_n \geq 0, \sum^N_{n=1}\alpha_n y_n = 0$

The kernel matrix enables SVM to operate in $Z$-space implicitly

The kernel increases $d_{vc}$ but the margin and sparse $\alpha_n$ (support vectors) reduce effective complexity

**The final hypothesis** 

The SVM hypothesis is:

$$g(x) = \text{sign}(w^\intercal z + b), \quad w= \sum_\text{SVs}\alpha_n y_n z_n$$

Using the kernel:

$$g(x) = \text{sign}\left(  \sum_{\alpha_n > 0} \alpha_n y_n K(x_n, x) + b\right)$$

Compute $b$ using any support vector $x_m$


$$b = y_m - \sum_{\alpha_n > 0}\alpha_n y_n K(x_n, x_m)$$

The hypothesis depends only on support vectors and kernel evaluations, making it computationally efficient


The sparsity of $\alpha_n > 0$ (few support vector is a form of implicit regularization, limitting model complexity)

**How do we know Z exists?**

Three ways to validate a kernel:

1. **Construction** - explicitly define $\Phi(x)$
2. **Mercer's condition** - mathematical properties ensure $K$ is a valid inner product
3. **Who cares?** - if the kernel works empirically, use it

Mercer’s condition is rigorous, but construction or empirical success often suffices

Valid kernels ensure the hypothesis set is well-defined, allowing the VC bound to apply

**Design the own kernel**

A kernel $K(x, x')$ is valid if:
1. **Symmetric** - $K(x, x') = K(x', x)$
2. **Positive semi-definite** - the kernel matrix:

$$[K(x_n, x_m)]^N_{n,m = 1}$$

is positive semi-definite for any $x_1, ... , x_n$ (Mercer's condition)

These conditions guarantee $K$ corresponds to an inner product in some $Z$-space

Custom kernels can be designed to control complexity (e.g., smoother kernels reduce effective $d_{vc}$)

**Soft-margin SVM**

Soft-margin SVMs handle non-separable data by allowing margin violations, introducing a regularization parameter to balance margin size and error

**Two types of non-separable** 

Non-separable data can be:

1. **Slightly non-separable** - a few points violate the margin (fixable with small adjustment)

2. **Seriously non-separable** - many points overlap, requiring significant relaxation

Solf-margin SVMs address both cases by allowing violations.

--

**Purpose of kernel methods**

The purpose of kernel methods is to enable machine learning models, particularly SVMs, to **efficiently handle non-linearly separable data** by mapping data into a higher-dimensional feature space where linear separation is possible, without the computational cost of explicitly computing the transformed features.

This allows models to learn complex, non-linear decision boundaries while maintaining computational feasibility and controlling overfitting through regularization

Kernel methods extend the hard-margin SVM and soft-margin SVM to address cases where data can not be separated by a linear hyperplane in the original input space $\chi$

Enable SVMs to classify data with non-linear patterns(e.g., circular or intertwined classes) by leveraging the kernel trick

**Main idea of kernel methods**

The main idea of kernel methods is the kernel trick, which allows SVMs to operate in a high-dimensional (or infinite-dimensional) feature space $Z$ by computing inner products between transformed data points using a kernel function $K(x, x')$, without explicitly computing the transformation $\Phi(x)$. This achieves two goals:
1. **Non-linear separation** - maps data to a space where a linear separator exists, enabling complex decision boundaries in the original space

2. **Computational efficiency** - avoids the computational burden of working in high-dimensional spaces by using kernel functions that directly compute $\Phi(x)^\intercal \Phi(x')$

So instead of transforming $x \to \Phi(x)$ and computing $z^\intercal z'$, use a kernel function $K(x, x')$ that directly gives the inner product in $Z$-space

The kernel integrates seamlessly into the SVM framework,  maintaining the same optimization structure but operating in $Z$-space implicitly

The hypothesis depends on a sparse set of support vectors, ensuring simplicity

This enables SVMs to find a linear separator in $Z$, which corresponds to a non-linear boundary in $\chi$

**Purpose**

- Handle non-linear data - for data that's not linearly separable, the kernel maps it to a space where separation is possible

- Efficiency - avoids the computational cost of high-dimensional transformation (e.g., for polynomial kernel of degree $Q$, the feature space can have the combination of $d + Q$ and $Q$ dimensions)