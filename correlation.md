# **Pearson's Correlation**

One of the most widely used measures of linear relationship between two variable.

**What is Pearson's Correlation?**

Pearson's correlation coefficient ($r$) quantifies the **strength** and **direction** of the **linear relationship** between two continuos variables.

**Range** $-1 \leq r \leq 1$

- $r = 1 \rightarrow$ Perfect positive linear relationship
- $r = -1 \rightarrow$ Perfect negative linear relationship
- $r = 0 \rightarrow$ No linear relationship 

**Assumptions**
- Both variables are continuous and normally distributed
- The relationship is linear
- Equal variance across the range

**Formula for Pearson's $r$**

The formula for Pearson's correlation is 

$$r = \frac{\sum (X- \bar{X})(Y - \bar{Y})}{\sqrt{\sum(X-\bar{X})^2 (Y - \bar{Y})^2}}$$

where

- $X, Y$ = Data points
- $\bar{X}, \bar{Y}$ = Means of $X$ and $Y$

**Simpler explanation**

Pearson's $r$ is the covariance of $X$ and $Y$ divided by the product of their std (standard deviations)

$$r = \frac{\operatorname{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

**Example Calculation**

Given 

$X = [1, 2, 3, 4 ,5]$

$Y = [2, 4, 6, 8, 10]$

**Step 1** - Compute means $\bar{X} = 3, \bar{Y} = 6$

**Step 2** - Compute deviations $(X - \bar{X})$ and $(Y - \bar{Y})$

**Step 3** - Multiply and sum them

$$\sum (X - \bar{X})(Y - \bar{Y}) = (1- 3)(2-6) + (2-3)(4-6)+ ... = 20$$

**Step 4** - Compute squared deviations

$$\sum (X - \bar{X})^2 = 10, \space \space \sum(Y - \bar{Y})^2 = 40$$

**Step 5** - Plug into formula

$$r = \frac{20}{\sqrt{10 \times 40}} = 1$$

**Limitation of Pearson's $r$**
- **Only measures linear relationships** (misses non-linear patterns)
- Sensitive to outliers
- Does not imply causation (correlation $\neq$ causation)

**When to use Pearson's $r$**
- Comparing two continuous variables
- Testing for linear trends

**When not to use Pearson's $r$**
- If data is ordinal or categorical (use **Spearman's rank correlation** instead)
- If the relationship is non-linear (consider **polynomial regression**)

---- 

# **Spearman’s Rank Correlation**

Spearman's rank correlation, often called Spearman's rho (denoted as $\rho$ or $r_s$), is a statistical measure that assesses how well the relationship between two variables can be described using a *monotonic* function without assuming that the relationship is perfectly linear.

> A *monotonic* function is a function that consistently increase or decreases across its entire domain, meaning it's either **non-decreasing** or **non-increasing**
> - **Non-decreasing** (or **monotonically increasing**) - if $x_1 < x_2$ then $f(x_1) \leq f(x_2)$
> - **Non-increasing** (or **monotonically decreasing**) - if $x_1 < x_2$ then $f(x_1) \geq f(x_2)$

Unlike Pearson's correlation (which measures linear relationships and requires numerical data that's normally distributed), Spearman's is *non-parametric*. This means it works with ranked data $\rightarrow$ make it more flexible $\rightarrow$ can handle ordinal data (like rankings or categories with an order) and does not care if the data follows a normal distribution.

**When use it?**

Imagine that two guys are judging a baking contest, the first one ranks 10 cakes based on taste (1st, 2nd, 3rd, ... ) and the other one ranks them based on appearance.

The question is "Do the rankings tend to agree"? 

$\Rightarrow$ Use Spearman's rank correlation for
- Ranked or ordinal data (e.g., good, better, best)
- Situations where the relationship is not necessarily a straight line but still trends up or down
- Data with outliers [Spearman's rank correlation is less sensitive to extreme values than Pearson's]

**How does it work?**

Given the dataset

<center>

| **Superhero** | **Strength Rank** | **Speed Rank** | 
| --- | --- | --- |
| Superman (1) | 1 | 2 |
| Hulk (2) | 2 | 4 |
| Flash (3) | 3 | 1 |
| Thor (4) | 4 | 3 |
| Batman (5) | 5 | 5 |
 
</center>

1 _ The data ranked (1 = strongest/fastest, 5 = weakest / slowest)

2 _ Find the difference in ranks $(d)$ by for each pair, subtract the ranks and call it $(d)$
- (1): 1 - 2 = -1
- (2): 2 - 4 = -2
- (3): 3 - 1 = 2
- (4): 4 - 3 = 1
- (5): 5 - 5 = 0

3 _ Square the difference $(d^2)$ $\rightarrow$ it helps remove negative signs
- (1): 1
- (2): 4 
- (3): 4
- (4): 1
- (5): 0

4 _ Sum the squared differences $1 + 4 + 4 + 1 + 0 = 10$

5 _ Plug into the formula _ The Spearman's rank correlation coefficient is calculated as

$$r_s = 1 - \frac{6\sum d^2}{n(n^2 - 1)}$$

where
- $\sum d^2 = 10$
- $n = 5$ (number of samples)

So 

$$r_s = 1 - \frac{6 \times 10}{5(5^2 - 1)}= 0.5$$

This means there's a moderate positive correlation between strength and speed rankings. As strength rank increases (get worse), speed rank tends to increase too, but not perfectly.

**Note**
- $r_s = 1 \rightarrow$ Perfect positive monotonic relationship (as one rank increases, the other does too, exactly).

- $r_s = -1 \rightarrow$ Perfect negative monotonic relationship (as one increases, the other decreases, exactly).

- $r_s = 0 \rightarrow$ No consistent monotonic relationship.

**Ties** - if 2 items have the same rank (e.g. two cakes tied for 3rd) $\rightarrow$ assign them the average rank (e.g., 3.5 each). The formula still works though it gets slightly more complex.

**Significance** - to know if $r_s = 0.5$ is statistically meaningful but still need a significance test which is depends on sample size.

--- 

**Spearman vs. Pearson - Summary**

<center>


| **Aspect** | **Spearman’s ρ** |	**Pearson’s r** |
| --- | --- | --- |
| **Relationship Type** | Monotonic (linear or nonlinear) | Strictly linear |
| **Data Type** | Ordinal or continuous | Continuous |
| **Assumptions** | Non-parametric |	Normality, linearity |
| **Outliers** | Robust | Sensitive |

</center>

---

# **Cramér’s V Correlation**

Cramér’s V is a measure of association between two *categorical* variables. It tells how strongly the categories of one variable are related to the categories of another.

Think of it as a way to quantify how much knowing the value of one variable can predict the value of the other - perfect or data like *survey responses*, *eye colors* or favorite pizza toppings.

It's an **extension** of the chi-square ($\chi^2$) test which checks if two categorical variables are independent. 

While chi-square ($\chi^2$) test $\rightarrow$ if there's a relationship, Cramér’s V $\rightarrow$ how strong that relationship is. 

It's also normalized so its value always falls between 0 and 1 $\rightarrow$ easy to interpret.

**When use it?**

Given the case of analyzing a survey:
- Variable 1 - Favorite season (Spring, Summer, Fall, Winter)
- Variable 2 - Preferred activity (Hiking, Swimming, Skiing, Reading)

If running the chi-square test and find the variables are not independent $\rightarrow$ season and activity are related. But how strong is that link? $\rightarrow$ Use Cramér’s V and it's ideal for:

- Nominal data (categories with no inherent order such as colors or types of pets)
- Contingency tables (grids showing how often each combination of categories occurs)

**How does it work?**

Given the dataset - A survey of 100 people asks about pet ownership (Dog, Cat, None) and happiness (Happy, Neutral, Unhappy) and the contingency table is

<center>

| | **Happy** | **Neutral** | **Unhappy** | **Row total** |
| --- | --- | --- | --- | --- |
| **Dog** | 20 | 10 | 5 | 35 |
| **Cat** | 15 | 15 | 5 | 35 |
| **None** | 5 | 10 | 15 | 30 |
| **Column Total** | 40 | 35 | 25 | 100 |

</center>

1 _ Calculate the chi-square statistic ($\chi^2$) to see if pet ownership and happiness are related. This involves:

- **Expected counts** - if the variables were independent, what would the table look like? (e.g., Expected Dogs & Happy = $\frac{35 \times 40}{100} = 14$)
- Compare observed vs expected and  sum up the differences $\chi^2 = \sum \frac{(O - E)^2}{E} = 19.64$

2 _ Plug into Cramér’s V formula

$$V = \sqrt{\frac{\chi^2}{n(k-1)}}$$

where
- $\chi^2 = 19.64$ 
- $n = 100$ (total number of observations)
- $k = $ the smaller of the number of rows or columns (In the dataset, rows = 3 $\rightarrow$ Dog, Cat, None and columns = 3 $\rightarrow$ Happy, Neutral, Unhappy) $\rightarrow k = 3$ 

So 

$$V = \sqrt{\frac{19.64}{100 \times(3-1)}} \approx 0.313$$

$\Rightarrow V = 0.313$ $\rightarrow$ There's a moderate association between pet ownership and happiness. It's not super strong but it's not negligible either.

**Note** 
$V = 0 \rightarrow$ No association (the variables are independent).

$V = 1 \rightarrow$ Perfect association (knowing one variable perfectly predicts the other).

Rough guide (depends on context):
- ( 0.1 ): Weak

- ( 0.3 ): Moderate

- ( 0.5+ ): Strong

**Table Size** - For a 2x2 table, Cramér’s V is equivalent to the phi coefficient ($\phi$) and the formula simplifies. For larger tables, $k - 1$ adjusts for the degrees of freedom.

**Bias Correction** - Some versions of Cramér’s V include a small-sample correction but the basic formula works well for decent-sized datasets.

**Compare to Spearman’s**
- Spearman's rank correlation is for *ranked/ordinal* data and measures monotonic relationships.
- Cramér’s V is for *nominal* data and measure association strength, regardless of order.


