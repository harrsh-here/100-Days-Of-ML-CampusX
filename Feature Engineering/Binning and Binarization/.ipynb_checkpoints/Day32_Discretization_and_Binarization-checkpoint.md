# Day 32: Discretization (Binning) & Binarization
*Feature Engineering — Binning and Binarization*

---

## 1. What is Discretization (Binning)?

**Discretization (or Binning)** is the preprocessing technique of converting continuous numerical variables (like `Age` or `Fare`) into discrete categorical intervals (like `Low`, `Medium`, `High`).

### The Big Picture Intuition

Instead of forcing a machine learning model to study tiny, highly precise, and often noisy differences (e.g., the difference in survival between a passenger who is 22.4 vs 22.6 years old), binning groups them together. It tells the model:

> "Treat everyone between 18 and 30 as one single group."

This filters out background mathematical noise, helping models focus on robust patterns rather than memorizing decimal-level coincidences.

---

## 2. Why Do We Use Binning? (The Core Reasons)

### Reason 1: To Handle Outliers (Capping)

Extreme outliers can distort models like Linear/Logistic Regression because their optimization math tries to accommodate every single point, including the extreme ones.

- **The Problem:** A single passenger paying a massive ticket `Fare` of ₹512 on the Titanic pulls a linear model's decision boundary out of alignment, even though almost everyone else paid under ₹50.
- **How Binning Fixes It:** Setting up bins (e.g., `0–20 → Low`, `21–100 → Medium`, `101+ → High`) forces that ₹512 value into the `High` bin alongside a ₹150 value. The outlier is safely contained, preventing it from warping the model's coefficients.

![Outlier capping effect](images/outlier_capping.png)

*Left: the raw Fare values, with one point completely dominating the scale. Right: after binning, that same passenger simply lands in the "High" category — the model no longer has to stretch its decision boundary to accommodate one extreme number.*

### Reason 2: To Improve Value Spread (Smoothing)

When data is heavily clumped in one area with massive empty gaps elsewhere, models struggle to remain stable.

- **The Problem:** If you have 10 passengers with ages `[22.1, 22.3, 22.5, 22.8, 23.0, 23.1, 23.2, 23.4, 71.0, 72.0]`, there is a massive gap between `23.4` and `71.0`. A model will struggle to make stable predictions in that empty zone.
- **How Binning Fixes It:** By forcing these into 3 equal-frequency bins:

| Bin | Values |
|---|---|
| Bin 0 (Young) | 22.1, 22.3, 22.5, 22.8 |
| Bin 1 (Middle) | 23.0, 23.1, 23.2 |
| Bin 2 (Older) | 23.4, 71.0, 72.0 |

> **Note:** `23.4` ends up in the "Older" bin because equal-frequency binning focuses strictly on equal data **counts** per bin rather than even spacing. This evens out the spread so the model has balanced data across all categories.

### Reason 3: To Prevent Overfitting

High-precision decimal numbers encourage decision trees to make hyper-specific splits (e.g., `Age <= 23.37`). Binning acts as a **regularizer**, forcing the model to learn stable, generalized categories instead of memorizing specific noisy decimals that won't generalize to new data.

### Reason 4: To Model Non-Linear Relations with Linear Models

Linear models can only draw straight lines. If survival rates go up for children, down for young adults, and up again for seniors, a single continuous feature cannot be mapped effectively by a straight line. Slicing `Age` into discrete categories allows a linear model to assign a unique coefficient to each category, capturing complex, non-linear curves that a single slope never could.

---

## 3. The 3 Types of Unsupervised Binning (The Nitish Framework)

In **unsupervised binning**, the algorithm splits the data without looking at your target output column (`y`). It evaluates only the feature distribution (`X`).

### I. Equal Width (Uniform) Binning

- **The Logic:** Divides the entire range of data from minimum to maximum into `n` intervals of completely equal **width**.
- **The Math:** `bin_width = (max - min) / n`
- **Distribution Example:** Best suited for a **Uniform Distribution**, where data points are spread out evenly like a flat line across their range (e.g., rolling a fair die multiple times).
- **Major Pitfall:** If there is a massive outlier, the "Maximum" value skyrockets, forcing almost all normal data points into the very first bin, while the remaining bins sit completely empty.

![Equal width binning](images/equal_width_binning.png)

*Notice the bin boundaries (orange dashed lines) are all exactly the same width apart — but because the underlying data isn't perfectly uniform, the bins can still end up holding very different numbers of points.*

### II. Equal Frequency (Quantile) Binning

- **The Logic:** Divides the data such that every single bin contains roughly the **same number of data points** (rows).
- **Distribution Example:** Best suited for highly **Skewed Distributions** (like `Fare` or `Income` data, where many people are clustered at the low end and a few are at the high end).
- **How it looks:** The bin *widths* are completely uneven. Dense areas get very narrow bins, while sparse areas get very wide bins.

![Equal frequency binning](images/equal_frequency_binning.png)

*Each of the 5 bins holds about 20% of the rows. Notice how narrow Bin 0–2 are (data is dense there) compared to how wide Bin 4 is (data is sparse out in the tail).*

### III. K-Means Binning

- **The Logic:** Runs a 1D clustering algorithm (K-Means) on your numerical column. It groups data points that are mathematically close to one another into clusters.
- **How it works:** Centroids are found for each cluster, and boundaries are drawn halfway between adjacent centroids.
- **Distribution Example:** Best suited for **Multimodal (Bimodal) Distributions**, where your data naturally has multiple distinct peaks or clusters (e.g., geyser eruption intervals).

![K-means binning](images/kmeans_binning.png)

*The two blue lines mark the discovered centroids; the orange dashed line is the boundary K-Means draws exactly halfway between them — this is very different from equal-width or equal-frequency binning, which would ignore the natural gap between the two humps.*

---

## 4. Custom/Domain-Based Binning & Binarization

### Custom / Domain Binning

- **The Logic:** Real-world domain logic determines the boundaries, not an algorithm. Use this when you (or domain experts) already know meaningful cutoffs — e.g., legal adulthood, retirement age, tax brackets.
- **How to implement:** Use Pandas' built-in `pd.cut()` function to manually set the cuts:

```python
import pandas as pd

bins = [0, 12, 19, 59, 100]
labels = ['Child', 'Teenager', 'Adult', 'Senior']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
```

### Binarization

- **The Logic:** An extreme case of binning where you create exactly **2 bins** (`0` or `1`) based on a single threshold.
- **How to implement:** Use Scikit-Learn's `Binarizer(threshold=...)`.
- **Example:** On the Titanic, determining if someone is traveling alone:

```python
from sklearn.preprocessing import Binarizer

# family_size = SibSp + Parch
df['is_alone'] = Binarizer(threshold=0.0).fit_transform(df[['family_size']])
# 0 -> travelling with family, 1 -> travelling alone (family_size > 0)
```

---

## 5. How to Feed Binned Data to Your Models

Once data is binned, you must **encode** the categorical bins because models only understand numbers.

- **One-Hot Encoding (`encode='onehot'`)**
  - Creates a new binary column for each bin.
  - **Best for:** Linear models, as it prevents the model from assuming a mathematical scale between categories (e.g., that "High" is mathematically "3x" as important as "Low").

- **Ordinal Encoding (`encode='ordinal'`)**
  - Keeps a single column and outputs the raw bin index numbers (`0, 1, 2, ...`).
  - **Best for:** Decision Trees and Random Forests. Trees can split directly on ordinal integers, keeping your dataset compact without adding extra columns.

### Scikit-Learn Implementation

```python
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer

# KBinsDiscretizer handles both binning and encoding in one step
age_binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

preprocessor = ColumnTransformer([
    ('bin_age', age_binner, ['Age'])
], remainder='passthrough')
```

**Key `KBinsDiscretizer` parameters at a glance:**

| Parameter | Options | What it controls |
|---|---|---|
| `n_bins` | integer | How many bins to create |
| `strategy` | `'uniform'`, `'quantile'`, `'kmeans'` | Which of the 3 binning types above to use |
| `encode` | `'onehot'`, `'onehot-dense'`, `'ordinal'` | How the resulting bins are represented as numbers |

---

## 6. Real-World Datasets for Practice

To practice binning, load these datasets in your Jupyter environment:

- **`diabetes.csv`**
  - Practice Column: `Insulin` (heavily skewed). Apply `KBinsDiscretizer` with `strategy='quantile'` to bin it into `Normal`, `Elevated`, and `High` insulin categories.

- **`marketing_campaign.csv`**
  - Practice Column: `Income` (contains heavy outliers). Use `KBinsDiscretizer` to bin income into salary tiers to stabilize your models.

---

## 7. Model vs. Binning Decision Map

| Model Type | Does it need Binning? | Why? | Best Encoding to Use |
|---|---|---|---|
| Linear / Logistic Regression | Yes (Highly Recommended) | To handle extreme outliers and capture non-linear relationships. | `encode='onehot'` |
| Decision Trees / Random Forests | Optional (For speed/regularization) | Helps prevent overfitting on noisy decimal values. | `encode='ordinal'` |
| Naive Bayes | Yes | Works natively with categorical probabilities. | `encode='onehot'` or `'ordinal'` |
| Deep Learning (Neural Nets) | Yes (Occasionally) | Helps networks identify boundaries faster. | `encode='onehot'` |

---

## 8. Quick Reference: Choosing a Binning Strategy

```
Is your data roughly uniform across its range?
 ├─ Yes → Equal Width Binning (strategy='uniform')
 └─ No
     ├─ Is it heavily skewed (long tail, e.g. Income, Fare)?
     │    └─ Yes → Equal Frequency Binning (strategy='quantile')
     └─ Does it have multiple distinct "humps" (multimodal)?
          └─ Yes → K-Means Binning (strategy='kmeans')
```

---

### Notes on the images in this notebook

The diagrams above (`images/equal_width_binning.png`, `images/equal_frequency_binning.png`, `images/kmeans_binning.png`, `images/outlier_capping.png`) are stored in an `images/` subfolder next to this markdown file. As long as that folder travels together with the `.md`/`.ipynb` file (e.g., you keep them zipped together or in the same repo folder), the images will render both in Jupyter's markdown preview and on platforms like GitHub.
