# Day 33: Handling Mixed Variables
*Feature Engineering — Handling Mixed Variables (CampusX)*

---

## 1. What Are Mixed Variables?

**Mixed variables** are columns that contain a combination of **numbers and text/categories** — either across different rows, or packed together inside a single value. Real-world datasets are full of these, especially anything scraped from forms, tickets, IDs, or legacy databases.

Machine learning models expect a column to be *consistently* one type — purely numeric or purely categorical. Mixed variables violate that assumption and will break or silently confuse most preprocessing steps (scaling, encoding, imputing) if left as-is.

There are **two distinct types** of mixed variables, and they need different fixes.

---

## 2. Type 1 — Mixed Types Across Different Rows

Here, the **column itself** sometimes holds a number and sometimes holds text, depending on the row.

**Example:** a `number_of_bathrooms` column where most rows are clean integers, but a few rows contain the string `"More than 3"` instead of a number.

![Mixed variables type 1](images/mixed_type1_across_rows.png)

- Rows 1, 2, 3, 5 are purely **numeric**.
- Row 4 is purely **text** ("More than 3" isn't a number pandas can read as one).
- If you don't catch this, pandas will likely infer the entire column as `dtype: object` (string), silently disabling any numeric operations on the "clean" rows too.

### How to Handle Type 1

**Strategy A — Create an indicator/flag column, then convert the rest to numeric:**

```python
import pandas as pd
import numpy as np

# 1. Flag which rows were originally non-numeric
df['bathrooms_is_more_than_3'] = df['number_of_bathrooms'].apply(
    lambda x: 1 if x == 'More than 3' else 0
)

# 2. Replace the text with a sensible numeric placeholder (e.g., 4, or the max seen)
df['number_of_bathrooms'] = df['number_of_bathrooms'].replace('More than 3', 4)

# 3. Now safely convert to numeric
df['number_of_bathrooms'] = pd.to_numeric(df['number_of_bathrooms'])
```

**Strategy B — Bin the whole thing into categories instead** (if the "More than 3" case suggests the exact count doesn't matter much beyond a point):

```python
def bucket_bathrooms(val):
    if val in ['1', '2', '3', 1, 2, 3]:
        return str(val)
    return '4+'

df['bathroom_category'] = df['number_of_bathrooms'].apply(bucket_bathrooms)
```

---

## 3. Type 2 — Mixed Types Within a Single Value

Here, **one single cell** contains both a categorical part and a numeric part glued together — most often as an alphanumeric code.

**Classic Titanic example:** the `Cabin` column has values like `C85`, `B22`, `A10` — the **letter** represents the deck (categorical), and the **number** represents the room/position on that deck (numeric).

![Mixed variables type 2](images/mixed_type2_within_value.png)

Another common real-world example: the Titanic `Ticket` column, with entries like `STON/O2 3101282` or `PC 17599` — a text prefix (carrier/class code) followed by digits.

### How to Handle Type 2

The general fix is: **split the single mixed column into two clean columns** — one purely categorical, one purely numeric — using string extraction (regex).

![Splitting pipeline](images/split_pipeline.png)

```python
import pandas as pd
import numpy as np

# Example: Cabin column like 'C85', 'B22', 'A10', 'C123'
df['Cabin_Category'] = df['Cabin'].str.extract(r'([A-Za-z]+)')   # letters only
df['Cabin_Number']   = df['Cabin'].str.extract(r'(\d+)')          # digits only
df['Cabin_Number']   = pd.to_numeric(df['Cabin_Number'])

# Cabin_Category can now be one-hot / ordinal encoded like any categorical column
# Cabin_Number can now be binned, scaled, or used as-is like any numeric column
```

For values with missing patterns (e.g., some rows have no letter, some have no digit), always inspect with:

```python
df['Cabin'].str.extract(r'([A-Za-z]+)').isnull().sum()
df['Cabin'].str.extract(r'(\d+)').isnull().sum()
```
...and decide how to impute those (e.g., a dedicated `"Unknown"` category, or the column median for the numeric part).

---

## 4. Type 1 vs Type 2 — Side-by-Side Summary

| Aspect | Type 1 (across rows) | Type 2 (within a value) |
|---|---|---|
| Where the mix happens | Different rows of the same column are entirely numeric OR entirely text | The same cell contains both a text part and a numeric part |
| Example | `number_of_bathrooms`: `1, 2, 3, "More than 3"` | `Cabin`: `"C85"`, `"B22"` |
| Fix strategy | Flag column + replace text with numeric placeholder, or bucket into categories | Regex `str.extract()` to split into 2 new columns |
| Resulting columns | 1 numeric column (+ optional flag column) | 2 new columns: 1 categorical + 1 numeric |

---

## 5. Why This Matters (Interview / Practical Notes)

- **Always check `df['col'].apply(type).value_counts()`** or `df['col'].unique()` early in EDA — mixed variables often hide silently as `dtype: object` and only surface as bugs later during scaling/encoding.
- **Don't just drop the "weird" rows.** The mixed-in category (like `"More than 3"`) is often informative, not noise — losing it can throw away signal.
- **Regex extraction is fragile to format changes.** Always validate extracted columns don't have unexpected `NaN`s from patterns the regex didn't anticipate.
- **This step usually comes right after initial data loading**, before scaling, encoding, or binning — you can't bin or scale a column that's still secretly a mix of strings and numbers.

---

### Notes on the images in this notebook

Diagrams (`images/mixed_type1_across_rows.png`, `images/mixed_type2_within_value.png`, `images/split_pipeline.png`) live in an `images/` subfolder next to this markdown file. Keep that folder alongside the `.md`/`.ipynb` file for the images to render in Jupyter's markdown preview.
