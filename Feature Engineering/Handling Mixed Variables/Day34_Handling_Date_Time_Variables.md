# Day 34: Handling Date and Time Variables
*Feature Engineering — Handling Date and Time Variables (CampusX)*

---

## 1. Why Do We Need to Engineer Date/Time Features?

A raw datetime value like `2026-07-16 14:35:02` is essentially useless to a machine learning model as-is — models understand numbers, not calendar semantics. But that single timestamp is secretly packed with a huge amount of predictive signal: the year, the month, whether it's a weekend, the hour of day, how long ago it was, and more.

**Feature engineering on dates means extracting that hidden structure into explicit numeric/categorical columns the model can actually use.**

![Datetime decomposition](images/datetime_decomposition.png)

*A single datetime column can be exploded into many separate features — year, month, day, day of week, quarter, is_weekend, hour, week of year, and more — each of which may carry independent predictive value.*

---

## 2. Extracting Date Parts (pandas `.dt` accessor)

Pandas gives you a rich `.dt` accessor once a column is converted to `datetime64`:

```python
import pandas as pd

df['date'] = pd.to_datetime(df['date'])   # always convert first!

df['year']         = df['date'].dt.year
df['month']        = df['date'].dt.month
df['day']          = df['date'].dt.day
df['day_of_week']  = df['date'].dt.dayofweek      # Monday=0 ... Sunday=6
df['day_name']     = df['date'].dt.day_name()      # 'Monday', 'Tuesday', ...
df['quarter']      = df['date'].dt.quarter
df['week_of_year']  = df['date'].dt.isocalendar().week
df['is_weekend']   = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end']   = df['date'].dt.is_month_end.astype(int)
```

## 3. Extracting Time Parts

If the column also carries a time component (not just a date):

```python
df['hour']    = df['date'].dt.hour
df['minute']  = df['date'].dt.minute
df['second']  = df['date'].dt.second

# common derived flags
df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
df['time_of_day'] = pd.cut(
    df['hour'],
    bins=[-1, 5, 11, 16, 20, 23],
    labels=['Night', 'Morning', 'Afternoon', 'Evening', 'Late Night']
)
```

---

## 4. Time-Elapsed / Duration Features

Often the raw date matters less than **how far apart two dates are** — age, tenure, days-since-last-purchase, etc.

![Time elapsed features](images/time_elapsed.png)

```python
import pandas as pd
from datetime import datetime

today = pd.Timestamp.today()

df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
df['age_years'] = (today - df['date_of_birth']).dt.days // 365

# Difference between two event columns
df['days_between'] = (df['end_date'] - df['start_date']).dt.days

# "Recency" style feature — very common in churn/RFM models
df['days_since_last_purchase'] = (today - df['last_purchase_date']).dt.days
```

---

## 5. The Big Pitfall: Naive Numeric Encoding of Cyclical Features

Calendar features like `month`, `day_of_week`, and `hour` are **cyclical** — they wrap around. December (12) is right next to January (1) in real life, but a plain numeric encoding puts them 11 units apart, which actively misleads models like linear regression or KNN that rely on numeric distance.

![Cyclical encoding](images/cyclical_encoding.png)

*Left: encoding month as a plain integer 1–12 makes December and January look maximally far apart — the exact opposite of reality. Right: sin/cos (cyclical) encoding maps each month onto a circle, so December and January end up right next to each other, correctly preserving the "wraparound" nature of time.*

### How to Implement Cyclical Encoding

```python
import numpy as np

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Drop the original raw integer column afterward — it's now redundant
df.drop(columns=['month', 'hour', 'day_of_week'], inplace=True)
```

> **Why sin AND cos, not just one?** A single sine value repeats itself twice per cycle (e.g., sin(month) gives the same value for two different months), so the model can't tell them apart. Using both sin and cos together gives every point on the circle a unique (x, y) coordinate, so no information is lost.

---

## 6. Full Worked Example

```python
import pandas as pd
import numpy as np

df['order_date'] = pd.to_datetime(df['order_date'])

# Basic parts
df['order_year']  = df['order_date'].dt.year
df['order_month'] = df['order_date'].dt.month
df['order_dow']   = df['order_date'].dt.dayofweek
df['order_hour']  = df['order_date'].dt.hour
df['is_weekend']  = df['order_dow'].isin([5, 6]).astype(int)

# Cyclical encode the periodic ones
for col, period in [('order_month', 12), ('order_dow', 7), ('order_hour', 24)]:
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)

df.drop(columns=['order_month', 'order_dow', 'order_hour'], inplace=True)

# Time-elapsed feature
df['days_since_order'] = (pd.Timestamp.today() - df['order_date']).dt.days
```

---

## 7. Common Pitfalls & Interview Notes

- **Always `pd.to_datetime()` first.** If pandas reads your date column as plain text/object, none of the `.dt` accessors will work.
- **Watch out for mixed date formats** (`DD/MM/YYYY` vs `MM/DD/YYYY`) — specify `format=` explicitly or use `dayfirst=True` to avoid silently misparsed dates.
- **Timezone awareness** — comparing a timezone-aware timestamp to a naive one will raise errors; standardize with `.dt.tz_localize()` / `.dt.tz_convert()`.
- **Don't leave raw cyclical integers in the model alongside their sin/cos versions** — that's redundant and can confuse coefficient interpretation in linear models; drop the original column after encoding.
- **Time-elapsed features can leak the future** if computed against "today" for training data collected long ago — for time-series/forecasting tasks, always compute elapsed time relative to a fixed, consistent reference point (e.g., the date the row was created), not `pd.Timestamp.today()`.
- **`day_of_week` numbering can vary by library** — pandas uses `Monday=0`, but some other tools use `Sunday=0`. Always double check.

---

### Notes on the images in this notebook

Diagrams (`images/datetime_decomposition.png`, `images/cyclical_encoding.png`, `images/time_elapsed.png`) live in an `images/` subfolder next to this markdown file. Keep that folder alongside the `.md`/`.ipynb` file for the images to render in Jupyter's markdown preview.
