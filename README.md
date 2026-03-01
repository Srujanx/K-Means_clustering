#  Customer Segmentation with K-Means Clustering

A machine learning project that segments **4,338 online retail customers** into actionable business groups using RFM analysis and K-Means clustering.

---

##  Overview

Raw transaction data from a UK-based e-commerce retailer is transformed into a three-dimensional **RFM (Recency · Frequency · Monetary)** feature space, cleaned, scaled, and clustered into four customer segments — each mapped to a distinct marketing strategy.

---

##  Project Structure

```
├── K-Means.ipynb          # Main notebook
├── README.md              # This file
```

---

##  Dataset

**Source:** [Online Retail Customer Clustering — Kaggle](https://www.kaggle.com/datasets/hellbuoy/online-retail-customer-clustering)

| Property | Value |
|---|---|
| Raw rows | 541,909 transactions |
| Customers after cleaning | 4,338 |
| Time period | Dec 2010 – Dec 2011 |
| Region | United Kingdom |

---

##  Pipeline

```
Raw Transactions
      │
      ▼
 Data Cleaning
 • Remove cancelled invoices (InvoiceNo starts with 'C')
 • Drop null CustomerIDs
 • Remove Quantity ≤ 0 and UnitPrice ≤ 0
      │
      ▼
 RFM Feature Engineering
 • Recency   — days since last purchase
 • Frequency — unique invoices per customer
 • Monetary  — total spend (£)
      │
      ▼
 Outlier Capping (1st / 99th percentile)
      │
      ▼
 log1p Transform → StandardScaler
      │
      ▼
 K-Means (k=4, n_init=25, random_state=42)
      │
      ▼
 Cluster Profiling & Business Naming
```

---

##  Optimal k Selection

Three validation metrics evaluated for k = 2 to 10:

| k | WCSS | Silhouette ↑ | Calinski-Harabasz ↑ |
|---|---|---|---|
| 2 | 6319.60 | **0.4362** | **4593.16** |
| 3 | 4712.48 | 0.3395 | 3818.35 |
| **4** | **3772.42** | 0.3393 | 3539.14 |
| 5 | 3155.97 | 0.3160 | 3383.66 |

> All metrics peak at k=2 statistically. **k=4 was chosen** for business interpretability — two segments is not a usable marketing strategy.

---

##  Customer Segments

| Segment | Size | % of Base | Profile |
|---|---|---|---|
|  Champions | 736 | 17.0% | High spend, recent, frequent |
|  Loyal Customers | 1,173 | 27.0% | Moderate spend, engaged |
|  At-Risk | 838 | 19.3% | Good history, going quiet |
|  Lost / Inactive | 1,591 | 36.7% | Low spend, not seen in months |

---

##  Business Actions

| Segment | Recommended Action |
|---|---|
| Champions | Loyalty rewards, early product access, referral programs |
| Loyal Customers | Upsell campaigns, membership tiers |
| At-Risk | Win-back emails, time-limited discounts |
| Lost / Inactive | Last-ditch reactivation or remove from active list |

---

##  Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-teal)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

##  Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/your-username/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
```

**3. Run the notebook**
```bash
jupyter notebook K-Means.ipynb
```

> The dataset downloads automatically via `kagglehub` — no manual setup needed.

---

##  Key Design Decisions

**Why log1p before StandardScaler?**
RFM distributions are heavily right-skewed (Pareto-like). Log compression reduces skew before standardisation so that Euclidean distance is meaningful. Outlier capping happens before the log transform — not after — to prevent extreme values from distorting the fitted scale.

**Why cap outliers instead of removing them?**
Capping at the 99th percentile preserves every customer record while neutralising the centroid-pulling effect of extreme values (e.g., one customer with £77,183 in spend).

**Why k=4 over k=2?**
Statistical metrics favour k=2, but two segments — "active" and "inactive" — have no actionable granularity. Four segments align with standard RFM marketing literature and give each group a distinct, executable CRM strategy.

**Why n_init=25?**
K-Means is sensitive to random initialisation. Running 25 independent attempts and keeping the lowest WCSS result significantly reduces the chance of converging to a local minimum.

---

##  Limitations

- K-Means assumes spherical, equally-sized clusters — retail customer distributions are neither
- Hard cluster boundaries may misassign customers near segment edges
- RFM ignores product category, geography, and demographics
- Segments should be validated with a business stakeholder before deployment

---

##  License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

**Srujan** — AI Student, Durham College  
[GitHub](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/your-profile)
