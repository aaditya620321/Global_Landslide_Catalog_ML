<img width="1536" height="1024" alt="ChatGPT Image Apr 20, 2026, 03_21_10 AM" src="https://github.com/user-attachments/assets/b77b65eb-cd25-4da3-8dcd-e495cc8842fd" />

# Global Landslide Catalog — End-to-End Machine Learning Project

A clean, beginner-friendly end-to-end machine learning pipeline on the **NASA Global Landslide Catalog**. The project walks through data understanding, missing-value analysis, cleaning, outlier handling, exploratory data analysis, feature engineering, and training **8 machine learning models** to predict landslide size and fatality count.

---

## 📂 Dataset

- **Name:** Global Landslide Catalog
- **Source:** [catalog.data.gov — Global Landslide Catalog Export](https://catalog.data.gov/dataset/global-landslide-catalog-export)
- **Publisher:** NASA
- **Rows:** 11,033
- **Columns:** 31
- **File format:** CSV (encoding: `latin-1`)

The catalog records landslide events reported worldwide, with information on location, trigger, size, category, impact (fatalities / injuries), and temporal metadata.

---

## 🗂️ Dataset Structure

| # | Column | Type | Description |
|---|---|---|---|
| 1 | `source_name` | object | Name of the reporting source |
| 2 | `source_link` | object | URL to the source article |
| 3 | `event_id` | int64 | Unique event identifier |
| 4 | `event_date` | object → datetime | Date the landslide occurred |
| 5 | `event_time` | float64 | Time of event (100% missing) |
| 6 | `event_title` | object | Short title of the event |
| 7 | `event_description` | object | Free-text description |
| 8 | `location_description` | object | Free-text location |
| 9 | `location_accuracy` | object | Accuracy bucket (exact, 1km, 5km, 25km, etc.) |
| 10 | `landslide_category` | object | Type (landslide, mudslide, rock_fall, etc.) |
| 11 | `landslide_trigger` | object | What triggered the event (rain, earthquake, etc.) |
| 12 | `landslide_size` | object | Size class (small, medium, large, very_large, catastrophic) |
| 13 | `landslide_setting` | object | Geological setting |
| 14 | `fatality_count` | float64 | Number of fatalities |
| 15 | `injury_count` | float64 | Number of injuries |
| 16 | `storm_name` | object | Associated storm, if any |
| 17 | `photo_link` | object | URL to photo |
| 18 | `notes` | object | Additional notes |
| 19 | `event_import_source` | object | Import metadata |
| 20 | `event_import_id` | float64 | Import ID |
| 21 | `country_name` | object | Country where event occurred |
| 22 | `country_code` | object | ISO country code |
| 23 | `admin_division_name` | object | Sub-national region |
| 24 | `admin_division_population` | float64 | Population of that region |
| 25 | `gazeteer_closest_point` | object | Nearest named place |
| 26 | `gazeteer_distance` | float64 | Distance (km) to nearest named place |
| 27 | `submitted_date` | object → datetime | Date the record was submitted |
| 28 | `created_date` | object → datetime | Date the record was created in DB |
| 29 | `last_edited_date` | object → datetime | Date the record was last edited |
| 30 | `longitude` | float64 | Event longitude |
| 31 | `latitude` | float64 | Event latitude |

---

## 🎯 Project Objectives

### Exploratory Data Analysis (10 objectives, 10 graphs)

| # | Objective | Graph Type | Key Insight |
|---|---|---|---|
| 1 | Distribution of landslide categories | Bar plot | "landslide" dominates; rare types like topple / creep barely appear |
| 2 | Landslide trigger frequency | Bar plot | Water-based triggers (downpour, rain, monsoon) cause the vast majority |
| 3 | Fatality count distribution | Histogram | Extremely right-skewed — most events kill nobody |
| 4 | Injury count distribution | Histogram | Same skew — most events report zero injuries |
| 5 | Landslide size vs frequency | Bar plot | "medium" is most common; catastrophic events are rare |
| 6 | Country-wise landslide occurrences | Bar plot | USA leads, followed by India, Philippines, China |
| 7 | Location accuracy distribution | Bar plot | Most events pinned within 1–25 km (news-report based) |
| 8 | Distance vs landslide impact | Scatter plot | No strong relationship between gazeteer distance and fatalities |
| 9 | Time-based trend (per year) | Line plot | Sharp rise from 2007, peak around 2010–2011 |
| 10 | Population vs landslide impact | Scatter plot | No clear linear relationship |

### Modeling Objectives

- **Classification:** Predict `landslide_size` (small / medium / large / very_large) from event features
- **Regression:** Predict `fatality_count` from event features

---

## 🧹 Data Cleaning Steps

### 1. Dropped Irrelevant Columns (13 columns removed)

| Reason | Columns Dropped |
|---|---|
| High missing values (>85%) | `event_time` (100%), `notes` (97%), `storm_name` (95%), `photo_link` (86%) |
| Free text / URLs | `event_description`, `event_title`, `location_description`, `source_link`, `source_name` |
| Identifiers | `event_id`, `event_import_id`, `event_import_source`, `gazeteer_closest_point` |

### 2. Kept Important Columns
- `injury_count` and `fatality_count` were **kept** even though they had ~51% and ~13% missing values, because they are the core impact measurements of the dataset.

### 3. Missing Value Treatment

| Column Type | Strategy |
|---|---|
| Impact counts (`fatality_count`, `injury_count`) | Filled with **0** (assume unreported = zero impact) |
| Numerical columns | Filled with **median** |
| Categorical columns | Filled with **"unknown"** |
| `event_date` (unparseable rows) | **Dropped** |

### 4. Datetime Parsing
The dataset has **two mixed date formats** (`08-01-2008 00:00` and `01/19/2007 12:00:00 AM`). We handle both:
```python
def parse_dates(series):
    s1 = pd.to_datetime(series, format='%m-%d-%Y %H:%M', errors='coerce')
    mask = s1.isnull()
    s2 = pd.to_datetime(series[mask], errors='coerce')
    s1[mask] = s2
    return s1
```

### 5. Duplicate Removal
- Removed 14 exact-duplicate rows.

### 6. Final Shape
- **Before cleaning:** 11,033 rows × 31 columns
- **After cleaning:** 11,019 rows × 18 columns

---

## 📊 Outlier Handling

Impact columns are heavily zero-inflated. A pure IQR rule fails here because Q1 = Q3 = 0, which would clip every non-zero value. Instead we use **99th percentile capping (winsorization)** for:

- `fatality_count`
- `injury_count`
- `admin_division_population`
- `gazeteer_distance`

---

## 🧠 Feature Engineering

- Extracted `year` and `month` from `event_date`
- Dropped all raw datetime columns before modeling
- Encoded categorical columns using `LabelEncoder`
- Prepared two separate dataframes: one for classification, one for regression
- Merged tiny `catastrophic` class (3 rows) into `very_large` to allow stratified split

---

## 🔀 Train-Test Split

- 80 / 20 split
- `random_state = 42`
- `stratify` on the target class for classification
- `StandardScaler` applied for distance / margin-based models (KNN, SVM, Logistic Regression)

---

## 🤖 Machine Learning Models

Eight models trained with **default parameters** (no hyperparameter tuning):

| Model | Task |
|---|---|
| Linear Regression | Regression |
| Logistic Regression | Classification |
| K-Nearest Neighbors | Classification |
| Gaussian Naive Bayes | Classification |
| Support Vector Machine | Classification |
| Decision Tree | Classification |
| Random Forest | Classification |
| XGBoost (with GradientBoosting fallback) | Classification |

---

## 📈 Results

### Regression (predicting `fatality_count`)

| Metric | Value |
|---|---|
| R² Score | 0.0806 |
| MAE | 2.0323 |
| RMSE | 4.4045 |

### Classification (predicting `landslide_size`)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Random Forest** | **0.7579** | **0.7462** | **0.7579** | **0.7394** |
| Gradient Boosting / XGBoost | 0.7406 | 0.7201 | 0.7406 | 0.7222 |
| SVM | 0.7318 | 0.7085 | 0.7318 | 0.7073 |
| Logistic Regression | 0.7032 | 0.6833 | 0.7032 | 0.6720 |
| KNN | 0.7018 | 0.6799 | 0.7018 | 0.6842 |
| Naive Bayes | 0.6841 | 0.7301 | 0.6841 | 0.6656 |
| Decision Tree | 0.6732 | 0.6709 | 0.6732 | 0.6720 |

🏆 **Best Model: Random Forest** with **75.79% accuracy**

---

## 💡 Why Random Forest Performed Best

Tree-ensemble methods lead on tabular data like this because:

- They handle mixed numeric + encoded categorical features natively
- They capture non-linear interactions (e.g. *rain × hilly setting × monsoon month*) without manual feature engineering
- They are robust to outliers and don't require feature scaling
- They resist overfitting better than a single tree by averaging many trees

Linear and distance-based models (Logistic Regression, KNN, Naive Bayes, SVM) assume linearly separable or Gaussian-shaped features, which doesn't match the heavily skewed and categorical nature of this dataset.

---

## 🔑 Key Insights

- **Water is the #1 driver of landslides** — downpour, rain, continuous_rain, tropical cyclones and monsoons together account for the vast majority of events
- **Top reporting countries:** United States, India, Philippines, China — reflects both real risk and reporting effort
- **Most events cause zero fatalities and zero injuries** — a small long tail drives all the human impact
- **Reporting activity peaked around 2010–2011** — reflects catalog maintenance more than actual frequency
- **"medium" is the most common landslide size**; catastrophic events are extremely rare
<img width="1536" height="1024" alt="ChatGPT Image Apr 20, 2026, 03_37_19 AM" src="https://github.com/user-attachments/assets/ad8202be-d4e9-4b55-9164-a3c82208e278" />


---

## 🗂️ Project Structure

```
Global-Landslide-ML-Project/
│
├── Global_Landslide_ML_Project.ipynb   # Main Jupyter notebook
├── Global_Landslide_Catalog.csv        # Dataset (downloaded from data.gov)
└── README.md                           # This file
```

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/Global-Landslide-ML-Project.git
cd Global-Landslide-ML-Project
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib scikit-learn xgboost jupyter
```

### 3. Download the dataset
Download the CSV from [catalog.data.gov](https://catalog.data.gov/dataset/global-landslide-catalog-export) and place it in the project folder as `Global_Landslide_Catalog.csv`.

### 4. Launch the notebook
```bash
jupyter notebook Global_Landslide_ML_Project.ipynb
```

Run all cells top to bottom.

> **Note:** If `xgboost` is not installed, the notebook automatically falls back to `GradientBoostingClassifier` — no code changes needed.

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas** — data manipulation
- **NumPy** — numerical computing
- **Matplotlib** — visualization (default plots, no styling)
- **scikit-learn** — ML models and metrics
- **XGBoost** — gradient boosting (optional)
- **Jupyter Notebook** — interactive environment

---

## 📝 Final Learnings

- Real-world catalogs are **messy** — missing values, mixed date formats, free-text columns, long-tailed targets. Cleaning is the biggest part of the job.
- For skewed count data, **IQR capping is too aggressive** — percentile capping works better.
- For imbalanced multi-class targets, **stratified splits and weighted metrics** matter — naive accuracy hides poor performance on rare classes.
- **Tree ensembles with default parameters** are a strong first baseline for any tabular ML problem.

---

## 📄 License

This project uses the Global Landslide Catalog dataset, which is publicly available through data.gov under NASA's open data policy.

---

## 🙋 Author

Feel free to fork, improve, or raise issues!
