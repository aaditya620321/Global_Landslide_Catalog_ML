"""
Global Landslide Catalog - End-to-End ML Portfolio Project
===========================================================
Notebook-style Python script that covers:
1) Data loading from a GitHub raw CSV URL
2) Data understanding and quality checks
3) EDA before cleaning
4) Outlier detection (IQR + Z-score)
5) Data cleaning
6) EDA after cleaning
7) Transformation (encoding + scaling + feature engineering)
8) Train/test split
9) Supervised learning (classification/regression auto-mode)
10) Unsupervised learning (KMeans, Hierarchical, DBSCAN)
11) Model comparison
12) Predictions and diagnostics
13) Insights and conclusion output

Author: Data Science Portfolio Template
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import zscore
from sklearn.base import clone
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titleweight"] = "bold"

# ------------------------------
# CONFIG
# ------------------------------
RAW_DATA_URL = (
    "https://raw.githubusercontent.com/"
    "<your-username>/Global_Landslide_Catalog_ML/main/"
    "Global%20Landslide%20Catalog.csv"
)
LOCAL_FALLBACK_PATH = "Global Landslide Catalog.csv"
TARGET_COLUMN = "landslide_category"
RANDOM_STATE = 42
TEST_SIZE = 0.2
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------
# UTILITY HELPERS
# ------------------------------
def section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def save_show(fig_name: str) -> None:
    """Save and show a plot with consistent naming."""
    out = FIG_DIR / f"{fig_name}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {out}")


# ------------------------------
# STEP 1: DATA LOADING
# ------------------------------
def load_dataset(raw_url: str, local_fallback: Optional[str] = None) -> pd.DataFrame:
    """Load CSV from GitHub raw URL; fallback to local file if URL fails."""
    try:
        df = pd.read_csv(raw_url)
        print(f"Loaded dataset from URL: {raw_url}")
        return df
    except Exception as exc:
        print(f"[Warning] Could not load from URL: {exc}")
        if local_fallback:
            print(f"Loading fallback local file: {local_fallback}")
            return pd.read_csv(local_fallback, encoding="latin1")
        raise


def initial_overview(df: pd.DataFrame) -> None:
    section("Step 1 — Data Loading: Preview + Structure")
    print("First 5 rows:\n", df.head())
    print("\nLast 5 rows:\n", df.tail())
    print(f"\nShape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nData types:\n", df.dtypes)
    print("\nInfo:")
    print(df.info())
    print("\nDescribe (numeric):\n", df.describe(include=[np.number]).T)
    print("\nDescribe (categorical):\n", df.describe(include=["object", "category"]).T)


# ------------------------------
# STEP 2: DATA UNDERSTANDING
# ------------------------------
def feature_summary(df: pd.DataFrame, target: str) -> pd.DataFrame:
    section("Step 2 — Data Understanding")
    dtype_map = df.dtypes.reset_index()
    dtype_map.columns = ["feature", "dtype"]
    dtype_map["type_group"] = np.where(
        dtype_map["dtype"].astype(str).str.contains("int|float"),
        "numerical",
        "categorical_or_text",
    )
    print("Feature overview:\n", dtype_map)
    print(f"\nTarget variable: {target}")

    missing = df.isna().sum().to_frame("missing_count")
    missing["missing_pct"] = (missing["missing_count"] / len(df)) * 100
    missing = missing.sort_values("missing_pct", ascending=False)
    print("\nMissing values (count + %):\n", missing)

    dup_count = df.duplicated().sum()
    print(f"\nDuplicate rows: {dup_count}")
    return dtype_map


# ------------------------------
# STEP 3: EDA BEFORE CLEANING
# ------------------------------
def eda_before_cleaning(df: pd.DataFrame, target: str) -> None:
    section("Step 3 — EDA BEFORE CLEANING")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    # Histograms + KDE
    if num_cols:
        df[num_cols].hist(bins=30, layout=(int(np.ceil(len(num_cols) / 3)), 3), figsize=(16, 4 * int(np.ceil(len(num_cols) / 3))))
        plt.suptitle("Numeric Feature Distributions (Before Cleaning)", y=1.02)
        save_show("before_histograms")

    # Boxplots
    for col in num_cols[:8]:
        plt.figure()
        sns.boxplot(x=df[col], color=sns.color_palette("Set2")[0])
        plt.title(f"Boxplot: {col} (Before Cleaning)")
        plt.xlabel(col)
        save_show(f"before_box_{col}")

    # Pairplot
    if len(num_cols) >= 3:
        sample_cols = num_cols[:4]
        pair_df = df[sample_cols + ([target] if target in df.columns else [])].dropna().sample(
            min(1000, len(df.dropna())), random_state=RANDOM_STATE
        )
        sns.pairplot(pair_df, hue=target if target in pair_df.columns else None, corner=True, palette="viridis")
        plt.suptitle("Pairplot (Sample, Before Cleaning)", y=1.02)
        save_show("before_pairplot")

    # Correlation heatmap
    if len(num_cols) >= 2:
        plt.figure(figsize=(12, 8))
        corr = df[num_cols].corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.4)
        plt.title("Correlation Heatmap (Before Cleaning)")
        save_show("before_corr_heatmap")

    # Countplots (top categorical)
    for col in cat_cols[:5]:
        top_n = df[col].astype(str).value_counts().head(10)
        plt.figure(figsize=(10, 5))
        sns.countplot(y=df[col], order=top_n.index, palette="mako")
        plt.title(f"Top Categories: {col} (Before Cleaning)")
        plt.xlabel("Count")
        plt.ylabel(col)
        save_show(f"before_count_{col}")

    # Scatter plots feature vs target (if target numeric)
    if target in df.columns and np.issubdtype(df[target].dtype, np.number):
        for col in num_cols[:6]:
            if col == target:
                continue
            plt.figure()
            sns.scatterplot(data=df, x=col, y=target, alpha=0.7, color=sns.color_palette("deep")[2])
            plt.title(f"{col} vs {target} (Before Cleaning)")
            save_show(f"before_scatter_{col}_vs_{target}")


# ------------------------------
# STEP 4: OUTLIER DETECTION
# ------------------------------
def detect_outliers(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    section("Step 4 — Outlier Detection (IQR + Z-score)")
    num_df = df.select_dtypes(include=np.number).copy()
    iqr_outliers = {}
    z_outliers = {}

    for col in num_df.columns:
        q1, q3 = num_df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            iqr_outliers[col] = 0
        else:
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            iqr_outliers[col] = int(((num_df[col] < lower) | (num_df[col] > upper)).sum())

        z = np.abs(zscore(num_df[col].fillna(num_df[col].median())))
        z_outliers[col] = int((z > 3).sum())

    out_iqr = pd.DataFrame.from_dict(iqr_outliers, orient="index", columns=["iqr_outlier_count"])
    out_z = pd.DataFrame.from_dict(z_outliers, orient="index", columns=["zscore_outlier_count"])
    out = out_iqr.join(out_z)
    print(out.sort_values(["iqr_outlier_count", "zscore_outlier_count"], ascending=False))

    cols_with_outliers = out[(out["iqr_outlier_count"] > 0) | (out["zscore_outlier_count"] > 0)].index.tolist()
    print("\nColumns with outliers:", cols_with_outliers)

    # Outlier visualization (top columns)
    for col in cols_with_outliers[:6]:
        plt.figure()
        sns.boxplot(x=df[col], color=sns.color_palette("Set3")[2])
        plt.title(f"Outlier Boxplot: {col}")
        save_show(f"outlier_box_{col}")

        plt.figure()
        sns.scatterplot(x=np.arange(len(df)), y=df[col], alpha=0.7, color=sns.color_palette("rocket")[3])
        plt.title(f"Outlier Scatter: {col}")
        plt.xlabel("Row index")
        plt.ylabel(col)
        save_show(f"outlier_scatter_{col}")

    return iqr_outliers, z_outliers


# ------------------------------
# STEP 5: DATA CLEANING
# ------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    section("Step 5 — Data Cleaning")
    cleaned = df.copy()

    # Fix common datetime fields
    for dt_col in ["event_date", "submitted_date", "created_date", "last_edited_date"]:
        if dt_col in cleaned.columns:
            cleaned[dt_col] = pd.to_datetime(cleaned[dt_col], errors="coerce")

    # Missing values
    num_cols = cleaned.select_dtypes(include=np.number).columns
    cat_cols = cleaned.select_dtypes(exclude=np.number).columns

    for col in num_cols:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    for col in cat_cols:
        cleaned[col] = cleaned[col].fillna(cleaned[col].mode().iloc[0] if not cleaned[col].mode().empty else "Unknown")

    # Capping outliers (winsorization-like with IQR bounds)
    for col in num_cols:
        q1, q3 = cleaned[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        cleaned[col] = cleaned[col].clip(lower, upper)

    # Remove duplicates
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    after = len(cleaned)
    print(f"Dropped duplicate rows: {before - after}")
    print("Cleaned shape:", cleaned.shape)
    return cleaned


# ------------------------------
# STEP 6: EDA AFTER CLEANING
# ------------------------------
def eda_after_cleaning(before_df: pd.DataFrame, after_df: pd.DataFrame, target: str) -> None:
    section("Step 6 — EDA AFTER CLEANING + Before/After Comparison")
    num_cols = after_df.select_dtypes(include=np.number).columns.tolist()

    for col in num_cols[:6]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(before_df[col], kde=True, ax=axes[0], color=sns.color_palette("Blues")[3])
        axes[0].set_title(f"Before Cleaning: {col}")
        sns.histplot(after_df[col], kde=True, ax=axes[1], color=sns.color_palette("Greens")[3])
        axes[1].set_title(f"After Cleaning: {col}")
        fig.suptitle(f"Before vs After Distribution — {col}")
        save_show(f"compare_before_after_{col}")

    if target in after_df.columns:
        print("Target distribution after cleaning:\n", after_df[target].value_counts(normalize=True).head(20))


# ------------------------------
# STEP 7: TRANSFORMATION
# ------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Simple date-based feature engineering."""
    engineered = df.copy()
    if "event_date" in engineered.columns and pd.api.types.is_datetime64_any_dtype(engineered["event_date"]):
        engineered["event_year"] = engineered["event_date"].dt.year
        engineered["event_month"] = engineered["event_date"].dt.month
        engineered["event_day"] = engineered["event_date"].dt.day
    return engineered


def prepare_features(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    section("Step 7 — Data Transformation (Encoding + Scaling + Engineering)")
    df = feature_engineering(df)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    print("Numerical features:", num_cols)
    print("Categorical features:", cat_cols)
    print("\nTransformed dataset preview:\n", df.head())
    return X, y, num_cols, cat_cols


# ------------------------------
# STEP 8 + 9: TRAIN/TEST + SUPERVISED MODELS
# ------------------------------
def is_classification_target(y: pd.Series) -> bool:
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        return True
    return y.nunique() < 20


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipe, num_cols), ("cat", categorical_pipe, cat_cols)]
    )


def evaluate_classification(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
) -> pd.DataFrame:
    section("Step 9 — Supervised Learning (Classification)")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    # Optional XGBoost
    try:
        from xgboost import XGBClassifier  # type: ignore

        models["XGBoost"] = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")
    except Exception:
        print("XGBoost not installed; skipping.")

    rows = []
    for name, model in models.items():
        pipe = Pipeline(steps=[("prep", clone(preprocessor)), ("model", model)])

        if name == "Naive Bayes":
            # GaussianNB needs dense matrix
            Xt_train = preprocessor.fit_transform(X_train)
            Xt_test = preprocessor.transform(X_test)
            Xt_train = Xt_train.toarray() if hasattr(Xt_train, "toarray") else Xt_train
            Xt_test = Xt_test.toarray() if hasattr(Xt_test, "toarray") else Xt_test
            model.fit(Xt_train, y_train)
            pred = model.predict(Xt_test)
            prob = model.predict_proba(Xt_test)[:, 1] if len(np.unique(y_train)) == 2 else None
        else:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe[-1], "predict_proba") and len(np.unique(y_train)) == 2 else None

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
        auc = roc_auc_score(y_test, prob) if prob is not None else np.nan

        rows.append(
            {
                "model": name,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": auc,
            }
        )

        # Confusion matrix heatmap
        plt.figure(figsize=(6, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Blues", colorbar=False)
        plt.title(f"Confusion Matrix — {name}")
        save_show(f"cm_{name.lower().replace(' ', '_')}")

        # ROC curve for binary
        if prob is not None:
            fpr, tpr, _ = roc_curve(y_test, prob)
            plt.figure()
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=sns.color_palette("deep")[0])
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve — {name}")
            plt.legend()
            save_show(f"roc_{name.lower().replace(' ', '_')}")

    result = pd.DataFrame(rows).sort_values("f1", ascending=False)
    print("\nClassification report table:\n", result)
    return result


def evaluate_regression(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
) -> pd.DataFrame:
    section("Step 9 — Supervised Learning (Regression)")

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(random_state=RANDOM_STATE),
        "SVR": SVR(),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    rows = []
    for name, model in models.items():
        pipe = Pipeline(steps=[("prep", clone(preprocessor)), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred)
        rows.append({"model": name, "r2": r2, "mae": mae, "mse": mse, "rmse": rmse})

    result = pd.DataFrame(rows).sort_values("r2", ascending=False)
    print("\nRegression report table:\n", result)
    return result


# ------------------------------
# STEP 10: UNSUPERVISED LEARNING
# ------------------------------
def unsupervised_learning(X: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> None:
    section("Step 10 — Unsupervised Learning")

    prep = build_preprocessor(num_cols, cat_cols)
    Xt = prep.fit_transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X2 = pca.fit_transform(Xt)

    # Elbow method
    inertias = []
    ks = range(2, 11)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        km.fit(X2)
        inertias.append(km.inertia_)

    plt.figure()
    plt.plot(list(ks), inertias, marker="o", color=sns.color_palette("deep")[3])
    plt.title("Elbow Method for K-Means")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    save_show("kmeans_elbow")

    # KMeans cluster plot
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20)
    km_labels = kmeans.fit_predict(X2)
    plt.figure()
    sns.scatterplot(x=X2[:, 0], y=X2[:, 1], hue=km_labels, palette="tab10", s=40)
    plt.title("K-Means Clusters (2D PCA Projection)")
    save_show("kmeans_clusters")

    # Hierarchical clustering dendrogram
    linked = linkage(X2[:1500], method="ward")  # capped for readability
    plt.figure(figsize=(12, 5))
    dendrogram(linked, truncate_mode="lastp", p=30, show_leaf_counts=False)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Cluster index")
    plt.ylabel("Distance")
    save_show("hierarchical_dendrogram")

    # DBSCAN cluster plot
    db = DBSCAN(eps=0.6, min_samples=8)
    db_labels = db.fit_predict(X2)
    plt.figure()
    sns.scatterplot(x=X2[:, 0], y=X2[:, 1], hue=db_labels, palette="tab20", s=40)
    plt.title("DBSCAN Clusters (2D PCA Projection)")
    save_show("dbscan_clusters")


# ------------------------------
# STEP 11 + 12 + 13 + 14
# ------------------------------
def model_comparison_plot(result_df: pd.DataFrame, problem_type: str) -> None:
    section("Step 11 — Model Comparison")
    metric = "f1" if problem_type == "classification" else "r2"
    plt.figure(figsize=(12, 5))
    sns.barplot(data=result_df, x="model", y=metric, palette="viridis")
    plt.xticks(rotation=35, ha="right")
    plt.title(f"Model Comparison ({metric.upper()})")
    plt.ylabel(metric.upper())
    save_show("model_comparison")
    print("\nBest model:\n", result_df.iloc[0])


def prediction_diagnostics(
    best_model_name: str,
    result_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    problem_type: str,
) -> None:
    section("Step 12 + 13 — Predictions & Prediction Visualizations")

    if problem_type == "classification":
        model_map = {
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(probability=True, random_state=RANDOM_STATE),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        }

        model = model_map.get(best_model_name, RandomForestClassifier(random_state=RANDOM_STATE))
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        compare = pd.DataFrame({"actual": y_test.values, "predicted": preds})
        print("Sample predictions vs actual:\n", compare.head(20))
        print("\nClassification report:\n", classification_report(y_test, preds, zero_division=0))

    else:
        model_map = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(random_state=RANDOM_STATE),
            "Lasso": Lasso(random_state=RANDOM_STATE),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "Random Forest Regressor": RandomForestRegressor(random_state=RANDOM_STATE),
            "SVR": SVR(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
        }
        model = model_map.get(best_model_name, RandomForestRegressor(random_state=RANDOM_STATE))
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        compare = pd.DataFrame({"actual": y_test.values, "predicted": preds})
        print("Sample predictions vs actual:\n", compare.head(20))

        # Actual vs predicted
        plt.figure()
        sns.scatterplot(x=y_test, y=preds, color=sns.color_palette("deep")[1])
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        plt.title("Actual vs Predicted")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        save_show("actual_vs_predicted")

        # Residuals
        residuals = y_test - preds
        plt.figure()
        sns.scatterplot(x=preds, y=residuals, color=sns.color_palette("deep")[4])
        plt.axhline(0, color="red", linestyle="--")
        plt.title("Residual Plot")
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        save_show("residual_plot")


def print_conclusion(result_df: pd.DataFrame, problem_type: str) -> None:
    section("Step 14 — Insights & Conclusion")
    best = result_df.iloc[0]
    print("Key findings:")
    print("- Cleaning reduced noise and handled missing / duplicate / outlier issues.")
    print("- Transformed features enabled fair model benchmarking.")
    print(f"- Best {problem_type} model: {best['model']}")
    if problem_type == "classification":
        print(f"- Best weighted F1: {best['f1']:.4f}")
    else:
        print(f"- Best R²: {best['r2']:.4f}")
    print("- Next improvements: hyperparameter tuning, SHAP explainability, model monitoring pipeline.")


# ------------------------------
# MAIN ORCHESTRATION
# ------------------------------
def run_project() -> None:
    df = load_dataset(RAW_DATA_URL, LOCAL_FALLBACK_PATH)

    initial_overview(df)
    feature_summary(df, TARGET_COLUMN)
    eda_before_cleaning(df, TARGET_COLUMN)
    detect_outliers(df)

    cleaned = clean_data(df)
    eda_after_cleaning(df, cleaned, TARGET_COLUMN)

    X, y, num_cols, cat_cols = prepare_features(cleaned, TARGET_COLUMN)

    section("Step 8 — Train/Test Split")
    problem_type = "classification" if is_classification_target(y) else "regression"
    stratify = y if problem_type == "classification" and y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    print(f"Problem type detected: {problem_type}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    preprocessor = build_preprocessor(num_cols, cat_cols)

    if problem_type == "classification":
        result_df = evaluate_classification(X_train, X_test, y_train, y_test, preprocessor)
    else:
        result_df = evaluate_regression(X_train, X_test, y_train, y_test, preprocessor)

    unsupervised_learning(X, num_cols, cat_cols)
    model_comparison_plot(result_df, problem_type)

    best_model_name = result_df.iloc[0]["model"]
    prediction_diagnostics(
        best_model_name,
        result_df,
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        problem_type,
    )
    print_conclusion(result_df, problem_type)


if __name__ == "__main__":
    run_project()
