"""
pipeline_utils.py — Core logic for AIDA (AI Data Assistant)

Separated from app.py so that:
  1. Functions can be unit-tested without importing Streamlit.
  2. The Streamlit UI layer stays thin and readable.
  3. Reviewers can evaluate the ML logic independently of the UI.

Design notes:
  - All functions are pure or near-pure (no Streamlit calls, minimal side effects).
  - Evaluation functions deliberately avoid data leakage by taking pre-split
    X_train/X_test rather than a single DataFrame.
  - SMOTE is always fit on training data only — the wrapper enforces this.
"""

from __future__ import annotations

import io
import textwrap
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# ---------------------------------------------------------------------------
# Types / constants
# ---------------------------------------------------------------------------

ImputeStrategy = Literal["median", "mean", "knn"]
ScalerType = Literal["standard", "minmax", "none"]
OutlierMethod = Literal["iqr", "zscore"]
OutlierAction = Literal["cap", "remove", "ignore"]

STRATEGY_LABELS: dict[str, str] = {
    "median": "Median",
    "mean": "Mean",
    "knn": "KNN (k=5)",
}


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


# ---------------------------------------------------------------------------
# Imputation recommendation (heuristic)
# ---------------------------------------------------------------------------

def recommend_numeric_strategy(series: pd.Series) -> tuple[ImputeStrategy, str]:
    """
    Heuristic recommendation for a single numeric column's imputation strategy.

    Returns (strategy, reason) where strategy in {"median", "mean", "knn"}.

    Logic:
      - Too few observations → median (can't trust skew estimate)
      - High missingness (>35%) → median (KNN degrades badly; mean gets pulled by outliers)
      - Skewed distribution (|skew| > 1.0) → median (robust to extreme values)
      - Roughly symmetric + low missingness → mean
      - Moderate skew → median as safe default
    """
    s = series.dropna()
    missing_pct = series.isna().mean() * 100

    if len(s) < 8:
        return "median", "too few observed values to assess distribution reliably"

    try:
        sk = float(scipy_skew(s))
    except Exception:
        sk = 0.0

    if missing_pct > 35:
        return (
            "median",
            f"missingness is high ({missing_pct:.0f}%) — a robust simple stat is safer than KNN",
        )

    if abs(sk) > 1.0:
        return (
            "median",
            f"distribution is skewed (skew={sk:.2f}) — median resists outliers better",
        )

    if abs(sk) <= 0.5 and missing_pct <= 25:
        return "mean", f"distribution is roughly symmetric (skew={sk:.2f})"

    return (
        "median",
        f"distribution is moderately skewed (skew={sk:.2f}) — median is the safer default",
    )


# ---------------------------------------------------------------------------
# Cross-validated imputation comparison
# ---------------------------------------------------------------------------

def compare_imputation_strategies(
    df: pd.DataFrame,
    target_col: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    For each candidate imputation strategy (median, mean, knn), run k-fold
    cross-validation on a RandomForest and return mean CV scores.

    Only numeric columns with missing values are imputed; non-missing numerics
    are passed through as-is. Non-numeric columns are dropped for this
    comparison (they haven't been encoded yet at the point this may be called).

    Returns a DataFrame with columns ["strategy", "cv_f1_macro", "cv_roc_auc"].
    """
    numeric_cols = get_numeric_cols(df)
    if target_col in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != target_col]

    # Work with numeric features only for the comparison
    X_raw = df[numeric_cols].copy()
    y = df[target_col].copy()

    # Drop rows where target is missing
    mask = y.notna()
    X_raw = X_raw[mask]
    y = y[mask]

    # Encode target if needed
    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)

    results = []
    for strategy in ("median", "mean", "knn"):
        if strategy == "knn":
            imp = KNNImputer(n_neighbors=5)
        else:
            imp = SimpleImputer(strategy=strategy)

        X_imp = pd.DataFrame(imp.fit_transform(X_raw), columns=X_raw.columns)

        n_classes = y.nunique()
        scoring = "f1_macro"
        f1_scores = cross_val_score(model, X_imp, y, cv=cv, scoring=scoring)

        roc_scoring = "roc_auc" if n_classes == 2 else "roc_auc_ovr_weighted"
        try:
            roc_scores = cross_val_score(model, X_imp, y, cv=cv, scoring=roc_scoring)
            mean_roc = float(np.mean(roc_scores))
        except Exception:
            mean_roc = float("nan")

        results.append(
            {
                "strategy": strategy,
                "cv_f1_macro": round(float(np.mean(f1_scores)), 4),
                "cv_roc_auc": round(mean_roc, 4),
            }
        )

    return pd.DataFrame(results).sort_values("cv_f1_macro", ascending=False)


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split. Returns (X_train, X_test, y_train, y_test).

    Stratification is on the target so class proportions are preserved in both
    splits — important for imbalanced datasets where a random split could give
    the test set a very different class distribution.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def detect_outliers(
    df: pd.DataFrame,
    method: OutlierMethod = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Flag outliers in numeric columns using IQR or Z-score.

    Returns a DataFrame with columns: ['column', 'n_outliers', 'pct_outliers', 'method'].
    """
    numeric_cols = get_numeric_cols(df)
    records = []

    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 4:
            continue

        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            mask = (df[col] < lower) | (df[col] > upper)
        else:  # zscore
            z = (df[col] - s.mean()) / s.std()
            mask = z.abs() > threshold

        n_out = int(mask.sum())
        records.append(
            {
                "column": col,
                "n_outliers": n_out,
                "pct_outliers": round(n_out / len(df) * 100, 2),
                "method": method,
                "lower_bound": round(lower if method == "iqr" else float("nan"), 4),
                "upper_bound": round(upper if method == "iqr" else float("nan"), 4),
            }
        )

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["column", "n_outliers", "pct_outliers", "method", "lower_bound", "upper_bound"]
    )


def apply_outlier_action(
    df: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    action: OutlierAction,
    method: OutlierMethod = "iqr",
    threshold: float = 1.5,
) -> tuple[pd.DataFrame, str]:
    """
    Apply cap, remove, or ignore to outliers detected by `detect_outliers`.

    Returns (modified_df, code_snippet).
    """
    df = df.copy()
    code_lines = [f"# ---- Step: Outlier handling ({action}, method={method}) ----"]

    if action == "ignore":
        code_lines.append("# Outliers detected but no action taken (user chose to ignore).")
        return df, "\n".join(code_lines)

    cols_with_outliers = outlier_summary[outlier_summary["n_outliers"] > 0]["column"].tolist()

    if action == "remove":
        keep_mask = pd.Series(True, index=df.index)
        for col in cols_with_outliers:
            s = df[col].dropna()
            if method == "iqr":
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                keep_mask &= (df[col] >= lower) & (df[col] <= upper)
            else:
                z = (df[col] - s.mean()) / s.std()
                keep_mask &= z.abs() <= threshold

        rows_before = len(df)
        df = df[keep_mask].reset_index(drop=True)
        code_lines.append(
            f"# Removed {rows_before - len(df)} rows containing outliers in: {cols_with_outliers}"
        )

    elif action == "cap":
        for col in cols_with_outliers:
            s = df[col].dropna()
            if method == "iqr":
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
            else:
                lower = s.mean() - threshold * s.std()
                upper = s.mean() + threshold * s.std()
            df[col] = df[col].clip(lower=lower, upper=upper)
            code_lines.append(
                f"df['{col}'] = df['{col}'].clip(lower={lower:.4f}, upper={upper:.4f})"
            )

    return df, "\n".join(code_lines)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def apply_scaling(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: ScalerType,
) -> tuple[pd.DataFrame, pd.DataFrame, object | None]:
    """
    Fit scaler on training data, transform both train and test.

    Returns (X_train_scaled, X_test_scaled, fitted_scaler).
    Scaler is None when scaler_type == "none".

    Critical: fitting on X_train only prevents test-set leakage of
    distribution statistics (mean, std, min, max).
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    if scaler_type == "none" or not numeric_cols:
        return X_train.copy(), X_test.copy(), None

    scaler_cls = StandardScaler if scaler_type == "standard" else MinMaxScaler
    scaler = scaler_cls()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, scaler


def scaling_justification(scaler_type: ScalerType, model_type: str) -> str:
    """Return a human-readable justification for the chosen scaler + model pairing."""
    if scaler_type == "none":
        if model_type in ("random_forest", "tree"):
            return (
                "✅ No scaling needed — tree-based models (Random Forest, Decision Trees) "
                "split on thresholds and are invariant to monotonic feature transformations. "
                "Scaling won't change their predictions."
            )
        return (
            "⚠️ No scaling applied. If your downstream model is distance-based (KNN) or "
            "uses gradient descent (Logistic Regression, Neural Nets), unscaled features "
            "with different magnitudes will cause those models to underperform."
        )
    elif scaler_type == "standard":
        if model_type in ("logistic_regression", "knn", "svm"):
            return (
                "✅ StandardScaler is the right choice here — Logistic Regression uses "
                "gradient descent and KNN uses Euclidean distance. Both assume features "
                "are on comparable scales. Scaling to zero mean / unit variance prevents "
                "large-magnitude features from dominating."
            )
        return (
            "ℹ️ StandardScaler applied. For tree-based models this has no effect on "
            "predictions, but it won't hurt and keeps the pipeline consistent."
        )
    else:  # minmax
        return (
            "ℹ️ MinMaxScaler applied — maps all features to [0, 1]. "
            "Useful when you need bounded inputs (e.g. neural networks, some distance metrics). "
            "Note: MinMaxScaler is sensitive to outliers; if you have extreme values, "
            "StandardScaler or capping outliers first is preferable."
        )


# ---------------------------------------------------------------------------
# SMOTE — train-only wrapper
# ---------------------------------------------------------------------------

def apply_smote_train_only(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Apply SMOTE to training data only.

    Returns (X_train_resampled, y_train_resampled, code_snippet).

    Design decision: SMOTE synthesises new minority-class samples by
    interpolating between existing training points. If applied before
    splitting, synthetic points derived from test-set neighbours would
    end up in the training set, giving an optimistic — and invalid —
    estimate of generalisation performance. Fitting on X_train only
    avoids this leakage entirely.
    """
    from imblearn.over_sampling import SMOTE  # lazy import so imblearn is optional

    non_numeric = X_train.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"SMOTE requires all features to be numeric. "
            f"Non-numeric columns remaining: {non_numeric}. "
            f"Complete the Encoding step first."
        )

    min_class_count = int(y_train.value_counts().min())
    k_neighbors = max(1, min(5, min_class_count - 1))

    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)

    code = textwrap.dedent(f"""
        # ---- Step: SMOTE (training data only — no leakage) ----
        from imblearn.over_sampling import SMOTE

        # IMPORTANT: SMOTE is fit on X_train ONLY.
        # Applying SMOTE before splitting would let synthetic samples derived
        # from test-set neighbours leak into training, producing optimistic
        # (invalid) generalisation estimates. Fit on train → transform train.
        smote = SMOTE(random_state=42, k_neighbors={k_neighbors})
        X_train, y_train = smote.fit_resample(X_train, y_train)
    """).strip()

    return X_res, y_res, code


# ---------------------------------------------------------------------------
# Baseline model evaluation
# ---------------------------------------------------------------------------

def _choose_model(n_classes: int, n_samples: int):
    """
    Heuristic model selection:
      - Binary with enough samples: LogisticRegression (interpretable, fast)
      - Multiclass or small dataset: RandomForestClassifier (handles multiclass
        natively, robust to scale differences, no convergence issues)
    """
    if n_classes == 2 and n_samples >= 200:
        return (
            LogisticRegression(max_iter=1000, random_state=42),
            "logistic_regression",
            "LogisticRegression (binary target, sufficient samples)",
        )
    return (
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "random_forest",
        f"RandomForestClassifier ({'multiclass' if n_classes > 2 else 'small dataset'})",
    )


def evaluate_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    label: str = "model",
) -> dict:
    """
    Train a baseline classifier and evaluate on the test set.

    Returns a dict with keys:
      label, model_name, precision, recall, f1, roc_auc, report

    Uses macro-averaged precision/recall/F1 (treats all classes equally —
    appropriate for imbalanced problems where accuracy hides poor minority
    class performance).
    """
    # Encode targets if non-numeric
    le = None
    if not pd.api.types.is_numeric_dtype(y_train):
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train.astype(str))
        y_test_enc = le.transform(y_test.astype(str))
    else:
        y_train_enc = y_train.values
        y_test_enc = y_test.values

    n_classes = len(np.unique(y_train_enc))
    model, model_type, model_name = _choose_model(n_classes, len(X_train))

    # Impute any residual NaNs (raw baseline path may have them)
    imp = SimpleImputer(strategy="median")
    X_tr = pd.DataFrame(imp.fit_transform(X_train.select_dtypes(include=[np.number])),
                        columns=X_train.select_dtypes(include=[np.number]).columns)
    X_te = pd.DataFrame(imp.transform(X_test.select_dtypes(include=[np.number])),
                        columns=X_test.select_dtypes(include=[np.number]).columns)

    model.fit(X_tr, y_train_enc)
    y_pred = model.predict(X_te)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_enc, y_pred, average="macro", zero_division=0
    )

    # ROC-AUC: binary → standard, multiclass → OVR weighted
    try:
        if n_classes == 2:
            y_prob = model.predict_proba(X_te)[:, 1]
            roc_auc = float(roc_auc_score(y_test_enc, y_prob))
        else:
            y_prob = model.predict_proba(X_te)
            roc_auc = float(roc_auc_score(y_test_enc, y_prob, multi_class="ovr", average="weighted"))
    except Exception:
        roc_auc = float("nan")

    report = classification_report(y_test_enc, y_pred, zero_division=0)

    return {
        "label": label,
        "model_name": model_name,
        "model_type": model_type,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(roc_auc, 4),
        "report": report,
    }


def build_raw_baseline(
    df_raw: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Evaluate a model on the raw data with minimal preprocessing:
      - Drop rows where target is NaN
      - Keep numeric columns only (categorical can't be passed to sklearn without encoding)
      - Impute residual NaNs with median (so the model can train; this is intentionally
        minimal to show the 'before' baseline)

    Returns the dict from evaluate_model, plus 'split_info'.
    """
    df = df_raw.copy()
    df = df[df[target_col].notna()]

    numeric_cols = [c for c in get_numeric_cols(df) if c != target_col]
    X = df[numeric_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    result = evaluate_model(X_train, X_test, y_train, y_test, label="Raw baseline")
    result["split_info"] = f"{len(X_train)} train / {len(X_test)} test rows (numeric cols only, median-imputed NaNs)"
    return result


# ---------------------------------------------------------------------------
# Report generation (Markdown)
# ---------------------------------------------------------------------------

def generate_markdown_report(
    filename: str,
    df_raw: pd.DataFrame,
    df_processed: pd.DataFrame,
    target_col: Optional[str],
    pipeline_log: list[str],
    baseline_raw: Optional[dict],
    baseline_processed: Optional[dict],
    outlier_summary: Optional[pd.DataFrame],
    outlier_action: str,
    scaler_type: str,
    impute_strategies: Optional[dict],
) -> str:
    """
    Generate a comprehensive Markdown report documenting the entire pipeline.

    This is the artifact you'd attach to a portfolio or CV writeup — it proves
    the cleaning helped (before/after model comparison) and documents every
    decision made and why.
    """
    now_str = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# AIDA — Data Quality & Modeling Report",
        f"\n**Generated:** {now_str}  ",
        f"**Source file:** `{filename}`\n",
        "---\n",
    ]

    # 1. Profile summary
    lines += [
        "## 1. Dataset Profile\n",
        f"| Metric | Raw | Processed |",
        f"|--------|-----|-----------|",
        f"| Rows | {df_raw.shape[0]} | {df_processed.shape[0]} |",
        f"| Columns | {df_raw.shape[1]} | {df_processed.shape[1]} |",
        f"| Missing cells | {int(df_raw.isna().sum().sum())} | {int(df_processed.isna().sum().sum())} |",
        f"| Duplicate rows | {int(df_raw.duplicated().sum())} | {int(df_processed.duplicated().sum())} |",
        "",
    ]

    # 2. Missingness
    missing_raw = df_raw.isna().sum()
    missing_cols = missing_raw[missing_raw > 0]
    lines.append("## 2. Missingness Handling\n")
    if missing_cols.empty:
        lines.append("No missing values were present in the raw data.\n")
    else:
        lines.append("Columns with missing values in the raw data:\n")
        lines.append("| Column | Missing (raw) | Strategy chosen | Reason |")
        lines.append("|--------|--------------|-----------------|--------|")
        for col in missing_cols.index:
            pct = f"{missing_raw[col] / len(df_raw) * 100:.1f}%"
            if impute_strategies and col in impute_strategies:
                strat = impute_strategies[col]
                _, reason = recommend_numeric_strategy(df_raw[col])
            else:
                strat = "mode (auto)"
                reason = "categorical column — mode is the standard default"
            lines.append(f"| `{col}` | {missing_raw[col]} ({pct}) | {strat} | {reason} |")
        lines.append("")

    # 3. Encoding
    lines += [
        "## 3. Encoding Decisions\n",
        "Encoding log from pipeline:\n",
    ]
    encode_log = [l for l in pipeline_log if "Encoding" in l]
    if encode_log:
        lines.append("```python")
        lines.append(encode_log[0])
        lines.append("```\n")
    else:
        lines.append("No encoding step logged (no categorical columns or step skipped).\n")

    # 4. Outliers
    lines.append("## 4. Outlier Handling\n")
    if outlier_summary is not None and not outlier_summary.empty:
        n_flagged = outlier_summary["n_outliers"].sum()
        lines.append(f"**Total outlier instances flagged:** {n_flagged}  ")
        lines.append(f"**Action taken:** {outlier_action}\n")
        lines.append("| Column | Outliers | % of rows |")
        lines.append("|--------|----------|-----------|")
        for _, row in outlier_summary.iterrows():
            if row["n_outliers"] > 0:
                lines.append(f"| `{row['column']}` | {row['n_outliers']} | {row['pct_outliers']}% |")
        lines.append("")
    else:
        lines.append("No outlier detection was run, or no outliers were found.\n")

    # 5. Scaling
    lines += [
        "## 5. Feature Scaling\n",
        f"**Scaler applied:** {scaler_type}\n",
        scaling_justification(scaler_type, "logistic_regression"),
        "",
    ]

    # 6. Class balance
    if target_col and target_col in df_raw.columns:
        lines.append("## 6. Class Balance\n")
        raw_vc = df_raw[target_col].value_counts()
        raw_ratio = raw_vc.max() / raw_vc.min() if raw_vc.min() > 0 else float("inf")
        lines += [
            "**Before balancing:**\n",
            "| Class | Count |",
            "|-------|-------|",
        ]
        for cls, cnt in raw_vc.items():
            lines.append(f"| {cls} | {cnt} |")
        lines.append(f"\nImbalance ratio: **{raw_ratio:.2f}x**\n")

        if target_col in df_processed.columns:
            proc_vc = df_processed[target_col].value_counts()
            proc_ratio = proc_vc.max() / proc_vc.min() if proc_vc.min() > 0 else float("inf")
            lines += [
                "**After balancing:**\n",
                "| Class | Count |",
                "|-------|-------|",
            ]
            for cls, cnt in proc_vc.items():
                lines.append(f"| {cls} | {cnt} |")
            lines.append(f"\nImbalance ratio: **{proc_ratio:.2f}x**\n")

    # 7. Before/after model comparison
    lines.append("## 7. Baseline Model Comparison\n")
    if baseline_raw and baseline_processed:
        lines += [
            f"Model used: **{baseline_raw['model_name']}**  ",
            "Evaluation metrics are macro-averaged (treats all classes equally — "
            "appropriate for imbalanced data where accuracy misleads).\n",
            "| Metric | Raw baseline | After preprocessing | Δ |",
            "|--------|-------------|---------------------|---|",
        ]
        for metric in ("precision", "recall", "f1", "roc_auc"):
            raw_val = baseline_raw[metric]
            proc_val = baseline_processed[metric]
            delta = proc_val - raw_val if not (np.isnan(raw_val) or np.isnan(proc_val)) else float("nan")
            delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}" if not np.isnan(delta) else "N/A"
            lines.append(
                f"| {metric.upper()} | {raw_val:.4f} | {proc_val:.4f} | {delta_str} |"
            )
        lines += [
            "",
            f"**Raw baseline split:** {baseline_raw.get('split_info', 'N/A')}  ",
            f"**Processed split:** {baseline_processed.get('split_info', 'N/A')}\n",
            "> **Note on leakage prevention:** SMOTE was applied to the training fold only. "
            "The test set contains real, unaugmented samples. Evaluating on a clean test set "
            "gives a valid estimate of how the model would perform on new, unseen data.\n",
        ]
    else:
        lines.append("Model comparison not available — target column not selected or evaluation was skipped.\n")

    # 8. Pipeline steps
    lines += [
        "## 8. Reproducible Pipeline\n",
        "The following steps were applied (auto-generated from in-app choices):\n",
        "```python",
        "\n\n".join(pipeline_log) if pipeline_log else "# No steps logged",
        "```\n",
    ]

    lines += [
        "---",
        "_Report generated by AIDA — AI Data Assistant_",
    ]

    return "\n".join(lines)
