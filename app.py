"""
AIDA - AI Data Assistant
Guided preprocessing pipeline: Upload → Profile → Impute → Encode → Scale → Outliers → Balance → Evaluate → Export

Architecture:
  app.py          — Streamlit UI layer (this file)
  pipeline_utils.py — Pure ML/DS logic, independently unit-testable
  test_app.py     — pytest unit tests for core transformation functions
"""

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# All heavy ML logic lives in pipeline_utils so this file stays readable
import pipeline_utils as pu

st.set_page_config(page_title="AIDA - AI Data Assistant", page_icon="🧠", layout="wide")

# ----------------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------------
DEFAULTS = {
    "df_raw": None,
    "df_work": None,
    "filename": None,
    # Step index: 0=upload, 1=profile, 2=impute, 3=encode,
    #             4=scale, 5=outliers, 6=balance, 7=evaluate, 8=export
    "step": 0,
    "pipeline_log": [],
    "impute_done": False,
    "encode_done": False,
    "scale_done": False,
    "outlier_done": False,
    "balance_done": False,
    "evaluate_done": False,
    "target_col": None,
    # Train/test split
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "test_size": 0.2,
    # Evaluation results
    "baseline_raw": None,
    "baseline_processed": None,
    # Outlier tracking
    "outlier_summary": None,
    "outlier_action": "ignore",
    # Scaling
    "scaler_type": "none",
    # Imputation choices (for report)
    "impute_strategies_chosen": {},
    # CV comparison result
    "cv_comparison": None,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_all():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def log_step(text: str):
    st.session_state.pipeline_log.append(text)


# ----------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 AIDA")
    st.caption("AI Data Assistant")
    st.divider()

    steps = [
        "1. Upload", "2. Profile", "3. Impute", "4. Encode",
        "5. Scale", "6. Outliers", "7. Balance", "8. Evaluate", "9. Export",
    ]
    for i, s in enumerate(steps):
        if i < st.session_state.step:
            st.markdown(f"✅ {s}")
        elif i == st.session_state.step:
            st.markdown(f"**▶️ {s}**")
        else:
            st.markdown(f"⬜ {s}")

    st.divider()
    if st.session_state.df_raw is not None:
        st.caption(f"File: `{st.session_state.filename}`")
        st.caption(
            f"Shape: {st.session_state.df_work.shape[0]} rows "
            f"× {st.session_state.df_work.shape[1]} cols"
        )
        if st.session_state.target_col:
            st.caption(f"Target: `{st.session_state.target_col}`")

    if st.button("🔄 Start Over", use_container_width=True):
        reset_all()
        st.rerun()


# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------
def load_file(uploaded_file):
    name = uploaded_file.name
    if name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")
    return df, name


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def generate_pipeline_script() -> str:
    header = f'''\"""
AIDA Generated Pipeline Script
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Source file: {st.session_state.filename}

Reproduces every transformation applied in the AIDA app.
Run against the original raw file to regenerate the cleaned output.

DESIGN NOTE — Train/Test Leakage:
  SMOTE is fit on X_train ONLY. Any scaler is also fit on X_train only.
  This prevents synthetic samples or distribution statistics from the
  training set from leaking information about the test set.
\"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ---- Load data ----
df = pd.read_csv("{st.session_state.filename}")  # or pd.read_excel(...)

'''
    body = "\n\n".join(st.session_state.pipeline_log)
    footer = '''

# ---- Save result ----
df.to_csv("aida_cleaned_output.csv", index=False)
print("Pipeline complete. Cleaned shape:", df.shape)
'''
    return header + body + footer


def metric_delta_color(delta: float) -> str:
    """Return 'normal', 'inverse', or 'off' for st.metric delta_color."""
    if delta > 0:
        return "normal"
    elif delta < 0:
        return "inverse"
    return "off"


# ----------------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------------
st.header("🧠 AIDA — AI Data Assistant")
st.caption("Upload → Profile → Impute → Encode → Scale → Outliers → Balance → Evaluate → Export")
st.divider()

# ----------------------------------------------------------------------------------
# STEP 0: Upload
# ----------------------------------------------------------------------------------
if st.session_state.df_raw is None:
    st.subheader("Step 1 · Upload your dataset")
    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            df, name = load_file(uploaded)
            if df.empty:
                st.error("The uploaded file appears to be empty.")
            else:
                st.session_state.df_raw = df.copy()
                st.session_state.df_work = df.copy()
                st.session_state.filename = name
                st.session_state.step = 1
                st.rerun()
        except Exception as e:
            st.error(f"Could not read file: {e}")
    st.stop()

df = st.session_state.df_work

# ----------------------------------------------------------------------------------
# STEP 1: Profile
# ----------------------------------------------------------------------------------
st.subheader("Step 2 · Automated Profile")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Missing cells", int(df.isna().sum().sum()))
c4.metric("Duplicate rows", int(df.duplicated().sum()))

with st.expander("📋 Preview data", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("🔍 Column-level profile", expanded=True):
    profile = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().sum() / len(df) * 100).round(1),
        "unique_values": df.nunique(),
    })
    st.dataframe(profile, use_container_width=True)

    missing_cols = profile[profile["missing"] > 0]
    if not missing_cols.empty:
        st.warning(
            f"⚠️ {len(missing_cols)} column(s) contain missing values: "
            f"{', '.join(missing_cols.index.tolist())}"
        )
    else:
        st.success("✅ No missing values detected.")

if st.session_state.step == 1:
    if st.button("Continue to Imputation ➡️", type="primary"):
        st.session_state.step = 2
        st.rerun()
    st.stop()

st.divider()

# ----------------------------------------------------------------------------------
# STEP 2: Imputation
# ----------------------------------------------------------------------------------
st.subheader("Step 3 · Imputation (handle missing values)")

num_cols_missing = [c for c in pu.get_numeric_cols(df) if df[c].isna().sum() > 0]
cat_cols_missing = [c for c in pu.get_categorical_cols(df) if df[c].isna().sum() > 0]

if not st.session_state.impute_done:
    if not num_cols_missing and not cat_cols_missing:
        st.success("✅ No missing values to impute — skipping this step.")
        st.session_state.impute_done = True
        st.session_state.step = 3
        st.rerun()

    if cat_cols_missing:
        st.caption(
            f"Categorical columns with missing values use **Mode (most frequent)** automatically: "
            f"{', '.join(cat_cols_missing)}"
        )

    chosen_strategy: dict[str, str] = {}

    if num_cols_missing:
        # ---- Optional: CV-based strategy comparison ----
        all_cols_for_target = df.columns.tolist()
        with st.expander("🔬 Compare imputation strategies via cross-validation (optional)", expanded=False):
            st.info(
                "**Rigorous mode:** runs k-fold cross-validation with a RandomForest under "
                "each candidate imputation strategy (median vs mean vs KNN) and reports which "
                "one actually produces the best macro-F1 on a chosen target. "
                "The fast heuristic below is still the default — this mode is for when you want "
                "evidence, not just a rule-of-thumb."
            )
            cv_target = st.selectbox(
                "Select target column for CV comparison",
                options=["-- Select --"] + all_cols_for_target,
                key="cv_target",
            )
            if cv_target != "-- Select --":
                if st.button("Run CV comparison ▶️", key="run_cv"):
                    with st.spinner("Running cross-validation (this may take ~15 seconds)…"):
                        try:
                            cv_result = pu.compare_imputation_strategies(df, cv_target)
                            st.session_state.cv_comparison = cv_result
                        except Exception as e:
                            st.error(f"CV comparison failed: {e}")

            if st.session_state.cv_comparison is not None:
                st.write("**Cross-validation results** (macro-F1, higher is better):")
                st.dataframe(st.session_state.cv_comparison, use_container_width=True)
                best = st.session_state.cv_comparison.iloc[0]["strategy"]
                st.success(f"Best strategy by CV: **{best}**")

        # ---- Heuristic recommendations ----
        st.write("AIDA recommends a strategy per column based on distribution & missingness:")

        auto_col, _ = st.columns([1, 3])
        with auto_col:
            auto_clicked = st.button("⚡ Auto-impute everything", type="primary", use_container_width=True)

        for c in num_cols_missing:
            rec_strategy, reason = pu.recommend_numeric_strategy(df[c])
            missing_pct = df[c].isna().mean() * 100
            row_left, row_right = st.columns([2, 2])
            with row_left:
                st.markdown(f"**{c}**  \n`{missing_pct:.0f}% missing`")
            with row_right:
                options = ["Median", "Mean", "KNN (k=5)"]
                rec_label = pu.STRATEGY_LABELS[rec_strategy]
                labeled_options = [
                    f"{opt} (recommended)" if opt == rec_label else opt for opt in options
                ]
                default_idx = options.index(rec_label)
                picked = st.selectbox(
                    f"Strategy for {c}",
                    labeled_options,
                    index=default_idx,
                    key=f"impute_strategy_{c}",
                    label_visibility="collapsed",
                )
                picked_clean = picked.replace(" (recommended)", "")
                chosen_strategy[c] = picked_clean.lower().split(" ")[0]
            st.caption(f"💡 {reason}")
    else:
        auto_clicked = False

    if num_cols_missing:
        run_clicked = st.button("Run Imputation ▶️")
    else:
        run_clicked = st.button("Run Imputation ▶️", type="primary")
        auto_clicked = False

    if run_clicked or auto_clicked:
        code_lines = ["# ---- Step: Imputation ----"]
        strategies_applied = {}

        for strat_key in ("median", "mean", "knn"):
            cols_for_strat = [
                c for c in num_cols_missing
                if (chosen_strategy.get(c) == strat_key if not auto_clicked
                    else pu.recommend_numeric_strategy(df[c])[0] == strat_key)
            ]
            if not cols_for_strat:
                continue
            if strat_key == "knn":
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=5)
                df[cols_for_strat] = imputer.fit_transform(df[cols_for_strat])
                code_lines.append(
                    f"num_cols = {cols_for_strat}\n"
                    f"imputer = KNNImputer(n_neighbors=5)\n"
                    f"df[num_cols] = imputer.fit_transform(df[num_cols])"
                )
            else:
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy=strat_key)
                df[cols_for_strat] = imputer.fit_transform(df[cols_for_strat])
                code_lines.append(
                    f"num_cols = {cols_for_strat}\n"
                    f"imputer = SimpleImputer(strategy='{strat_key}')\n"
                    f"df[num_cols] = imputer.fit_transform(df[num_cols])"
                )
            for c in cols_for_strat:
                strategies_applied[c] = strat_key

        if cat_cols_missing:
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy="most_frequent")
            df[cat_cols_missing] = imputer.fit_transform(df[cat_cols_missing])
            code_lines.append(
                f"cat_cols = {cat_cols_missing}\n"
                f"imputer = SimpleImputer(strategy='most_frequent')\n"
                f"df[cat_cols] = imputer.fit_transform(df[cat_cols])"
            )
            for c in cat_cols_missing:
                strategies_applied[c] = "mode"

        log_step("\n".join(code_lines))
        st.session_state.df_work = df
        st.session_state.impute_strategies_chosen = strategies_applied
        st.session_state.impute_done = True
        st.success("Imputation complete.")
        st.rerun()
    st.stop()
else:
    st.success("✅ Imputation complete — no missing values remain.")
    st.dataframe(df.isna().sum().to_frame("missing").T, use_container_width=True)

if st.session_state.step == 2:
    if st.button("Continue to Encoding ➡️", type="primary"):
        st.session_state.step = 3
        st.rerun()
    st.stop()

st.divider()

# ----------------------------------------------------------------------------------
# STEP 3: Encoding
# ----------------------------------------------------------------------------------
st.subheader("Step 4 · Encoding (categorical → numeric)")

cat_cols = pu.get_categorical_cols(df)

if not st.session_state.encode_done:
    if not cat_cols:
        st.success("✅ No categorical columns detected — skipping encoding.")
        st.session_state.encode_done = True
        st.session_state.step = 4
        st.rerun()

    cardinality = df[cat_cols].nunique().sort_values()
    st.write("Categorical columns and their cardinality (unique value count):")
    st.dataframe(cardinality.to_frame("unique_values"), use_container_width=True)

    threshold = st.slider(
        "One-Hot Encoding threshold (max unique values). Columns above this use Label Encoding instead.",
        min_value=2, max_value=50, value=10,
    )
    st.caption("Low-cardinality columns → One-Hot Encoding. High-cardinality columns → Label Encoding.")

    if st.button("Run Encoding ▶️", type="primary"):
        code_lines = ["# ---- Step: Encoding ----"]
        onehot_cols = [c for c in cat_cols if df[c].nunique() <= threshold]
        label_cols = [c for c in cat_cols if df[c].nunique() > threshold]

        if onehot_cols:
            df = pd.get_dummies(df, columns=onehot_cols, drop_first=False)
            new_dummy_cols = [c for c in df.columns if c not in cat_cols and df[c].dtype == bool]
            df[new_dummy_cols] = df[new_dummy_cols].astype(int)
            code_lines.append(
                f"onehot_cols = {onehot_cols}\n"
                f"df = pd.get_dummies(df, columns=onehot_cols, drop_first=False)\n"
                f"bool_cols = df.select_dtypes(include='bool').columns\n"
                f"df[bool_cols] = df[bool_cols].astype(int)"
            )

        if label_cols:
            from sklearn.preprocessing import LabelEncoder
            le_code = [f"label_cols = {label_cols}", "label_encoders = {}"]
            for c in label_cols:
                le = LabelEncoder()
                df[c] = le.fit_transform(df[c].astype(str))
                le_code.append(
                    f"le = LabelEncoder()\n"
                    f"df['{c}'] = le.fit_transform(df['{c}'].astype(str))\n"
                    f"label_encoders['{c}'] = le"
                )
            code_lines.append("\n".join(le_code))

        log_step("\n".join(code_lines))
        st.session_state.df_work = df
        st.session_state.encode_done = True
        st.info(f"One-Hot encoded: {onehot_cols or 'none'}  |  Label encoded: {label_cols or 'none'}")
        st.success("Encoding complete.")
        st.rerun()
    st.stop()
else:
    st.success("✅ Encoding complete.")
    st.dataframe(df.head(10), use_container_width=True)

if st.session_state.step == 3:
    if st.button("Continue to Scaling ➡️", type="primary"):
        st.session_state.step = 4
        st.rerun()
    st.stop()

st.divider()

# ----------------------------------------------------------------------------------
# STEP 4 (NEW): Feature Scaling
# ----------------------------------------------------------------------------------
st.subheader("Step 5 · Feature Scaling")

if not st.session_state.scale_done:
    st.info(
        "**Why scaling matters:** distance-based algorithms (KNN) and gradient-descent "
        "models (Logistic Regression) are sensitive to feature magnitude. A salary column "
        "in the tens of thousands and a binary flag in {0,1} will cause those models to "
        "treat salary as overwhelmingly more important. Tree-based models (Random Forest, "
        "XGBoost) split on thresholds and are invariant to monotonic transformations — "
        "scaling won't change their predictions but won't hurt either."
    )

    scaler_choice = st.radio(
        "Choose a scaling strategy:",
        options=["none", "standard", "minmax"],
        format_func=lambda x: {
            "none": "No scaling",
            "standard": "StandardScaler — zero mean, unit variance (recommended for LR / KNN)",
            "minmax": "MinMaxScaler — maps to [0, 1] (use when bounded inputs are required)",
        }[x],
        index=0,
        horizontal=True,
    )

    # Model-aware justification
    st.caption(pu.scaling_justification(scaler_choice, "logistic_regression"))

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Apply Scaling ▶️", type="primary"):
            st.session_state.scaler_type = scaler_choice

            numeric_cols = pu.get_numeric_cols(df)
            code_lines = [f"# ---- Step: Scaling ({scaler_choice}) ----"]

            if scaler_choice != "none" and numeric_cols:
                from sklearn.preprocessing import StandardScaler, MinMaxScaler
                scaler_cls = StandardScaler if scaler_choice == "standard" else MinMaxScaler
                scaler = scaler_cls()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                code_lines.append(
                    f"# NOTE: In the train/test context the scaler is fit on X_train only\n"
                    f"# (see balance step). Here we scale the full working DataFrame for export.\n"
                    f"from sklearn.preprocessing import {scaler_cls.__name__}\n"
                    f"numeric_cols = {numeric_cols}\n"
                    f"scaler = {scaler_cls.__name__}()\n"
                    f"df[numeric_cols] = scaler.fit_transform(df[numeric_cols])"
                )
            else:
                code_lines.append("# Scaling skipped by user choice.")

            log_step("\n".join(code_lines))
            st.session_state.df_work = df
            st.session_state.scale_done = True
            st.success(f"Scaling complete ({scaler_choice}).")
            st.rerun()

    with col_b:
        if st.button("Skip Scaling ➡️"):
            log_step("# ---- Step: Scaling ----\n# Skipped by user choice.")
            st.session_state.scale_done = True
            st.session_state.step = 5
            st.rerun()
    st.stop()
else:
    st.success(f"✅ Scaling complete — {st.session_state.scaler_type}.")

if st.session_state.step == 4:
    if st.button("Continue to Outlier Detection ➡️", type="primary"):
        st.session_state.step = 5
        st.rerun()
    st.stop()

st.divider()

# ----------------------------------------------------------------------------------
# STEP 5 (NEW): Outlier Detection & Handling
# ----------------------------------------------------------------------------------
st.subheader("Step 6 · Outlier Detection & Handling")

if not st.session_state.outlier_done:
    st.info(
        "Outlier detection flags numeric values that deviate far from the bulk of the "
        "distribution. You can then cap them (Winsorize), remove the rows, or ignore. "
        "This choice is logged in the pipeline script and the final report."
    )

    col_method, col_thresh = st.columns(2)
    with col_method:
        outlier_method = st.radio(
            "Detection method:",
            options=["iqr", "zscore"],
            format_func=lambda x: {
                "iqr": "IQR (Q1 − 1.5×IQR … Q3 + 1.5×IQR) — robust, distribution-free",
                "zscore": "Z-score |z| > threshold — assumes roughly normal distribution",
            }[x],
        )
    with col_thresh:
        if outlier_method == "iqr":
            outlier_thresh = st.slider("IQR multiplier (k)", min_value=1.0, max_value=3.0, value=1.5, step=0.25)
        else:
            outlier_thresh = st.slider("Z-score threshold", min_value=1.5, max_value=4.0, value=3.0, step=0.25)

    if st.button("Scan for outliers 🔍"):
        summary = pu.detect_outliers(df, method=outlier_method, threshold=outlier_thresh)
        st.session_state.outlier_summary = summary

    if st.session_state.outlier_summary is not None:
        summary = st.session_state.outlier_summary
        total_flagged = summary["n_outliers"].sum()
        cols_affected = (summary["n_outliers"] > 0).sum()

        mc1, mc2 = st.columns(2)
        mc1.metric("Outlier instances flagged", int(total_flagged))
        mc2.metric("Columns affected", int(cols_affected))

        with st.expander("📊 Outlier detail by column", expanded=True):
            st.dataframe(summary[summary["n_outliers"] > 0], use_container_width=True)

        if total_flagged > 0:
            outlier_action = st.radio(
                "What should AIDA do with flagged outliers?",
                options=["cap", "remove", "ignore"],
                format_func=lambda x: {
                    "cap": "Cap (Winsorize) — clip values to the boundary (preserves rows, reduces extreme influence)",
                    "remove": f"Remove rows — delete {total_flagged} flagged instances ({total_flagged/len(df)*100:.1f}% of data)",
                    "ignore": "Ignore — flag them in the report but don't modify the data",
                }[x],
                index=2,
            )
        else:
            outlier_action = "ignore"
            st.success("✅ No outliers detected with the chosen settings.")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Apply & Continue ▶️", type="primary"):
                df_new, code = pu.apply_outlier_action(
                    df, summary, action=outlier_action,
                    method=outlier_method, threshold=outlier_thresh,
                )
                log_step(code)
                st.session_state.df_work = df_new
                st.session_state.outlier_action = outlier_action
                st.session_state.outlier_done = True
                st.session_state.step = 6
                st.success(f"Outlier step complete — action: {outlier_action}.")
                st.rerun()
        with col_b:
            if st.button("Skip Outliers ➡️"):
                log_step("# ---- Step: Outliers ----\n# Skipped by user choice.")
                st.session_state.outlier_done = True
                st.session_state.step = 6
                st.rerun()
    else:
        st.caption("Click 'Scan for outliers' above to see results, then choose an action.")
        if st.button("Skip Outliers ➡️"):
            log_step("# ---- Step: Outliers ----\n# Skipped by user choice.")
            st.session_state.outlier_done = True
            st.session_state.step = 6
            st.rerun()

    st.stop()
else:
    st.success(f"✅ Outlier step complete — action: {st.session_state.outlier_action}.")

if st.session_state.step == 5:
    if st.button("Continue to Balancing ➡️", type="primary"):
        st.session_state.step = 6
        st.rerun()
    st.stop()

st.divider()

# ----------------------------------------------------------------------------------
# STEP 6: Train/Test Split + Balancing (SMOTE on train only)
# ----------------------------------------------------------------------------------
st.subheader("Step 7 · Train/Test Split & Class Balancing")

df = st.session_state.df_work

if not st.session_state.balance_done:
    all_cols = df.columns.tolist()

    st.info(
        "**⚠️ Leakage note:** SMOTE synthesises new samples by interpolating between "
        "existing training points. If applied *before* splitting, synthetic points could "
        "be derived from test-set neighbours, giving an optimistic and invalid estimate "
        "of generalisation. AIDA splits first, then fits SMOTE on the training fold only. "
        "Your test set stays completely untouched."
    )

    target_col = st.selectbox(
        "Select the target column (the class you want to predict):",
        options=["-- Select --"] + all_cols,
    )

    test_size = st.slider(
        "Test set size (%)", min_value=10, max_value=40, value=20, step=5
    ) / 100.0

    if target_col != "-- Select --":
        st.session_state.target_col = target_col
        vc = df[target_col].value_counts()
        st.write("**Class distribution (pre-split):**")
        st.bar_chart(vc)

        n_unique = df[target_col].nunique()
        looks_categorical = n_unique <= max(20, int(len(df) * 0.05))
        if not looks_categorical:
            st.warning(
                f"⚠️ Target column has {n_unique} unique values — looks more like a "
                f"regression target. SMOTE applies to classification (discrete classes)."
            )

        imbalance_ratio = vc.max() / vc.min() if vc.min() > 0 else float("inf")
        st.metric("Imbalance ratio (majority / minority)", f"{imbalance_ratio:.2f}")

        if imbalance_ratio > 1.5:
            st.warning("⚠️ Class imbalance detected.")
            run_smote = st.checkbox("Run SMOTE to balance training classes", value=True)
        else:
            st.success("✅ Classes look reasonably balanced.")
            run_smote = st.checkbox("Run SMOTE anyway", value=False)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Apply & Continue ▶️", type="primary"):
                code_lines = ["# ---- Step: Train/Test Split ----"]

                # --- Raw baseline: evaluate BEFORE all preprocessing ---
                with st.spinner("Training raw baseline model…"):
                    try:
                        raw_result = pu.build_raw_baseline(
                            st.session_state.df_raw, target_col, test_size=test_size
                        )
                        st.session_state.baseline_raw = raw_result
                    except Exception as e:
                        st.warning(f"Raw baseline skipped: {e}")

                # --- Stratified split ---
                try:
                    X_train, X_test, y_train, y_test = pu.split_data(
                        df, target_col, test_size=test_size
                    )
                except ValueError as e:
                    st.error(f"Split failed (stratification needs at least 2 samples per class): {e}")
                    st.stop()

                code_lines.append(
                    f"target_col = '{target_col}'\n"
                    f"X = df.drop(columns=[target_col])\n"
                    f"y = df[target_col]\n"
                    f"X_train, X_test, y_train, y_test = train_test_split(\n"
                    f"    X, y, test_size={test_size}, random_state=42, stratify=y\n"
                    f")"
                )

                # --- Apply scaler on train-only splits ---
                if st.session_state.scaler_type != "none":
                    X_train, X_test, _ = pu.apply_scaling(
                        X_train, X_test, st.session_state.scaler_type
                    )
                    code_lines.append(
                        f"# Scaler fit on X_train only — prevents distribution leakage\n"
                        f"from sklearn.preprocessing import StandardScaler\n"
                        f"scaler = StandardScaler()\n"
                        f"X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])\n"
                        f"X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])"
                    )

                # --- SMOTE on train only ---
                if run_smote:
                    non_numeric = X_train.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
                    if non_numeric:
                        st.error(
                            f"Cannot run SMOTE: columns {non_numeric} are still non-numeric. "
                            f"Go back and encode them first."
                        )
                        st.stop()

                    try:
                        X_train, y_train, smote_code = pu.apply_smote_train_only(X_train, y_train)
                        code_lines.append(smote_code)
                        st.success(
                            f"✅ SMOTE applied to training data only — "
                            f"training shape now {X_train.shape}. "
                            f"Test set untouched ({X_test.shape[0]} rows)."
                        )
                    except Exception as e:
                        st.error(f"SMOTE failed: {e}")
                        st.stop()
                else:
                    code_lines.append("# SMOTE skipped by user choice.")

                log_step("\n".join(code_lines))

                # Persist split
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.test_size = test_size

                # Rebuild df_work from the full (non-split) processed data for export
                # (the test split is kept separate for evaluation)
                balanced_df = pd.concat([X_train, y_train], axis=1)
                st.session_state.df_work = balanced_df
                st.session_state.balance_done = True
                st.session_state.step = 7
                st.rerun()

        with col_b:
            if st.button("Skip Balancing ➡️"):
                log_step("# ---- Step: Balancing ----\n# Skipped.")
                st.session_state.balance_done = True
                st.session_state.step = 7
                st.rerun()
    st.stop()
else:
    st.success("✅ Train/test split and balancing complete.")
    if st.session_state.target_col and st.session_state.y_train is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Training set class distribution (after SMOTE if applied):")
            st.bar_chart(st.session_state.y_train.value_counts())
        with col2:
            st.caption("Test set class distribution (untouched):")
            st.bar_chart(st.session_state.y_test.value_counts())

if st.session_state.step == 6:
    if st.button("Continue to Evaluation ➡️", type="primary"):
        st.session_state.step = 7
        st.rerun()
    st.stop()

st.divider()

# ----------------------------------------------------------------------------------
# STEP 7 (NEW): Baseline Model Evaluation — Before vs After
# ----------------------------------------------------------------------------------
st.subheader("Step 8 · Baseline Model Evaluation")

if not st.session_state.evaluate_done:
    if st.session_state.X_train is None or st.session_state.target_col is None:
        st.warning(
            "⚠️ No train/test split found. Go back to the Balancing step and select a target column."
        )
        if st.button("Skip Evaluation ➡️"):
            st.session_state.evaluate_done = True
            st.session_state.step = 8
            st.rerun()
        st.stop()

    st.info(
        "AIDA trains a simple baseline classifier twice:\n\n"
        "1. **Before preprocessing** — on raw data (numeric cols only, median-imputed NaNs, no encoding/balancing)\n"
        "2. **After preprocessing** — on the fully processed training fold, evaluated on the untouched test set\n\n"
        "Metrics are macro-averaged (every class weighted equally). "
        "This matters on imbalanced data — **accuracy hides poor minority-class performance**. "
        "A model that always predicts the majority class scores high accuracy but zero recall on the minority."
    )

    if st.button("Run evaluation ▶️", type="primary"):
        with st.spinner("Training and evaluating models…"):
            # After-preprocessing evaluation
            try:
                proc_result = pu.evaluate_model(
                    st.session_state.X_train,
                    st.session_state.X_test,
                    st.session_state.y_train,
                    st.session_state.y_test,
                    label="After preprocessing",
                )
                proc_result["split_info"] = (
                    f"{len(st.session_state.X_train)} train / "
                    f"{len(st.session_state.X_test)} test rows"
                )
                st.session_state.baseline_processed = proc_result
            except Exception as e:
                st.error(f"Post-processing evaluation failed: {e}")

        st.session_state.evaluate_done = True
        st.rerun()
    st.stop()

else:
    raw = st.session_state.baseline_raw
    proc = st.session_state.baseline_processed

    if raw and proc:
        st.success("✅ Evaluation complete.")

        st.write(f"**Model:** {raw['model_name']}")
        st.caption(
            "Macro-averaged metrics — each class is weighted equally regardless of frequency."
        )

        # Comparison table
        metrics = ["precision", "recall", "f1", "roc_auc"]
        compare_data = {
            "Metric": [m.upper() for m in metrics],
            "Raw baseline": [raw[m] for m in metrics],
            "After preprocessing": [proc[m] for m in metrics],
            "Δ": [
                round(proc[m] - raw[m], 4)
                if not (np.isnan(raw[m]) or np.isnan(proc[m]))
                else "N/A"
                for m in metrics
            ],
        }
        compare_df = pd.DataFrame(compare_data)
        st.dataframe(compare_df, use_container_width=True)

        # Visual: side-by-side bar chart
        chart_data = pd.DataFrame(
            {
                "Raw baseline": [raw[m] for m in ["precision", "recall", "f1", "roc_auc"]],
                "After preprocessing": [proc[m] for m in ["precision", "recall", "f1", "roc_auc"]],
            },
            index=["Precision", "Recall", "F1", "ROC-AUC"],
        )
        st.bar_chart(chart_data)

        # Detailed reports
        with st.expander("📋 Raw baseline — full classification report"):
            st.code(raw["report"])
        with st.expander("📋 After preprocessing — full classification report"):
            st.code(proc["report"])

        with st.expander("🎓 Why these metrics?"):
            st.markdown(
                """
**Why not accuracy?**  
On an imbalanced dataset (e.g. 90% negative, 10% positive), a model that *always* predicts negative
scores 90% accuracy — but zero recall on the positive class. This is why AIDA reports:

- **Precision** — of the positive predictions, how many were correct?  
- **Recall** — of the actual positives, how many did the model find?  
- **F1** — harmonic mean of precision and recall (penalises imbalance between the two)  
- **ROC-AUC** — area under the ROC curve; a score of 0.5 is random, 1.0 is perfect.
  For multiclass, AIDA uses one-vs-rest (OVR) weighted average.

All metrics are **macro-averaged** here — each class contributes equally, which makes  
performance on rare classes visible instead of buried in the majority class.
                """
            )

    elif proc:
        st.success("✅ Post-processing evaluation complete.")
        st.write(f"F1: {proc['f1']:.4f} | ROC-AUC: {proc['roc_auc']:.4f}")
    else:
        st.warning("Evaluation results not available.")

if st.session_state.step == 7:
    if st.button("Continue to Export ➡️", type="primary"):
        st.session_state.step = 8
        st.rerun()
    st.stop()

st.divider()

# ----------------------------------------------------------------------------------
# STEP 8: Export
# ----------------------------------------------------------------------------------
st.subheader("Step 9 · Export")

df = st.session_state.df_work
st.write(f"Final processed training shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
st.dataframe(df.head(20), use_container_width=True)

# Generate report
raw_result = st.session_state.baseline_raw
proc_result = st.session_state.baseline_processed
markdown_report = pu.generate_markdown_report(
    filename=st.session_state.filename,
    df_raw=st.session_state.df_raw,
    df_processed=df,
    target_col=st.session_state.target_col,
    pipeline_log=st.session_state.pipeline_log,
    baseline_raw=raw_result,
    baseline_processed=proc_result,
    outlier_summary=st.session_state.outlier_summary,
    outlier_action=st.session_state.outlier_action,
    scaler_type=st.session_state.scaler_type,
    impute_strategies=st.session_state.impute_strategies_chosen,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.download_button(
        "⬇️ Download Clean CSV",
        data=to_csv_bytes(df),
        file_name=f"aida_cleaned_{st.session_state.filename.rsplit('.', 1)[0]}.csv",
        mime="text/csv",
        use_container_width=True,
    )
with col2:
    script = generate_pipeline_script()
    st.download_button(
        "⬇️ Download Pipeline Script (.py)",
        data=script.encode("utf-8"),
        file_name="aida_pipeline.py",
        mime="text/x-python",
        use_container_width=True,
    )
with col3:
    st.download_button(
        "⬇️ Download Data Quality Report (.md)",
        data=markdown_report.encode("utf-8"),
        file_name="aida_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

if st.session_state.X_test is not None:
    st.download_button(
        "⬇️ Download Test Set CSV (untouched — use for final evaluation)",
        data=to_csv_bytes(
            pd.concat([st.session_state.X_test, st.session_state.y_test], axis=1)
        ),
        file_name="aida_test_set.csv",
        mime="text/csv",
        use_container_width=True,
    )

with st.expander("📜 Preview generated pipeline script"):
    st.code(generate_pipeline_script(), language="python")

with st.expander("📄 Preview data quality report"):
    st.markdown(markdown_report)

st.divider()
st.caption("AIDA — AI Data Assistant. Built with Streamlit, pandas, scikit-learn, and imbalanced-learn.")
