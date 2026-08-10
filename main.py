from sklearn import pipeline
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from catboost import CatBoostClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# Read the YAML file 
def load_config(config_path="config.yaml"):
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

# Load the three CSV files expected in the data folder
def load_data():
    data_dir = Path("data")
    development = pd.read_csv(data_dir / "dev.csv")
    evaluation = pd.read_csv(data_dir / "eval.csv")
    sample_submission = pd.read_csv(data_dir / "submission.csv")
    return development, evaluation, sample_submission

# Basic checks
def validate_data(development, evaluation, config):
    target = config["columns"]["target"]
    id_candidates = config["columns"]["id_candidates"]
    if target not in development.columns:
        raise ValueError(f"Missing target column in development data: {target}")
    if target in evaluation.columns:
        raise ValueError(f"Evaluation data must not contain the target column: {target}")

    dev_id = next((col for col in id_candidates if col in development.columns), None)
    eval_id = next((col for col in id_candidates if col in evaluation.columns), None)
    if dev_id is None or eval_id is None:
        raise ValueError(f"Could not find ID columns. Tried: {id_candidates}")
    return dev_id, eval_id

# Keep configured cols
def existing_columns(df, columns):
    return [column for column in columns if column in df.columns]

# Put rare/invalid EDUCATION and MARRIAGE codes to standard 
def normalize_categorical_codes(df, config):
    preprocessing = config.get("preprocessing", {})
    if not preprocessing.get("normalize_unknown_categorical_codes", True):
        return df.copy()

    normalized = df.copy()
    if "EDUCATION" in normalized.columns:
        normalized["EDUCATION"] = normalized["EDUCATION"].where(
            normalized["EDUCATION"].isin({1, 2, 3, 4}),
            preprocessing.get("education_unknown_to", 4),
        )
    if "MARRIAGE" in normalized.columns:
        normalized["MARRIAGE"] = normalized["MARRIAGE"].where(
            normalized["MARRIAGE"].isin({1, 2, 3}),
            preprocessing.get("marriage_unknown_to", 3),
        )
    return normalized


# Ratio helper used for LIMIT_BAL-based features
def safe_divide(numerator, denominator, epsilon):
    result = numerator / (denominator + epsilon)
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan)
    return pd.Series(result).replace([np.inf, -np.inf], np.nan)

# For bill/payment ratios, divide only when the bill is positive 
def safe_positive_divide(numerator, denominator, epsilon):
    denominator = pd.Series(denominator)
    valid = denominator > epsilon
    result = pd.Series(np.nan, index=denominator.index, dtype=float)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result.replace([np.inf, -np.inf], np.nan)

# Drop disabled engineered features
def apply_feature_flags(engineered, original_columns, config):
    flags = config.get("feature_engineering", {}).get("feature_flags", {})
    disabled = [
        column
        for column in engineered.columns
        if column not in original_columns and flags.get(column) is False
    ]
    return engineered.drop(columns=disabled)

# Clip unstable ratio features
def clip_engineered_ratios(df, config):
    preprocessing = config.get("preprocessing", {})
    if not preprocessing.get("clip_engineered_ratios", True):
        return df

    ratio_columns = [
        column
        for column in df.columns
        if "_to_" in column or column.startswith("payment_coverage_")
    ]
    if not ratio_columns:
        return df

    clipped = df.copy()
    clipped[ratio_columns] = clipped[ratio_columns].clip(
        lower=preprocessing.get("ratio_lower_clip", -1.0),
        upper=preprocessing.get("ratio_upper_clip", 20.0),
    )
    return clipped

# Create engineered features
def add_features(df, config):
    engineered = normalize_categorical_codes(df, config)
    if not config.get("feature_engineering", {}).get("enabled", True):
        return engineered

    engineered = engineered.copy()
    original_columns = set(engineered.columns)
    epsilon = config["feature_engineering"].get("epsilon", 1e-9)
    pay_cols = existing_columns(engineered, config["columns"]["pay_status"])
    bill_cols = existing_columns(engineered, config["columns"]["bill_amount"])
    payment_cols = existing_columns(engineered, config["columns"]["payment_amount"])

    # Repayment status is the strongest signal, so most delay summaries come from here
    if pay_cols:
        pay = engineered[pay_cols]
        positive_pay = pay.clip(lower=0)
        engineered["max_delay"] = pay.max(axis=1)
        engineered["mean_delay"] = pay.mean(axis=1)
        engineered["min_delay"] = pay.min(axis=1)
        engineered["std_delay"] = pay.std(axis=1)
        engineered["max_positive_delay"] = positive_pay.max(axis=1)
        engineered["mean_positive_delay"] = positive_pay.mean(axis=1)
        engineered["sum_positive_delay"] = positive_pay.sum(axis=1)
        engineered["num_delayed_months"] = (pay > 0).sum(axis=1)
        engineered["num_duly_months"] = (pay <= 0).sum(axis=1)
        engineered["num_delay_ge_2"] = (pay >= 2).sum(axis=1)
        engineered["num_delay_ge_3"] = (pay >= 3).sum(axis=1)
        engineered["has_severe_delay"] = (pay >= 2).any(axis=1).astype(int)
        if "PAY_0" in engineered.columns:
            engineered["recent_delay"] = engineered["PAY_0"]
            engineered["recent_positive_delay"] = engineered["PAY_0"].clip(lower=0)
            engineered["has_recent_delay"] = (engineered["PAY_0"] > 0).astype(int)
            engineered["recent_delay_ge_2"] = (engineered["PAY_0"] >= 2).astype(int)
        if {"PAY_0", "PAY_6"}.issubset(engineered.columns):
            engineered["delay_trend"] = engineered["PAY_0"] - engineered["PAY_6"]
            engineered["positive_delay_trend"] = (
                engineered["PAY_0"].clip(lower=0) - engineered["PAY_6"].clip(lower=0)
            )
        last3_pay = existing_columns(engineered, ["PAY_0", "PAY_2", "PAY_3"])
        if last3_pay:
            engineered["last3_num_delay_ge_2"] = (engineered[last3_pay] >= 2).sum(axis=1)
            engineered["last3_max_delay"] = engineered[last3_pay].max(axis=1)
        engineered["has_any_delay"] = (pay > 0).any(axis=1).astype(int)

    # Bill amount features describe exposure and whether the customer balance is moving
    if bill_cols:
        bills = engineered[bill_cols]
        engineered["mean_bill"] = bills.mean(axis=1)
        engineered["max_bill"] = bills.max(axis=1)
        engineered["min_bill"] = bills.min(axis=1)
        engineered["std_bill"] = bills.std(axis=1)
        engineered["total_bill"] = bills.sum(axis=1)
        engineered["num_zero_bills"] = (bills == 0).sum(axis=1)
        engineered["num_negative_bills"] = (bills < 0).sum(axis=1)
        engineered["has_negative_bill"] = (bills < 0).any(axis=1).astype(int)
        if {"BILL_AMT1", "BILL_AMT6"}.issubset(engineered.columns):
            engineered["bill_trend"] = engineered["BILL_AMT1"] - engineered["BILL_AMT6"]
        if "LIMIT_BAL" in engineered.columns:
            if "BILL_AMT1" in engineered.columns:
                engineered["recent_bill_to_limit"] = safe_divide(
                    engineered["BILL_AMT1"], engineered["LIMIT_BAL"], epsilon
                )
            engineered["mean_bill_to_limit"] = safe_divide(
                engineered["mean_bill"], engineered["LIMIT_BAL"], epsilon
            )
            engineered["max_bill_to_limit"] = safe_divide(
                engineered["max_bill"], engineered["LIMIT_BAL"], epsilon
            )

    # Payment amount features describe recent repayment capacity
    if payment_cols:
        payments = engineered[payment_cols]
        engineered["mean_payment"] = payments.mean(axis=1)
        engineered["max_payment"] = payments.max(axis=1)
        engineered["min_payment"] = payments.min(axis=1)
        engineered["std_payment"] = payments.std(axis=1)
        engineered["total_payment"] = payments.sum(axis=1)
        engineered["num_zero_payments"] = (payments == 0).sum(axis=1)
        engineered["has_zero_payment"] = (payments == 0).any(axis=1).astype(int)
        if {"PAY_AMT1", "PAY_AMT6"}.issubset(engineered.columns):
            engineered["payment_trend"] = engineered["PAY_AMT1"] - engineered["PAY_AMT6"]
        if "LIMIT_BAL" in engineered.columns:
            if "PAY_AMT1" in engineered.columns:
                engineered["recent_payment_to_limit"] = safe_divide(
                    engineered["PAY_AMT1"], engineered["LIMIT_BAL"], epsilon
                )
            engineered["mean_payment_to_limit"] = safe_divide(
                engineered["mean_payment"], engineered["LIMIT_BAL"], epsilon
            )

    # These ratios are clipped later because very small bills can create huge values
    if bill_cols and payment_cols:
        engineered["total_payment_to_total_bill"] = safe_positive_divide(
            engineered["total_payment"], engineered["total_bill"], epsilon
        )
        engineered["mean_payment_to_mean_bill"] = safe_positive_divide(
            engineered["mean_payment"], engineered["mean_bill"], epsilon
        )
        if {"PAY_AMT1", "BILL_AMT1"}.issubset(engineered.columns):
            engineered["recent_payment_to_recent_bill"] = safe_positive_divide(
                engineered["PAY_AMT1"], engineered["BILL_AMT1"], epsilon
            )

        previous_bill_coverage_cols = []
        for index in range(1, 7):
            pay_col = f"PAY_AMT{index}"
            bill_col = f"BILL_AMT{index}"
            if {pay_col, bill_col}.issubset(engineered.columns):
                engineered[f"payment_coverage_{index}"] = safe_positive_divide(
                    engineered[pay_col], engineered[bill_col], epsilon
                )
                engineered[f"zero_payment_positive_bill_{index}"] = (
                    (engineered[pay_col] == 0) & (engineered[bill_col] > 0)
                ).astype(int)

        for index in range(1, 6):
            pay_col = f"PAY_AMT{index}"
            bill_col = f"BILL_AMT{index + 1}"
            coverage_col = f"payment_to_previous_bill_{index}"
            if {pay_col, bill_col}.issubset(engineered.columns):
                engineered[coverage_col] = safe_positive_divide(
                    engineered[pay_col], engineered[bill_col], epsilon
                )
                previous_bill_coverage_cols.append(coverage_col)

        if previous_bill_coverage_cols:
            previous = engineered[previous_bill_coverage_cols]
            engineered["mean_payment_to_previous_bill"] = previous.mean(axis=1)
            engineered["min_payment_to_previous_bill"] = previous.min(axis=1)
            engineered["max_payment_to_previous_bill"] = previous.max(axis=1)
            engineered["num_low_previous_bill_coverage"] = (previous < 0.10).sum(axis=1)
            engineered["has_low_previous_bill_coverage"] = (
                engineered["num_low_previous_bill_coverage"] > 0
            ).astype(int)

        zero_cols = [
            f"zero_payment_positive_bill_{index}"
            for index in range(1, 7)
            if f"zero_payment_positive_bill_{index}" in engineered.columns
        ]
        if zero_cols:
            engineered["num_zero_payment_positive_bill"] = engineered[zero_cols].sum(axis=1)
            engineered["has_zero_payment_positive_bill"] = (
                engineered["num_zero_payment_positive_bill"] > 0
            ).astype(int)

    if {"recent_positive_delay", "recent_bill_to_limit"}.issubset(engineered.columns):
        engineered["recent_delay_x_recent_bill_to_limit"] = (
            engineered["recent_positive_delay"] * engineered["recent_bill_to_limit"]
        )
    if {"num_delayed_months", "mean_bill_to_limit"}.issubset(engineered.columns):
        engineered["num_delayed_months_x_mean_bill_to_limit"] = (
            engineered["num_delayed_months"] * engineered["mean_bill_to_limit"]
        )
    if {"has_recent_delay", "recent_payment_to_recent_bill"}.issubset(engineered.columns):
        engineered["has_recent_delay_x_recent_payment_to_recent_bill"] = (
            engineered["has_recent_delay"] * engineered["recent_payment_to_recent_bill"]
        )

    engineered = engineered.replace([np.inf, -np.inf], np.nan)
    engineered = apply_feature_flags(engineered, original_columns, config)
    return clip_engineered_ratios(engineered, config)


# Build preprocessor (median for nums, oh for cat)
def build_preprocessor(X, config):
    categorical = [column for column in config["columns"]["categorical"] if column in X.columns]
    if not config["preprocessing"].get("one_hot_encode_categorical", True):
        categorical = []
    numeric = [column for column in X.columns if column not in categorical]

    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(
                                    handle_unknown=config["preprocessing"].get("handle_unknown_categories", "ignore"),
                                    sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


# Train the 5 OOF CatBoost models and average evaluation probabilities
def run_oof_catboost(X, y, X_eval, config):
    oof_config = config["oof_ensemble"]
    variants = oof_config.get("variants", {})
    if len(variants) != 1:
        raise ValueError("This final script expects exactly one OOF CatBoost variant.")

    variant_name, variant_config = next(iter(variants.items()))
    params = dict(variant_config.get("params") or {})
    if params.get("classifier__auto_class_weights") is None:
        params.pop("classifier__auto_class_weights", None)
    params.setdefault("classifier__thread_count", int(oof_config.get("thread_count", 2)))

    fixed_threshold = float(oof_config["fixed_threshold"])
    cv = StratifiedKFold(
        n_splits=int(oof_config.get("n_splits", 5)),
        shuffle=bool(oof_config.get("shuffle", True)),
        random_state=config["random_state"] if oof_config.get("shuffle", True) else None,
    )

    oof_proba = np.zeros(len(y), dtype=float)
    eval_fold_probabilities = []
    print(f"\nOOF CatBoost variant: {variant_name}")

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(X_train, config)),
                (
                    "classifier",
                    CatBoostClassifier(
                        random_seed=config["random_state"],
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )

        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)

        oof_proba[valid_idx] = pipeline.predict_proba(X_valid)[:, 1]
        eval_fold_probabilities.append(pipeline.predict_proba(X_eval)[:, 1])
        print(f"{variant_name} fold {fold}: completed")

    eval_proba = np.mean(eval_fold_probabilities, axis=0)
    fixed_predictions = (oof_proba >= fixed_threshold).astype(int)
    eval_predictions = (eval_proba >= fixed_threshold).astype(int)

    print(
        f"{variant_name}: fixed threshold={fixed_threshold:.6f}, "
        f"OOF Macro F1={f1_score(y, fixed_predictions, average='macro'):.6f}, "
        f"ROC-AUC={roc_auc_score(y, oof_proba):.6f}"
        )
    return eval_predictions, fixed_threshold


# Run the reproducible pipeline
def main():
    config = load_config()
    development, evaluation, sample_submission = load_data()
    dev_id, eval_id = validate_data(development, evaluation, config)

    target = config["columns"]["target"]
    evaluation_ids = evaluation[eval_id].copy()
    X = development.drop(columns=[target, dev_id])
    y = development[target].astype(int)
    X_eval = evaluation.drop(columns=[eval_id])

    X = add_features(X, config)
    X_eval = add_features(X_eval, config)

    predictions, threshold = run_oof_catboost(X, y, X_eval, config)
    output_path = config["oof_ensemble"].get(
        "output_submission", config["paths"]["submission_output"]
    )
    submission = sample_submission.copy()
    submission.iloc[:, 0] = evaluation_ids.values
    submission.iloc[:, 1] = np.asarray(predictions).astype(int)
    submission.to_csv(output_path, index=False)
    prediction_column = submission.columns[-1]
    prediction_column = submission.columns[-1]

    print("\nFinal run summary")
    print(f"Development shape: {development.shape}")
    print(f"Evaluation shape: {evaluation.shape}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Threshold: {threshold:.6f}")
    print("Predicted class distribution:")
    print(submission[prediction_column].value_counts().sort_index().to_string())
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
