"""AINT351: Body Performance Classification — improved Milestone 2 workflow.

Place bodyPerformance.csv in the same directory, then run:
    pip install -r requirements.txt
    python milestone2_improved.py

The reported project scores are described in README.md. This script is a
cleaned, reproducible implementation based on the student's Milestone 2 work.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from xgboost import XGBClassifier


def make_preprocessor(features: pd.DataFrame, tree: bool) -> ColumnTransformer:
    """Preprocess numeric and categorical features within CV folds."""
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    encoder = (
        OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        if tree
        else OneHotEncoder(handle_unknown="ignore")
    )
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", encoder, categorical),
        ]
    )


def main() -> None:
    csv_path = Path(__file__).resolve().parent / "bodyPerformance.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Download it from the Kaggle "
            "Body Performance Data page linked in README.md."
        )

    # 1. Load and inspect data.
    df = pd.read_csv(csv_path)
    print("Original shape:", df.shape)
    print("Missing values by column:\n", df.isnull().sum())
    print("Empty strings by column:\n", (df == "").sum())

    # 2. Exclude the two blood-pressure predictors for Milestone 2.
    df_clean = df.drop(columns=["diastolic", "systolic"])
    print("Cleaned shape:", df_clean.shape)
    inconsistent = df_clean.isnull() | (df_clean == "")
    print("Inconsistent values by column:\n", inconsistent.sum())

    # 3. Encode the four target classes and create a stratified split.
    label_encoder = LabelEncoder()
    X = df_clean.drop(columns=["class"])
    y = label_encoder.fit_transform(df_clean["class"])
    print("Target classes:", label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Train/test sizes:", len(X_train), len(X_test))

    # 4. Logistic Regression baseline.
    baseline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train, tree=False)),
            (
                "classifier",
                LogisticRegression(max_iter=1000, solver="lbfgs"),
            ),
        ]
    )
    baseline_cv = cross_val_score(
        baseline, X_train, y_train, cv=5, scoring="roc_auc_ovr"
    )
    print("\nBaseline mean CV ROC-AUC:", baseline_cv.mean())
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    print(classification_report(y_test, baseline_predictions))
    print(confusion_matrix(y_test, baseline_predictions))

    # 5. Use a shallow Random Forest as a feature-importance diagnostic.
    forest = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train, tree=True)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300, max_depth=3,
                    random_state=42, n_jobs=-1
                ),
            ),
        ]
    )
    forest_cv = cross_val_score(
        forest, X_train, y_train, cv=5, scoring="roc_auc_ovr"
    )
    print("\nRandom Forest mean CV ROC-AUC:", forest_cv.mean())
    forest.fit(X_train, y_train)
    feature_importance = pd.Series(
        forest.named_steps["classifier"].feature_importances_,
        index=X_train.columns,
    ).sort_values(ascending=False)
    print("Feature importance:\n", feature_importance)
    print("Below 0.02:", feature_importance[feature_importance < 0.02])

    # 6. Age ablation: remove age, retrain, and compare with baseline.
    X_train_no_age = X_train.drop(columns=["age"])
    X_test_no_age = X_test.drop(columns=["age"])
    ablated = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train_no_age, tree=False)),
            (
                "classifier",
                LogisticRegression(max_iter=1000, solver="lbfgs"),
            ),
        ]
    )
    ablated_cv = cross_val_score(
        ablated, X_train_no_age, y_train,
        cv=5, scoring="roc_auc_ovr"
    )
    print("\nAge-ablation mean CV ROC-AUC:", ablated_cv.mean())
    ablated.fit(X_train_no_age, y_train)
    ablated_predictions = ablated.predict(X_test_no_age)
    print(classification_report(y_test, ablated_predictions))
    print(confusion_matrix(y_test, ablated_predictions))

    # 7. XGBoost multiclass classification.
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y)),
        eval_metric="mlogloss",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    xgb_pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X, tree=True)),
            ("classifier", xgb),
        ]
    )
    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )
    xgb_probabilities = cross_val_predict(
        xgb_pipeline, X, y, cv=cv, method="predict_proba"
    )
    xgb_predictions = np.argmax(xgb_probabilities, axis=1)
    xgb_auc = roc_auc_score(
        y, xgb_probabilities, multi_class="ovr"
    )
    print("\nXGBoost out-of-fold ROC-AUC:", xgb_auc)
    print(classification_report(
        y, xgb_predictions, target_names=label_encoder.classes_
    ))
    print(confusion_matrix(y, xgb_predictions))

    # 8. Hyperparameter search (3 x 3 combinations, 5 folds each).
    grid = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid={
            "classifier__max_depth": [3, 4, 5],
            "classifier__min_child_weight": [1, 3, 5],
        },
        scoring="roc_auc_ovr",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X, y)
    best_model = grid.best_estimator_
    print("\nBest parameters:", grid.best_params_)
    print("Best GridSearchCV ROC-AUC:", grid.best_score_)

    # This repeats CV on data used for tuning; see limitations in README.
    tuned_probabilities = cross_val_predict(
        best_model, X, y, cv=cv, method="predict_proba"
    )
    tuned_predictions = np.argmax(tuned_probabilities, axis=1)
    print("Tuned CV ROC-AUC:", roc_auc_score(
        y, tuned_probabilities, multi_class="ovr"
    ))
    print(classification_report(
        y, tuned_predictions, target_names=label_encoder.classes_
    ))
    print(confusion_matrix(y, tuned_predictions))

    # 9. Save preprocessing and classifier together.
    output = Path(__file__).resolve().parent / (
        "xgboost_body_performance.joblib"
    )
    joblib.dump(best_model, output)
    print("Saved model:", output)


if __name__ == "__main__":
    main()
