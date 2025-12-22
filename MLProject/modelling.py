# -*- coding: utf-8 -*-
import argparse
import inspect
import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
def safe_predict_proba(model, X):
    """Return P(class=1) if available, else None."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba is not None and proba.shape[1] >= 2:
            return proba[:, 1]
    return None
def save_confusion_matrix(cm, out_path: str) -> None:
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
def parse_args():
    p = argparse.ArgumentParser(description="Train Titanic model + log to MLflow")
    p.add_argument("--data_path", type=str, default="titanic_preprocessing.csv")
    p.add_argument("--output_dir", type=str, default="artifacts")
    p.add_argument("--target", type=str, default="Survived")
    p.add_argument("--experiment_name", type=str, default="workflow_ci_titanic")
    p.add_argument("--run_name", type=str, default="ci_logreg_training")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)

    args, _unknown = p.parse_known_args()
    return args
def log_model_compat(model, name: str = "model"):
    """
    MLflow versi baru prefer 'name', versi lama pakai 'artifact_path'.
    """
    sig = inspect.signature(mlflow.sklearn.log_model)
    if "name" in sig.parameters:
        mlflow.sklearn.log_model(model, name=name)
    else:
        mlflow.sklearn.log_model(model, artifact_path=name)
def main():
    args = parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"File tidak ditemukan: {args.data_path}. "
            f"Pastikan CSV ada di folder MLProject/ atau ubah --data_path.")
    df = pd.read_csv(args.data_path)
    if args.target not in df.columns:
        raise ValueError(
            f"Kolom target '{args.target}' tidak ada. Kolom yang ada: {list(df.columns)}")

    X = df.drop(columns=[args.target]).copy()
    y = df[args.target].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if y.nunique() == 2 else None,
    )

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Preprocessing
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ],
        remainder="drop",
    )
    # Model
    clf = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
    )
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", clf),
    ])
    # Train
    pipe.fit(X_train, y_train)
    # Evaluate
    y_pred = pipe.predict(X_test)
    y_proba = safe_predict_proba(pipe, X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    # Build local artifacts
    extras_dir = os.path.join(args.output_dir, "extras")
    ensure_dir(extras_dir)
    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(extras_dir, "confusion_matrix.png")
    save_confusion_matrix(cm, cm_path)
    rep = classification_report(y_test, y_pred, digits=4, zero_division=0)
    rep_path = os.path.join(extras_dir, "classification_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(rep)
    meta = {
        "timestamp": datetime.now().isoformat(),
        "data_path": args.data_path,
        "target": args.target,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "features": list(X.columns),
        "num_numeric_cols": int(len(num_cols)),
        "num_categorical_cols": int(len(cat_cols)),
    }
    meta_path = os.path.join(extras_dir, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    # MLflow logging
    env_run_id = os.environ.get("MLFLOW_RUN_ID")
    if env_run_id:
        with mlflow.start_run(run_id=env_run_id):
            mlflow.log_param("data_path", args.data_path)
            mlflow.log_param("target", args.target)
            mlflow.log_param("test_size", args.test_size)
            mlflow.log_param("random_state", args.random_state)
            mlflow.log_param("model", "LogisticRegression")
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_param("num_features", int(len(X.columns)))
            mlflow.log_param("num_numeric_cols", int(len(num_cols)))
            mlflow.log_param("num_categorical_cols", int(len(cat_cols)))
            mlflow.log_metric("test_accuracy", float(acc))
            mlflow.log_metric("test_precision", float(prec))
            mlflow.log_metric("test_recall", float(rec))
            mlflow.log_metric("test_f1", float(f1))
            if auc is not None:
                mlflow.log_metric("test_roc_auc", float(auc))
            mlflow.log_artifacts(extras_dir, artifact_path="extras")
            log_model_compat(pipe, name="model")
    else:
        mlflow.set_experiment(args.experiment_name)
        with mlflow.start_run(run_name=args.run_name):
            mlflow.log_param("data_path", args.data_path)
            mlflow.log_param("target", args.target)
            mlflow.log_param("test_size", args.test_size)
            mlflow.log_param("random_state", args.random_state)
            mlflow.log_param("model", "LogisticRegression")
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_param("num_features", int(len(X.columns)))
            mlflow.log_param("num_numeric_cols", int(len(num_cols)))
            mlflow.log_param("num_categorical_cols", int(len(cat_cols)))
            mlflow.log_metric("test_accuracy", float(acc))
            mlflow.log_metric("test_precision", float(prec))
            mlflow.log_metric("test_recall", float(rec))
            mlflow.log_metric("test_f1", float(f1))
            if auc is not None:
                mlflow.log_metric("test_roc_auc", float(auc))
            mlflow.log_artifacts(extras_dir, artifact_path="extras")
            log_model_compat(pipe, name="model")
    print("DONE: training + MLflow logging berhasil.")

if __name__ == "__main__":
    main()
