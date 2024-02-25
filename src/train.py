"""Train the model and evaluate it using K-Fold"""

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn import metrics
from sklearn.calibration import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def main() -> None:
    """
    A function to train a model using the provided input data and parameters.
    It takes in the input train data, output model path, and params.yml path.
    It trains the model, saves the metrics and model, and generates confusion matrices.
    """
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("train", help="path to the input train data", type=str)
    parser.add_argument("model", help="path to the output model", type=str)
    parser.add_argument("params", help="params.yml path", type=str)
    args = parser.parse_args()

    # Check inputs
    params_file = Path(args.params)
    if not params_file.is_file():
        raise FileNotFoundError(f"Provided params file does not exist: {params_file}")

    with open(params_file, "r", encoding="utf-8") as yaml_file:
        params = yaml.safe_load(yaml_file)["train"]

    train_data = Path(args.train)
    if not params_file.is_file():
        raise FileNotFoundError("Provided train data file does not exist: {train_data}")

    # Load the data
    data = pd.read_csv(train_data)
    y = data[[params["target_col"]]]
    X = data.drop(params["target_col"], axis=1)

    # Train the model
    # Model is determinist -> no need for random state
    model = XGBClassifier(**params["xgb_params"], verbosity=0)
    k_fold = StratifiedKFold(
        n_splits=params["k_fold"], shuffle=True, random_state=params["random_state"]
    )
    cross_val_preds = cross_val_predict(model, X, y, cv=k_fold)

    # Save the metrics
    metrics_dict = {}
    metrics_dict["recall"] = metrics.recall_score(y, cross_val_preds)
    metrics_dict["precision"] = metrics.precision_score(y, cross_val_preds)
    metrics_dict["f1"] = metrics.f1_score(y, cross_val_preds)
    metrics_dict["accuracy"] = metrics.accuracy_score(y, cross_val_preds)
    # AUC is quite useless in case of imbalanced dataset -> prc is better
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    metrics_dict["prc"] = metrics.average_precision_score(y, cross_val_preds)
    with open("metrics/train.json", "w", encoding="utf-8") as json_file:
        json.dump(metrics_dict, json_file)

    # Save the model
    model.fit(X, y)
    model.save_model(Path(args.model))
    train_preds = model.predict(X)
    y_labels = ["fraud" if y_ else "No fraud" for y_ in y.values]

    val_preds_labels = ["fraud" if y_ else "No fraud" for y_ in cross_val_preds]
    val_confusion_matrix = [("actual", "predicted")]
    val_confusion_matrix.extend(list(zip(y_labels, val_preds_labels)))

    train_preds_labels = ["fraud" if y_ else "No fraud" for y_ in train_preds]
    train_confusion_matrix = [("actual", "predicted")]
    train_confusion_matrix.extend(list(zip(y_labels, train_preds_labels)))

    Path("plots/cm").mkdir(parents=True, exist_ok=True)
    with open("plots/cm/validation.csv", "w", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(val_confusion_matrix)

    with open("plots/cm/train.csv", "w", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(train_confusion_matrix)


if __name__ == "__main__":
    main()
