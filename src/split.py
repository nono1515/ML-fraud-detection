"""Seperate the data into a training and a testing sets"""

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def main() -> None:
    """The main function called by DVC, with its parser"""
    parser = argparse.ArgumentParser(description="Train/test split")
    parser.add_argument("dataset", help="creditcard.csv path", type=str)
    parser.add_argument("split", help="path to the train and test data", type=str)
    parser.add_argument("params", help="params.yml path", type=str)
    args = parser.parse_args()

    # Check inputs
    params_file = Path(args.params)
    if not params_file.is_file():
        raise FileNotFoundError(f"Provided params file does not exist: {params_file}")

    with open(params_file, "r", encoding="utf-8") as yaml_file:
        params = yaml.safe_load(yaml_file)["split"]

    dataset_path = Path(args.dataset)
    if not params_file.is_file():
        raise FileNotFoundError("Provided dataset file does not exist: {dataset_path}")

    # Output folders
    output_path = Path(args.split)
    output_path.mkdir(parents=True, exist_ok=True)

    # Split the data
    data = pd.read_csv(dataset_path)
    train, test = train_test_split(
        data, test_size=params["test_size"], random_state=params["random_state"]
    )

    # Save the data
    train.to_csv(output_path / "train.csv", index=False)
    test.to_csv(output_path / "test.csv", index=False)


if __name__ == "__main__":
    main()
