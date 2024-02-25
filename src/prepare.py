"""Perform feature selection"""

import argparse
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    """The main function called by DVC, with its parser"""
    parser = argparse.ArgumentParser(description="Train/test split")
    parser.add_argument("split", help="path to the splitted data", type=str)
    parser.add_argument("prepare", help="path to the outpud folder", type=str)
    parser.add_argument("params", help="params.yml path", type=str)
    args = parser.parse_args()

    # Check inputs
    params_file = Path(args.params)
    if not params_file.is_file():
        raise FileNotFoundError(f"Provided params file does not exist: {params_file}")

    with open(params_file, "r", encoding="utf-8") as yaml_file:
        params = yaml.safe_load(yaml_file)["prepare"]

    split_path = Path(args.split)
    if not params_file.is_file():
        raise FileNotFoundError("Provided dataset file does not exist: {split_path}")

    # Output folders
    output_path = Path(args.prepare)
    output_path.mkdir(parents=True, exist_ok=True)

    # Split the data
    train_data = pd.read_csv(split_path / "train.csv")
    test_data = pd.read_csv(split_path / "test.csv")

    features_to_keep = params["features_to_keep"]
    if features_to_keep:
        train_data = train_data[features_to_keep]
        test_data = test_data[features_to_keep]

    # Save the data
    train_data.to_csv(output_path / "train.csv", index=False)
    test_data.to_csv(output_path / "test.csv", index=False)


if __name__ == "__main__":
    main()
