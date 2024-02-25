"""Grid-search module for the different model configurations"""

import itertools
import subprocess

# Automated grid search experiments
eta_list = [0.1, 0.3, 0.5]
max_depth_list = [4, 6]
subsample_list = [0.5, 1]
alpha_list = [0]

features_to_keep_list = [
    [],  # keep all
    [
        "Class",
        "V4",
        "V14",
        "V10",
        "V12",
        "Amount",
        "V18",
        "V7",
        "V17",
        "V16",
        "V19",
        "V15",
        "V26",
    ],
]

# Iterate over all combinations of hyperparameter values.
for features_to_keep, eta, max_depth, subsample, alpha in itertools.product(
    features_to_keep_list, eta_list, max_depth_list, subsample_list, alpha_list
):
    subprocess.run(
        [
            "dvc",
            "exp",
            "run",
            "dvc.yaml",
            "--queue",
            "--set-param",
            f"params.yaml:prepare.features_to_keep={features_to_keep}",
            "--set-param",
            f"params.yaml:train.xgb_params.eta={eta}",
            "--set-param",
            f"params.yaml:train.xgb_params.max_depth={max_depth}",
            "--set-param",
            f"params.yaml:train.xgb_params.subsample={subsample}",
            "--set-param",
            f"params.yaml:train.xgb_params.alpha={alpha}",
        ],
        check=True,
    )
