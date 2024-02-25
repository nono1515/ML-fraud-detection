"""Grid-search module for the different model configurations"""

import itertools
import subprocess

# Automated grid search experiments
eta_list = [0.1, 0.3]
max_depth_list = [4, 6]
# subsample_list = [0.5, 1]
alpha_list = [0, 1]

# Iterate over all combinations of hyperparameter values.
for eta, max_depth, alpha in itertools.product(eta_list, max_depth_list, alpha_list):
    subprocess.run(
        [
            "dvc",
            "exp",
            "run",
            "dvc.yaml",
            # "--queue",
            "--set-param",
            f"params.yaml:train.xgb_params.eta={eta}",
            "--set-param",
            f"params.yaml:train.xgb_params.max_depth={max_depth}",
            "--set-param",
            f"params.yaml:train.xgb_params.alpha={alpha}",
        ],
        check=True,
    )
