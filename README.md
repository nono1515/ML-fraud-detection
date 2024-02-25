# ML-fraud-detection

## Desciption
This repo is a small usecase of [XGBoost](https://xgboost.readthedocs.io/en/stable/)
for anomaly detection. It uses [DVC](https://dvc.org/) for data, model and experiment
tracking.

#### Data pipeline
```bash
dvc dag
+-------------------------+  
| data\creditcard.csv.dvc |  
+-------------------------+  
              *
              *
              *
         +-------+
         | split |
         +-------+
              *
              *
              *
        +---------+
        | prepare |
        +---------+
              *
              *
              *
         +-------+
         | train |
         +-------+
```
The pipeline consists of 3 stages:
- split
- prepare
- train

The split stage seperates the data into training and testing sets.
The prepare stage only keeps selected features. The train stage loads the train data, evaluates an XGBoost classifier model using cross-validation and saves the metrics. It also generates two plos alongside metrics:
- Confusion matrix
- Precision-Recall curve

The train stage does not use the test data as the latter are kept for final evalutaion of the model only. This will be done in another `eval` stage.

## Setup

#### Datatset
The `Credit Card Fraud Detection` dataset needs to be downloaded from [kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place in the data directory of this repo. It should be named `creditcard.csv`.
```
ML-fraud-detection
├── data
│   ├── creditcard.csv
│   ├── ...
├── ...
```

#### Python environnment
The code was tested with Python 3.11 on Windows and Windows Subsystem for Linux (WSL). One need to create a virtual environnement first and install all dependancies.
```bash
python3.11 -m venv venv
source venv/bin/activate  # WSL
venv\Scripts\activate     # Windows
pip install -U pip
pip install -r requirements.txt
```

#### DVC
This repo uses DVC for tracking data alongside experiments and metrics. After the previous steps, DVC is ready to run the ML pipeline and reprodruce the result of the current stage of repo.
```bash
dvc repro
dvc metrics show
# Metrics will be printed in the terminal
dvc plots show
# Plots will be generated in ./dvc_plots/index.html
```

## Usage

#### Run experiment
One can easily modify the parameter of the model by directly changing it in the `params.yaml` file and relaunch `dvc repro`. However, this will directly modify the workspace and make difficult the comparison between different models. A better way is to use `dvc exp run` ([doc here](https://dvc.org/doc/command-reference/exp/run)) with the `--set-param flag`. For instance
```bash
dvc exp run dvc.yaml --name baseline
dvc exp run dvc.yaml --name my-new-exp --set-param params.yaml:train.xgb_params.eta=0.1
dvc metrics diff baseline my-new-exp
dvc plots diff baseline my-new-exp
```
The DVC extension for VSCode can also be used to "Run, compare, visualize, and track machine learning experiments right in VS Code".

#### Grid-search
One can also perform a grid-search over different sets of parameters, using the script `scripts/grid_search.py`. It works the same way as experiment, except for the fact that the latter are only put in queue, they must still be launched afterwards.
```bash
python scripts/grid_search.py
dvc exp run --run-all -j 4  # run 4 threads in parallel
```
If you want more detail, see this [blog post](https://dvc.ai/blog/hyperparam-tuning).

#### Notebooks
The notebooks folder contain utility notebooks for quick experiment and easy visualization.

- features.ipynb presents differents ways to measure features importance
- base_model.ipynb allows to quickly load data, train and evaluate a model

## Contributing
#### Pre-commit
If you want to contribute to this repo, you have to follow specific guidelines. Fortunately, they are already all implemented as pre-commit hooks (more information about this [here](https://pre-commit.com)). So you just have to install them and your code will be automatically checked at each commit.
```bash
pre-commit install
```
Those includes:
- JSON, YAML and TOML syntax checker
- DVC hook to ckeck the status of your data before commits
- DVC hook to automatize `dvc checkout` after each `git checkout`
- Jupyter notebook strip out (only commit notebooks with cleared outputs)

And for Python files:
- `black` formatter
- `isort` import sorting
- `pylint` linter
- `mypy` for static types checking
