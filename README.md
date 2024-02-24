# ML-fraud-detection

## Desciption

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
The code was tested with Python 3.11 and Windows Subsystem for Linux (WSL). One need to create a virtual environnement first and install all dependancies.
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

#### DVC
This repo uses DVC for tracking data alongside experiments and metrics.
-> DVC init ?


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

And for Python files:
- `black` formatter
- `isort` import sorting
- `pylint` linter
- `mypy` for static types checking
