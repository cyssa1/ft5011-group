# FT5011 Group Project

This repository contains the data pipeline, model training code, backtesting utilities, notebook-based visualization, and report files for the stock signal prediction project.

## Project Structure

- `data/`
  - Main experiment datasets used by the current pipeline, such as `ta_only.csv`, `ta_sentiment.csv`, and `all_features.csv`
- `rawdata/`
  - Original or less-processed source files kept for reference
- `scripts/`
  - Main Python scripts
  - `model_training.py`: training and evaluation pipeline for `xgboost`, `cnn`, `lstm`, `lstm_ic`, and `lstm_attention`
  - `run_models.py`: batch runner that trains models on selected datasets and saves outputs
  - `backtest_saved_model.py`: backtests saved models on the test split
  - `model_comparison_notebook_helper.py`: helper functions used by the notebook
- `results/`
  - Saved training logs
  - `results/models/`: saved trained model artifacts as `.pkl`
- `reports/tex/`
  - LaTeX report source and compiled PDF
- `model_structure_comparison.ipynb`
  - Notebook for comparing model performance and backtest results visually

## How To Run

### 1. Train models and save outputs

Run:

```bash
python scripts/run_models.py
```

This will:

- train the models listed in `scripts/run_models.py`
- run them on the datasets listed in `scripts/run_models.py`
- print progress to terminal
- save a training log to `results/`
- save the trained model to `results/models/`

Typical output files:

- `results/training_output_<dataset>_<model>_<timestamp>.txt`
- `results/models/trained_model_<dataset>_<model>.pkl`

### 2. Backtest a saved model

Run:

```bash
python scripts/backtest_saved_model.py --model-path results/models/trained_model_ta_sentiment_lstm_attention.pkl
```

This will:

- reload the saved model
- regenerate predictions on the test split
- run the trading backtest
- save backtest outputs to `results/`

Typical backtest outputs include:

- test predictions CSV
- daily strategy returns CSV
- strategy metrics JSON

### 3. Compare models visually

Open:

- `model_structure_comparison.ipynb`

The notebook is used to:

- compare multiple models on the same dataset
- visualize classification metrics
- visualize confusion matrices
- visualize backtest equity curves

## What To Expect

After a normal run, you should have:

- printed training and evaluation results in the terminal
- saved log files in `results/`
- saved trained models in `results/models/`
- optional backtest result files in `results/`

The main report is located at:

- `reports/tex/main.tex`
- `reports/tex/main.pdf`

To compile the report from the LaTeX source:

```bash
cd reports/tex
latexmk -pdf main.tex
```
