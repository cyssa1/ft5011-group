from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import sys

from model_training import main


RESULTS_DIR = Path("results")
DATASETS_TO_RUN = [
    "ta_sentiment",
]
MODELS_TO_RUN = [
    "xgboost",
    # "lstm_attention",
    # "lstm",
    # "lstm_ic",
]


class Tee:
    """Write output to both the terminal and a file."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def run_and_save(model_name: str, dataset_name: str) -> Path:
    """Run one model on one dataset and save the full output."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"training_output_{dataset_name}_{model_name}_{timestamp}.txt"

    with output_path.open("w", encoding="utf-8") as output_file:
        tee_stream = Tee(sys.__stdout__, output_file)
        with redirect_stdout(tee_stream), redirect_stderr(tee_stream):
            print(f"Training on dataset: {dataset_name}")
            main(model_name=model_name, data_set=dataset_name)

    print(f"Saved output to {output_path}")
    return output_path


if __name__ == "__main__":
    for dataset_name in DATASETS_TO_RUN:
        for model_name in MODELS_TO_RUN:
            run_and_save(model_name=model_name, dataset_name=dataset_name)
