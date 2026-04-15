from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import sys

from model_training import build_model_artifact, main, save_model_artifact


RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"
DATASETS_TO_RUN = [
    # "ta_sentiment",
    "all_features",
    "ta_only"
]
MODELS_TO_RUN = [
    "xgboost",
    #  "lstm_attention",
    "lstm",
    # "cnn",
]


class Tee:
    """Write output to both the terminal and a file."""

    def __init__(self, *streams) -> None:
        self.streams = streams
        self.primary_stream = streams[0]

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.primary_stream, "isatty", lambda: False)())

    @property
    def encoding(self) -> str | None:
        return getattr(self.primary_stream, "encoding", None)

    def fileno(self) -> int:
        return self.primary_stream.fileno()


def run_and_save(model_name: str, dataset_name: str) -> Path:
    """Run one model on one dataset and save the full output and trained model."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"training_output_{dataset_name}_{model_name}_{timestamp}.txt"
    model_path = MODELS_DIR / f"trained_model_{dataset_name}_{model_name}.pkl"

    with output_path.open("w", encoding="utf-8") as output_file:
        tee_stdout = Tee(sys.__stdout__, output_file)
        tee_stderr = Tee(sys.__stderr__, output_file)
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            print(f"Training on dataset: {dataset_name}")
            run_bundle = main(
                model_name=model_name,
                data_set=dataset_name,
                return_run_bundle=True,
            )
            artifact = build_model_artifact(
                run_bundle["model_bundle"],
                data_set=dataset_name,
                sequence_bundle=run_bundle["sequence_bundle"],
            )
            save_model_artifact(artifact, model_path)
            print(f"Saved trained model to {model_path}")

    print(f"Saved output to {output_path}")
    return output_path


if __name__ == "__main__":
    for dataset_name in DATASETS_TO_RUN:
        for model_name in MODELS_TO_RUN:
            run_and_save(model_name=model_name, dataset_name=dataset_name)
