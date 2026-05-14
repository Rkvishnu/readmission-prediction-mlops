from readmission.models.train import run_training


def run() -> dict[str, float]:
    return run_training()

