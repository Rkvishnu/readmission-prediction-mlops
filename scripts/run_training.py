from readmission.models.train import run_training


if __name__ == "__main__":
    metrics = run_training()
    print(metrics)

