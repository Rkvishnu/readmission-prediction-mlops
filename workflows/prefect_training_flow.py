from prefect import flow, task

from readmission.models.train import run_training


@task
def train_task():
    return run_training()


@flow(name="readmission-training")
def training_flow():
    return train_task()


if __name__ == "__main__":
    training_flow()

