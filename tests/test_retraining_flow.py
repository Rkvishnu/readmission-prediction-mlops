from workflows.prefect_retraining_flow import should_retrain_task


def test_should_retrain_when_drift_detected():
    assert should_retrain_task.fn({"status": "drift_detected"}) is True


def test_should_not_retrain_without_drift():
    assert should_retrain_task.fn({"status": "no_drift_detected"}) is False


def test_force_retrain_overrides_drift_status():
    assert should_retrain_task.fn({"status": "no_drift_detected"}, force_retrain=True) is True
