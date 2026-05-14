from prometheus_client import Counter, Histogram

PREDICTION_COUNTER = Counter("readmission_predictions_total", "Total prediction requests")
PREDICTION_LATENCY = Histogram("readmission_prediction_latency_seconds", "Prediction latency")

