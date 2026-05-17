from readmission.monitoring.drift_report import summarize_drift


def test_summarize_drift_handles_distance_and_p_value_metrics():
    summary = summarize_drift(
        {
            "metrics": [
                {
                    "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                    "value": {"count": 2.0, "share": 0.5},
                },
                {
                    "config": {
                        "type": "evidently:metric_v2:ValueDrift",
                        "column": "time_in_hospital",
                        "method": "Wasserstein distance (normed)",
                        "threshold": 0.1,
                    },
                    "value": 0.2,
                },
                {
                    "config": {
                        "type": "evidently:metric_v2:ValueDrift",
                        "column": "num_medications",
                        "method": "K-S p_value",
                        "threshold": 0.05,
                    },
                    "value": 0.01,
                },
                {
                    "config": {
                        "type": "evidently:metric_v2:ValueDrift",
                        "column": "num_lab_procedures",
                        "method": "Wasserstein distance (normed)",
                        "threshold": 0.1,
                    },
                    "value": 0.02,
                },
            ]
        }
    )

    assert summary["status"] == "drift_detected"
    assert summary["drifted_columns_count"] == 2
    assert [column["column"] for column in summary["drifted_columns"]] == [
        "time_in_hospital",
        "num_medications",
    ]
