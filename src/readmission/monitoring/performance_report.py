def summarize_performance(metrics: dict[str, float]) -> str:
    return ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())

