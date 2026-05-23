"""Paper-facing metric labels and formatting helpers."""

METRIC_LABELS = {
    "principal_CVaR95_loss_episode": "CVaR95 loss",
    "crowding": "Crowding",
    "co_crash": "Co-crash",
    "mean_effort": "Mean effort",
}


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def round_value(value: float, digits: int = 5) -> float:
    return round(float(value), digits)
