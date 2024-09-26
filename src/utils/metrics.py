from torchmetrics import MetricCollection
from torchmetrics.retrieval import RetrievalPrecision


def get_metrics(**kwargs) -> MetricCollection:
    return MetricCollection({"RetrievalPrecision": RetrievalPrecision(**kwargs)})
