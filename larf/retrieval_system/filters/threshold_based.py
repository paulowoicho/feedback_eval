from ...data_models.query import Query
from ..base import Component


class ThresholdBasedFilter(Component):
    """Filter to remove passages based on a threshold score."""

    def __init__(self, threshold: int) -> None:
        """Initialize the ThresholdBasedFilter.

        Args:
            threshold (float): The relevance threshold for filtering passages.
        """
        self.threshold = threshold

    def run(self, queries: list[Query]) -> list[Query]:
        for query in queries:
            filtered = [
                p for p in query.passages if (p.relevance_assessment or 0) >= self.threshold
            ]
            query.set_passages(filtered)
        return queries
