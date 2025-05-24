from dataclasses import dataclass, field


@dataclass(frozen=True)
class Passage:
    """Represents a passage of text."""

    id: str | int
    text: str
    score: float
    # Simulated relevance assessment on a 0-3 scale.
    relevance_assessment: int | None = field(default=None, init=False)

    def set_relevance_assessment(self, relevance_assessment: int) -> None:
        """Sets the relevance assessment for the passage.

        Args:
            relevance_assessment (int): The relevance assessment to set.
        """
        object.__setattr__(self, "relevance_assessment", relevance_assessment)


@dataclass(frozen=True)
class Query:
    """Represents a query with an ID and text."""

    id: str | int
    text: str
    passages: list[Passage] = field(default_factory=list, init=False)

    def set_passages(self, passages: list[Passage]) -> None:
        """Sets the passages for the query.

        Args:
            passages (list[Passage]): The passages to set.
        """
        object.__setattr__(self, "passages", passages)
