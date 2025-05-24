from abc import ABC, abstractmethod
from pathlib import Path

import jsonlines

from ..data_models.query import Passage, Query


class BaseQueryLoader(ABC):
    """Base class for all query loaders."""

    def __init__(self, queries_path: str | Path, resume_path: str | Path | None = None) -> None:
        """Initialize the query loader with the path to the queries.

        Args:
            queries_path (str | Path): The path to the queries file.
            resume_path (str | Path | None): A path to a previous experiment
                run with passages stored in a jsonlines file per query. Defaults to None.
        """
        self.queries_path = Path(queries_path)
        if not self.queries_path.exists():
            raise FileNotFoundError(f"Queries file not found: {self.queries_path}")

        self.resume_path = Path(resume_path) if resume_path else None
        if self.resume_path and not self.resume_path.exists():
            raise FileNotFoundError(f"Resume path not found: {self.resume_path}")

    @abstractmethod
    def get_queries(self) -> list[Query]:
        """Returns a list of query objects.

        Returns:
            list[Query]: A list of query objects.
        """

    @property
    def name(self) -> str:
        """Returns the name of the query loader.

        Returns:
            str: The name of the query loader.
        """
        return self.__class__.__name__

    def _load_passages(self, qid: int | str) -> list[Passage]:
        assert self.resume_path, "Resume path is None, so nothing to resume from."
        path = self.resume_path / f"{qid}.jsonl"
        passages: list[Passage] = []

        with jsonlines.open(path) as f:
            for line in f:
                p = Passage(id=line["id"], text=line["text"], score=line["score"])
                p.set_relevance_assessment(line["relevance_assessment"])
                passages.append(p)

        return passages
