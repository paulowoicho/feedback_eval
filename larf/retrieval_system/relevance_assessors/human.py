from pathlib import Path
import random

from ...data_models.query import Query
from ..base import Component


def _load_qrel(path: Path) -> dict[str | int, dict[str | int, int]]:
    """Load a TREC‐style qrel file into a nested lookup dict."""
    if not path.exists():
        raise FileNotFoundError(f"Qrel file not found: {path}")
    lookup: dict[str | int, dict[str | int, int]] = {}
    for line in path.read_text().splitlines():
        query_id, _, doc_id, score = line.strip().split()
        lookup.setdefault(query_id, {})[doc_id] = int(score)
    return lookup


class HumanRelevanceAssessor(Component):
    def __init__(self, qrel_path: str | Path) -> None:
        """Initialize the HumanRelevanceAssessor.

        Args:
            qrel_path (str | Path): Path to the qrel file. Qrels are expected to be in TREC style
                format, i,e (query_id, dummy, doc_id, relevance_score).
        """
        self.relevance_lookup = _load_qrel(Path(qrel_path))

    def run(self, queries: list[Query]) -> list[Query]:
        for query in queries:
            self._assess_single(query)
        return queries

    def _assess_single(self, query: Query) -> None:
        """Assess the passages for a given query using the qrel file.

        Args:
            query (Query): The query object containing the passages to be assessed.
        """

        for passage in query.passages:
            passage.set_relevance_assessment(
                self.relevance_lookup.get(query.id, {}).get(passage.id, 0)
            )


class NoisyHumanRelevanceAssessor(HumanRelevanceAssessor):
    def __init__(
        self,
        qrel_path: str | Path,
        noise_probability: float,
        max_score: int = 3,
        seed: int | None = None,
    ) -> None:
        """Creates a NoisyCopyCatAssessor instance.

        Args:
            qrel_path (str | Path): Path to the qrel file. Qrels are expected to
                be in TREC style format, i.e (query_id, dummy, doc_id, relevance_score).
            noise_probability (float): A probability indicating
                how often the assessor is likely to be wrong about a judgement
            max_score (int): The maximum score a sample can have as its
                relevance assessment.
            seed (int | None): Seed for reproducible noise. If None, uses a random seed.
        """
        super().__init__(qrel_path)
        self.noise_probability = noise_probability
        self.max_score = max_score
        self.rng = random.Random(seed)
        self._other_scores: dict[int, list[int]] = {
            gt: [s for s in range(max_score + 1) if s != gt] for gt in range(max_score + 1)
        }

    def _assess_single(self, query: Query) -> None:
        for passage in query.passages:
            groundtruth_relevance = self.relevance_lookup.get(query.id, {}).get(passage.id, 0)
            if self.rng.random() <= self.noise_probability:
                alt_scores = self._other_scores[groundtruth_relevance]
                passage.set_relevance_assessment(random.choice(alt_scores))
            else:
                passage.set_relevance_assessment(groundtruth_relevance)


class LLMJudgeSimulator(HumanRelevanceAssessor):
    def __init__(
        self,
        qrel_path: str | Path,
        probability_profile: list[list[float]],
        max_score: int = 3,
        seed: int | None = None,
    ) -> None:
        """Creates an LLM Simulator Relevance Assessor based on a probability profile.

        Args:
            qrel_path (str | Path): Path to the qrel file.
            probability_profile (list[list[float]]): A 2D list where row `i` gives the distribution over predicted
                scores when the true score is `i`.
            max_score (int, optional): Maximum score index in the profile.
            seed (int | None, optional): Seed for reproducible sampling. Defaults to None.
        """
        super().__init__(qrel_path)
        self.rng = random.Random(seed)
        self.probability_profile = probability_profile
        self.max_score = max_score
        self.rng = random.Random(seed)

    def _assess_single(self, query: Query) -> None:
        for passage in query.passages:
            groundtruth_relevance = self.relevance_lookup.get(query.id, {}).get(passage.id, 0)
            groundtruth_relevance = min(groundtruth_relevance, self.max_score)
            predicted_relevance = random.choices(
                list(range(self.max_score + 1)),
                weights=self.probability_profile[groundtruth_relevance],
            )[0]
            passage.set_relevance_assessment(predicted_relevance)
