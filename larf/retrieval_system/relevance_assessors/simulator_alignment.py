from simulator_alignment.data_models.sample import Sample  # type: ignore[import-untyped]
from simulator_alignment.simulators.base import BaseSimulator  # type: ignore[import-untyped]

from ...data_models.query import Passage, Query
from ..base import Component


class SimulatorAlignmentAssessor(Component):
    """Wrapper for simulators experimented with in the Simulator Alignment project.

    For more details: https://github.com/paulowoicho/simulator_alignment
    """

    def __init__(self, simulator: BaseSimulator) -> None:
        """Initialize the Simulator Alignment Assessor.

        Args:
            simulator (BaseSimulator): The simulator to be used for relevance assessment.
        """
        self.simulator = simulator

    def run(self, queries: list[Query]) -> list[Query]:
        qp_pairs: list[tuple[Query, Passage]] = [
            (query, passage) for query in queries for passage in (query.passages)
        ]

        samples = [
            Sample(
                query=query.text,
                passage=passage.text,
                groundtruth_relevance=0,
            )
            for query, passage in qp_pairs
        ]

        assessed_samples = self.simulator.assess(samples)

        for (_, p), s in zip(qp_pairs, assessed_samples):
            p.set_relevance_assessment(s.predicted_relevance)

        return queries
