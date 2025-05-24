from dataclasses import asdict
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Self

import ir_measures
from ir_measures import Measure, ScoredDoc
import jsonlines
from pyserini.search.lucene import LuceneSearcher  # type: ignore[import-untyped]
import torch
import tqdm  # type: ignore[import-untyped]
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast

from larf.data_models.query import Query
from larf.query_loaders.base import BaseQueryLoader
from larf.query_loaders.ikat import iKAT2023
from larf.retrieval_system.base import Component
from larf.retrieval_system.rerankers.monot5 import MonoT5Reranker
from larf.retrieval_system.searchers.pyserini import PyseriniLuceneSearcher

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ExperimentManager:
    def __init__(
        self,
        pipeline: list[Component],
        query_loaders: list[BaseQueryLoader],
        measures_by_loader: list[list[Measure]],
        qrels: list[str],
        experiment_name: str | None = None,
    ) -> None:
        """Initialize the ExperimentManager with a pipeline, query loaders, measures, and qrels.

        Args:
            pipeline (list[Component]): Pipeline of components to process queries.
            query_loaders (list[BaseQueryLoader]): List of query loaders to load queries.
            measures_by_loader (list[list[Measure]]): List of measures for each query loader to evaluate results.
            qrels (list[str]): List of qrel file paths for each query loader.
            experiment_name (str | None, optional): The name of the experiment

        Raises:
            ValueError: If the number of sets of measures does not match the number of query loaders or qrels.
            ValueError: If the number of qrels does not match the number of query loaders.
        """
        if len(measures_by_loader) != len(query_loaders):
            raise ValueError("Number of measure-lists must match number of query loaders")
        if len(qrels) != len(query_loaders):
            raise ValueError("Number of qrels must match number of query loaders")

        self.pipeline = pipeline
        self.query_loaders = query_loaders
        self.measures_and_qrel_lookup = self._create_measures_and_qrel_lookup(
            query_loaders, measures_by_loader, qrels
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = experiment_name or f"experiment_results_{ts}"
        self.output_path = Path.cwd() / "reports" / experiment_name
        self.output_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_flat_measures(
        cls,
        pipeline: list[Component],
        query_loaders: list[BaseQueryLoader],
        flat_measures: list[Measure],
        qrels: list[str],
    ) -> Self:
        """Convenience constructor for one flat list of measures to apply to every loader."""
        # replicate the same flat list for each loader
        measures_by_loader = [flat_measures for _ in query_loaders]
        return cls(pipeline, query_loaders, measures_by_loader, qrels)

    def _create_measures_and_qrel_lookup(
        self,
        query_loaders: list[BaseQueryLoader],
        measures: list[list[Measure]],
        qrels: list[str],
    ) -> dict[str, tuple[list[Measure], str]]:
        """Create a lookup dictionary for measures and qrels based on query loaders.

        Args:
            query_loaders (list[BaseQueryLoader]): A list of query loaders.
            measures (list[list[BaseMeasure]]): A list of measures.
            qrels (list[Path]): A list of qrel file paths.

        Returns:
            dict[str, tuple[list[measures], Path]]: A dictionary mapping query loader names to their corresponding measures and qrel paths.
        """
        return {
            query_loader.name: (measure_list, qrel)
            for query_loader, measure_list, qrel in zip(query_loaders, measures, qrels)
        }

    def run_experiment(self, rounds: int = 1) -> None:
        """Run the experiment by loading queries and passing them through the pipeline.

        Args:
            rounds (int): The number of times to run the pipeline for.
        """
        for query_loader in tqdm.tqdm(self.query_loaders):
            queries = query_loader.get_queries()
            logger.info(f"Loaded {len(queries)} queries from {query_loader.name}")

            for round in range(1, rounds + 1):
                logger.info(f"Starting Round {round}/{rounds}...")

                for component in self.pipeline:
                    logger.info(f"Running {component.name} on {query_loader.name}")
                    queries = component.run(queries)
                    self._save_results(queries, component.name, query_loader.name, round)
                    # Can think of doing component specific evaluation here.
                self._evaluate_aggregate(queries, query_loader.name, round)

    def _save_results(
        self, queries: list[Query], component_name: str, query_loader_name: str, round: int
    ) -> None:
        """Save the results of the experiment to a file.

        Args:
            queries (list[Query]): The list of queries with results.
            component_name (str): The name of the component that processed the queries.
            query_loader_name (str): The name of the query loader used.
            round (int): The current iteration of the pipeline run
        """
        output_path = self.output_path / query_loader_name / str(round) / component_name
        output_path.mkdir(parents=True, exist_ok=True)

        for query in queries:
            query_file_path = output_path / f"{query.id}.jsonl"
            with jsonlines.open(query_file_path, mode="w") as writer:
                passages = [asdict(p) for p in query.passages or []]
                writer.write_all(passages)

    def _evaluate_aggregate(
        self, queries: list[Query], query_loader_name: str, round: int
    ) -> None:
        """Evaluate the results of the experiment using the specified measures.

        Args:
            queries (list[Query]): The list of queries with results.
            query_loader_name (str): The name of the query loader used.
            round (int): The current iteration of the pipeline run
        """
        measures, qrel_path = self.measures_and_qrel_lookup[query_loader_name]
        qrels = list(ir_measures.read_trec_qrels(qrel_path))
        scored_docs = [
            ScoredDoc(str(query.id), str(passage.id), passage.score)
            for query in queries
            for passage in query.passages or []
        ]
        results = ir_measures.calc_aggregate(measures, qrels, scored_docs)
        parsed_results = {str(k): v for k, v in results.items()}

        output_path = self.output_path / query_loader_name / str(round) / "aggregate_results.json"
        with open(output_path, "w") as f:
            json.dump(parsed_results, f, indent=4)


if __name__ == "__main__":
    ikat_search_engine = LuceneSearcher("..")
    ikat_search_engine.set_bm25(4.46, 0.82)
    monoT5_model = T5ForConditionalGeneration.from_pretrained("castorini/monot5-base-msmarco-10k")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    monoT5_model = monoT5_model.to(device)
    monoT5_tokenizer = T5TokenizerFast.from_pretrained("castorini/monot5-base-msmarco-10k")

    query_loaders: list[BaseQueryLoader] = [iKAT2023("...")]

    # Passages are json strings with an id and contents field. What we care about is in contents.
    def result_parser_fn(res_text: str) -> str:
        return json.loads(res_text).get("contents", "") if res_text else ""

    pipeline = [
        PyseriniLuceneSearcher(
            search_engine=ikat_search_engine, result_parser_fn=result_parser_fn
        ),
        MonoT5Reranker(model=monoT5_model, tokenizer=monoT5_tokenizer),
    ]

    qrels = [".."]
    measures = [
        [
            ir_measures.R @ 1000,
            ir_measures.R @ 20,
            ir_measures.P @ 20,
            ir_measures.MAP @ 1000,
            ir_measures.nDCG @ 3,
            ir_measures.nDCG @ 5,
            ir_measures.nDCG @ 1000,
        ]
    ]

    experiment_manager = ExperimentManager(
        pipeline=pipeline,
        query_loaders=query_loaders,
        measures_by_loader=measures,
        qrels=qrels,
        experiment_name="ikat/baseline",
    )

    with open(experiment_manager.output_path / "README.md", "w") as f:
        f.write(
            f"iKAT Baseline\n"
            "Ran with the following components:\n"
            f"{'\n'.join([component.name for component in pipeline])}"
        )
    experiment_manager.run_experiment()
    logger.info("Finished running experiment")
