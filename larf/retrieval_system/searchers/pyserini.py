from logging import getLogger
from typing import Callable

from pyserini.search.lucene import LuceneSearcher  # type: ignore[import-untyped]

from ...data_models.query import Passage, Query
from ..base import Component

logger = getLogger(__name__)


class PyseriniLuceneSearcher(Component):
    """Wrapper for the Pyserini Lucene searcher."""

    def __init__(
        self,
        search_engine: LuceneSearcher,
        num_results: int = 1000,
        result_parser_fn: Callable[[str], str] | None = None,
        num_threads: int = 1024,
    ) -> None:
        """Initialize the Pyserini Lucene searcher.

        Args:
            search_enine (LuceneSearcher): An instance of LuceneSearcher to use for retrieval.
            num_results (int, optional): The number of documents to retrieve for each query. Defaults to 1000.
            result_parser_fn (Callable[[Any], str] | None, optional): Function to parse out the content from a search result. Defaults to None.
            num_threads (int, optional): Number of threads to run batch search on. Defaults to 64.
        """
        self.search_engine = search_engine
        self.num_results = num_results
        self.num_threads = num_threads

        if not result_parser_fn:
            logger.warning(
                "No result parser function provided. Using default parser that may include noisy metadata in parsed results. If you want better control, pass in your own parser."
            )
            self._result_parser_fn: Callable[[str], str] = lambda res_txt: res_txt
        else:
            self._result_parser_fn = result_parser_fn

    def run(self, queries: list[Query]) -> list[Query]:
        texts, qids = zip(*((q.text, q.id) for q in queries))

        results = self.search_engine.batch_search(
            queries=list(texts), qids=list(qids), k=self.num_results, threads=self.num_threads
        )
        for query in queries:
            prev_passages = query.passages
            prev_pids = {p.id for p in prev_passages}

            hits = results.get(query.id, [])
            passages = [
                Passage(
                    id=hit.docid,
                    text=self._result_parser_fn(hit.lucene_document.get("raw") or ""),
                    score=hit.score,
                )
                for hit in hits
                if hit.docid not in prev_pids
            ]

            updated_passages = prev_passages + passages
            query.set_passages(updated_passages[: self.num_results])

        return queries
