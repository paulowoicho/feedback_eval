from enum import StrEnum, auto
import logging
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI
from pyserini.search.lucene import LuceneSearcher  # type: ignore[import-untyped]
from retry import retry  # type: ignore[import-untyped]

from larf.data_models.query import Passage, Query

from ...base import Component

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()

openai_client = OpenAI()


@retry(tries=3, delay=0.1)
def _get_contextual_summary(query: str, passage: str) -> str:
    """Generate a summary of the passage that answers the query.

    Args:
        query (str): The question to be answered.
        passage (str): The passage to be summarized.

    Returns:
        str: The generated summary.
    """
    prompt = f"Generate a concise summary of the Passage so that it completely answers the Question: \n\nQuestion: {query}\n\nPassage: {passage}\n\nSummary:"
    messages = [{"role": "user", "content": prompt}]

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,  # type: ignore[arg-type]
        temperature=0,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0,
    )

    return response.choices[0].message.content or ""


class ExpansionMethod(StrEnum):
    """Enum for for the different expansion methods."""

    FULL_TEXT = auto()
    SUMMARY = auto()
    RM3 = auto()


EXPANSION_METHOD_TO_FN: dict[ExpansionMethod, Callable[[str, str], str]] = {
    ExpansionMethod.FULL_TEXT: lambda query, passage: passage,
    ExpansionMethod.SUMMARY: _get_contextual_summary,
    ExpansionMethod.RM3: lambda query, passage: passage,
}


class PyseriniBasedQBDExpander(Component):
    """Expander that uses a Pyserini-based search engine to expand the passage pool for a query."""

    def __init__(
        self,
        search_engine: LuceneSearcher,
        num_neighbours: int = 10,
        rerank_budget: int = 1000,
        result_parser_fn: Callable[[str], str] | None = None,
        expansion_method: ExpansionMethod = ExpansionMethod.FULL_TEXT,
        num_threads: int = 1024,
    ) -> None:
        """Initialize the PyseriniBasedExpander.

        Args:
            search_engine (LuceneSearcher): An instance of LuceneSearcher to use for retrieval.
            num_neighbours (int, optional): The number of extra passages to retrieve per k in top_k. Defaults to 10.
            rerank_budget (int, optional): The number of passages to collect in the document pool through expansion. Defaults to 1000.
            result_parser_fn (Callable[[str], str], optional): A function to parse the result text. If None, a default parser is used. Defaults to None.
            expansion_method (ExpansionMethod, optional): The method to use for expansion. Defaults to ExpansionMethod.NAIVE.
        """
        self.search_engine = search_engine
        self.num_neighbours = num_neighbours
        self.rerank_budget = rerank_budget
        self.num_threads = num_threads

        if not result_parser_fn:
            logger.warning(
                "No result parser function provided. Using default parser that may include noisy metadata in parsed results. If you want better control, pass in your own parser."
            )
            self._result_parser_fn: Callable[[str], str] = lambda res_txt: res_txt
        else:
            self._result_parser_fn = result_parser_fn
        self.expansion_method = expansion_method

        if expansion_method not in EXPANSION_METHOD_TO_FN:
            raise ValueError(
                f"Invalid expansion method: {expansion_method}. Supported methods are: {list(EXPANSION_METHOD_TO_FN.keys())}"
            )
        self._expansion_fn: Callable[[str, str], str] = EXPANSION_METHOD_TO_FN[expansion_method]

    def run(self, queries: list[Query]) -> list[Query]:
        if self.expansion_method == ExpansionMethod.RM3 and not self.search_engine.is_using_rm3():
            logger.info("Setting RM3 for the search engine.")
            self.search_engine.set_rm3()

        for query in queries:
            self._expand_single(query)

        if self.search_engine.is_using_rm3():
            self.search_engine.unset_rm3()

        return queries

    def _expand_single(self, query: Query) -> None:
        """Expand the passages for a given query using the Pyserini search engine.

        Args:
            query (Query): The query object containing the passages to be expanded.
        """

        if len(query.passages) >= self.rerank_budget:
            logger.info("Already reached rerank budget for query: %s", query.id)
            return

        seeds = list(query.passages)
        expansion_queries = {s.id: self._expansion_fn(query.text, s.text) for s in seeds}

        neighbours = self.search_engine.batch_search(
            queries=list(expansion_queries.values()),
            qids=list(expansion_queries.keys()),
            k=self.num_neighbours,
            threads=self.num_threads,
        )

        seen_ids = set(expansion_queries.keys())
        new_passages: list[Passage] = []

        for hits in neighbours.values():
            for hit in hits:
                if hit.docid in seen_ids:
                    continue
                new_passages.append(
                    Passage(
                        id=hit.docid,
                        text=self._result_parser_fn(hit.lucene_document.get("raw") or ""),
                        score=hit.score,
                    )
                )
                seen_ids.add(hit.docid)

                if len(seeds) + len(new_passages) >= self.rerank_budget:
                    query.set_passages(seeds + new_passages)
                    return
        # If we reach here, we have not reached the rerank budget
        # but we have exhausted the seed set of results to use for expansion.
        logger.info("Exhausted seed set of results to expand for query: %s", query.id)
        query.set_passages(seeds + new_passages)
