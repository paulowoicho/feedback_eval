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

        self._result_parser_fn: Callable[[str], str] = result_parser_fn or (lambda raw: raw)
        self.expansion_method = expansion_method

        self.expansion_fn: Callable[[str, str], str] = EXPANSION_METHOD_TO_FN[expansion_method]

    def _get_expansion_inputs(
        self, queries: list[Query]
    ) -> dict[tuple[str | int, str | int], str]:
        expansion_inputs: dict[tuple[str | int, str | int], str] = {}
        for query in queries:
            if len(query.passages) >= self.rerank_budget:
                logger.info("Already reached rerank budget for query: %s", query.id)
                continue
            for passage in query.passages:
                key = (query.id, passage.id)
                expansion_inputs[key] = self.expansion_fn(query.text, passage.text)

        return expansion_inputs

    def _get_new_passages(
        self, queries: list[Query], neighbours: dict[str, list]
    ) -> dict[str | int, list[Passage]]:
        new_passages_map: dict[str | int, list[Passage]] = {q.id: [] for q in queries}
        seen_map: dict[str | int, set] = {q.id: set(p.id for p in q.passages) for q in queries}

        for full_qid, hits in neighbours.items():
            qid, _ = full_qid.split(":::", 1)
            for hit in hits:
                if hit.docid in seen_map[qid]:
                    continue
                new_passages_map[qid].append(
                    Passage(
                        id=hit.docid,
                        text=self._result_parser_fn(hit.lucene_document.get("raw") or ""),
                        score=hit.score,
                    )
                )
                seen_map[qid].add(hit.docid)
                total = len(seen_map[qid])
                if total >= self.rerank_budget:
                    break

        return new_passages_map

    def run(self, queries: list[Query]) -> list[Query]:
        if self.expansion_method == ExpansionMethod.RM3 and not self.search_engine.is_using_rm3():
            logger.info("Setting RM3 for the search engine.")
            self.search_engine.set_rm3()

        expansion_inputs = self._get_expansion_inputs(queries)

        qids = [f"{qid}:::{pid}" for (qid, pid) in expansion_inputs.keys()]
        q_texts = list(expansion_inputs.values())
        neighbours = self.search_engine.batch_search(
            queries=q_texts,
            qids=qids,
            k=self.num_neighbours,
            threads=self.num_threads,
        )

        new_passages_map = self._get_new_passages(queries, neighbours)

        for query in queries:
            additions = new_passages_map.get(query.id, [])
            if additions:
                query.set_passages(query.passages + additions)

        if self.search_engine.is_using_rm3():
            self.search_engine.unset_rm3()

        return queries
