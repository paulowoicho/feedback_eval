from enum import StrEnum, auto
from logging import getLogger
from typing import Callable, cast

from dotenv import load_dotenv
from openai import OpenAI
from pyserini.search.lucene import LuceneSearcher  # type: ignore[import-untyped]
from retry import retry  # type: ignore[import-untyped]

from larf.data_models.query import Passage, Query
from larf.retrieval_system.base import Component

load_dotenv()

openai_client = OpenAI()

logger = getLogger(__name__)


class ReformulationMethod(StrEnum):
    """Enum for the different reformulation methods."""

    QUERY_REWRITE = auto()
    FULL_TEXT = auto()
    SUMMARY = auto()


@retry(tries=3, delay=0.1)
def _call_openai(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0,
    )

    return response.choices[0].message.content or ""


def _generate_query_rewrite(query: str, passages: list[str]) -> str:
    """
    Rewrite a user query so that it is answerable using only the provided passages.

    Args:
        query: The original user question.
        passages: A list of text passages containing the available information.
        model: The OpenAI model to use for rewriting.

    Returns:
        A rewritten question that can be answered by the given passages.
    """
    prompt = (
        "You are a Question-Rewriter whose job is to take an arbitrary user question "
        "and turn it into a new question that can be answered using the information "
        "contained in a given set of passages.\n\n"
        "Your output must satisfy these requirements:\n"
        "1. Answerability: The rewritten question must be fully answerable by the facts, names, dates, "
        "and relationships explicitly stated in the passages.\n"
        "2. Fidelity: Do not introduce any new facts or assumptions that are not present in the passages.\n"
        "3. Clarity: Make the question as clear and specific as possible, referencing the same concepts "
        "used in the passages.\n"
        "4. Conciseness: Keep the question brief; only include what is needed to ensure answerability.\n\n"
        "Passages:\n" + "\n".join(f"- {p}" for p in passages) + "\n\n"
        "Original Question:\n" + query + "\n"
        "Rewritten Question: "
    )

    return _call_openai(prompt=prompt)


def _generate_summary(query: str, passages: list[str]) -> str:
    """
    Generate a concise answer by summarizing the provided passages to address the user question.

    Args:
        query: The user question to answer.
        passages: A list of text passages containing relevant information.
        model: The OpenAI model to use for summarization.

    Returns:
        A concise summary that answers the question based solely on the passages.
    """
    # Build the summarization prompt
    prompt = (
        "You are a Passage Summarizer whose job is to read a set of passages and produce a concise, accurate answer to the user’s question using only the information provided.\n\n"
        "Your output must satisfy these requirements:\n"
        "1. Completeness: Include all key facts from the passages that directly answer the question.\n"
        "2. Fidelity: Do not add any information or assumptions not present in the passages.\n"
        "3. Clarity: Write in clear, direct language, referencing the same names and terms used in the passages.\n"
        "4. Brevity: Keep the answer as short as possible while fully answering the question.\n\n"
        "Question:\n" + query + "\n\n"
        "Passages:\n" + "\n".join(f"- {p}" for p in passages) + "\n"
        "Answer:"
    )

    return _call_openai(prompt=prompt)


def _expand_query(query: str, passages: list[str]) -> str:
    return "\n".join(passages)


REFORMULATION_METHOD_TO_FN: dict[ReformulationMethod, Callable[[str, list[str]], str]] = {
    ReformulationMethod.QUERY_REWRITE: _generate_query_rewrite,
    ReformulationMethod.SUMMARY: _generate_summary,
    ReformulationMethod.FULL_TEXT: _expand_query,
}


class PyseriniBasedQRExpander(Component):
    """Expander that uses a Pyserini-based search engine to expand the passage pool for a query."""

    def __init__(
        self,
        search_engine: LuceneSearcher,
        top_n: int = 10,
        rerank_budget: int = 1000,
        num_threads: int = 1024,
        result_parser_fn: Callable[[str], str] | None = None,
        reformulation_method: ReformulationMethod = ReformulationMethod.QUERY_REWRITE,
    ) -> None:
        """Initialize the ThesholdBasedQueryReformulator.

        Args:
            search_engine (LuceneSearcher): An instance of LuceneSearcher to use for retrieval.
            top_n (int, optional): The number of top passages to consider for reformulation. In a RAG
                system, this would be the number of candidate passages that are considered to generate a
                response to the query. Defaults to 10.
            rerank_budget (int, optional): The number of documents to return for each query. Defaults to 1000.
            num_threads (int, optional): Number of threads to run batch search on. Defaults to 1024.
            result_parser_fn (Callable[[Any], str] | None, optional): Function to parse out the content from a search result. Defaults to None.
            reformulation_method (ReformulationMethod, optional): The type of query reformulation to perform.
        """
        self.search_engine = search_engine
        self.top_n = top_n
        self.rerank_budget = rerank_budget
        self.num_threads = num_threads
        self.reformulation_fn = REFORMULATION_METHOD_TO_FN[reformulation_method]

        if not result_parser_fn:
            logger.warning(
                "No result parser function provided. Using default parser that may include noisy metadata in parsed results. If you want better control, pass in your own parser."
            )
            self._result_parser_fn: Callable[[str], str] = lambda res_txt: res_txt
        else:
            self._result_parser_fn = result_parser_fn

    def run(self, queries: list[Query]) -> list[Query]:
        """Run the query reformulator on the given queries.

        Args:
            queries (list[Query]): The list of queries to reformulate.

        Returns:
            list[Query]: The list of reformulated queries.
        """
        new_queries = {q.id: self._get_new_queries(q) for q in queries}
        qids, texts = list(new_queries.keys()), list(new_queries.values())
        results = self.search_engine.batch_search(
            queries=texts, qids=qids, k=self.rerank_budget, threads=self.num_threads
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
            query.set_passages(updated_passages[: self.rerank_budget])

        return queries

    def _get_new_queries(self, query: Query) -> str:
        """Reformulate the query based on the passages.

        Args:
            query (Query): The query object to reformulate.

        Returns:
            str: A reformulated query to search with.
        """
        query_text = query.text
        passages = cast(list[Passage], query.passages)[: self.top_n]
        rel_passages = [p.text for p in passages]

        if not rel_passages:
            return query_text

        return self.reformulation_fn(query_text, rel_passages)
