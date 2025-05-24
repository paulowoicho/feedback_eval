import pytest

import larf.retrieval_system.expanders.query_reformulation.pyserini as mod
from larf.retrieval_system.expanders.query_reformulation.pyserini import (
    PyseriniBasedQRExpander,
    ReformulationMethod,
)


@pytest.mark.parametrize(
    "method,stub_fn",
    [
        (ReformulationMethod.FULL_TEXT, lambda q, ps: ps),
        (ReformulationMethod.QUERY_REWRITE, lambda q, ps: "REWQ"),
        (ReformulationMethod.SUMMARY, lambda q, ps: "SUMQ"),
    ],
)
def test_expansion(method, stub_fn, make_query, make_passage, make_search_engine, make_hit):
    mod.REFORMULATION_METHOD_TO_FN[method] = stub_fn

    existing = make_passage("E", "old", 1.0)
    q = make_query("QR", passages=[existing])

    hits = [make_hit("E", 0.5, "r"), make_hit("N", 0.8, "new")]
    searcher = make_search_engine({"QR": hits})
    exp = PyseriniBasedQRExpander(
        search_engine=searcher,
        rerank_budget=5,
        reformulation_method=method,
    )
    out = exp.run([q])
    res = out[0].passages
    assert len(res) == 2
    assert res[1].id == "N"
    assert res[1].score == 0.8
