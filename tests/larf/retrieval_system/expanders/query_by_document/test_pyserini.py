import pytest

import larf.retrieval_system.expanders.query_by_document.pyserini as mod
from larf.retrieval_system.expanders.query_by_document.pyserini import (
    ExpansionMethod,
    PyseriniBasedQBDExpander,
)


@pytest.mark.parametrize(
    "method,stub_fn",
    [
        (ExpansionMethod.FULL_TEXT, lambda q, ps: ps),
        (ExpansionMethod.RM3, lambda q, ps: "RM3"),
        (ExpansionMethod.SUMMARY, lambda q, ps: "SUMQ"),
    ],
)
def test_expansion(method, stub_fn, make_query, make_passage, make_hit, make_search_engine):
    mod.EXPANSION_METHOD_TO_FN[method] = stub_fn
    seeds = [make_passage("A1", "s1"), make_passage("A2", "s2")]
    q = make_query("QF", passages=seeds.copy())

    hits_map = {"QF:::A1": [make_hit("B1", 0.1, "r1")], "QF:::A2": [make_hit("B2", 0.2, "r2")]}
    searcher = make_search_engine(hits_map, using_rm3=True)
    expander = PyseriniBasedQBDExpander(
        search_engine=searcher,
        num_neighbours=1,
        rerank_budget=10,
        num_threads=2,
        expansion_method=method,
    )
    out = expander.run([q])

    ids = [p.id for p in out[0].passages]
    assert ids == ["A1", "A2", "B1", "B2"]
