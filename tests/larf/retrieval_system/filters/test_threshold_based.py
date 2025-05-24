from larf.retrieval_system.filters.threshold_based import ThresholdBasedFilter


def test_filter_removes_below_threshold(make_query, make_passage):
    # passages with relevance 0,1,2,3
    ps = [
        make_passage("p0", "t0", 0.1, 0),
        make_passage("p1", "t1", 0.2, 1),
        make_passage("p2", "t2", 0.3, 2),
        make_passage("p3", "t3", 0.4, 3),
    ]
    q = make_query("Q", ps.copy())
    filt = ThresholdBasedFilter(threshold=2)

    out = filt.run([q])
    assert out is not None and isinstance(out, list)
    assert out[0] is q

    kept = q.passages
    assert [p.id for p in kept] == ["p2", "p3"]


def test_filter_treats_none_as_zero(make_query, make_passage):
    # one passage has relevance_assessment unset (None), one has 1
    p_none = make_passage("p_none", "tn", 0.1, None)
    p1 = make_passage("p1", "t1", 0.2, 1)
    q = make_query("Q2", [p_none, p1])
    filt = ThresholdBasedFilter(threshold=1)

    filt.run([q])
    assert [p.id for p in q.passages] == ["p1"]


def test_filter_empty_passages(make_query):
    q = make_query("Q3", [])
    filt = ThresholdBasedFilter(threshold=0)
    result = filt.run([q])
    assert result == [q]
    assert q.passages == []


def test_filter_multiple_queries(make_query, make_passage):
    # Two queries, each with different passages
    q1 = make_query("A", [make_passage("a1", "", 0, 2), make_passage("a2", "", 0, 1)])
    q2 = make_query("B", [make_passage("b1", "", 0, 3), make_passage("b2", "", 0, 0)])
    filt = ThresholdBasedFilter(threshold=2)

    _ = filt.run([q1, q2])

    assert [p.id for p in q1.passages] == ["a1"]
    assert [p.id for p in q2.passages] == ["b1"]
