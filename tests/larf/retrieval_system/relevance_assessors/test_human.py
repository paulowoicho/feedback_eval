from larf.retrieval_system.relevance_assessors.human import (
    HumanRelevanceAssessor,
    LLMJudgeSimulator,
    NoisyHumanRelevanceAssessor,
)


def test_human_relevance_assessor(make_query, make_passage, tmp_path):
    lines = ["QA 0 P1 1", "QA 0 P2 2"]
    qrel = tmp_path / "hra.qrel"
    qrel.write_text("\n".join(lines), encoding="utf-8")

    p1 = make_passage("P1")
    p2 = make_passage("P2")
    p3 = make_passage("P3")
    q = make_query("QA", [p1, p2, p3])

    hra = HumanRelevanceAssessor(str(qrel))
    out = hra.run([q])
    assert out is not None and out[0] is q

    # check assignments
    assert p1.relevance_assessment == 1
    assert p2.relevance_assessment == 2
    # default when missing
    assert p3.relevance_assessment == 0


def test_noisy_assessor_zero_prob(make_query, make_passage, tmp_path):
    lines = ["QZ 0 A 0", "QZ 0 B 1"]
    qrel = tmp_path / "noisy0.qrel"
    qrel.write_text("\n".join(lines), encoding="utf-8")

    pA = make_passage("A")
    pB = make_passage("B")
    q = make_query("QZ", [pA, pB])

    na = NoisyHumanRelevanceAssessor(str(qrel), noise_probability=0.0, max_score=1, seed=123)
    na.run([q])

    # noise_probability=0 ⇒ always ground truth
    assert pA.relevance_assessment == 0
    assert pB.relevance_assessment == 1


def test_noisy_assessor_one_prob(make_query, make_passage, tmp_path):
    lines = ["QO 0 X 0", "QO 0 Y 1"]
    qrel = tmp_path / "noisy1.qrel"
    qrel.write_text("\n".join(lines), encoding="utf-8")

    pX = make_passage("X")
    pY = make_passage("Y")
    q = make_query("QO", [pX, pY])

    na = NoisyHumanRelevanceAssessor(str(qrel), noise_probability=1.0, max_score=1, seed=123)
    na.run([q])

    assert pX.relevance_assessment == 1
    assert pY.relevance_assessment == 0


def test_llm_simulator(make_query, make_passage, tmp_path):
    lines = ["QI 0 P1 0", "QI 0 P2 1", "QI 0 P3 5"]
    qrel = tmp_path / "llm.qrel"
    qrel.write_text("\n".join(lines), encoding="utf-8")

    profile = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    p1 = make_passage("P1")
    p2 = make_passage("P2")
    p3 = make_passage("P3")
    q = make_query("QI", [p1, p2, p3])

    sim = LLMJudgeSimulator(str(qrel), probability_profile=profile, max_score=2, seed=999)
    sim.run([q])

    assert p1.relevance_assessment == 0
    assert p2.relevance_assessment == 1
    assert p3.relevance_assessment == 2
