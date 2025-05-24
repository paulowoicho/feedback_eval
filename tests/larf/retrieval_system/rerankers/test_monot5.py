import torch

from larf.retrieval_system.rerankers.monot5 import MonoT5Reranker


class DummyModel:
    """Minimal stub of T5ForConditionalGeneration with a .parameters() method."""

    def __init__(self, device="cpu"):
        self._param = torch.tensor([0.0], device=device)

    def parameters(self):
        yield self._param


class DummyTokenizer:
    """Stub of T5TokenizerFast that returns fixed token IDs."""

    def __init__(self, false_id=0, true_id=1):
        self._false_id = false_id
        self._true_id = true_id

    def __call__(self, *args, **kwargs):
        class Batch:
            def __init__(self, false_id, true_id):
                self.input_ids = [[false_id], [true_id]]

            def to(self, device):
                return self

        return Batch(self._false_id, self._true_id)


def test_rerank(make_query, make_passage):
    model = DummyModel()
    tok = DummyTokenizer()

    reranker = MonoT5Reranker(model=model, tokenizer=tok, batch_size=2, num_passages_to_return=3)

    texts = ["short", "tiny", "medium length", "a much longer document", "mid"]
    passages = [make_passage(f"P{i}", text) for i, text in enumerate(texts)]
    q = make_query("Y", passages)

    def fake_score(prompts):
        return [float(len(p)) for p in prompts]

    reranker._score_batch = fake_score

    reranker.run([q])

    sorted_texts = [p.text for p in q.passages]
    expected = ["a much longer document", "medium length", "short"]
    assert sorted_texts == expected
    assert len(q.passages) == 3
