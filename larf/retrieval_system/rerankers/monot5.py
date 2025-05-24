import torch
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast

from ...data_models.query import Passage, Query
from ..base import Component


class MonoT5Reranker(Component):
    """Wrapper for the MonoT5 Reranker available on the HuggingFace Hub."""

    def __init__(
        self,
        model: T5ForConditionalGeneration,
        tokenizer: T5TokenizerFast,
        batch_size: int = 64,
        num_passages_to_return: int = 1000,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the MonoT5 Reranker.

        Args:
            model (T5ForConditionalGeneration): The pre-trained MonoT5 model.
            tokenizer (T5TokenizerFast): The tokenizer for the MonoT5 model.
            batch_size (int, optional): The batch size for processing. Defaults to 64.
            num_passages_to_return (int, optional): The number of passages to return. Defaults to 1000.
            device: Torch device (e.g., 'cuda' or 'cpu'). Defaults to None, which uses the model's device.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_passages_to_return = num_passages_to_return
        self.device = device or next(model.parameters()).device
        self.false_id, self.true_id = tokenizer(
            ["false", "true"], add_special_tokens=False
        ).input_ids

    def run(self, queries: list[Query]) -> list[Query]:
        for query in queries:
            self._rerank_single(query)
        return queries

    def _rerank_single(self, query: Query) -> None:
        """Rerank the passages for a given query using the MonoT5 model.

        Args:
            query (Query): The query object containing the passages to be reranked.
        """
        prompts = [
            f"Query: {query.text} Document: {p.text} Relevant:" for p in query.passages or []
        ]

        scores = []
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[start : start + self.batch_size]
            scores.extend(self._score_batch(batch_prompts))

        scored_passages = [
            Passage(id=p.id, text=p.text, score=s)
            for p, s in zip(query.passages, scores, strict=True)
        ]

        scored_passages = sorted(scored_passages, key=lambda p: p.score, reverse=True)[
            : self.num_passages_to_return
        ]
        query.set_passages(scored_passages)

    def _score_batch(self, prompts: list[str]) -> list[float]:
        """Call the model to get the logits for the given prompts.

        Args:
            prompts (list[str]): The list of prompts to be passed to the model.
        """
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=512,
            return_attention_mask=True,
        ).to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
        )

        if not hasattr(output, "scores") or output.scores is None:
            raise RuntimeError(
                "MonoT5Reranker.generate() did not return scores; cannot compute relevance probabilities."
            )

        batch_logits = output.scores[0]
        # Select false/true logits and compute log-softmax
        pair_logits = batch_logits[:, [self.false_id, self.true_id]]
        log_probs = torch.log_softmax(pair_logits, dim=1)[:, 1]
        probs = log_probs.tolist()
        return [p[0] for p in probs]
