"""Dense retriever using facebook/contriever-msmarco.

Encodes queries with Contriever and performs ANN search against a pre-built
FAISS index to return the top-k passage texts.
"""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

from rag.retrieval.index import FAISSIndex

_CONTRIEVER_MODEL = "facebook/contriever-msmarco"


def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings, ignoring padding tokens."""
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class ContrieverRetriever:
    """Encode queries with Contriever-MSMARCO and search a FAISS index."""

    def __init__(
        self,
        index: FAISSIndex,
        model_name: str = _CONTRIEVER_MODEL,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.index = index
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, use_safetensors=True).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _encode(self, texts: list[str]) -> torch.Tensor:
        """Return L2-normalised embeddings for a list of texts."""
        all_embeddings: list[torch.Tensor] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            embeddings = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

    def retrieve_batch(self, queries: list[str], top_k: int = 10) -> list[list[str]]:
        """Return top-k passage texts for each query in a batch."""
        q_embs = self._encode(queries).numpy()
        return [self.index.search(q_embs[i : i + 1], top_k) for i in range(len(queries))]
