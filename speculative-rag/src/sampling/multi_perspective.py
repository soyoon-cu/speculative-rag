import torch

import random

import numpy as np

from sklearn.cluster import KMeans

from transformers import AutoModel, AutoTokenizer


class MultiPerspectiveSampler:
    """
    Implements Multi-Perspective Sampling from ICLR 2025.
    Clusters retrieved documents and creates m subsets of size k.
    """

    def __init__(self, model_name="facebook/contriever-msmarco", device=None, load_model=True):

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        else:
            self.device = device

        self._model_name = model_name

        self.tokenizer = None

        self.model = None

        if load_model:
            print(f"Initializing MultiPerspectiveSampler on {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.model = AutoModel.from_pretrained(model_name).to(self.device)

            self.model.eval()

    @torch.inference_mode()
    def _get_embeddings(self, question, passages):
        """
        Embeds strings with regard to the question instruction.
        Uses mean-pooling consistent with index.py.
        """

        if self.model is None:
            raise RuntimeError(
                "MultiPerspectiveSampler was created with load_model=False. "
                "Pass precomputed_emb to generate_subsets() instead."
            )

        texts = [f"Question: {question} Document: {p}" for p in passages]

        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)

        mask = (
            inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        )

        summed = (outputs.last_hidden_state * mask).sum(dim=1)

        counts = mask.sum(dim=1).clamp(min=1e-9)

        embeddings = summed / counts

        return torch.nn.functional.normalize(embeddings, p=2, dim=-1).cpu().numpy()

    def generate_subsets(self, question, passages, m=5, k=2, precomputed_emb=None):
        """
        Strict implementation of Algorithm 1.
        m: total drafts to generate.
        k: documents per subset.
        precomputed_embs : receive embeddings from FAISS index(no re-encoding)
        """

        if not passages:
            return [[] for _ in range(m)]

        if precomputed_emb is not None:
            assert len(precomputed_emb) == len(passages), (
                f"precomputed_embeddings length {len(precomputed_emb)} "
                f"must match passages length {len(passages)}"
            )

            embeddings = precomputed_emb

        else:
            embeddings = self._get_embeddings(question, passages)

        n_clusters = min(k, len(passages))

        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)

        cluster_labels = kmeans.fit_predict(embeddings)

        clusters = [[] for _ in range(n_clusters)]

        for idx, label in enumerate(cluster_labels):
            clusters[label].append(passages[idx])

        delta_set = []

        seen = set()

        max_attempts = m * 50

        attempts = 0

        while len(delta_set) < m and attempts < max_attempts:
            subset = [random.choice(c) for c in clusters if c]

            subset_key = tuple(sorted(subset))

            if subset_key not in seen:
                seen.add(subset_key)

                delta_set.append(subset)

            attempts += 1

        return delta_set


if __name__ == "__main__":
    test_question = "Who wrote the play Hamlet?"

    test_passages = [
        "William Shakespeare wrote Hamlet in the early 17th century.",
        "Hamlet is a tragedy by Shakespeare set in Denmark.",
        "The play Hamlet features characters like Ophelia and Polonius.",
        "Shakespeare was born in Stratford-upon-Avon.",
        "Danish history inspired the story of Prince Hamlet.",
    ]

    sampler = MultiPerspectiveSampler()

    subsets = sampler.generate_subsets(test_question, test_passages, m=3, k=2)

    print(f"\nGenerated {len(subsets)} unique subsets:")

    for i, s in enumerate(subsets):
        print(f"Subset {i + 1}: {len(s)} documents")
