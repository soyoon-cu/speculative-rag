"""Build and load a FAISS index."""

from __future__ import annotations

import contextlib
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
import torch
import typer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

_CONTRIEVER_MODEL = "facebook/contriever-msmarco"
_PASSAGE_BATCH = 256
_DIM = 768


def load_passages(
    tsv_path: str | Path, max_passages: int | None = None
) -> tuple[list[str], list[str], list[str]]:
    """Parse the DPR psgs_w100.tsv file.

    Columns: id \\t text \\t title
    Returns (ids, texts, titles).
    """
    ids, texts, titles = [], [], []
    with open(tsv_path, encoding="utf-8") as fh:
        next(fh)
        for i, line in enumerate(fh):
            if max_passages is not None and i >= max_passages:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            pid = parts[0]
            text = parts[1]
            title = parts[2] if len(parts) > 2 else ""
            ids.append(pid)
            texts.append(text)
            titles.append(title)
    return ids, texts, titles


def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.inference_mode()
def embed_passages(
    texts: list[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int = _PASSAGE_BATCH,
) -> np.ndarray:
    all_embs: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding passages"):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        outputs = model(**inputs)
        embs = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
        all_embs.append(embs.cpu().float().numpy())
    return np.vstack(all_embs)


class FAISSIndex:
    """Wraps a flat FAISS index (inner-product) with passage text lookup."""

    def __init__(self, index: faiss.Index, texts: list[str], titles: list[str]) -> None:
        self._index = index
        self._texts = texts
        self._titles = titles

    def search(self, query_emb: np.ndarray, top_k: int) -> list[str]:
        """Return top-k passage texts for a single (1, D) query embedding."""
        q = np.ascontiguousarray(query_emb, dtype=np.float32)
        distances, indices = self._index.search(q, top_k)
        passages: list[str] = []
        for idx in indices[0]:
            if idx == -1:
                continue
            title = self._titles[idx]
            text = self._texts[idx]
            passages.append(f"{title}\n{text}" if title else text)
        return passages

    def save(self, index_path: str | Path, meta_path: str | Path) -> None:
        faiss.write_index(self._index, str(index_path))
        with open(meta_path, "wb") as fh:
            pickle.dump({"texts": self._texts, "titles": self._titles}, fh)
        logger.info("Saved FAISS index → %s  metadata → %s", index_path, meta_path)

    @classmethod
    def load(cls, index_path: str | Path, meta_path: str | Path) -> FAISSIndex:
        index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as fh:
            meta = pickle.load(fh)
        logger.info("Loaded FAISS index with %d vectors from %s", index.ntotal, index_path)
        return cls(index, meta["texts"], meta["titles"])


def build_index(
    passages_path: str | Path,
    index_path: str | Path,
    meta_path: str | Path,
    passages_subset: int | None = None,
    device: str | None = None,
) -> FAISSIndex:
    """Embed all passages and write FAISS flat-IP index to disk."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading passages from %s (subset=%s)…", passages_path, passages_subset)
    _, texts, titles = load_passages(passages_path, max_passages=passages_subset)
    logger.info("Loaded %d passages. Loading Contriever model…", len(texts))

    tokenizer = AutoTokenizer.from_pretrained(_CONTRIEVER_MODEL)
    model = AutoModel.from_pretrained(_CONTRIEVER_MODEL, use_safetensors=True).to(device)
    model.eval()

    embeddings = embed_passages(texts, model, tokenizer, device)

    logger.info("Building FAISS flat IP index (dim=%d, n=%d)…", _DIM, len(texts))
    index = faiss.IndexFlatIP(_DIM)
    if device.startswith("cuda"):
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception:
            logger.warning("faiss-gpu not available; falling back to CPU FAISS index.")
    index.add(embeddings)

    if device.startswith("cuda"):
        with contextlib.suppress(Exception):
            index = faiss.index_gpu_to_cpu(index)

    fi = FAISSIndex(index, texts, titles)
    fi.save(index_path, meta_path)
    return fi


app = typer.Typer(add_completion=False)


@app.command()
def main(
    passages: Path = typer.Option(..., help="Path to psgs_w100.tsv"),
    output: Path = typer.Option(..., help="Output path for FAISS index file"),
    meta: Path = typer.Option(..., help="Output path for passages metadata pickle"),
    passages_subset: int | None = typer.Option(
        None,
        "--passages-subset",
        help="Index only the first N passages (useful for testing on CPU).",
    ),
    device: str | None = typer.Option(None, help="torch device, e.g. 'cuda' or 'cpu'"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_index(passages, output, meta, passages_subset=passages_subset, device=device)
    print(f"Index written to {output}")


if __name__ == "__main__":
    app()
