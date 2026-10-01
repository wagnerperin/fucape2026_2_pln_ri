"""
LSA — Análise Semântica Latente (Latent Semantic Analysis).

Aplica decomposição em valores singulares truncada (``TruncatedSVD``) à
matriz termo-documento TF-IDF, projetando documentos e consultas em um espaço
latente de ``k`` dimensões (padrão: 100).

A intuição é que as ``k`` primeiras direções singulares capturam padrões de
coocorrência: termos que aparecem nos mesmos contextos acabam próximos no
espaço reduzido. Isso dá ao modelo uma forma rudimentar de tratar sinonímia
— ``notebook`` e ``laptop`` passam a ter representações parecidas mesmo sem
nunca terem sido declarados sinônimos — ao custo de perder a precisão
literal dos modelos de termo exato.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import config
from src import utils
from src.indexer import Timer
from src.retrieval.vector_search import VectorIndex, build_vector_index

logger = logging.getLogger(__name__)


@dataclass
class LSAIndex:
    """
    Índice do modelo LSA.

    Attributes:
        vectorizer: ``TfidfVectorizer`` ajustado ao corpus.
        svd_model: ``TruncatedSVD`` ajustado à matriz TF-IDF.
        doc_embeddings: Matriz densa ``(n_docs, k)`` já normalizada em L2, o
            que reduz o cosseno a um produto interno.
        product_ids: ``doc_idx -> product_id``.
        n_components: Dimensionalidade efetiva do espaço latente.
        explained_variance: Fração da variância retida pelas ``k`` dimensões.
        indexing_time_ms: Tempo gasto nesta chamada. Quando uma matriz
            TF-IDF pronta é reaproveitada, mede apenas a decomposição SVD —
            o custo total do modelo é somado em
            :meth:`src.engine.SearchEngine.build`.
    """

    vectorizer: TfidfVectorizer
    svd_model: TruncatedSVD
    doc_embeddings: np.ndarray
    product_ids: List[int]
    n_components: int
    explained_variance: float = 0.0
    indexing_time_ms: float = 0.0

    @property
    def n_docs(self) -> int:
        """Número de documentos indexados."""
        return int(self.doc_embeddings.shape[0])


def build_lsa_index(
    corpus: Sequence[Dict],
    texts: Optional[Sequence[str]] = None,
    k: int = config.LSA_COMPONENTS,
    vector_index: Optional[VectorIndex] = None,
    seed: int = config.RANDOM_SEED,
) -> LSAIndex:
    """
    Ajusta o TF-IDF e o SVD truncado sobre o corpus.

    Args:
        corpus: Lista de produtos.
        texts: Textos já concatenados, se disponíveis.
        k: Número de dimensões latentes. É truncado automaticamente para
            ``min(k, n_docs - 1, |V| - 1)``, limite do SVD.
        vector_index: Índice TF-IDF pronto, para reaproveitamento. Quando
            informado, apenas o SVD é cronometrado.
        seed: Semente do solver randomizado, para reprodutibilidade.

    Returns:
        O :class:`LSAIndex` construído.

    Exemplo:
        >>> corpus = [
        ...     {"product_id": i, "name": f"Mouse {i}",
        ...      "description": "mouse óptico sem fio para notebook",
        ...      "category": "Informática"}
        ...     for i in range(1, 6)
        ... ]
        >>> index = build_lsa_index(corpus, k=2)
        >>> index.n_docs
        5
    """
    if vector_index is None:
        vector_index = build_vector_index(corpus, texts)
        base_time = vector_index.indexing_time_ms
    else:
        base_time = 0.0

    tfidf_matrix = vector_index.tfidf_matrix
    n_docs, n_terms = tfidf_matrix.shape
    n_components = max(1, min(k, n_docs - 1, n_terms - 1))
    if n_components != k:
        logger.warning(
            "k=%d reduzido para %d (limite: min(n_docs, |V|) - 1).", k, n_components
        )

    with Timer() as timer:
        svd_model = TruncatedSVD(
            n_components=n_components, random_state=seed, algorithm="randomized"
        )
        embeddings = svd_model.fit_transform(tfidf_matrix)
        # Normalizar em L2 permite calcular o cosseno como produto interno.
        doc_embeddings = normalize(embeddings).astype(np.float32)

    explained = float(svd_model.explained_variance_ratio_.sum())
    index = LSAIndex(
        vectorizer=vector_index.vectorizer,
        svd_model=svd_model,
        doc_embeddings=doc_embeddings,
        product_ids=list(vector_index.product_ids),
        n_components=n_components,
        explained_variance=explained,
        indexing_time_ms=base_time + timer.elapsed_ms,
    )
    logger.info(
        "Índice LSA: %d docs em %d dimensões (%.1f%% da variância), %.1f ms.",
        index.n_docs,
        n_components,
        explained * 100.0,
        index.indexing_time_ms,
    )
    return index


def lsa_search(
    query: str,
    corpus: Optional[Sequence[Dict]],
    svd_model: TruncatedSVD,
    tfidf_vectorizer: TfidfVectorizer,
    k: int = config.LSA_COMPONENTS,
    top_k: int = config.TOP_K,
    doc_embeddings: Optional[np.ndarray] = None,
    product_ids: Optional[Sequence[int]] = None,
) -> List[Tuple[int, float]]:
    """
    Busca por similaridade de cosseno no espaço semântico latente.

    Args:
        query: Consulta em texto livre. Operadores booleanos são ignorados.
        corpus: Lista de documentos, usada apenas para recuperar os
            ``product_id`` quando ``product_ids`` não é informado.
        svd_model: ``TruncatedSVD`` já ajustado.
        tfidf_vectorizer: ``TfidfVectorizer`` já ajustado.
        k: Dimensionalidade do espaço latente (informativo; a dimensão real é
            a do ``svd_model`` recebido).
        top_k: Número de resultados a retornar.
        doc_embeddings: Embeddings normalizados dos documentos. Se ausentes,
            são recalculados — o que é bem mais lento e só faz sentido em
            testes.
        product_ids: Identificadores dos documentos.

    Returns:
        Lista de ``(product_id, similaridade)`` ordenada por similaridade
        decrescente. A similaridade fica em ``[-1, 1]``.

    Raises:
        ValueError: Se não for possível determinar os ``product_id``.

    Exemplo:
        >>> corpus = [
        ...     {"product_id": i, "name": f"Mouse {i}",
        ...      "description": "mouse óptico sem fio", "category": "Informática"}
        ...     for i in range(1, 6)
        ... ]
        >>> index = build_lsa_index(corpus, k=2)
        >>> results = lsa_search("mouse", corpus, index.svd_model,
        ...                      index.vectorizer, doc_embeddings=index.doc_embeddings,
        ...                      product_ids=index.product_ids)
        >>> len(results) > 0
        True
    """
    if product_ids is None:
        if corpus is None:
            raise ValueError("Informe 'product_ids' ou 'corpus'.")
        product_ids = [int(product["product_id"]) for product in corpus]

    cleaned = utils.strip_operators(query)
    if not cleaned:
        return []

    query_tfidf = tfidf_vectorizer.transform([cleaned])
    if query_tfidf.nnz == 0:
        logger.debug("Nenhum termo de %r está no vocabulário TF-IDF.", query)
        return []

    query_latent = svd_model.transform(query_tfidf)
    norm = np.linalg.norm(query_latent)
    if norm == 0.0:
        # A consulta caiu no núcleo da projeção: não há direção latente.
        return []
    query_latent = query_latent / norm

    if doc_embeddings is None:
        logger.warning("doc_embeddings ausente; recalculando a projeção do corpus.")
        if corpus is None:
            raise ValueError("Sem 'doc_embeddings', o 'corpus' é obrigatório.")
        documents = utils.build_corpus_texts(corpus)
        doc_embeddings = normalize(
            svd_model.transform(tfidf_vectorizer.transform(documents))
        )

    scores = doc_embeddings @ query_latent.ravel().astype(doc_embeddings.dtype)
    return utils.top_k_from_scores(scores, product_ids, top_k, min_score=0.0)


def lsa_search_with_index(
    query: str,
    index: LSAIndex,
    top_k: int = config.TOP_K,
) -> List[Tuple[int, float]]:
    """
    Atalho de :func:`lsa_search` a partir de um :class:`LSAIndex`.

    Args:
        query: Consulta em texto livre.
        index: Índice LSA construído por :func:`build_lsa_index`.
        top_k: Número de resultados.

    Returns:
        Lista de ``(product_id, similaridade)``.
    """
    return lsa_search(
        query=query,
        corpus=None,
        svd_model=index.svd_model,
        tfidf_vectorizer=index.vectorizer,
        k=index.n_components,
        top_k=top_k,
        doc_embeddings=index.doc_embeddings,
        product_ids=index.product_ids,
    )
