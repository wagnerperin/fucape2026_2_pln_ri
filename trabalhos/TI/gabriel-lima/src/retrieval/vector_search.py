"""
Modelo de Espaço Vetorial — TF-IDF com similaridade de cosseno.

Documentos e consulta são representados como vetores em um espaço de
dimensão ``|V|`` (o vocabulário). Cada coordenada é o peso TF-IDF do termo, e
a relevância é o cosseno do ângulo entre o vetor da consulta e o do documento.

Como o ``TfidfVectorizer`` do scikit-learn já normaliza as linhas em L2, o
cosseno se reduz a um produto interno — o que permite pontuar o corpus inteiro
com uma única multiplicação de matriz esparsa.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from src import utils
from src.indexer import Timer

logger = logging.getLogger(__name__)


@dataclass
class VectorIndex:
    """
    Índice do modelo vetorial.

    Attributes:
        vectorizer: ``TfidfVectorizer`` já ajustado ao corpus.
        tfidf_matrix: Matriz esparsa ``(n_docs, |V|)`` normalizada em L2.
        product_ids: ``doc_idx -> product_id``.
        indexing_time_ms: Tempo de construção, em milissegundos.
    """

    vectorizer: TfidfVectorizer
    tfidf_matrix: object
    product_ids: List[int]
    indexing_time_ms: float = 0.0

    @property
    def vocabulary(self) -> Dict[str, int]:
        """Mapa ``termo -> coluna`` aprendido pelo vetorizador."""
        return self.vectorizer.vocabulary_

    @property
    def n_docs(self) -> int:
        """Número de documentos indexados."""
        return self.tfidf_matrix.shape[0]


def build_vector_index(
    corpus: Sequence[Dict],
    texts: Optional[Sequence[str]] = None,
    min_df: int = config.TFIDF_MIN_DF,
    max_df: float = config.TFIDF_MAX_DF,
) -> VectorIndex:
    """
    Ajusta o TF-IDF sobre o corpus.

    O ``analyzer`` é :func:`src.utils.preprocess`, o mesmo do índice
    invertido: sem isso, o modelo vetorial enxergaria um vocabulário diferente
    do BM25 e a comparação perderia validade.

    Args:
        corpus: Lista de produtos.
        texts: Textos já concatenados, se disponíveis.
        min_df: Frequência documental mínima para um termo entrar no
            vocabulário.
        max_df: Proporção máxima de documentos em que um termo pode aparecer
            antes de ser descartado como termo genérico demais.

    Returns:
        O :class:`VectorIndex` construído.

    Exemplo:
        >>> corpus = [{"product_id": 1, "name": "Mouse",
        ...            "description": "mouse óptico", "category": "Informática"}]
        >>> index = build_vector_index(corpus)
        >>> index.n_docs
        1
    """
    with Timer() as timer:
        documents = (
            list(texts) if texts is not None else utils.build_corpus_texts(corpus)
        )
        vectorizer = TfidfVectorizer(
            analyzer=utils.preprocess,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        tfidf_matrix = vectorizer.fit_transform(documents)
        product_ids = [int(product["product_id"]) for product in corpus]

    index = VectorIndex(
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        product_ids=product_ids,
        indexing_time_ms=timer.elapsed_ms,
    )
    logger.info(
        "Índice TF-IDF: %d docs x %d termos, %.1f ms.",
        tfidf_matrix.shape[0],
        tfidf_matrix.shape[1],
        timer.elapsed_ms,
    )
    return index


def vector_search(
    query: str,
    tfidf_matrix: object,
    vocabulary: VectorIndex,
    top_k: int = config.TOP_K,
) -> List[Tuple[int, float]]:
    """
    Busca por similaridade de cosseno no espaço TF-IDF.

    Args:
        query: Consulta em texto livre. Operadores booleanos são ignorados.
        tfidf_matrix: Matriz TF-IDF do corpus (``index.tfidf_matrix``).
        vocabulary: O :class:`VectorIndex` completo — carrega o vetorizador
            ajustado e o mapa de ``product_id``, ambos necessários para
            projetar a consulta no mesmo espaço dos documentos.
        top_k: Número de resultados a retornar.

    Returns:
        Lista de ``(product_id, similaridade)`` ordenada por similaridade
        decrescente. A similaridade fica em ``[0, 1]``.

    Exemplo:
        >>> corpus = [{"product_id": 1, "name": "Mouse",
        ...            "description": "mouse óptico", "category": "Informática"}]
        >>> index = build_vector_index(corpus)
        >>> pid, score = vector_search("mouse", index.tfidf_matrix, index)[0]
        >>> pid
        1
    """
    # Passa-se o texto bruto (apenas sem operadores booleanos): o analyzer do
    # vetorizador já é utils.preprocess, e reenviar radicais aplicaria
    # tokenização e stemming duas vezes sobre os mesmos termos.
    cleaned = utils.strip_operators(query)
    if not cleaned:
        return []

    query_vector = vocabulary.vectorizer.transform([cleaned])
    if query_vector.nnz == 0:
        logger.debug("Nenhum termo de %r está no vocabulário TF-IDF.", query)
        return []

    scores = (tfidf_matrix @ query_vector.T).toarray().ravel()
    return utils.top_k_from_scores(scores, vocabulary.product_ids, top_k)
