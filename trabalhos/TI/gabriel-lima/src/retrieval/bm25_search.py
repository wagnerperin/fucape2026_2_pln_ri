"""
Okapi BM25.

Modelo probabilístico de ranqueamento. Para uma consulta ``q`` e um documento
``d``::

    score(d, q) = Σ_{t ∈ q}  IDF(t) · ( f(t,d) · (k1 + 1) )
                             ---------------------------------------------
                             f(t,d) + k1 · ( 1 - b + b · |d| / avgdl )

com o IDF na forma suavizada de Robertson-Spärck Jones::

    IDF(t) = ln( 1 + ( N - df(t) + 0.5 ) / ( df(t) + 0.5 ) )

Os dois hiperparâmetros têm papéis distintos:

* ``k1`` controla a **saturação** da frequência do termo. Com ``k1`` baixo, a
  décima ocorrência de um termo acrescenta quase nada em relação à segunda.
* ``b`` controla a **normalização por comprimento**. Com ``b = 0`` o tamanho
  do documento é ignorado; com ``b = 1`` a normalização é total.

A pontuação percorre apenas as *postings* dos termos da consulta, o que torna
o custo proporcional a ``Σ df(t)`` e não a ``N``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import config
from src import utils
from src.indexer import InvertedIndex

logger = logging.getLogger(__name__)


def compute_idf(doc_freq: int, n_docs: int) -> float:
    """
    Calcula o IDF probabilístico suavizado de um termo.

    Args:
        doc_freq: Número de documentos que contêm o termo.
        n_docs: Total de documentos no corpus.

    Returns:
        Valor de IDF, sempre positivo graças ao ``1 +`` dentro do logaritmo.

    Exemplo:
        >>> round(compute_idf(1, 100), 3)
        4.205
    """
    return float(np.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5)))


def bm25_search(
    query: str,
    inverted_index: InvertedIndex,
    corpus: Optional[Sequence[Dict]] = None,
    k1: float = config.BM25_K1,
    b: float = config.BM25_B,
    top_k: int = config.TOP_K,
) -> List[Tuple[int, float]]:
    """
    Busca usando o algoritmo Okapi BM25.

    Args:
        query: Consulta em texto livre. Operadores booleanos são ignorados.
        inverted_index: Índice invertido pré-computado.
        corpus: Lista de documentos. Não é necessária para o cálculo — todas
            as estatísticas vêm do índice — e é aceita apenas para manter a
            assinatura pedida na especificação.
        k1: Parâmetro de saturação de frequência (padrão 1.5).
        b: Parâmetro de normalização de comprimento (padrão 0.75).
        top_k: Número de resultados a retornar.

    Returns:
        Lista de ``(product_id, relevance_score)`` ordenada por score
        decrescente.

    Raises:
        ValueError: Se ``k1`` for negativo ou ``b`` estiver fora de ``[0, 1]``.

    Exemplo:
        >>> from src.indexer import build_inverted_index
        >>> corpus = [{"product_id": 42, "name": "Mouse Óptico",
        ...            "description": "mouse óptico de precisão",
        ...            "category": "Informática"}]
        >>> index = build_inverted_index(corpus)
        >>> pid, score = bm25_search("mouse óptico", index, corpus)[0]
        >>> pid
        42
    """
    if k1 < 0:
        raise ValueError(f"k1 deve ser não negativo, recebido {k1}.")
    if not 0.0 <= b <= 1.0:
        raise ValueError(f"b deve estar em [0, 1], recebido {b}.")

    terms = utils.preprocess_query(query)
    if not terms:
        return []

    n_docs = inverted_index.n_docs
    if n_docs == 0:
        return []

    scores = np.zeros(n_docs, dtype=np.float64)

    # Denominador da normalização por comprimento: constante por documento,
    # portanto calculado uma única vez por consulta.
    average_length = inverted_index.avg_doc_length or 1.0
    length_norm = k1 * (
        1.0 - b + b * inverted_index.doc_lengths / average_length
    )

    for term in set(terms):
        doc_freq = inverted_index.doc_freq.get(term, 0)
        if doc_freq == 0:
            continue

        doc_indices, frequencies = inverted_index.postings_arrays(term)
        idf = compute_idf(doc_freq, n_docs)

        numerator = frequencies * (k1 + 1.0)
        denominator = frequencies + length_norm[doc_indices]
        scores[doc_indices] += idf * (numerator / denominator)

    return utils.top_k_from_scores(scores, inverted_index.product_ids, top_k)


def bm25_search_default(
    query: str,
    inverted_index: InvertedIndex,
    corpus: Optional[Sequence[Dict]] = None,
    top_k: int = config.TOP_K,
) -> List[Tuple[int, float]]:
    """
    Atalho para o BM25 com os hiperparâmetros padrão (``k1=1.5``, ``b=0.75``).

    Args:
        query: Consulta em texto livre.
        inverted_index: Índice invertido pré-computado.
        corpus: Lista de documentos (não utilizada).
        top_k: Número de resultados.

    Returns:
        Lista de ``(product_id, relevance_score)``.
    """
    return bm25_search(
        query, inverted_index, corpus, config.BM25_K1, config.BM25_B, top_k
    )


def explain_score(
    query: str,
    product_id: int,
    inverted_index: InvertedIndex,
    k1: float = config.BM25_K1,
    b: float = config.BM25_B,
) -> Dict[str, float]:
    """
    Decompõe o score BM25 de um documento, termo a termo.

    Útil para justificar no relatório por que um produto subiu ou desceu no
    ranking.

    Args:
        query: Consulta em texto livre.
        product_id: Identificador do produto a explicar.
        inverted_index: Índice invertido pré-computado.
        k1: Parâmetro de saturação.
        b: Parâmetro de normalização de comprimento.

    Returns:
        Mapa ``termo -> contribuição``, acrescido da chave ``"_total"``.

    Raises:
        KeyError: Se ``product_id`` não estiver no índice.
    """
    try:
        doc_idx = inverted_index.product_ids.index(product_id)
    except ValueError as exc:
        raise KeyError(f"product_id {product_id} não está indexado.") from exc

    contributions: Dict[str, float] = {}
    doc_length = inverted_index.doc_lengths[doc_idx]
    norm = k1 * (1.0 - b + b * doc_length / (inverted_index.avg_doc_length or 1.0))

    for term in set(utils.preprocess_query(query)):
        doc_freq = inverted_index.doc_freq.get(term, 0)
        if doc_freq == 0:
            continue
        doc_indices, frequencies = inverted_index.postings_arrays(term)
        position = np.searchsorted(doc_indices, doc_idx)
        if position >= doc_indices.size or doc_indices[position] != doc_idx:
            continue
        frequency = frequencies[position]
        idf = compute_idf(doc_freq, inverted_index.n_docs)
        contributions[term] = float(
            idf * (frequency * (k1 + 1.0)) / (frequency + norm)
        )

    contributions["_total"] = float(sum(contributions.values()))
    return contributions
