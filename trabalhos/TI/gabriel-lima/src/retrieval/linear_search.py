"""
Busca linear ingênua — o *baseline* O(N).

Não existe índice: a cada consulta o algoritmo varre sequencialmente todos os
documentos, normaliza o texto na hora e conta quantos termos da query
aparecem. É o comportamento de um ``LIKE '%termo%'`` sem índice invertido, e
serve de piso de comparação para os modelos indexados.

Complexidade por consulta: ``O(N * |d| * |q|)``, onde ``N`` é o número de
documentos, ``|d|`` o comprimento médio do documento e ``|q|`` o número de
termos da consulta.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import config
from src import utils

logger = logging.getLogger(__name__)


def _score_document(text: str, terms: Sequence[str]) -> float:
    """
    Pontua um documento pela cobertura dos termos da consulta.

    O score combina duas informações: quantos termos distintos da query
    aparecem (peso maior, pois define a relevância) e quantas ocorrências no
    total (desempate). Isso dá ao baseline um ranqueamento mínimo, sem o qual
    a comparação de SERPs não faria sentido.

    Args:
        text: Texto do documento já em caixa baixa.
        terms: Termos da consulta, em caixa baixa e sem operadores.

    Returns:
        Score não negativo; zero quando nenhum termo casa.
    """
    matched = 0
    occurrences = 0
    for term in terms:
        count = text.count(term)
        if count:
            matched += 1
            occurrences += count
    if not matched:
        return 0.0
    return matched + min(occurrences, 50) / 100.0


def linear_search(
    query: str,
    corpus: Sequence[Dict],
    top_k: int = config.TOP_K,
) -> List[Dict]:
    """
    Busca sequencial por substring sobre o corpus inteiro, sem índice.

    Args:
        query: Consulta em texto livre. Operadores booleanos, se presentes,
            são tratados como termos comuns e descartados.
        corpus: Lista de produtos a varrer.
        top_k: Número máximo de produtos retornados.

    Returns:
        Lista de produtos (os próprios dicionários do corpus) ordenada por
        score decrescente, limitada a ``top_k``.

    Exemplo:
        >>> corpus = [
        ...     {"product_id": 1, "name": "Mouse Óptico",
        ...      "description": "mouse para escritório", "category": "Informática"},
        ...     {"product_id": 2, "name": "Cafeteira",
        ...      "description": "café coado", "category": "Casa"},
        ... ]
        >>> [p["product_id"] for p in linear_search("mouse", corpus)]
        [1]
    """
    return [corpus[idx] for idx, _ in _ranked_indices(query, corpus, top_k)]


def linear_search_scored(
    query: str,
    corpus: Sequence[Dict],
    top_k: int = config.TOP_K,
) -> List[Tuple[int, float]]:
    """
    Variante de :func:`linear_search` com a assinatura comum ao benchmark.

    Args:
        query: Consulta em texto livre.
        corpus: Lista de produtos a varrer.
        top_k: Número máximo de resultados.

    Returns:
        Lista de ``(product_id, score)`` ordenada por score decrescente.

    Exemplo:
        >>> corpus = [{"product_id": 7, "name": "Mouse",
        ...            "description": "mouse óptico", "category": "Informática"}]
        >>> linear_search_scored("mouse", corpus)[0][0]
        7
    """
    return [
        (int(corpus[idx]["product_id"]), score)
        for idx, score in _ranked_indices(query, corpus, top_k)
    ]


def _ranked_indices(
    query: str,
    corpus: Sequence[Dict],
    top_k: int,
) -> List[Tuple[int, float]]:
    """
    Varre o corpus e devolve os índices mais bem pontuados.

    Args:
        query: Consulta em texto livre.
        corpus: Lista de produtos.
        top_k: Número máximo de resultados.

    Returns:
        Lista de ``(índice no corpus, score)`` ordenada por score decrescente
        e, em caso de empate, por ``product_id`` crescente.
    """
    terms = [
        token
        for token in utils.tokenize(query)
        if token.upper() not in utils.BOOLEAN_OPERATORS
        and len(token) >= config.MIN_TOKEN_LENGTH
    ]
    if not terms:
        logger.debug("Consulta vazia após limpeza: %r", query)
        return []

    scored: List[Tuple[int, float]] = []
    for idx, product in enumerate(corpus):
        # Sem índice: o texto é montado e normalizado a cada consulta.
        text = (
            f"{product['name']} {product.get('category', '')} "
            f"{product['description']}"
        ).lower()
        score = _score_document(text, terms)
        if score > 0.0:
            scored.append((idx, score))

    scored.sort(key=lambda pair: (-pair[1], corpus[pair[0]]["product_id"]))
    return scored[:top_k]
