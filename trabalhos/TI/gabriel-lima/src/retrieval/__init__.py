"""
Modelos clássicos de Recuperação da Informação.

Cinco algoritmos, todos sobre o mesmo pré-processamento
(:func:`src.utils.preprocess`), de modo que as diferenças observadas no
benchmark venham do modelo e não da tokenização:

======================  ====================================================
Módulo                  Modelo
======================  ====================================================
``linear_search``       Varredura sequencial sem índice (baseline O(N))
``boolean_search``      Modelo booleano sobre matriz de incidência binária
``vector_search``       Espaço vetorial: TF-IDF + similaridade de cosseno
``bm25_search``         Okapi BM25 (probabilístico)
``lsa_search``          Análise Semântica Latente (TF-IDF + TruncatedSVD)
======================  ====================================================
"""

from src.retrieval.bm25_search import bm25_search, bm25_search_default
from src.retrieval.boolean_search import (
    BooleanQueryError,
    boolean_search,
    boolean_search_scored,
    parse_boolean_query,
)
from src.retrieval.linear_search import linear_search, linear_search_scored
from src.retrieval.lsa_search import (
    LSAIndex,
    build_lsa_index,
    lsa_search,
    lsa_search_with_index,
)
from src.retrieval.vector_search import VectorIndex, build_vector_index, vector_search

__all__ = [
    "BooleanQueryError",
    "LSAIndex",
    "VectorIndex",
    "bm25_search",
    "bm25_search_default",
    "boolean_search",
    "boolean_search_scored",
    "build_lsa_index",
    "build_vector_index",
    "linear_search",
    "linear_search_scored",
    "lsa_search",
    "lsa_search_with_index",
    "parse_boolean_query",
    "vector_search",
]
