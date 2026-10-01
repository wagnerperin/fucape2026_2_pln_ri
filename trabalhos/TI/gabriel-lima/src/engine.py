"""
Fachada única sobre os cinco modelos de recuperação.

Os dois benchmarks precisam construir os mesmos índices e disparar as mesmas
consultas; este módulo concentra essa montagem para que ambos meçam
exatamente a mesma coisa.

Sobre a contabilidade de tempo de indexação: estruturas compartilhadas são
construídas uma vez, mas **cobradas de cada modelo que dependeria delas** em
produção. O índice invertido entra no tempo do BM25 e no do Booleano; a
matriz TF-IDF entra no do Vetorial e no do LSA. Assim o número reportado
responde à pergunta prática "quanto custa colocar este modelo no ar?".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import config
from src import utils
from src.indexer import (
    BooleanIndex,
    InvertedIndex,
    build_boolean_index,
    build_inverted_index,
)
from src.retrieval.bm25_search import bm25_search
from src.retrieval.boolean_search import boolean_search_scored
from src.retrieval.linear_search import linear_search_scored
from src.retrieval.lsa_search import LSAIndex, build_lsa_index, lsa_search_with_index
from src.retrieval.vector_search import VectorIndex, build_vector_index, vector_search

logger = logging.getLogger(__name__)

SearchResult = List[Tuple[int, float]]


@dataclass
class SearchEngine:
    """
    Motor de busca com os cinco modelos indexados sobre o mesmo corpus.

    Attributes:
        corpus: Produtos indexados.
        inverted_index: Índice invertido (Booleano e BM25).
        boolean_index: Índice invertido + matriz de incidência.
        vector_index: Índice TF-IDF (Vetorial e LSA).
        lsa_index: Índice do espaço latente.
        indexing_times_ms: Tempo de indexação atribuído a cada algoritmo.
        bm25_k1: ``k1`` usado pelo BM25 nas buscas deste motor.
        bm25_b: ``b`` usado pelo BM25 nas buscas deste motor.
    """

    corpus: Sequence[Dict]
    inverted_index: InvertedIndex
    boolean_index: BooleanIndex
    vector_index: VectorIndex
    lsa_index: LSAIndex
    indexing_times_ms: Dict[str, float] = field(default_factory=dict)
    bm25_k1: float = config.BM25_K1
    bm25_b: float = config.BM25_B

    # ------------------------------------------------------------------ #
    # Construção
    # ------------------------------------------------------------------ #

    @classmethod
    def build(
        cls,
        corpus: Sequence[Dict],
        lsa_components: int = config.LSA_COMPONENTS,
        bm25_k1: float = config.BM25_K1,
        bm25_b: float = config.BM25_B,
    ) -> "SearchEngine":
        """
        Constrói todos os índices sobre um corpus.

        Args:
            corpus: Lista de produtos.
            lsa_components: Dimensionalidade do espaço latente.
            bm25_k1: Parâmetro de saturação do BM25.
            bm25_b: Parâmetro de normalização de comprimento do BM25.

        Returns:
            O :class:`SearchEngine` pronto para consultas.

        Exemplo:
            >>> corpus = [{"product_id": 1, "name": "Mouse",
            ...            "description": "mouse óptico sem fio",
            ...            "category": "Informática"}]
            >>> engine = SearchEngine.build(corpus, lsa_components=1)
            >>> engine.search("BM25", "mouse")[0][0]
            1
        """
        texts = utils.build_corpus_texts(corpus)

        inverted_index = build_inverted_index(corpus, texts)
        boolean_index = build_boolean_index(
            corpus, texts, inverted_index=inverted_index
        )
        vector_index = build_vector_index(corpus, texts)
        lsa_index = build_lsa_index(
            corpus, texts, k=lsa_components, vector_index=vector_index
        )

        # Como as estruturas compartilhadas foram reaproveitadas, os índices
        # derivados cronometraram apenas o trabalho incremental. Aqui se soma
        # de volta o custo da dependência, para que cada linha responda
        # "quanto custa subir este modelo sozinho em produção?".
        indexing_times_ms = {
            # A busca linear é o baseline sem índice: nada a construir.
            "Linear": 0.0,
            # Booleano = índice invertido + matriz de incidência.
            "Boolean": inverted_index.indexing_time_ms
            + boolean_index.indexing_time_ms,
            "Vetorial": vector_index.indexing_time_ms,
            "BM25": inverted_index.indexing_time_ms,
            # LSA = matriz TF-IDF + decomposição SVD.
            "LSA": vector_index.indexing_time_ms + lsa_index.indexing_time_ms,
        }

        return cls(
            corpus=corpus,
            inverted_index=inverted_index,
            boolean_index=boolean_index,
            vector_index=vector_index,
            lsa_index=lsa_index,
            indexing_times_ms=indexing_times_ms,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )

    # ------------------------------------------------------------------ #
    # Consulta
    # ------------------------------------------------------------------ #

    def _searchers(self) -> Dict[str, Callable[[str, int], SearchResult]]:
        """
        Mapeia o nome de cada algoritmo à sua função de busca.

        Returns:
            Mapa ``nome -> f(query, top_k) -> [(product_id, score)]``.
        """
        return {
            "Linear": lambda query, top_k: linear_search_scored(
                query, self.corpus, top_k
            ),
            "Boolean": lambda query, top_k: boolean_search_scored(
                query, self.inverted_index, self.boolean_index.matrix, top_k
            ),
            "Vetorial": lambda query, top_k: vector_search(
                query, self.vector_index.tfidf_matrix, self.vector_index, top_k
            ),
            "BM25": lambda query, top_k: bm25_search(
                query,
                self.inverted_index,
                self.corpus,
                k1=self.bm25_k1,
                b=self.bm25_b,
                top_k=top_k,
            ),
            "LSA": lambda query, top_k: lsa_search_with_index(
                query, self.lsa_index, top_k
            ),
        }

    def search(
        self,
        algorithm: str,
        query: str,
        top_k: int = config.TOP_K,
    ) -> SearchResult:
        """
        Executa uma consulta com o algoritmo indicado.

        Args:
            algorithm: Um de ``"Linear"``, ``"Boolean"``, ``"Vetorial"``,
                ``"BM25"`` ou ``"LSA"``.
            query: Consulta em texto livre (ou booleana).
            top_k: Número de resultados.

        Returns:
            Lista de ``(product_id, score)`` ordenada por score decrescente.

        Raises:
            KeyError: Se o algoritmo não existir.
        """
        searchers = self._searchers()
        if algorithm not in searchers:
            raise KeyError(
                f"Algoritmo desconhecido: {algorithm!r}. "
                f"Opções: {sorted(searchers)}"
            )
        return searchers[algorithm](query, top_k)

    def search_all(
        self,
        query: str,
        top_k: int = config.TOP_K,
    ) -> Dict[str, SearchResult]:
        """
        Executa a mesma consulta nos cinco algoritmos.

        Args:
            query: Consulta em texto livre (ou booleana).
            top_k: Número de resultados por algoritmo.

        Returns:
            Mapa ``algoritmo -> [(product_id, score)]``.
        """
        return {
            name: searcher(query, top_k)
            for name, searcher in self._searchers().items()
        }

    # ------------------------------------------------------------------ #
    # Utilidades
    # ------------------------------------------------------------------ #

    @property
    def n_docs(self) -> int:
        """Número de documentos indexados."""
        return len(self.corpus)

    def product(self, product_id: int) -> Optional[Dict]:
        """
        Recupera um produto pelo identificador.

        Args:
            product_id: Identificador do produto.

        Returns:
            O dicionário do produto, ou ``None`` se não estiver no corpus.
        """
        return self._by_id().get(product_id)

    def _by_id(self) -> Dict[int, Dict]:
        """Constrói (e memoriza) o mapa ``product_id -> produto``."""
        cached = getattr(self, "_id_cache", None)
        if cached is None:
            cached = {int(p["product_id"]): p for p in self.corpus}
            object.__setattr__(self, "_id_cache", cached)
        return cached
