"""
Módulo genérico de indexação.

Constrói as estruturas compartilhadas pelos modelos de recuperação:

* **Índice invertido** — ``termo -> [(doc_idx, tf), ...]``, base do modelo
  booleano e do BM25.
* **Matriz de incidência binária** — bits empacotados (``np.packbits``), sobre
  a qual o modelo booleano opera com ``&``, ``|`` e ``~``.

Também expõe o utilitário :class:`Timer`, usado para medir tempos de
indexação e de consulta em milissegundos.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from src import utils

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Cronometragem
# --------------------------------------------------------------------------- #


class Timer:
    """
    Cronômetro de alta resolução, em milissegundos.

    Usa ``time.perf_counter`` e serve tanto como gerenciador de contexto
    quanto como objeto consultável depois do bloco.

    Exemplo:
        >>> with Timer() as timer:
        ...     _ = sum(range(1000))
        >>> timer.elapsed_ms >= 0.0
        True
    """

    def __init__(self) -> None:
        """Cria um cronômetro zerado, ainda não iniciado."""
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        """Inicia a contagem."""
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        """Encerra a contagem e preenche ``elapsed_ms``."""
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


# --------------------------------------------------------------------------- #
# Matriz de incidência binária
# --------------------------------------------------------------------------- #


@dataclass
class IncidenceMatrix:
    """
    Matriz termo-documento binária com bits empacotados.

    Cada linha corresponde a um termo do vocabulário e carrega ``n_docs`` bits
    (um por documento), armazenados em ``ceil(n_docs / 8)`` bytes. As operações
    booleanas são feitas diretamente sobre os bytes com ``&``, ``|`` e ``~``,
    processando 8 documentos por byte em vez de um por vez.

    Attributes:
        packed: Array ``(n_terms, n_bytes)`` de ``uint8``.
        vocabulary: Mapa ``termo -> índice da linha``.
        n_docs: Número de documentos representados.
    """

    packed: np.ndarray
    vocabulary: Dict[str, int]
    n_docs: int

    @property
    def n_terms(self) -> int:
        """Quantidade de termos (linhas) da matriz."""
        return self.packed.shape[0]

    @property
    def n_bytes(self) -> int:
        """Quantidade de bytes por linha."""
        return self.packed.shape[1]

    def nbytes(self) -> int:
        """Memória ocupada pela matriz, em bytes."""
        return int(self.packed.nbytes)

    def empty_row(self) -> np.ndarray:
        """Retorna uma linha de bits toda zerada (conjunto vazio)."""
        return np.zeros(self.n_bytes, dtype=np.uint8)

    def full_row(self) -> np.ndarray:
        """
        Retorna uma linha com todos os documentos marcados.

        Os bits de preenchimento do último byte (além de ``n_docs``) ficam em
        zero, para que ``~`` nunca produza documentos inexistentes.
        """
        row = np.full(self.n_bytes, 0xFF, dtype=np.uint8)
        return self._mask_tail(row)

    def _mask_tail(self, row: np.ndarray) -> np.ndarray:
        """
        Zera os bits de preenchimento do último byte.

        Args:
            row: Linha de bits empacotados.

        Returns:
            A mesma linha, com o excedente do último byte zerado.
        """
        surplus = self.n_bytes * 8 - self.n_docs
        if surplus:
            # O deslocamento precisa voltar para a faixa de 8 bits antes de
            # virar uint8, senão 0xFF << 4 já estoura o tipo.
            row[-1] &= np.uint8((0xFF << surplus) & 0xFF)
        return row

    def term_row(self, term: str) -> np.ndarray:
        """
        Devolve a linha de incidência de um termo.

        Args:
            term: Termo já pré-processado (radical).

        Returns:
            Cópia da linha de bits; linha zerada se o termo não existir.
        """
        index = self.vocabulary.get(term)
        if index is None:
            return self.empty_row()
        return self.packed[index].copy()

    def complement(self, row: np.ndarray) -> np.ndarray:
        """
        Complemento booleano de uma linha (operador ``NOT``).

        Args:
            row: Linha de bits.

        Returns:
            ``~row``, com os bits de preenchimento zerados.
        """
        return self._mask_tail(~row)

    def to_doc_indices(self, row: np.ndarray) -> np.ndarray:
        """
        Converte uma linha de bits nos índices de documento correspondentes.

        Args:
            row: Linha de bits empacotados.

        Returns:
            Array de índices de documento (ordenado, crescente).

        Exemplo:
            >>> matrix = build_incidence_matrix({"a": [(0, 1), (2, 1)]}, 3)
            >>> matrix.to_doc_indices(matrix.term_row("a")).tolist()
            [0, 2]
        """
        bits = np.unpackbits(row)[: self.n_docs]
        return np.flatnonzero(bits)


def build_incidence_matrix(
    postings: Dict[str, List[Tuple[int, int]]], n_docs: int
) -> IncidenceMatrix:
    """
    Constrói a matriz de incidência binária a partir do índice invertido.

    Args:
        postings: Mapa ``termo -> [(doc_idx, tf), ...]``.
        n_docs: Número de documentos do corpus.

    Returns:
        A :class:`IncidenceMatrix` correspondente.

    Exemplo:
        >>> matrix = build_incidence_matrix({"mouse": [(0, 3), (1, 1)]}, 2)
        >>> matrix.n_terms, matrix.n_docs
        (1, 2)
    """
    vocabulary = {term: idx for idx, term in enumerate(sorted(postings))}
    dense = np.zeros((len(vocabulary), n_docs), dtype=np.uint8)
    for term, term_idx in vocabulary.items():
        doc_indices = [doc_idx for doc_idx, _ in postings[term]]
        dense[term_idx, doc_indices] = 1

    packed = np.packbits(dense, axis=1)
    logger.debug(
        "Matriz de incidência: %d termos x %d docs (%.1f KB empacotados).",
        len(vocabulary),
        n_docs,
        packed.nbytes / 1024,
    )
    return IncidenceMatrix(packed=packed, vocabulary=vocabulary, n_docs=n_docs)


# --------------------------------------------------------------------------- #
# Índice invertido
# --------------------------------------------------------------------------- #


@dataclass
class InvertedIndex:
    """
    Índice invertido com as estatísticas exigidas pelo BM25.

    Attributes:
        postings: ``termo -> [(doc_idx, tf), ...]``, ordenado por ``doc_idx``.
        doc_freq: ``termo -> número de documentos que contêm o termo``.
        doc_lengths: Comprimento (em tokens) de cada documento.
        avg_doc_length: Comprimento médio dos documentos.
        product_ids: ``doc_idx -> product_id``.
        n_docs: Número de documentos indexados.
        indexing_time_ms: Tempo gasto na construção, em milissegundos.
    """

    postings: Dict[str, List[Tuple[int, int]]]
    doc_freq: Dict[str, int]
    doc_lengths: np.ndarray
    avg_doc_length: float
    product_ids: List[int]
    n_docs: int
    indexing_time_ms: float = 0.0
    _arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, repr=False
    )

    @property
    def vocabulary_size(self) -> int:
        """Número de termos distintos indexados."""
        return len(self.postings)

    def postings_arrays(self, term: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Devolve as *postings* de um termo como arrays NumPy, com cache.

        A lista de tuplas é a forma canônica exigida pela interface pública,
        mas o BM25 pontua milhares de documentos por consulta e se beneficia
        de operações vetorizadas.

        Args:
            term: Termo já pré-processado.

        Returns:
            Par ``(doc_indices, term_frequencies)``; arrays vazios se o termo
            não estiver no índice.
        """
        cached = self._arrays.get(term)
        if cached is not None:
            return cached

        entries = self.postings.get(term)
        if not entries:
            empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))
            self._arrays[term] = empty
            return empty

        docs = np.fromiter((d for d, _ in entries), dtype=np.int64, count=len(entries))
        freqs = np.fromiter(
            (f for _, f in entries), dtype=np.float64, count=len(entries)
        )
        self._arrays[term] = (docs, freqs)
        return docs, freqs


def build_inverted_index(
    corpus: Sequence[Dict],
    texts: Optional[Sequence[str]] = None,
) -> InvertedIndex:
    """
    Constrói o índice invertido de um corpus de produtos.

    O pré-processamento (tokenização, remoção de stopwords e stemming) é o de
    :func:`src.utils.preprocess`, o mesmo usado pelos modelos vetoriais — o
    que garante comparabilidade entre os algoritmos.

    Args:
        corpus: Lista de produtos.
        texts: Textos já concatenados (evita refazer o trabalho quando o
            chamador já os possui). Se ``None``, são derivados do corpus.

    Returns:
        O :class:`InvertedIndex` construído, com ``indexing_time_ms``
        preenchido.

    Exemplo:
        >>> corpus = [
        ...     {"product_id": 1, "name": "Mouse", "description": "mouse óptico",
        ...      "category": "Informática"},
        ... ]
        >>> index = build_inverted_index(corpus)
        >>> index.n_docs, index.doc_freq["mous"]
        (1, 1)
    """
    with Timer() as timer:
        documents = (
            list(texts) if texts is not None else utils.build_corpus_texts(corpus)
        )
        product_ids = [int(product["product_id"]) for product in corpus]

        postings: Dict[str, List[Tuple[int, int]]] = {}
        doc_lengths = np.zeros(len(documents), dtype=np.float64)

        for doc_idx, text in enumerate(documents):
            tokens = utils.preprocess(text)
            doc_lengths[doc_idx] = len(tokens)

            counts: Dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1

            for term, term_frequency in counts.items():
                postings.setdefault(term, []).append((doc_idx, term_frequency))

        doc_freq = {term: len(entries) for term, entries in postings.items()}
        avg_doc_length = float(doc_lengths.mean()) if len(doc_lengths) else 0.0

    index = InvertedIndex(
        postings=postings,
        doc_freq=doc_freq,
        doc_lengths=doc_lengths,
        avg_doc_length=avg_doc_length,
        product_ids=product_ids,
        n_docs=len(documents),
        indexing_time_ms=timer.elapsed_ms,
    )
    logger.info(
        "Índice invertido: %d docs, %d termos, |d| médio %.1f, %.1f ms.",
        index.n_docs,
        index.vocabulary_size,
        index.avg_doc_length,
        index.indexing_time_ms,
    )
    return index


@dataclass
class BooleanIndex:
    """
    Par índice invertido + matriz de incidência usado pelo modelo booleano.

    Attributes:
        inverted_index: Índice invertido do corpus.
        matrix: Matriz de incidência binária empacotada.
        indexing_time_ms: Tempo gasto nesta chamada. Quando um índice
            invertido pronto é reaproveitado, mede apenas a construção da
            matriz — o custo total do modelo é somado em
            :meth:`src.engine.SearchEngine.build`.
    """

    inverted_index: InvertedIndex
    matrix: IncidenceMatrix
    indexing_time_ms: float = 0.0


def build_boolean_index(
    corpus: Sequence[Dict],
    texts: Optional[Sequence[str]] = None,
    inverted_index: Optional[InvertedIndex] = None,
) -> BooleanIndex:
    """
    Constrói o índice completo do modelo booleano.

    Args:
        corpus: Lista de produtos.
        texts: Textos já concatenados, se disponíveis.
        inverted_index: Índice invertido pronto, para reaproveitamento. Quando
            informado, apenas a matriz de incidência é cronometrada.

    Returns:
        O :class:`BooleanIndex` correspondente.
    """
    if inverted_index is None:
        inverted_index = build_inverted_index(corpus, texts)
        base_time = inverted_index.indexing_time_ms
    else:
        base_time = 0.0

    with Timer() as timer:
        matrix = build_incidence_matrix(inverted_index.postings, inverted_index.n_docs)

    return BooleanIndex(
        inverted_index=inverted_index,
        matrix=matrix,
        indexing_time_ms=base_time + timer.elapsed_ms,
    )
