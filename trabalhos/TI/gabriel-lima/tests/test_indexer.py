"""
Testes do pré-processamento e das estruturas de indexação.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import utils  # noqa: E402
from src.indexer import (  # noqa: E402
    Timer,
    build_boolean_index,
    build_incidence_matrix,
    build_inverted_index,
)


@pytest.fixture(scope="module")
def tiny_corpus() -> List[Dict]:
    """Corpus mínimo, com polissemia e sinonímia plantadas."""
    return [
        {
            "product_id": 1,
            "name": "Mouse Óptico Sem Fio",
            "description": "Mouse óptico sem fio para notebook e escritório.",
            "category": "Informática",
            "price": 89.9,
            "rating": 4.5,
        },
        {
            "product_id": 2,
            "name": "Teclado Mecânico RGB",
            "description": "Teclado mecânico com iluminação RGB para digitação.",
            "category": "Informática",
            "price": 349.0,
            "rating": 4.7,
        },
        {
            "product_id": 3,
            "name": "Rato de Pelúcia para Gato",
            "description": "Brinquedo rato de pelúcia, o mouse de brinquedo do gato.",
            "category": "Pet Shop",
            "price": 19.9,
            "rating": 4.2,
        },
    ]


# --------------------------------------------------------------------------- #
# utils
# --------------------------------------------------------------------------- #


class TestPreprocessing:
    """Comportamento do pipeline de texto."""

    def test_tokenize_splits_hyphen_and_keeps_decimals(self) -> None:
        """``sem-fio`` vira dois tokens e ``5.0`` permanece inteiro."""
        assert utils.tokenize("Mouse sem-fio Bluetooth 5.0") == [
            "mouse",
            "sem",
            "fio",
            "bluetooth",
            "5.0",
        ]

    def test_strip_accents(self) -> None:
        """Diacríticos são removidos, demais caracteres preservados."""
        assert utils.strip_accents("óptico ergonômico") == "optico ergonomico"

    def test_sem_is_not_a_stopword(self) -> None:
        """``sem`` é preservado: distingue ``mouse sem fio`` de ``mouse fio``."""
        assert "sem" not in utils.get_stopwords()
        assert "sem" in utils.preprocess("mouse sem fio")

    def test_common_stopwords_are_removed(self) -> None:
        """Stopwords sem valor discriminativo saem do índice."""
        tokens = utils.preprocess("o mouse para a mesa de trabalho")
        assert "de" not in tokens
        assert "para" not in tokens

    def test_stemming_groups_inflections(self) -> None:
        """Singular e plural colapsam no mesmo radical."""
        assert utils.preprocess("teclado")[0] == utils.preprocess("teclados")[0]

    def test_preprocess_query_drops_boolean_operators(self) -> None:
        """Operadores não viram termos de conteúdo."""
        tokens = utils.preprocess_query("teclado AND (mecânico OR rgb) AND NOT gamer")
        assert "and" not in tokens and "or" not in tokens and "not" not in tokens
        assert utils.preprocess("rgb")[0] in tokens

    def test_strip_operators_keeps_raw_words(self) -> None:
        """A limpeza para o TF-IDF não aplica stemming."""
        assert utils.strip_operators("teclado AND (mecânico OR rgb)") == (
            "teclado mecânico rgb"
        )

    def test_build_document_text_boosts_the_name(self) -> None:
        """O nome entra duas vezes, funcionando como *field boost*."""
        text = utils.build_document_text(
            {"name": "Mouse", "description": "desc", "category": "cat"}
        )
        assert text.split().count("Mouse") == 2

    def test_top_k_from_scores_orders_and_filters(self) -> None:
        """Ordena por score e descarta zeros."""
        result = utils.top_k_from_scores(np.array([0.0, 0.9, 0.5]), [10, 20, 30], 5)
        assert result == [(20, 0.9), (30, 0.5)]

    def test_top_k_breaks_ties_by_product_id(self) -> None:
        """Empates são resolvidos de forma determinística."""
        result = utils.top_k_from_scores(np.array([0.5, 0.5]), [77, 22], 2)
        assert [pid for pid, _ in result] == [22, 77]


# --------------------------------------------------------------------------- #
# Timer
# --------------------------------------------------------------------------- #


def test_timer_measures_elapsed_time() -> None:
    """O cronômetro devolve um valor não negativo em milissegundos."""
    with Timer() as timer:
        sum(range(10_000))
    assert timer.elapsed_ms >= 0.0


# --------------------------------------------------------------------------- #
# Índice invertido
# --------------------------------------------------------------------------- #


class TestInvertedIndex:
    """Estrutura e estatísticas do índice invertido."""

    def test_indexes_every_document(self, tiny_corpus: List[Dict]) -> None:
        """Todos os documentos entram no índice."""
        index = build_inverted_index(tiny_corpus)
        assert index.n_docs == 3
        assert index.product_ids == [1, 2, 3]

    def test_document_frequency_matches_postings(self, tiny_corpus: List[Dict]) -> None:
        """``doc_freq`` é o tamanho da lista de postings."""
        index = build_inverted_index(tiny_corpus)
        for term, entries in index.postings.items():
            assert index.doc_freq[term] == len(entries)

    def test_postings_are_sorted_by_document(self, tiny_corpus: List[Dict]) -> None:
        """As postings saem ordenadas, como exige a busca por interseção."""
        index = build_inverted_index(tiny_corpus)
        for entries in index.postings.values():
            docs = [doc for doc, _ in entries]
            assert docs == sorted(docs)

    def test_polysemous_term_links_both_senses(self, tiny_corpus: List[Dict]) -> None:
        """``mouse`` aparece no periférico e no brinquedo de pet shop."""
        index = build_inverted_index(tiny_corpus)
        stem = utils.preprocess("mouse")[0]
        docs = {doc for doc, _ in index.postings[stem]}
        assert docs == {0, 2}

    def test_average_length_is_the_mean(self, tiny_corpus: List[Dict]) -> None:
        """``avg_doc_length`` corresponde à média dos comprimentos."""
        index = build_inverted_index(tiny_corpus)
        assert index.avg_doc_length == pytest.approx(index.doc_lengths.mean())

    def test_postings_arrays_match_postings(self, tiny_corpus: List[Dict]) -> None:
        """A visão vetorizada reflete a lista de tuplas."""
        index = build_inverted_index(tiny_corpus)
        stem = utils.preprocess("mouse")[0]
        docs, freqs = index.postings_arrays(stem)
        assert docs.tolist() == [doc for doc, _ in index.postings[stem]]
        assert freqs.tolist() == [float(tf) for _, tf in index.postings[stem]]

    def test_unknown_term_yields_empty_arrays(self, tiny_corpus: List[Dict]) -> None:
        """Termo fora do vocabulário não quebra a busca."""
        index = build_inverted_index(tiny_corpus)
        docs, freqs = index.postings_arrays("termoinexistentexyz")
        assert docs.size == 0 and freqs.size == 0

    def test_indexing_time_is_recorded(self, tiny_corpus: List[Dict]) -> None:
        """O tempo de indexação é preenchido."""
        assert build_inverted_index(tiny_corpus).indexing_time_ms >= 0.0


# --------------------------------------------------------------------------- #
# Matriz de incidência
# --------------------------------------------------------------------------- #


class TestIncidenceMatrix:
    """Operações bitwise sobre a matriz binária."""

    def test_term_row_recovers_document_indices(self) -> None:
        """A linha de bits devolve exatamente os documentos da posting."""
        matrix = build_incidence_matrix({"a": [(0, 1), (2, 1)]}, 3)
        assert matrix.to_doc_indices(matrix.term_row("a")).tolist() == [0, 2]

    def test_unknown_term_is_the_empty_set(self) -> None:
        """Termo ausente devolve conjunto vazio."""
        matrix = build_incidence_matrix({"a": [(0, 1)]}, 3)
        assert matrix.to_doc_indices(matrix.term_row("zzz")).size == 0

    @pytest.mark.parametrize("n_docs", [1, 7, 8, 9, 15, 16, 17, 100])
    def test_complement_never_invents_documents(self, n_docs: int) -> None:
        """
        O ``NOT`` respeita a borda do último byte.

        Com ``n_docs`` que não é múltiplo de 8, os bits de preenchimento
        precisam ficar zerados — senão a negação passaria a "encontrar"
        documentos inexistentes.
        """
        matrix = build_incidence_matrix({"a": [(0, 1)]}, n_docs)
        row = matrix.term_row("a")
        complement = matrix.to_doc_indices(matrix.complement(row))
        assert complement.size == n_docs - 1
        assert complement.max(initial=-1) < n_docs

    def test_full_row_covers_every_document(self) -> None:
        """A linha cheia contém todos os documentos e nada além deles."""
        matrix = build_incidence_matrix({"a": [(0, 1)]}, 13)
        assert matrix.to_doc_indices(matrix.full_row()).tolist() == list(range(13))

    def test_and_or_are_set_operations(self) -> None:
        """``&`` e ``|`` correspondem a interseção e união."""
        matrix = build_incidence_matrix(
            {"a": [(0, 1), (1, 1)], "b": [(1, 1), (2, 1)]}, 4
        )
        row_a, row_b = matrix.term_row("a"), matrix.term_row("b")
        assert matrix.to_doc_indices(row_a & row_b).tolist() == [1]
        assert matrix.to_doc_indices(row_a | row_b).tolist() == [0, 1, 2]

    def test_matrix_covers_the_whole_vocabulary(self, tiny_corpus: List[Dict]) -> None:
        """Toda entrada do índice invertido vira uma linha da matriz."""
        boolean_index = build_boolean_index(tiny_corpus)
        assert boolean_index.matrix.n_terms == len(
            boolean_index.inverted_index.postings
        )
        assert boolean_index.matrix.n_docs == 3

    def test_boolean_index_reuses_inverted_index(self, tiny_corpus: List[Dict]) -> None:
        """Reaproveitar o índice invertido não o reconstrói."""
        inverted = build_inverted_index(tiny_corpus)
        boolean_index = build_boolean_index(tiny_corpus, inverted_index=inverted)
        assert boolean_index.inverted_index is inverted
