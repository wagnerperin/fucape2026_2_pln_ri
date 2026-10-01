"""
Testes dos cinco modelos de recuperação e do gerador de corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import utils  # noqa: E402
from src.corpus_generator import (  # noqa: E402
    build_qrels,
    corpus_statistics,
    generate_corpus,
    generate_products,
)
from src.engine import SearchEngine  # noqa: E402
from src.indexer import build_boolean_index, build_inverted_index  # noqa: E402
from src.retrieval.bm25_search import (  # noqa: E402
    bm25_search,
    compute_idf,
    explain_score,
)
from src.retrieval.boolean_search import (  # noqa: E402
    BooleanQueryError,
    boolean_match_count,
    boolean_search,
    parse_boolean_query,
)
from src.retrieval.linear_search import (  # noqa: E402
    linear_search,
    linear_search_scored,
)
from src.retrieval.lsa_search import (  # noqa: E402
    build_lsa_index,
    lsa_search_with_index,
)
from src.retrieval.vector_search import build_vector_index, vector_search  # noqa: E402


@pytest.fixture(scope="module")
def corpus() -> List[Dict]:
    """Corpus sintético pequeno, porém suficiente para ranquear."""
    return generate_corpus(size=120)


@pytest.fixture(scope="module")
def engine(corpus: List[Dict]) -> SearchEngine:
    """Motor com os cinco índices construídos uma única vez."""
    return SearchEngine.build(corpus, lsa_components=20)


# --------------------------------------------------------------------------- #
# Gerador de corpus
# --------------------------------------------------------------------------- #


class TestCorpusGenerator:
    """Contrato do corpus sintético."""

    def test_schema_of_every_product(self, corpus: List[Dict]) -> None:
        """Todo produto tem exatamente os seis campos exigidos."""
        expected = {
            "product_id",
            "name",
            "description",
            "category",
            "price",
            "rating",
        }
        assert all(set(product) == expected for product in corpus)

    def test_ids_are_sequential(self, corpus: List[Dict]) -> None:
        """Os identificadores vão de 1 a N, sem buracos."""
        assert [p["product_id"] for p in corpus] == list(range(1, len(corpus) + 1))

    def test_description_length_is_within_range(self, corpus: List[Dict]) -> None:
        """As descrições respeitam a faixa de 200 a 500 palavras."""
        for product in corpus:
            words = len(product["description"].split())
            assert config.DESCRIPTION_MIN_WORDS <= words <= config.DESCRIPTION_MAX_WORDS

    def test_prices_and_ratings_are_plausible(self, corpus: List[Dict]) -> None:
        """Preço positivo e nota entre 1 e 5."""
        for product in corpus:
            assert product["price"] > 0
            assert 1.0 <= product["rating"] <= 5.0

    def test_generation_is_reproducible(self) -> None:
        """A mesma semente produz exatamente o mesmo corpus."""
        assert generate_corpus(30, seed=7) == generate_corpus(30, seed=7)

    def test_corpus_is_a_stable_superset(self) -> None:
        """
        Gerar mais produtos preserva os anteriores.

        É o que permite comparar N = 100 e N = 10.000 sabendo que o menor é
        um prefixo exato do maior.
        """
        assert generate_corpus(60)[:20] == generate_corpus(20)

    def test_different_seeds_differ(self) -> None:
        """Sementes distintas geram catálogos distintos."""
        assert generate_corpus(20, seed=1) != generate_corpus(20, seed=2)

    def test_polysemy_is_planted(self) -> None:
        """``mouse`` ocorre como periférico e como brinquedo de pet shop."""
        items = generate_products(600)
        categories = {
            item.record["category"]
            for item in items
            if "mouse" in utils.build_document_text(item.record).lower()
        }
        assert "Informática" in categories
        assert "Pet Shop" in categories

    def test_qrels_cover_every_query(self) -> None:
        """Há julgamentos para as cinco consultas qualitativas."""
        qrels = build_qrels(generate_products(600))
        assert set(qrels) == {spec["id"] for spec in config.QUALITATIVE_QUERIES}
        assert all(judgments for judgments in qrels.values())

    def test_qrels_exclude_the_pet_toy_from_q1(self) -> None:
        """O rato de pelúcia não é resposta para a busca por ``mouse``."""
        items = generate_products(600)
        qrels = build_qrels(items)
        for item in items:
            if item.ptype == "brinquedo_rato":
                assert str(item.record["product_id"]) not in qrels["Q1"]

    def test_name_never_claims_an_attribute_the_product_lacks(self) -> None:
        """
        Nome, descrição e julgamento contam a mesma história.

        Um produto chamado "Teclado Gamer" que não tivesse o atributo
        ``gamer`` apareceria como resposta correta para
        ``NOT gamer`` — contradição visível na SERP e ground truth errado.
        """
        from src.corpus_generator import ATTRIBUTE_NAME_FRAGMENT, _fold

        for item in generate_products(800):
            words = {_fold(word) for word in item.record["name"].split()}
            for attribute, fragment in ATTRIBUTE_NAME_FRAGMENT.items():
                head = _fold(fragment.split()[0])
                if head in words:
                    assert attribute in item.attrs, (
                        f"{item.record['name']!r} anuncia {fragment!r} "
                        f"sem possuir o atributo {attribute!r}"
                    )

    def test_statistics_report_the_corpus(self, corpus: List[Dict]) -> None:
        """As estatísticas descritivas batem com o corpus."""
        stats = corpus_statistics(corpus)
        assert stats["n_produtos"] == len(corpus)
        assert stats["palavras_min"] >= config.DESCRIPTION_MIN_WORDS


# --------------------------------------------------------------------------- #
# A. Busca linear
# --------------------------------------------------------------------------- #


class TestLinearSearch:
    """Baseline sem índice."""

    def test_returns_product_dictionaries(self, corpus: List[Dict]) -> None:
        """A assinatura devolve os próprios produtos."""
        results = linear_search("mouse", corpus, top_k=5)
        assert all(isinstance(item, dict) and "product_id" in item for item in results)

    def test_respects_top_k(self, corpus: List[Dict]) -> None:
        """Nunca devolve mais que ``top_k``."""
        assert len(linear_search("mouse", corpus, top_k=3)) <= 3

    def test_scores_are_monotonically_decreasing(self, corpus: List[Dict]) -> None:
        """O ranking sai ordenado."""
        scores = [score for _, score in linear_search_scored("mouse", corpus, 10)]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_returns_nothing(self, corpus: List[Dict]) -> None:
        """Consulta vazia não inventa resultados."""
        assert linear_search("", corpus) == []

    def test_unmatched_query_returns_nothing(self, corpus: List[Dict]) -> None:
        """Termo ausente do catálogo devolve lista vazia."""
        assert linear_search("xyzabcinexistente", corpus) == []

    def test_substring_matching_is_the_known_weakness(
        self, corpus: List[Dict]
    ) -> None:
        """
        Documenta a fragilidade do baseline: por casar substring, ``mouse``
        também casa dentro de ``mousepad``.
        """
        names = " ".join(
            product["name"].lower() for product in linear_search("mouse", corpus, 10)
        )
        assert "mousepad" in names


# --------------------------------------------------------------------------- #
# B. Modelo booleano
# --------------------------------------------------------------------------- #


class TestBooleanSearch:
    """Parser e avaliação bitwise."""

    def test_parses_operators_and_parentheses(self) -> None:
        """A gramática aceita a consulta do enunciado."""
        assert parse_boolean_query("mouse AND (óptico OR sem-fio) NOT gamer")

    @pytest.mark.parametrize("query", ["", "   ", "mouse AND", "(mouse", "mouse)"])
    def test_rejects_malformed_queries(self, query: str) -> None:
        """Consultas malformadas levantam erro explícito."""
        with pytest.raises(BooleanQueryError):
            parse_boolean_query(query)

    def test_and_is_a_subset_of_or(self, engine: SearchEngine) -> None:
        """O conjunto do ``AND`` está contido no do ``OR``."""
        args = (engine.inverted_index, engine.boolean_index.matrix)
        conjunctive = boolean_match_count("mouse AND óptico", *args)
        disjunctive = boolean_match_count("mouse OR óptico", *args)
        assert conjunctive <= disjunctive

    def test_not_excludes_the_term(self, engine: SearchEngine) -> None:
        """Nenhum resultado de ``NOT gamer`` contém ``gamer``."""
        results = boolean_search(
            "teclado AND NOT gamer",
            engine.inverted_index,
            engine.boolean_index.matrix,
            top_k=20,
        )
        gamer_stem = utils.preprocess("gamer")[0]
        for product_id in results:
            tokens = utils.preprocess(
                utils.build_document_text(engine.product(product_id))
            )
            assert gamer_stem not in tokens

    def test_or_is_the_default_for_free_text(self, engine: SearchEngine) -> None:
        """Sem operadores, os termos são combinados com ``OR``."""
        args = (engine.inverted_index, engine.boolean_index.matrix)
        assert boolean_match_count("mouse teclado", *args) == boolean_match_count(
            "mouse OR teclado", *args
        )

    def test_de_morgan_holds(self, engine: SearchEngine) -> None:
        """``NOT (a OR b)`` equivale a ``NOT a AND NOT b``."""
        args = (engine.inverted_index, engine.boolean_index.matrix)
        assert boolean_match_count("NOT (mouse OR teclado)", *args) == (
            boolean_match_count("NOT mouse AND NOT teclado", *args)
        )

    def test_returns_product_ids(self, engine: SearchEngine) -> None:
        """A assinatura devolve identificadores inteiros."""
        results = boolean_search(
            "mouse", engine.inverted_index, engine.boolean_index.matrix, top_k=5
        )
        assert all(isinstance(pid, int) for pid in results)
        assert all(engine.product(pid) is not None for pid in results)

    def test_unknown_term_returns_empty(self, engine: SearchEngine) -> None:
        """Termo fora do vocabulário devolve conjunto vazio."""
        assert (
            boolean_search(
                "xyzabcinexistente",
                engine.inverted_index,
                engine.boolean_index.matrix,
            )
            == []
        )


# --------------------------------------------------------------------------- #
# C. Modelo vetorial
# --------------------------------------------------------------------------- #


class TestVectorSearch:
    """TF-IDF e cosseno."""

    def test_similarity_is_bounded(self, engine: SearchEngine) -> None:
        """O cosseno de vetores não negativos fica em [0, 1]."""
        results = vector_search(
            "mouse óptico", engine.vector_index.tfidf_matrix, engine.vector_index, 10
        )
        assert all(0.0 <= score <= 1.0 + 1e-6 for _, score in results)

    def test_results_are_sorted(self, engine: SearchEngine) -> None:
        """O ranking sai ordenado por similaridade."""
        scores = [
            score
            for _, score in vector_search(
                "mouse", engine.vector_index.tfidf_matrix, engine.vector_index, 10
            )
        ]
        assert scores == sorted(scores, reverse=True)

    def test_matrix_is_l2_normalized(self, engine: SearchEngine) -> None:
        """As linhas têm norma unitária, o que reduz o cosseno a um produto."""
        import numpy as np

        norms = np.sqrt(engine.vector_index.tfidf_matrix.multiply(
            engine.vector_index.tfidf_matrix
        ).sum(axis=1)).A1
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_out_of_vocabulary_query_returns_empty(self, engine: SearchEngine) -> None:
        """Consulta sem termo conhecido devolve lista vazia."""
        assert (
            vector_search(
                "xyzabcinexistente",
                engine.vector_index.tfidf_matrix,
                engine.vector_index,
            )
            == []
        )

    def test_boolean_operators_are_ignored(self, engine: SearchEngine) -> None:
        """Os operadores não poluem o vetor da consulta."""
        with_ops = vector_search(
            "mouse AND óptico", engine.vector_index.tfidf_matrix, engine.vector_index, 5
        )
        without = vector_search(
            "mouse óptico", engine.vector_index.tfidf_matrix, engine.vector_index, 5
        )
        assert [pid for pid, _ in with_ops] == [pid for pid, _ in without]


# --------------------------------------------------------------------------- #
# D. BM25
# --------------------------------------------------------------------------- #


class TestBM25:
    """Fórmula, hiperparâmetros e validação."""

    def test_idf_decreases_with_document_frequency(self) -> None:
        """Termo raro tem IDF maior que termo comum."""
        assert compute_idf(1, 1000) > compute_idf(500, 1000)

    def test_idf_is_positive_even_for_ubiquitous_terms(self) -> None:
        """A suavização evita IDF negativo quando o termo está em todo lugar."""
        assert compute_idf(1000, 1000) > 0.0

    def test_results_are_sorted(self, engine: SearchEngine, corpus: List[Dict]) -> None:
        """O ranking sai ordenado por score."""
        scores = [
            s for _, s in bm25_search("mouse óptico", engine.inverted_index, corpus)
        ]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_positive(
        self, engine: SearchEngine, corpus: List[Dict]
    ) -> None:
        """Documentos retornados têm score estritamente positivo."""
        results = bm25_search("mouse", engine.inverted_index, corpus, top_k=10)
        assert results and all(score > 0 for _, score in results)

    @pytest.mark.parametrize("k1,b", [(0.0, 0.0), (1.2, 0.5), (2.0, 1.0)])
    def test_accepts_the_valid_parameter_range(
        self, engine: SearchEngine, corpus: List[Dict], k1: float, b: float
    ) -> None:
        """A parametrização cobre toda a faixa válida."""
        assert bm25_search("mouse", engine.inverted_index, corpus, k1=k1, b=b)

    @pytest.mark.parametrize("k1,b", [(-1.0, 0.75), (1.5, -0.1), (1.5, 1.5)])
    def test_rejects_invalid_parameters(
        self, engine: SearchEngine, corpus: List[Dict], k1: float, b: float
    ) -> None:
        """Valores fora do domínio falham cedo, com mensagem clara."""
        with pytest.raises(ValueError):
            bm25_search("mouse", engine.inverted_index, corpus, k1=k1, b=b)

    def test_b_zero_disables_length_normalization(
        self, engine: SearchEngine, corpus: List[Dict]
    ) -> None:
        """``b = 0`` muda o ranking em relação ao padrão ``b = 0.75``."""
        without = bm25_search(
            "mouse óptico", engine.inverted_index, corpus, b=0.0, top_k=10
        )
        with_norm = bm25_search(
            "mouse óptico", engine.inverted_index, corpus, b=1.0, top_k=10
        )
        assert [pid for pid, _ in without] != [pid for pid, _ in with_norm]

    def test_explain_score_sums_to_total(
        self, engine: SearchEngine, corpus: List[Dict]
    ) -> None:
        """A decomposição por termo soma o score total."""
        product_id, score = bm25_search(
            "mouse óptico", engine.inverted_index, corpus
        )[0]
        breakdown = explain_score("mouse óptico", product_id, engine.inverted_index)
        assert breakdown["_total"] == pytest.approx(score, rel=1e-6)

    def test_unknown_term_returns_empty(
        self, engine: SearchEngine, corpus: List[Dict]
    ) -> None:
        """Termo ausente do índice devolve lista vazia."""
        assert bm25_search("xyzabcinexistente", engine.inverted_index, corpus) == []


# --------------------------------------------------------------------------- #
# E. LSA
# --------------------------------------------------------------------------- #


class TestLSA:
    """Projeção latente."""

    def test_embeddings_have_the_requested_rank(self, corpus: List[Dict]) -> None:
        """A matriz de documentos tem ``k`` colunas."""
        index = build_lsa_index(corpus, k=15)
        assert index.doc_embeddings.shape == (len(corpus), 15)

    def test_k_is_capped_by_the_corpus(self) -> None:
        """``k`` maior que o corpus é truncado em vez de quebrar."""
        tiny = generate_corpus(6)
        assert build_lsa_index(tiny, k=500).n_components <= 5

    def test_similarity_is_bounded(self, engine: SearchEngine) -> None:
        """O cosseno fica em [-1, 1]."""
        results = lsa_search_with_index("mouse", engine.lsa_index, 10)
        assert all(-1.0 - 1e-6 <= score <= 1.0 + 1e-6 for _, score in results)

    def test_results_are_sorted(self, engine: SearchEngine) -> None:
        """O ranking sai ordenado."""
        scores = [
            s for _, s in lsa_search_with_index("mouse óptico", engine.lsa_index, 10)
        ]
        assert scores == sorted(scores, reverse=True)

    def test_is_reproducible(self, corpus: List[Dict]) -> None:
        """Mesma semente, mesmo ranking — o SVD randomizado é determinístico."""
        first = build_lsa_index(corpus, k=15, seed=42)
        second = build_lsa_index(corpus, k=15, seed=42)
        assert [p for p, _ in lsa_search_with_index("mouse", first, 5)] == [
            p for p, _ in lsa_search_with_index("mouse", second, 5)
        ]

    def test_explained_variance_is_a_fraction(self, engine: SearchEngine) -> None:
        """A variância explicada é uma proporção válida."""
        assert 0.0 < engine.lsa_index.explained_variance <= 1.0


# --------------------------------------------------------------------------- #
# Motor unificado
# --------------------------------------------------------------------------- #


class TestSearchEngine:
    """Contrato comum aos cinco modelos."""

    def test_all_algorithms_are_available(self, engine: SearchEngine) -> None:
        """Os cinco modelos respondem à mesma consulta."""
        results = engine.search_all("mouse óptico", top_k=5)
        assert set(results) == set(config.ALGORITHMS)

    @pytest.mark.parametrize("algorithm", config.ALGORITHMS)
    def test_result_shape_is_uniform(
        self, engine: SearchEngine, algorithm: str
    ) -> None:
        """Todo modelo devolve ``(product_id, score)`` com ids válidos."""
        for product_id, score in engine.search(algorithm, "mouse", top_k=5):
            assert isinstance(product_id, int)
            assert isinstance(score, float)
            assert engine.product(product_id) is not None

    @pytest.mark.parametrize("algorithm", config.ALGORITHMS)
    def test_top_k_is_respected(self, engine: SearchEngine, algorithm: str) -> None:
        """Nenhum modelo ultrapassa o limite pedido."""
        assert len(engine.search(algorithm, "mouse", top_k=3)) <= 3

    def test_unknown_algorithm_raises(self, engine: SearchEngine) -> None:
        """Nome inválido falha com erro explícito."""
        with pytest.raises(KeyError):
            engine.search("Inexistente", "mouse")

    def test_indexing_times_are_recorded(self, engine: SearchEngine) -> None:
        """O baseline não indexa; os demais custam tempo positivo."""
        assert engine.indexing_times_ms["Linear"] == 0.0
        for algorithm in ("Boolean", "Vetorial", "BM25", "LSA"):
            assert engine.indexing_times_ms[algorithm] > 0.0

    def test_ranked_models_agree_on_the_obvious_case(
        self, engine: SearchEngine
    ) -> None:
        """
        Para uma consulta inequívoca, os modelos ranqueados convergem.

        Não se exige ranking idêntico — cada um pondera diferente —, apenas
        que o TOP-5 se sobreponha, o que indica que todos leem o mesmo
        vocabulário.
        """
        vetorial = {p for p, _ in engine.search("Vetorial", "teclado mecânico", 5)}
        bm25 = {p for p, _ in engine.search("BM25", "teclado mecânico", 5)}
        assert vetorial & bm25
