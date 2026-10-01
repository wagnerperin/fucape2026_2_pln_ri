"""
Análise qualitativa de relevância — SERP comparativa (Tarefa 5).

Executa as cinco consultas de ``config.QUALITATIVE_QUERIES`` nos cinco
modelos, registra o TOP-5 de cada um e avalia a qualidade do ranking contra
os julgamentos graduados de ``data/qrels.json``.

Como o corpus é sintético, a relevância é conhecida por construção: cada
produto sabe se é um mouse periférico, um rato de pelúcia ou um teclado
musical. Isso permite medir P@5 e nDCG@5 objetivamente, em vez de julgar as
SERPs no olho.

Saídas:

* ``results/serp_analysis.csv`` — TOP-5 de cada (consulta, algoritmo).
* ``results/relevance_metrics.csv`` — P@5, nDCG@5 e taxa de irrelevantes.
* ``results/bm25_tuning.csv`` — varredura de ``k1`` e ``b``.

Uso:
    python benchmark/benchmark_relevance.py [--skip-tuning]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import utils  # noqa: E402
from src.corpus_generator import load_corpus, load_qrels  # noqa: E402
from src.engine import SearchEngine  # noqa: E402
from src.retrieval.bm25_search import bm25_search  # noqa: E402
from src.retrieval.boolean_search import boolean_match_count  # noqa: E402

logger = logging.getLogger(__name__)

Ranking = Sequence[Tuple[int, float]]
Judgments = Dict[str, int]


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #


def precision_at_k(ranking: Ranking, judgments: Judgments, k: int) -> float:
    """
    Precisão em ``k``: fração dos ``k`` primeiros que é relevante.

    Um documento conta como relevante se seu grau de julgamento for maior que
    zero.

    Args:
        ranking: Lista de ``(product_id, score)``.
        judgments: Mapa ``product_id (str) -> grau``.
        k: Profundidade de corte.

    Returns:
        Valor em ``[0, 1]``.

    Exemplo:
        >>> precision_at_k([(1, 0.9), (2, 0.5)], {"1": 3}, 2)
        0.5
    """
    if k <= 0:
        return 0.0
    top = list(ranking)[:k]
    if not top:
        return 0.0
    hits = sum(1 for pid, _ in top if judgments.get(str(pid), 0) > 0)
    return hits / k


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """
    Ganho cumulativo descontado.

    Args:
        gains: Ganhos na ordem do ranking.
        k: Profundidade de corte.

    Returns:
        O DCG@k, com desconto logarítmico ``log2(i + 2)``.
    """
    gains = list(gains)[:k]
    if not gains:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(np.asarray(gains, dtype=np.float64) * discounts))


def ndcg_at_k(ranking: Ranking, judgments: Judgments, k: int) -> float:
    """
    nDCG em ``k``, com ganho exponencial ``2^grau - 1``.

    Diferente da precisão, o nDCG é sensível à **ordem**: colocar o produto
    mais relevante em quinto lugar vale menos do que em primeiro. É a métrica
    que melhor reflete o comportamento de clique em uma SERP de e-commerce.

    Args:
        ranking: Lista de ``(product_id, score)``.
        judgments: Mapa ``product_id (str) -> grau``.
        k: Profundidade de corte.

    Returns:
        Valor em ``[0, 1]``; zero se não houver documento relevante algum.

    Exemplo:
        >>> round(ndcg_at_k([(1, 0.9)], {"1": 3}, 1), 3)
        1.0
    """
    gains = [2.0 ** judgments.get(str(pid), 0) - 1.0 for pid, _ in list(ranking)[:k]]
    ideal_grades = sorted(judgments.values(), reverse=True)[:k]
    ideal_gains = [2.0**grade - 1.0 for grade in ideal_grades]

    ideal = dcg_at_k(ideal_gains, k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(gains, k) / ideal


def irrelevant_rate(ranking: Ranking, judgments: Judgments, k: int) -> float:
    """
    Fração dos ``k`` primeiros resultados com grau zero.

    Args:
        ranking: Lista de ``(product_id, score)``.
        judgments: Mapa de julgamentos.
        k: Profundidade de corte.

    Returns:
        Valor em ``[0, 1]``. Posições vazias contam como irrelevantes, pois
        para o usuário uma SERP curta demais também é uma falha.
    """
    if k <= 0:
        return 0.0
    top = list(ranking)[:k]
    misses = sum(1 for pid, _ in top if judgments.get(str(pid), 0) == 0)
    misses += k - len(top)
    return misses / k


# --------------------------------------------------------------------------- #
# Execução das consultas
# --------------------------------------------------------------------------- #


def run_serp_analysis(
    engine: SearchEngine,
    qrels: Dict[str, Judgments],
    top_k: int = config.SERP_TOP_K,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executa as consultas qualitativas em todos os algoritmos.

    O modelo booleano recebe a variante com sintaxe explícita declarada em
    ``config.QUALITATIVE_QUERIES``; os demais recebem o texto livre. É assim
    que cada modelo é avaliado no regime para o qual foi projetado.

    Args:
        engine: Motor com os índices construídos.
        qrels: Julgamentos de relevância por identificador de consulta.
        top_k: Profundidade da SERP.

    Returns:
        Par ``(serp, metrics)`` de ``DataFrame``: o detalhamento posição a
        posição e o resumo de métricas por consulta e algoritmo.
    """
    serp_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []

    for spec in config.QUALITATIVE_QUERIES:
        query_id = spec["id"]
        free_text = spec["query"]
        boolean_text = spec["boolean_query"]
        judgments = qrels.get(query_id, {})

        matched = boolean_match_count(
            boolean_text, engine.inverted_index, engine.boolean_index.matrix
        )
        logger.info(
            "%s (%s) %r — conjunto booleano: %d docs; relevantes no corpus: %d",
            query_id,
            spec["tipo"],
            free_text,
            matched,
            len(judgments),
        )

        for algorithm in config.ALGORITHMS:
            query = boolean_text if algorithm == "Boolean" else free_text
            ranking = engine.search(algorithm, query, top_k)

            for position, (product_id, score) in enumerate(ranking, start=1):
                product = engine.product(product_id) or {}
                description = product.get("description", "")
                serp_rows.append(
                    {
                        "query_id": query_id,
                        "query": free_text,
                        "tipo": spec["tipo"],
                        "algoritmo": algorithm,
                        "posicao": position,
                        "product_id": product_id,
                        "name": product.get("name", ""),
                        "category": product.get("category", ""),
                        "price": product.get("price", float("nan")),
                        "rating": product.get("rating", float("nan")),
                        "score": round(float(score), 4),
                        "relevancia": judgments.get(str(product_id), 0),
                        "description": description[:300],
                    }
                )

            metric_rows.append(
                {
                    "query_id": query_id,
                    "query": free_text,
                    "tipo": spec["tipo"],
                    "algoritmo": algorithm,
                    "resultados": len(ranking),
                    "P@5": round(precision_at_k(ranking, judgments, top_k), 4),
                    "nDCG@5": round(ndcg_at_k(ranking, judgments, top_k), 4),
                    "irrelevantes@5": round(
                        irrelevant_rate(ranking, judgments, top_k), 4
                    ),
                }
            )

    return pd.DataFrame(serp_rows), pd.DataFrame(metric_rows)


# --------------------------------------------------------------------------- #
# Ajuste do BM25
# --------------------------------------------------------------------------- #


def tune_bm25(
    engine: SearchEngine,
    qrels: Dict[str, Judgments],
    k1_values: Sequence[float] = config.BM25_TUNING_GRID["k1"],
    b_values: Sequence[float] = config.BM25_TUNING_GRID["b"],
    top_k: int = config.SERP_TOP_K,
) -> pd.DataFrame:
    """
    Varre a grade de ``k1`` e ``b`` maximizando o nDCG@5 médio.

    Args:
        engine: Motor com os índices construídos.
        qrels: Julgamentos de relevância.
        k1_values: Valores de saturação a testar.
        b_values: Valores de normalização de comprimento a testar.
        top_k: Profundidade de avaliação.

    Returns:
        ``DataFrame`` com uma linha por combinação, ordenado por nDCG@5
        decrescente.
    """
    rows: List[Dict[str, float]] = []

    for k1 in k1_values:
        for b in b_values:
            scores = []
            precisions = []
            for spec in config.QUALITATIVE_QUERIES:
                judgments = qrels.get(spec["id"], {})
                ranking = bm25_search(
                    spec["query"], engine.inverted_index, engine.corpus,
                    k1=k1, b=b, top_k=top_k,
                )
                scores.append(ndcg_at_k(ranking, judgments, top_k))
                precisions.append(precision_at_k(ranking, judgments, top_k))

            rows.append(
                {
                    "k1": k1,
                    "b": b,
                    "nDCG@5": round(float(np.mean(scores)), 4),
                    "P@5": round(float(np.mean(precisions)), 4),
                }
            )

    frame = pd.DataFrame(rows).sort_values("nDCG@5", ascending=False)
    best = frame.iloc[0]
    default = frame[(frame["k1"] == config.BM25_K1) & (frame["b"] == config.BM25_B)]
    default_ndcg = (
        float(default["nDCG@5"].iloc[0]) if not default.empty else float("nan")
    )

    logger.info(
        "BM25 padrão (k1=%.1f, b=%.2f): nDCG@5 %.4f | melhor (k1=%.1f, b=%.2f): %.4f",
        config.BM25_K1,
        config.BM25_B,
        default_ndcg,
        best["k1"],
        best["b"],
        best["nDCG@5"],
    )
    return frame


# --------------------------------------------------------------------------- #
# Apresentação
# --------------------------------------------------------------------------- #


def lexical_coverage(
    engine: SearchEngine,
    query: str,
    ranking: Ranking,
) -> float:
    """
    Fração do TOP-K que contém literalmente **todos** os termos da consulta.

    É o diagnóstico que separa correspondência léxica de correspondência
    semântica. Se um modelo devolve documentos relevantes cujo texto não
    contém todos os termos buscados, ele os encontrou por outra via — no caso
    do LSA, pela proximidade no espaço latente.

    Args:
        engine: Motor com o corpus indexado.
        query: Consulta em texto livre.
        ranking: Resultado do modelo.

    Returns:
        Valor em ``[0, 1]``; ``0.0`` para ranking vazio.
    """
    terms = set(utils.preprocess_query(query))
    if not terms or not ranking:
        return 0.0

    full_matches = 0
    for product_id, _ in ranking:
        product = engine.product(product_id)
        if product is None:
            continue
        tokens = set(utils.preprocess(utils.build_document_text(product)))
        if terms <= tokens:
            full_matches += 1
    return full_matches / len(ranking)


def coverage_analysis(
    engine: SearchEngine,
    qrels: Dict[str, Judgments],
    top_k: int = config.SERP_TOP_K,
) -> pd.DataFrame:
    """
    Cruza relevância e cobertura léxica para cada consulta e algoritmo.

    Args:
        engine: Motor com os índices construídos.
        qrels: Julgamentos de relevância.
        top_k: Profundidade da SERP.

    Returns:
        ``DataFrame`` com ``P@5`` e ``cobertura_lexica@5`` por consulta e
        algoritmo.
    """
    rows: List[Dict[str, object]] = []
    for spec in config.QUALITATIVE_QUERIES:
        judgments = qrels.get(spec["id"], {})
        for algorithm in config.ALGORITHMS:
            query = spec["boolean_query"] if algorithm == "Boolean" else spec["query"]
            ranking = engine.search(algorithm, query, top_k)
            rows.append(
                {
                    "query_id": spec["id"],
                    "algoritmo": algorithm,
                    "P@5": round(precision_at_k(ranking, judgments, top_k), 4),
                    "cobertura_lexica@5": round(
                        lexical_coverage(engine, spec["query"], ranking), 4
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega as métricas por algoritmo.

    Args:
        metrics: ``DataFrame`` por consulta e algoritmo.

    Returns:
        ``DataFrame`` indexado por algoritmo, na ordem de
        ``config.ALGORITHMS``.
    """
    summary = (
        metrics.groupby("algoritmo")[["P@5", "nDCG@5", "irrelevantes@5"]]
        .mean()
        .round(4)
        .reindex(list(config.ALGORITHMS))
    )
    return summary


def pivot_serp(serp: pd.DataFrame) -> pd.DataFrame:
    """
    Monta a tabela comparativa ``Query x Algoritmo`` com o TOP-5 resumido.

    Args:
        serp: Detalhamento posição a posição.

    Returns:
        ``DataFrame`` com uma linha por consulta e uma coluna por algoritmo.
    """
    def _cell(group: pd.DataFrame) -> str:
        """Resume o TOP-5 de um par (consulta, algoritmo) em uma célula."""
        parts = []
        for _, row in group.sort_values("posicao").iterrows():
            flag = "✓" if row["relevancia"] > 0 else "✗"
            parts.append(f"{flag} {row['name']}")
        return "\n".join(parts) if parts else "(vazio)"

    cells = (
        serp.groupby(["query_id", "algoritmo"], sort=False)
        .apply(_cell, include_groups=False)
        .rename("top5")
        .reset_index()
    )
    return cells.pivot(index="query_id", columns="algoritmo", values="top5").reindex(
        columns=list(config.ALGORITHMS)
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Ponto de entrada da análise qualitativa.

    Args:
        argv: Argumentos de linha de comando.

    Returns:
        Código de saída do processo.
    """
    parser = argparse.ArgumentParser(
        description="Análise qualitativa de relevância (SERP)."
    )
    parser.add_argument(
        "--skip-tuning", action="store_true", help="Não varrer a grade do BM25."
    )
    parser.add_argument(
        "--top-k", type=int, default=config.SERP_TOP_K, help="Profundidade da SERP."
    )
    args = parser.parse_args(argv)

    config.setup_logging()
    config.ensure_directories()

    corpus = load_corpus()
    qrels = load_qrels()
    if not qrels:
        logger.error(
            "Sem julgamentos de relevância. "
            "Rode 'python src/corpus_generator.py' antes."
        )
        return 1

    utils.warm_up(utils.build_corpus_texts(corpus[:50]))
    engine = SearchEngine.build(corpus)

    serp, metrics = run_serp_analysis(engine, qrels, args.top_k)
    serp.to_csv(config.SERP_CSV_PATH, index=False, encoding="utf-8-sig")
    metrics.to_csv(config.RELEVANCE_CSV_PATH, index=False, encoding="utf-8-sig")
    logger.info("SERP salva em %s", config.SERP_CSV_PATH)
    logger.info("Métricas salvas em %s", config.RELEVANCE_CSV_PATH)

    print("\n=== Métricas por consulta (nDCG@5) ===")
    print(
        metrics.pivot(index="query_id", columns="algoritmo", values="nDCG@5")
        .reindex(columns=list(config.ALGORITHMS))
        .to_string(float_format=lambda v: f"{v:6.3f}")
    )

    print("\n=== Média por algoritmo ===")
    print(summarize(metrics).to_string(float_format=lambda v: f"{v:7.4f}"))

    coverage = coverage_analysis(engine, qrels, args.top_k)
    coverage.to_csv(config.COVERAGE_CSV_PATH, index=False, encoding="utf-8-sig")
    logger.info("Cobertura léxica salva em %s", config.COVERAGE_CSV_PATH)
    print(
        "\n=== Cobertura léxica@5 (fração do TOP-5 que contém TODOS os termos) ==="
    )
    print(
        coverage.pivot(
            index="query_id", columns="algoritmo", values="cobertura_lexica@5"
        )
        .reindex(columns=list(config.ALGORITHMS))
        .to_string(float_format=lambda v: f"{v:6.2f}")
    )

    if not args.skip_tuning:
        tuning = tune_bm25(engine, qrels, top_k=args.top_k)
        tuning.to_csv(config.BM25_TUNING_CSV_PATH, index=False)
        logger.info("Grade do BM25 salva em %s", config.BM25_TUNING_CSV_PATH)
        print("\n=== BM25: melhores combinações ===")
        print(tuning.head(8).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
