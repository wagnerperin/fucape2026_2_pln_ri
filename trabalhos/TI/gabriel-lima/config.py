"""
Configuração centralizada do benchmark de modelos clássicos de Recuperação da
Informação aplicado a um catálogo de e-commerce.

Todos os caminhos, hiperparâmetros e constantes de reprodutibilidade usados
pelos scripts de geração de corpus, indexação, busca e benchmark vivem aqui.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Final, List, Tuple

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"

PRODUCTS_PATH: Final[Path] = DATA_DIR / "products.json"
QRELS_PATH: Final[Path] = DATA_DIR / "qrels.json"

SCALABILITY_PLOT_PATH: Final[Path] = RESULTS_DIR / "benchmark_scalability.png"
SCALABILITY_CSV_PATH: Final[Path] = RESULTS_DIR / "benchmark_scalability.csv"
INDEXING_CSV_PATH: Final[Path] = RESULTS_DIR / "benchmark_indexing.csv"
SERP_CSV_PATH: Final[Path] = RESULTS_DIR / "serp_analysis.csv"
RELEVANCE_CSV_PATH: Final[Path] = RESULTS_DIR / "relevance_metrics.csv"
BM25_TUNING_CSV_PATH: Final[Path] = RESULTS_DIR / "bm25_tuning.csv"
COVERAGE_CSV_PATH: Final[Path] = RESULTS_DIR / "lexical_coverage.csv"
REPORT_PATH: Final[Path] = RESULTS_DIR / "REPORT.md"

# --------------------------------------------------------------------------- #
# Reprodutibilidade
# --------------------------------------------------------------------------- #

RANDOM_SEED: Final[int] = 42

# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #

CORPUS_SIZE: Final[int] = 5_000
"""Número de produtos gravados em ``data/products.json``."""

DESCRIPTION_MIN_WORDS: Final[int] = 200
DESCRIPTION_MAX_WORDS: Final[int] = 500

CATEGORIES: Final[Tuple[str, ...]] = (
    "Eletrônicos",
    "Informática",
    "Roupas",
    "Casa",
    "Esportes",
    "Livros",
    "Pet Shop",
    "Instrumentos Musicais",
    "Alimentos",
    "Beleza",
)

# --------------------------------------------------------------------------- #
# Pré-processamento
# --------------------------------------------------------------------------- #

LANGUAGE: Final[str] = "portuguese"

STOPWORDS_KEEP: Final[frozenset] = frozenset({"sem", "não", "nao"})
"""
Palavras que a lista padrão do NLTK considera stopwords mas que carregam
significado no domínio de e-commerce (``mouse sem fio`` != ``mouse fio``).
"""

USE_STEMMING: Final[bool] = True
STRIP_ACCENTS: Final[bool] = True
MIN_TOKEN_LENGTH: Final[int] = 2

# --------------------------------------------------------------------------- #
# Hiperparâmetros dos modelos
# --------------------------------------------------------------------------- #

TOP_K: Final[int] = 10
"""Número padrão de documentos retornados por uma busca."""

SERP_TOP_K: Final[int] = 5
"""Profundidade da SERP usada na análise qualitativa."""

BM25_K1: Final[float] = 1.5
BM25_B: Final[float] = 0.75

BM25_TUNING_GRID: Final[Dict[str, Tuple[float, ...]]] = {
    "k1": (0.6, 0.9, 1.2, 1.5, 2.0, 2.5),
    "b": (0.0, 0.25, 0.5, 0.75, 1.0),
}

LSA_COMPONENTS: Final[int] = 100
"""Dimensionalidade do espaço latente do TruncatedSVD."""

TFIDF_MIN_DF: Final[int] = 1
TFIDF_MAX_DF: Final[float] = 0.95

# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #

BENCHMARK_SIZES: Final[Tuple[int, ...]] = (100, 500, 1_000, 5_000, 10_000)
BENCHMARK_REPEATS: Final[int] = 7
"""
Repetições por consulta. Reporta-se a mediana das repetições de cada
consulta e, depois, a média entre consultas — ver
``benchmark.benchmark_scalability.measure_query_latency``.
"""

ALGORITHMS: Final[Tuple[str, ...]] = ("Linear", "Boolean", "Vetorial", "BM25", "LSA")

BENCHMARK_QUERIES: Final[Tuple[str, ...]] = (
    "mouse",
    "teclado mecânico",
    "notebook",
    "mouse óptico sem fio",
    "fone de ouvido bluetooth",
    "camiseta algodão masculina",
    "cafeteira elétrica para casa",
    "tênis de corrida amortecimento",
    "mouse ergonômico bluetooth 5.0 recarregável",
    "periférico para notebook com conexão usb-c",
)
"""Dez queries de complexidade crescente (1 a 7 termos) usadas na latência."""

# --------------------------------------------------------------------------- #
# Queries da análise qualitativa (Tarefa 5)
# --------------------------------------------------------------------------- #

QUALITATIVE_QUERIES: Final[Tuple[Dict[str, str], ...]] = (
    {
        "id": "Q1",
        "tipo": "Termo genérico / polissemia",
        "query": "mouse",
        "boolean_query": "mouse",
        "objetivo": (
            "Verificar se o periférico se separa do sentido animal "
            "('rato'/'mouse' de brinquedos de pet shop)."
        ),
    },
    {
        "id": "Q2",
        "tipo": "Combinação multi-termo",
        "query": "mouse óptico sem fio",
        "boolean_query": "mouse AND óptico AND (sem-fio OR wireless)",
        "objetivo": "Medir a qualidade do casamento de múltiplos termos.",
    },
    {
        "id": "Q3",
        "tipo": "Termo raro / específico",
        "query": "mouse ergonômico bluetooth 5.0",
        "boolean_query": "mouse AND ergonômico AND bluetooth",
        "objetivo": "Avaliar o comportamento com atributos de baixa frequência.",
    },
    {
        "id": "Q4",
        "tipo": "Booleano",
        "query": "teclado AND (mecânico OR rgb) AND NOT gamer",
        "boolean_query": "teclado AND (mecânico OR rgb) AND NOT gamer",
        "objetivo": "Contrastar o modelo booleano com os modelos ranqueados.",
    },
    {
        "id": "Q5",
        "tipo": "Semântico / sinonímia",
        "query": "periférico para notebook",
        "boolean_query": "periférico AND (notebook OR laptop)",
        "objetivo": "Testar se o LSA recupera sinônimos não presentes na query.",
    },
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

LOG_LEVEL: Final[int] = logging.INFO
LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
LOG_DATEFMT: Final[str] = "%H:%M:%S"


def setup_logging(level: int = LOG_LEVEL) -> None:
    """
    Configura o logging global do projeto.

    Também força UTF-8 na saída padrão: o console do Windows usa cp1252 por
    padrão, o que quebraria a impressão de acentos e dos símbolos usados nas
    tabelas de resultado.

    Args:
        level: Nível mínimo de log (padrão ``logging.INFO``).

    Exemplo:
        >>> setup_logging()
        >>> logging.getLogger(__name__).info("pronto")
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - stream exótico
                pass

    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT)


def ensure_directories() -> None:
    """Cria ``data/`` e ``results/`` caso ainda não existam."""
    for directory in (DATA_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


__all__: List[str] = [name for name in dir() if not name.startswith("_")]
