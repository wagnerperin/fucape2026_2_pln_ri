"""
Benchmark quantitativo de escalabilidade (Tarefa 4).

Para cada tamanho de corpus ``N`` em ``config.BENCHMARK_SIZES``:

1. Recorta (ou gera) um corpus com ``N`` produtos.
2. Indexa cada um dos cinco modelos, uma única vez, medindo o tempo.
3. Dispara as dez consultas padronizadas, repetidas
   ``config.BENCHMARK_REPEATS`` vezes, e mede a latência média por consulta.

Saídas:

* ``results/benchmark_scalability.csv`` — latência média (ms) por ``N``.
* ``results/benchmark_indexing.csv`` — tempo de indexação (ms) por ``N``.
* ``results/benchmark_scalability.png`` — gráfico log-log das duas grandezas.

Uso:
    python benchmark/benchmark_scalability.py [--sizes 100 500 1000] [--repeats 3]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from statistics import median
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import utils  # noqa: E402
from src.corpus_generator import generate_corpus, load_corpus  # noqa: E402
from src.engine import SearchEngine  # noqa: E402

logger = logging.getLogger(__name__)


def build_subcorpus(size: int, base_corpus: Sequence[Dict]) -> List[Dict]:
    """
    Obtém um corpus com exatamente ``size`` produtos.

    O gerador semeia cada produto pelo próprio ``product_id``, de modo que o
    catálogo é um superconjunto estável: o prefixo de ``N`` produtos é sempre
    o mesmo, e para ``N`` maior que o corpus salvo basta gerar mais itens sem
    invalidar os anteriores.

    Args:
        size: Número de produtos desejado.
        base_corpus: Corpus carregado do disco.

    Returns:
        Lista com ``size`` produtos.
    """
    if size <= len(base_corpus):
        return list(base_corpus[:size])

    logger.info(
        "N=%d excede o corpus salvo (%d); estendendo com o mesmo gerador.",
        size,
        len(base_corpus),
    )
    return generate_corpus(size)


def measure_query_latency(
    engine: SearchEngine,
    algorithm: str,
    queries: Sequence[str],
    repeats: int,
    top_k: int = config.TOP_K,
) -> float:
    """
    Mede a latência média de um algoritmo sobre um conjunto de consultas.

    Args:
        engine: Motor com os índices já construídos.
        algorithm: Nome do algoritmo.
        queries: Consultas a executar.
        repeats: Repetições por consulta.
        top_k: Profundidade do ranking.

    Returns:
        Tempo médio por consulta, em milissegundos.

    Notas:
        As latências dos modelos indexados ficam na casa das centenas de
        microssegundos, faixa em que o escalonador do sistema operacional
        introduz ruído da ordem da própria medida. Por isso toma-se a
        **mediana** das repetições de cada consulta — que descarta os picos
        esporádicos — e só então a média entre as consultas.
    """
    # Uma execução de aquecimento evita contabilizar custos de primeira
    # chamada (alocações internas do NumPy, caches do SciPy).
    engine.search(algorithm, queries[0], top_k)

    per_query_medians: List[float] = []
    for query in queries:
        samples: List[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            engine.search(algorithm, query, top_k)
            samples.append(time.perf_counter() - start)
        per_query_medians.append(median(samples))

    return (sum(per_query_medians) / len(per_query_medians)) * 1000.0


def run_benchmark(
    sizes: Sequence[int] = config.BENCHMARK_SIZES,
    queries: Sequence[str] = config.BENCHMARK_QUERIES,
    repeats: int = config.BENCHMARK_REPEATS,
    base_corpus: Optional[Sequence[Dict]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Executa o benchmark completo de escalabilidade.

    Args:
        sizes: Tamanhos de corpus a avaliar.
        queries: Consultas padronizadas.
        repeats: Repetições por consulta.
        base_corpus: Corpus base; carregado do disco se ``None``.

    Returns:
        Mapa com dois ``DataFrame``: ``"latency"`` (ms por consulta) e
        ``"indexing"`` (ms de indexação), ambos indexados por ``N``.
    """
    corpus = list(base_corpus) if base_corpus is not None else load_corpus()

    latency_rows: List[Dict[str, float]] = []
    indexing_rows: List[Dict[str, float]] = []

    for size in sizes:
        subcorpus = build_subcorpus(size, corpus)
        logger.info("=" * 68)
        logger.info("N = %d documentos", len(subcorpus))

        # Aquece stopwords, stemmer e cache de radicais: é custo único de
        # processo, não de indexação.
        utils.warm_up(utils.build_corpus_texts(subcorpus[: min(50, len(subcorpus))]))

        engine = SearchEngine.build(subcorpus)

        latency_row: Dict[str, float] = {"N": size}
        for algorithm in config.ALGORITHMS:
            latency_ms = measure_query_latency(engine, algorithm, queries, repeats)
            latency_row[algorithm] = round(latency_ms, 4)
            logger.info(
                "  %-9s busca %9.3f ms/query | indexação %9.1f ms",
                algorithm,
                latency_ms,
                engine.indexing_times_ms[algorithm],
            )

        latency_rows.append(latency_row)
        indexing_rows.append(
            {
                "N": size,
                **{
                    a: round(engine.indexing_times_ms[a], 2)
                    for a in config.ALGORITHMS
                },
            }
        )

    latency = pd.DataFrame(latency_rows).set_index("N")
    indexing = pd.DataFrame(indexing_rows).set_index("N")
    return {"latency": latency, "indexing": indexing}


# --------------------------------------------------------------------------- #
# Gráfico
# --------------------------------------------------------------------------- #

# Paleta categórica validada (adjacent pairlist, modo claro): pior par
# CVD ΔE 9.1 e visão normal ΔE 19.6, ambos acima do piso. Três slots ficam
# abaixo de 3:1 de contraste com a superfície, o que obriga rótulo visível —
# por isso cada série é rotulada diretamente na ponta da linha, além do CSV
# que acompanha o gráfico como table view.
SERIES_COLORS: Dict[str, str] = {
    "Linear": "#2a78d6",
    "Boolean": "#eb6834",
    "Vetorial": "#1baf7a",
    "BM25": "#eda100",
    "LSA": "#e87ba4",
}

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#dcdcd8"


def _place_end_labels(
    axis: object,
    endpoints: Sequence[Tuple[str, float, float]],
    min_gap_fraction: float = 0.052,
) -> None:
    """
    Rotula cada série na ponta da linha, evitando sobreposição.

    Séries que terminam em valores próximos — caso de Vetorial e BM25 no
    painel de indexação — teriam os rótulos impressos um sobre o outro. Aqui
    eles são afastados no eixo logarítmico até respeitarem um espaçamento
    mínimo, preservando a ordem vertical das curvas.

    Args:
        axis: Eixo do matplotlib onde desenhar.
        endpoints: Triplas ``(rótulo, x final, y final)``.
        min_gap_fraction: Espaçamento mínimo entre rótulos, como fração da
            altura do eixo em escala log.
    """
    import numpy as np

    if not endpoints:
        return

    bottom, top = axis.get_ylim()
    span = np.log10(top) - np.log10(bottom)
    min_gap = span * min_gap_fraction

    ordered = sorted(endpoints, key=lambda item: item[2])
    positions = [np.log10(value) for _, _, value in ordered]

    # Varredura de baixo para cima: empurra cada rótulo o mínimo necessário.
    for i in range(1, len(positions)):
        if positions[i] - positions[i - 1] < min_gap:
            positions[i] = positions[i - 1] + min_gap

    # Se a pilha estourou o topo, desloca o conjunto todo para baixo.
    overflow = positions[-1] - (np.log10(top) - min_gap * 0.5)
    if overflow > 0:
        positions = [value - overflow for value in positions]

    for (label, x_end, _), y_position in zip(ordered, positions):
        axis.annotate(
            label,
            xy=(x_end, 10.0**y_position),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT_SECONDARY,
            zorder=4,
        )


def plot_scalability(
    latency: pd.DataFrame,
    indexing: pd.DataFrame,
    output_path: Path = config.SCALABILITY_PLOT_PATH,
) -> Path:
    """
    Desenha o gráfico log-log de latência e de tempo de indexação.

    A escala log-log é o que torna a complexidade legível: uma reta de
    inclinação 1 é ``O(N)``, e uma curva que achata indica custo
    independente de ``N``.

    Args:
        latency: ``DataFrame`` de latência por ``N``.
        indexing: ``DataFrame`` de tempo de indexação por ``N``.
        output_path: Caminho do PNG de saída.

    Returns:
        O caminho do arquivo gravado.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), facecolor=SURFACE)

    panels = (
        (axes[0], latency, "Latência de consulta", "Tempo médio por consulta (ms)"),
        (axes[1], indexing, "Custo de indexação", "Tempo de indexação (ms)"),
    )

    for axis, frame, title, ylabel in panels:
        axis.set_facecolor(SURFACE)
        endpoints: List[Tuple[str, float, float]] = []
        for algorithm in config.ALGORITHMS:
            if algorithm not in frame.columns:
                continue
            series = frame[algorithm]
            # Escala log não representa zero: o baseline linear não indexa.
            plotted = series[series > 0]
            if plotted.empty:
                continue
            axis.plot(
                plotted.index,
                plotted.values,
                color=SERIES_COLORS[algorithm],
                linewidth=2.0,
                marker="o",
                markersize=5.5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
                label=algorithm,
                zorder=3,
            )
            # Rótulo direto na ponta: relief exigido pelo contraste da paleta.
            endpoints.append(
                (algorithm, float(plotted.index[-1]), float(plotted.values[-1]))
            )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(title, fontsize=12.5, color=TEXT_PRIMARY, pad=10, loc="left")
        axis.set_xlabel("Número de documentos (N)", fontsize=10, color=TEXT_SECONDARY)
        axis.set_ylabel(ylabel, fontsize=10, color=TEXT_SECONDARY)
        axis.grid(True, which="major", color=GRID_COLOR, linewidth=0.8, zorder=0)
        axis.grid(
            True, which="minor", color=GRID_COLOR, linewidth=0.4, alpha=0.6, zorder=0
        )
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color(GRID_COLOR)
        axis.set_xticks(list(latency.index))
        axis.set_xticklabels([f"{n:,}".replace(",", ".") for n in latency.index])
        axis.margins(x=0.16)
        # Só depois dos limites definidos é possível espaçar os rótulos.
        _place_end_labels(axis, endpoints)

    # Guia de inclinação O(N) no painel de latência: referência visual para
    # separar "cresce com o corpus" de "praticamente constante".
    first_n = float(latency.index[0])
    last_n = float(latency.index[-1])
    anchor = float(latency["Linear"].iloc[0])
    axes[0].plot(
        [first_n, last_n],
        [anchor, anchor * (last_n / first_n)],
        color=TEXT_SECONDARY,
        linewidth=1.0,
        linestyle=(0, (4, 4)),
        alpha=0.55,
        zorder=1,
    )
    axes[0].annotate(
        "inclinação O(N)",
        xy=(last_n, anchor * (last_n / first_n)),
        xytext=(-4, 8),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        color=TEXT_SECONDARY,
        alpha=0.9,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=9.5,
        labelcolor=TEXT_SECONDARY,
        bbox_to_anchor=(0.5, -0.01),
    )

    figure.suptitle(
        "Escalabilidade dos modelos clássicos de RI — catálogo de e-commerce",
        fontsize=14,
        color=TEXT_PRIMARY,
        x=0.007,
        ha="left",
        y=0.985,
    )
    figure.text(
        0.007,
        0.925,
        f"Escala log-log · mediana de {config.BENCHMARK_REPEATS} repetições por "
        f"consulta, média entre {len(config.BENCHMARK_QUERIES)} consultas padronizadas",
        fontsize=10,
        color=TEXT_SECONDARY,
        ha="left",
    )

    figure.tight_layout(rect=(0, 0.06, 1, 0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)

    logger.info("Gráfico salvo em %s", output_path)
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Ponto de entrada do benchmark de escalabilidade.

    Args:
        argv: Argumentos de linha de comando.

    Returns:
        Código de saída do processo.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark de escalabilidade dos modelos de RI."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(config.BENCHMARK_SIZES),
        help="Tamanhos de corpus a avaliar.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=config.BENCHMARK_REPEATS,
        help="Repetições por consulta.",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Não gerar o gráfico."
    )
    args = parser.parse_args(argv)

    config.setup_logging()
    config.ensure_directories()

    frames = run_benchmark(sizes=args.sizes, repeats=args.repeats)
    latency, indexing = frames["latency"], frames["indexing"]

    latency.to_csv(config.SCALABILITY_CSV_PATH, float_format="%.4f")
    indexing.to_csv(config.INDEXING_CSV_PATH, float_format="%.2f")
    logger.info("Latência salva em %s", config.SCALABILITY_CSV_PATH)
    logger.info("Tempos de indexação salvos em %s", config.INDEXING_CSV_PATH)

    print("\n=== Latência média por consulta (ms) ===")
    print(latency.to_string(float_format=lambda v: f"{v:9.3f}"))
    print("\n=== Tempo de indexação (ms) ===")
    print(indexing.to_string(float_format=lambda v: f"{v:10.1f}"))

    if not args.no_plot:
        plot_scalability(latency, indexing)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
