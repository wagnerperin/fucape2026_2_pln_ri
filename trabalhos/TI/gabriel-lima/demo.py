"""
Demonstração interativa do buscador — *live demo*.

Carrega o catálogo, constrói os índices dos cinco modelos e abre um prompt
onde cada consulta digitada é executada **nos cinco algoritmos ao mesmo
tempo**, exibindo lado a lado o TOP-K de cada um com score e latência real.

Quando a consulta coincide com uma das cinco consultas avaliadas do estudo, os
resultados saem marcados com ✓ / ✗ segundo os julgamentos de relevância de
``data/qrels.json``, deixando visível na hora qual modelo acertou.

Uso:
    python demo.py                          # modo interativo
    python demo.py "mouse óptico sem fio"   # consulta única e sai
    python demo.py --size 1000              # catálogo menor (sobe mais rápido)
    python demo.py --top-k 3                # profundidade do ranking

Comandos do modo interativo:
    :ajuda      lista os comandos
    :exemplos   mostra as consultas de teste do estudo
    :topk N     muda a profundidade do ranking
    :produto N  exibe a ficha completa de um produto
    :sair       encerra
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from src import utils  # noqa: E402
from src.corpus_generator import load_corpus, load_qrels  # noqa: E402
from src.engine import SearchEngine  # noqa: E402

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Apresentação no terminal
# --------------------------------------------------------------------------- #


class Style:
    """
    Códigos ANSI de cor, desligados automaticamente quando não há suporte.

    A demo é feita para ser projetada em tela; a cor separa o nome do
    algoritmo do conteúdo do ranking. Mas redirecionar a saída para um arquivo
    (ou rodar em um terminal antigo) não pode encher o texto de lixo, então as
    sequências viram strings vazias quando a saída não é um terminal.
    """

    ENABLED: bool = False

    RESET = BOLD = DIM = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = GREY = ""

    @classmethod
    def setup(cls, force: bool = False) -> None:
        """
        Ativa as cores se o terminal suportar.

        Args:
            force: Liga as cores mesmo sem detectar um terminal.
        """
        supported = force or (
            sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
        )
        if supported and os.name == "nt":
            # Habilita o processamento de sequências ANSI no console do Windows.
            try:
                import colorama

                colorama.just_fix_windows_console()
            except Exception:  # pragma: no cover - colorama é opcional
                os.system("")

        if not supported:
            return

        cls.ENABLED = True
        cls.RESET = "\033[0m"
        cls.BOLD = "\033[1m"
        cls.DIM = "\033[2m"
        cls.RED = "\033[31m"
        cls.GREEN = "\033[32m"
        cls.YELLOW = "\033[33m"
        cls.BLUE = "\033[34m"
        cls.MAGENTA = "\033[35m"
        cls.CYAN = "\033[36m"
        cls.GREY = "\033[90m"


ALGORITHM_COLOR: Dict[str, str] = {
    "Linear": "BLUE",
    "Boolean": "YELLOW",
    "Vetorial": "GREEN",
    "BM25": "CYAN",
    "LSA": "MAGENTA",
}

ALGORITHM_NOTE: Dict[str, str] = {
    "Linear": "varredura sequencial, sem índice",
    "Boolean": "matriz de incidência, operações bitwise",
    "Vetorial": "TF-IDF + cosseno",
    "BM25": "probabilístico, índice invertido",
    "LSA": f"espaço latente, k={config.LSA_COMPONENTS}",
}

WIDTH = 78


def color(text: str, name: str) -> str:
    """
    Aplica uma cor nomeada a um texto.

    Args:
        text: Texto a colorir.
        name: Nome do atributo em :class:`Style` (ex.: ``"CYAN"``).

    Returns:
        O texto com os códigos ANSI, ou inalterado se as cores estiverem off.
    """
    prefix = getattr(Style, name, "")
    return f"{prefix}{text}{Style.RESET}" if prefix else text


def rule(char: str = "─") -> str:
    """Devolve uma linha horizontal da largura da saída."""
    return color(char * WIDTH, "GREY")


def truncate(text: str, limit: int) -> str:
    """
    Encurta um texto preservando a legibilidade.

    Args:
        text: Texto de entrada.
        limit: Comprimento máximo.

    Returns:
        O texto, com reticências se precisou ser cortado.
    """
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------- #
# Relevância conhecida
# --------------------------------------------------------------------------- #


def find_known_query(query: str) -> Optional[Dict[str, str]]:
    """
    Localiza a consulta avaliada do estudo que corresponde ao texto digitado.

    Args:
        query: Texto digitado pelo usuário.

    Returns:
        A especificação da consulta (``config.QUALITATIVE_QUERIES``) ou
        ``None`` se o texto não corresponder a nenhuma delas.
    """
    def fold(text: str) -> str:
        """Caixa baixa, sem acento e sem operadores — como se compara aqui."""
        return utils.strip_accents(utils.normalize(utils.strip_operators(text)))

    # Numa demo ao vivo é comum digitar sem acento ("periferico"); a consulta
    # deve ser reconhecida assim mesmo, senão as marcas de relevância somem.
    normalized = fold(query)
    for spec in config.QUALITATIVE_QUERIES:
        if normalized in (fold(spec["query"]), fold(spec["boolean_query"])):
            return spec
    return None


def relevance_mark(grade: Optional[int]) -> str:
    """
    Converte um grau de relevância no símbolo exibido na SERP.

    Args:
        grade: Grau 0-3, ou ``None`` quando a consulta não é avaliada.

    Returns:
        ``"✓"`` (relevante), ``"~"`` (parcial), ``"✗"`` (irrelevante) ou
        espaço em branco.
    """
    if grade is None:
        return " "
    if grade >= 3:
        return color("✓", "GREEN")
    if grade > 0:
        return color("~", "YELLOW")
    return color("✗", "RED")


# --------------------------------------------------------------------------- #
# Execução de uma consulta
# --------------------------------------------------------------------------- #


def run_query(
    engine: SearchEngine,
    query: str,
    qrels: Dict[str, Dict[str, int]],
    top_k: int,
) -> None:
    """
    Executa a consulta nos cinco modelos e imprime a comparação.

    Args:
        engine: Motor com os índices construídos.
        query: Consulta digitada.
        qrels: Julgamentos de relevância carregados.
        top_k: Profundidade do ranking exibido.
    """
    known = find_known_query(query)
    judgments: Dict[str, int] = {}
    if known is not None:
        judgments = qrels.get(known["id"], {})

    print()
    print(rule("═"))
    print(f" {color('CONSULTA', 'BOLD')}  {color(query, 'BOLD')}")
    if known is not None:
        print(
            f" {color(known['id'] + ' · ' + known['tipo'], 'GREY')}"
        )
        print(f" {color(known['objetivo'], 'GREY')}")
    stems = utils.preprocess_query(query)
    print(f" {color('radicais indexados: ' + ' · '.join(stems), 'GREY')}")
    print(rule("═"))

    timings: Dict[str, float] = {}

    for algorithm in config.ALGORITHMS:
        start = time.perf_counter()
        # O modelo booleano recebe a sintaxe explícita quando ela existe;
        # os demais recebem texto livre. Cada um no regime para que foi feito.
        text = (
            known["boolean_query"]
            if (algorithm == "Boolean" and known is not None)
            else query
        )
        results = engine.search(algorithm, text, top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings[algorithm] = elapsed_ms

        header = color(f"{algorithm:<9}", ALGORITHM_COLOR[algorithm])
        note = color(ALGORITHM_NOTE[algorithm], "GREY")
        timing = color(f"{elapsed_ms:7.2f} ms", "BOLD")
        print(f"\n{header} {timing}  {note}")

        if not results:
            print(f"   {color('(nenhum resultado)', 'GREY')}")
            continue

        for position, (product_id, score) in enumerate(results, start=1):
            product = engine.product(product_id) or {}
            # Documentos de grau zero são omitidos do qrels: num consulta
            # avaliada, ausência significa irrelevante (✗), não "sem
            # julgamento" — é o que precisa ficar visível na demo.
            grade = judgments.get(str(product_id), 0) if known is not None else None
            mark = relevance_mark(grade)
            name = truncate(product.get("name", "?"), 46)
            price = product.get("price")
            if isinstance(price, (int, float)):
                # Formato brasileiro: milhar com ponto, decimal com vírgula.
                price_text = (
                    f"R$ {price:>9,.2f}".replace(",", "X")
                    .replace(".", ",")
                    .replace("X", ".")
                )
            else:
                price_text = " " * 12
            print(
                f"   {position}. {mark} {name:<46} "
                f"{color(f'{score:8.4f}', 'GREY')} {color(price_text, 'GREY')}"
            )

    # Resumo comparativo de latência: é o ponto alto da demo ao vivo.
    print()
    print(rule())
    fastest = min(timings, key=timings.get)
    slowest = max(timings, key=timings.get)
    parts = []
    for algorithm in config.ALGORITHMS:
        value = f"{algorithm} {timings[algorithm]:.2f}ms"
        if algorithm == fastest:
            value = color(value, "GREEN")
        elif algorithm == slowest:
            value = color(value, "RED")
        parts.append(value)
    print(" " + "  ·  ".join(parts))
    if timings[fastest] > 0:
        ratio = timings[slowest] / timings[fastest]
        print(
            f" {color(f'{fastest} foi {ratio:.0f}x mais rápido que {slowest}', 'BOLD')}"
        )
    if known is not None:
        print(
            f" {color('✓ relevante  ~ parcialmente relevante  ✗ irrelevante', 'GREY')}"
        )
    print(rule())


def show_product(engine: SearchEngine, product_id: int) -> None:
    """
    Exibe a ficha completa de um produto.

    Args:
        engine: Motor com o corpus carregado.
        product_id: Identificador do produto.
    """
    product = engine.product(product_id)
    if product is None:
        print(color(f"  produto {product_id} não existe no catálogo", "RED"))
        return

    print()
    print(rule("═"))
    print(f" {color(product['name'], 'BOLD')}")
    print(
        f" {color('id', 'GREY')} {product['product_id']}   "
        f"{color('categoria', 'GREY')} {product['category']}   "
        f"{color('preço', 'GREY')} R$ {product['price']:.2f}   "
        f"{color('nota', 'GREY')} {product['rating']}"
    )
    print(rule())
    description = product["description"]
    words = description.split()
    print(f" {color(f'descrição ({len(words)} palavras)', 'GREY')}\n")
    # Quebra manual em parágrafo justificado à largura da demo.
    line = " "
    for word in words:
        if len(line) + len(word) + 1 > WIDTH:
            print(line)
            line = " "
        line += " " + word
    if line.strip():
        print(line)
    print(rule("═"))


# --------------------------------------------------------------------------- #
# Modo interativo
# --------------------------------------------------------------------------- #


def print_banner(engine: SearchEngine, build_seconds: float) -> None:
    """
    Imprime o cabeçalho da demo com o estado dos índices.

    Args:
        engine: Motor construído.
        build_seconds: Tempo total de construção dos índices.
    """
    print()
    print(rule("═"))
    print(
        f" {color('BUSCA EM CATÁLOGO DE E-COMMERCE', 'BOLD')} · "
        f"{color('5 modelos clássicos de RI lado a lado', 'GREY')}"
    )
    print(rule("═"))
    docs = f"{engine.n_docs:,}".replace(",", ".")
    vocabulary = f"{engine.inverted_index.vocabulary_size:,}".replace(",", ".")
    print(f" catálogo    {docs} produtos   ·   vocabulário {vocabulary} radicais")
    print(
        f" índices     construídos em {build_seconds:.1f}s   ·   "
        f"LSA k={engine.lsa_index.n_components} "
        f"({engine.lsa_index.explained_variance * 100:.1f}% da variância)"
    )
    print(" tempos de indexação por modelo:")
    for algorithm in config.ALGORITHMS:
        ms = engine.indexing_times_ms[algorithm]
        label = f"{ms:,.0f} ms".replace(",", ".") if ms else "— (sem índice)"
        print(
            f"   {color(f'{algorithm:<9}', ALGORITHM_COLOR[algorithm])} {label}"
        )
    print(rule())
    print(
        f" digite uma consulta e tecle Enter · "
        f"{color(':ajuda', 'BOLD')} para os comandos · "
        f"{color(':sair', 'BOLD')} para encerrar"
    )
    print(rule("═"))


def print_help() -> None:
    """Imprime a lista de comandos do modo interativo."""
    print()
    print(f" {color('COMANDOS', 'BOLD')}")
    for command, description in (
        (":ajuda", "mostra esta lista"),
        (":exemplos", "as 5 consultas avaliadas no estudo"),
        (":topk N", "muda a profundidade do ranking (atual exibido no prompt)"),
        (":produto N", "ficha completa do produto de id N"),
        (":sair", "encerra a demo (ou Ctrl+C)"),
    ):
        print(f"   {color(command, 'CYAN'):<28} {description}")
    print()
    print(f" {color('A consulta pode usar sintaxe booleana:', 'GREY')}")
    print(f"   {color('teclado AND (mecânico OR rgb) AND NOT gamer', 'GREY')}")
    print(
        " "
        + color(
            "Os operadores valem para o modelo booleano; os demais os ignoram.",
            "GREY",
        )
    )


def print_examples() -> None:
    """Imprime as consultas avaliadas do estudo, prontas para copiar."""
    print()
    print(f" {color('CONSULTAS AVALIADAS NO ESTUDO', 'BOLD')}")
    for spec in config.QUALITATIVE_QUERIES:
        print(
            f"   {color(spec['id'], 'CYAN')}  {color(spec['query'], 'BOLD')}"
        )
        print(f"       {color(spec['tipo'] + ' — ' + spec['objetivo'], 'GREY')}")


def interactive_loop(
    engine: SearchEngine,
    qrels: Dict[str, Dict[str, int]],
    top_k: int,
) -> int:
    """
    Executa o laço de leitura de consultas.

    Args:
        engine: Motor com os índices construídos.
        qrels: Julgamentos de relevância.
        top_k: Profundidade inicial do ranking.

    Returns:
        Código de saída do processo.
    """
    while True:
        try:
            raw = input(color(f"\nbusca[top {top_k}]> ", "BOLD")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + color(" encerrado.", "GREY"))
            return 0

        if not raw:
            continue

        if raw.startswith(":"):
            command, _, argument = raw[1:].partition(" ")
            command = command.lower().strip()
            argument = argument.strip()

            if command in ("sair", "quit", "exit", "q"):
                print(color(" encerrado.", "GREY"))
                return 0
            if command in ("ajuda", "help", "h", "?"):
                print_help()
                continue
            if command in ("exemplos", "examples"):
                print_examples()
                continue
            if command == "topk":
                if argument.isdigit() and int(argument) > 0:
                    top_k = min(int(argument), 20)
                    print(color(f" profundidade do ranking: {top_k}", "GREY"))
                else:
                    print(color(" uso: :topk N  (N inteiro positivo)", "RED"))
                continue
            if command in ("produto", "product"):
                if argument.isdigit():
                    show_product(engine, int(argument))
                else:
                    print(color(" uso: :produto N  (N = product_id)", "RED"))
                continue

            print(color(f" comando desconhecido: :{command}  (tente :ajuda)", "RED"))
            continue

        run_query(engine, raw, qrels, top_k)


# --------------------------------------------------------------------------- #
# Entrada
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Ponto de entrada da demonstração.

    Args:
        argv: Argumentos de linha de comando.

    Returns:
        Código de saída do processo.
    """
    parser = argparse.ArgumentParser(
        description="Demonstração interativa do buscador (5 modelos lado a lado)."
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Consulta a executar. Sem argumento, abre o modo interativo.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Usa apenas os N primeiros produtos (acelera a inicialização).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.SERP_TOP_K,
        help=f"Profundidade do ranking (padrão {config.SERP_TOP_K}).",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Força cores mesmo sem terminal detectado.",
    )
    args = parser.parse_args(argv)

    Style.setup(force=args.color)
    # A demo é visual: os logs de indexação só poluiriam a tela.
    config.setup_logging(logging.WARNING)

    print(color("\n carregando catálogo...", "GREY"), flush=True)
    corpus = load_corpus()
    if args.size is not None:
        corpus = corpus[: args.size]
    qrels = load_qrels()

    print(
        color(
            f" indexando {len(corpus):,} produtos nos 5 modelos...".replace(",", "."),
            "GREY",
        ),
        flush=True,
    )
    utils.warm_up(utils.build_corpus_texts(corpus[:50]))
    started = time.perf_counter()
    engine = SearchEngine.build(corpus)
    build_seconds = time.perf_counter() - started

    # Dispara uma consulta descartada em cada modelo. Sem isso, a primeira
    # busca da demo pagaria os custos de primeira chamada (alocações do NumPy,
    # caches do SciPy) e exibiria uma latência várias vezes maior que a real —
    # justamente no momento em que a plateia está olhando.
    for algorithm in config.ALGORITHMS:
        engine.search(algorithm, "mouse teclado", args.top_k)

    if args.query:
        run_query(engine, " ".join(args.query), qrels, args.top_k)
        return 0

    print_banner(engine, build_seconds)
    return interactive_loop(engine, qrels, args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
