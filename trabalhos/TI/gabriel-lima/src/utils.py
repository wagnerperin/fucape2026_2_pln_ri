"""
Utilitários de pré-processamento textual para o benchmark de RI.

Reúne as rotinas compartilhadas por todos os modelos de recuperação:
normalização, tokenização, remoção de stopwords e stemming em português
(RSLP). O pipeline é determinístico e fortemente cacheado, já que o mesmo
vocabulário é reprocessado milhares de vezes durante os benchmarks.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Tokenização
# --------------------------------------------------------------------------- #

TOKEN_PATTERN: re.Pattern = re.compile(r"\d+(?:[.,]\d+)+|[^\W_]+", re.UNICODE)
"""
Captura números com casas decimais como um único token (``bluetooth 5.0`` ->
``['bluetooth', '5.0']``) e, fora isso, sequências alfanuméricas. Hífens e
underscores são separadores, de modo que ``sem-fio`` vira ``['sem', 'fio']`` e
casa com a forma ``sem fio`` escrita nas descrições.
"""

BOOLEAN_OPERATORS: frozenset = frozenset({"AND", "OR", "NOT"})


# --------------------------------------------------------------------------- #
# Recursos externos (NLTK) com degradação graciosa
# --------------------------------------------------------------------------- #

_FALLBACK_STOPWORDS: frozenset = frozenset(
    """a à às ao aos aquela aquelas aquele aqueles aquilo as até com como da das de
    dela delas dele deles depois do dos e ela elas ele eles em entre era eram essa
    essas esse esses esta estas este estes eu foi fomos for foram há isso isto já
    lhe lhes mais mas me mesmo meu meus minha minhas muito na nas nem no nos nós
    nossa nossas nosso nossos num numa o os ou para pela pelas pelo pelos por
    qual quando que quem se seja sejam ser seu seus só sua suas também te tem
    temos ter teu teus tu tua tuas um uma umas uns você vocês""".split()
)


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset:
    """
    Retorna o conjunto de stopwords em português usado pelo pipeline.

    Parte da lista do NLTK (com fallback embutido caso o corpus não esteja
    baixado) e remove os termos listados em ``config.STOPWORDS_KEEP``, que são
    semanticamente relevantes no domínio (``mouse sem fio``).

    Returns:
        Conjunto imutável de stopwords, nas formas com e sem acento.

    Exemplo:
        >>> "sem" in get_stopwords()
        False
    """
    try:
        from nltk.corpus import stopwords  # import tardio: NLTK é opcional

        words: Set[str] = set(stopwords.words(config.LANGUAGE))
        logger.debug("Stopwords do NLTK carregadas (%d termos).", len(words))
    except Exception as exc:  # pragma: no cover - depende do ambiente
        logger.warning("NLTK stopwords indisponível (%s); usando fallback.", exc)
        words = set(_FALLBACK_STOPWORDS)

    keep = set(config.STOPWORDS_KEEP)
    keep |= {strip_accents(word) for word in config.STOPWORDS_KEEP}
    words = {w for w in words if w not in keep and strip_accents(w) not in keep}
    return frozenset(words | {strip_accents(w) for w in words})


class _NullStemmer:
    """Stemmer identidade, usado quando o RSLP do NLTK não está disponível."""

    @staticmethod
    def stem(word: str) -> str:
        """Devolve a palavra inalterada."""
        return word


@lru_cache(maxsize=1)
def get_stemmer() -> object:
    """
    Instancia o stemmer RSLP (português) do NLTK.

    Returns:
        Objeto com método ``stem(word) -> str``. Caso o recurso ``rslp`` não
        esteja instalado, devolve um stemmer identidade para que o pipeline
        continue funcionando (com perda de recall).
    """
    try:
        from nltk.stem import RSLPStemmer

        stemmer = RSLPStemmer()
        stemmer.stem("teclados")  # falha cedo se o recurso não existir
        logger.debug("RSLPStemmer carregado.")
        return stemmer
    except Exception as exc:  # pragma: no cover - depende do ambiente
        logger.warning("RSLPStemmer indisponível (%s); stemming desativado.", exc)
        return _NullStemmer()


# --------------------------------------------------------------------------- #
# Normalização
# --------------------------------------------------------------------------- #


def strip_accents(text: str) -> str:
    """
    Remove diacríticos preservando os demais caracteres.

    Args:
        text: Texto de entrada.

    Returns:
        Texto sem acentos.

    Exemplo:
        >>> strip_accents("óptico ergonômico")
        'optico ergonomico'
    """
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize(text: str) -> str:
    """
    Normaliza um texto para caixa baixa com espaços colapsados.

    Args:
        text: Texto de entrada.

    Returns:
        Texto em caixa baixa, sem espaços duplicados nem nas bordas.
    """
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> List[str]:
    """
    Quebra um texto em tokens alfanuméricos em caixa baixa.

    Não aplica stopwords nem stemming: é a etapa crua usada também pela busca
    linear, que por definição não possui índice.

    Args:
        text: Texto de entrada.

    Returns:
        Lista de tokens na ordem de ocorrência.

    Exemplo:
        >>> tokenize("Mouse sem-fio Bluetooth 5.0")
        ['mouse', 'sem', 'fio', 'bluetooth', '5.0']
    """
    return TOKEN_PATTERN.findall(text.lower())


_STEM_CACHE: Dict[str, str] = {}


def stem_token(token: str) -> str:
    """
    Aplica stemming RSLP a um token, com cache de dicionário.

    O cache é essencial: o benchmark reprocessa milhões de tokens sobre um
    vocabulário de poucos milhares de palavras distintas.

    Args:
        token: Token já em caixa baixa.

    Returns:
        Radical do token, sem acentos se ``config.STRIP_ACCENTS`` estiver ativo.

    Exemplo:
        >>> stem_token("ergonômicos")
        'ergonom'
    """
    cached = _STEM_CACHE.get(token)
    if cached is not None:
        return cached

    stem = token
    if config.USE_STEMMING and not token[0].isdigit():
        try:
            stem = get_stemmer().stem(token)
        except Exception:  # pragma: no cover - RSLP é tolerante, mas não infalível
            stem = token
    if config.STRIP_ACCENTS:
        stem = strip_accents(stem)

    _STEM_CACHE[token] = stem
    return stem


def preprocess(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Pipeline completo: caixa baixa, tokenização, stopwords, stemming e acentos.

    É o *analyzer* compartilhado por índice invertido, TF-IDF, BM25 e LSA, o
    que garante que todos os modelos enxerguem exatamente o mesmo vocabulário.

    Args:
        text: Texto livre (nome + descrição do produto, ou uma query).
        remove_stopwords: Se ``False``, mantém as stopwords (útil em testes).

    Returns:
        Lista de radicais na ordem de ocorrência.

    Exemplo:
        >>> preprocess("Mouse Óptico Sem Fio para Notebooks")
        ['mous', 'optic', 'sem', 'fio', 'notebook']
    """
    stops = get_stopwords() if remove_stopwords else frozenset()
    tokens: List[str] = []
    for token in tokenize(text):
        if len(token) < config.MIN_TOKEN_LENGTH:
            continue
        if token in stops:
            continue
        stem = stem_token(token)
        if not stem or stem in stops:
            continue
        tokens.append(stem)
    return tokens


def strip_operators(query: str) -> str:
    """
    Remove os operadores booleanos preservando o texto original dos termos.

    Diferente de :func:`preprocess_query`, devolve texto bruto — é o que deve
    ser entregue a um ``TfidfVectorizer`` cujo *analyzer* já é
    :func:`preprocess`, para que o pré-processamento seja aplicado uma única
    vez.

    Args:
        query: Consulta em texto livre ou com sintaxe booleana.

    Returns:
        A consulta sem ``AND``, ``OR``, ``NOT`` e sem parênteses.

    Exemplo:
        >>> strip_operators("teclado AND (mecânico OR rgb) AND NOT gamer")
        'teclado mecânico rgb gamer'
    """
    words = []
    for word in query.split():
        bare = word.strip("()")
        if bare in BOOLEAN_OPERATORS or not bare:
            continue
        words.append(bare)
    return " ".join(words)


def preprocess_query(query: str) -> List[str]:
    """
    Pré-processa uma consulta, descartando operadores booleanos.

    Assim a mesma string (``"teclado AND (mecânico OR rgb) AND NOT gamer"``)
    pode ser enviada tanto ao modelo booleano quanto aos modelos ranqueados.

    Args:
        query: Consulta em texto livre ou com sintaxe booleana.

    Returns:
        Lista de radicais dos termos de conteúdo.

    Exemplo:
        >>> preprocess_query("teclado AND (mecânico OR rgb)")
        ['tecl', 'mecan', 'rgb']
    """
    cleaned = " ".join(
        word for word in query.split() if word.strip("()") not in BOOLEAN_OPERATORS
    )
    return preprocess(cleaned)


def build_document_text(product: Dict) -> str:
    """
    Concatena os campos textuais de um produto em um único documento.

    O nome é repetido duas vezes como um *field boost* simples: um termo no
    título é mais indicativo de relevância do que o mesmo termo no corpo.

    Args:
        product: Dicionário de produto do corpus.

    Returns:
        Texto único a ser indexado.

    Exemplo:
        >>> build_document_text({"name": "Mouse", "description": "d", "category": "c"})
        'Mouse Mouse c d'
    """
    return (
        f"{product['name']} {product['name']} "
        f"{product.get('category', '')} {product['description']}"
    )


def build_corpus_texts(corpus: Sequence[Dict]) -> List[str]:
    """
    Aplica :func:`build_document_text` a um corpus inteiro.

    Args:
        corpus: Lista de produtos.

    Returns:
        Lista de textos alinhada por índice com ``corpus``.
    """
    return [build_document_text(product) for product in corpus]


def top_k_from_scores(
    scores: np.ndarray,
    product_ids: Sequence[int],
    top_k: int,
    min_score: float = 1e-12,
) -> List[Tuple[int, float]]:
    """
    Converte um vetor de scores no ranking final de ``(product_id, score)``.

    Usa ``np.argpartition`` (O(N)) em vez de ordenar o vetor inteiro
    (O(N log N)), o que importa quando N chega a 10.000 documentos.

    Args:
        scores: Vetor de scores alinhado por índice com ``product_ids``.
        product_ids: Identificadores dos documentos.
        top_k: Quantidade máxima de resultados.
        min_score: Scores menores ou iguais a este valor são descartados.

    Returns:
        Lista de ``(product_id, score)`` ordenada por score decrescente e, em
        caso de empate, por ``product_id`` crescente.
    """
    array = np.asarray(scores, dtype=np.float64).ravel()
    if array.size == 0 or top_k <= 0:
        return []

    k = min(top_k, array.size)
    candidates = np.argpartition(-array, k - 1)[:k]
    ordered = sorted(candidates, key=lambda i: (-array[i], product_ids[i]))
    return [
        (int(product_ids[i]), float(array[i])) for i in ordered if array[i] > min_score
    ]


def warm_up(texts: Optional[Iterable[str]] = None) -> None:
    """
    Pré-carrega stopwords, stemmer e cache de radicais.

    Chamado antes das medições para que o custo único de inicializar o NLTK
    não seja contabilizado como tempo de indexação.

    Args:
        texts: Textos opcionais usados para popular o cache de stemming.
    """
    get_stopwords()
    get_stemmer()
    for text in texts or ():
        preprocess(text)
