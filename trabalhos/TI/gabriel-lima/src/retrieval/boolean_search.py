"""
Modelo Booleano sobre matriz de incidência binária.

A consulta é convertida em uma árvore sintática (``AND``, ``OR``, ``NOT`` e
parênteses) e avaliada com operações bitwise sobre os bits empacotados da
:class:`src.indexer.IncidenceMatrix`. Cada operação processa 8 documentos por
byte, de modo que o custo depende do vocabulário da consulta e de ``N/8``, e
não da varredura do texto.

O modelo booleano clássico não ranqueia: ele devolve um conjunto. Para que a
SERP seja comparável à dos modelos ranqueados, o conjunto resultante é
ordenado por *coordination level match* — quantos termos distintos da consulta
o documento contém — o que é a extensão ranqueada tradicional do modelo.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import config
from src import utils
from src.indexer import IncidenceMatrix, InvertedIndex

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")


class BooleanQueryError(ValueError):
    """Erro de sintaxe em uma consulta booleana."""


# --------------------------------------------------------------------------- #
# Árvore sintática
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TermNode:
    """
    Folha da árvore: um termo de busca.

    Attributes:
        stems: Radicais que compõem o termo. Um termo escrito com hífen
            (``sem-fio``) gera vários radicais, combinados com ``AND``.
    """

    stems: Tuple[str, ...]


@dataclass(frozen=True)
class NotNode:
    """Negação de uma subexpressão."""

    child: "Node"


@dataclass(frozen=True)
class BinaryNode:
    """
    Operação binária entre duas subexpressões.

    Attributes:
        operator: ``"AND"`` ou ``"OR"``.
        left: Subexpressão à esquerda.
        right: Subexpressão à direita.
    """

    operator: str
    left: "Node"
    right: "Node"


Node = Union[TermNode, NotNode, BinaryNode]


# --------------------------------------------------------------------------- #
# Parser (descida recursiva)
# --------------------------------------------------------------------------- #


class _Parser:
    """
    Analisador sintático descendente recursivo para consultas booleanas.

    Gramática (precedência crescente)::

        expressao := termo_or
        termo_or  := termo_and ( "OR" termo_and )*
        termo_and := fator ( ("AND" | "NOT" | <implícito>) fator )*
        fator     := "NOT" fator | "(" expressao ")" | TERMO

    A justaposição (``mouse óptico``) é tratada como ``AND`` implícito e
    ``A NOT B`` é lido como ``A AND (NOT B)``, que é a forma usada no
    enunciado (``mouse AND (óptico OR sem-fio) NOT gamer``).
    """

    def __init__(self, tokens: Sequence[str]) -> None:
        """Prepara o analisador sobre a lista de tokens da consulta."""
        self._tokens = list(tokens)
        self._pos = 0

    def _peek(self) -> Optional[str]:
        """Devolve o token atual sem consumi-lo."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _next(self) -> str:
        """Consome e devolve o token atual."""
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def parse(self) -> Node:
        """
        Analisa a consulta completa.

        Returns:
            A raiz da árvore sintática.

        Raises:
            BooleanQueryError: Se a consulta estiver malformada.
        """
        node = self._parse_or()
        if self._peek() is not None:
            raise BooleanQueryError(f"Token inesperado: {self._peek()!r}")
        return node

    def _parse_or(self) -> Node:
        """Analisa uma sequência de ``OR``."""
        node = self._parse_and()
        while self._peek() == "OR":
            self._next()
            node = BinaryNode("OR", node, self._parse_and())
        return node

    def _parse_and(self) -> Node:
        """Analisa ``AND`` explícito, ``AND`` implícito e ``A NOT B``."""
        node = self._parse_factor()
        while True:
            token = self._peek()
            if token == "AND":
                self._next()
                node = BinaryNode("AND", node, self._parse_factor())
            elif token == "NOT":
                # "A NOT B" equivale a "A AND NOT B".
                self._next()
                node = BinaryNode("AND", node, NotNode(self._parse_factor()))
            elif token in (None, ")", "OR"):
                return node
            else:
                node = BinaryNode("AND", node, self._parse_factor())

    def _parse_factor(self) -> Node:
        """Analisa ``NOT`` unário, parênteses e termos."""
        token = self._peek()
        if token is None:
            raise BooleanQueryError("Consulta terminou de forma inesperada.")

        if token == "NOT":
            self._next()
            return NotNode(self._parse_factor())

        if token == "(":
            self._next()
            node = self._parse_or()
            if self._peek() != ")":
                raise BooleanQueryError("Parêntese ')' ausente.")
            self._next()
            return node

        if token == ")":
            raise BooleanQueryError("Parêntese ')' sem abertura correspondente.")

        self._next()
        stems = tuple(utils.preprocess(token))
        return TermNode(stems=stems)


def parse_boolean_query(query: str) -> Node:
    """
    Converte o texto da consulta em árvore sintática.

    Args:
        query: Consulta booleana, ex.: ``"mouse AND (óptico OR sem-fio) NOT gamer"``.

    Returns:
        Raiz da árvore sintática.

    Raises:
        BooleanQueryError: Se a consulta for vazia ou malformada.

    Exemplo:
        >>> node = parse_boolean_query("mouse AND óptico")
        >>> isinstance(node, BinaryNode)
        True
    """
    tokens = _TOKEN_RE.findall(query.strip())
    if not tokens:
        raise BooleanQueryError("Consulta booleana vazia.")
    return _Parser(tokens).parse()


# --------------------------------------------------------------------------- #
# Avaliação bitwise
# --------------------------------------------------------------------------- #


def _evaluate(node: Node, matrix: IncidenceMatrix) -> np.ndarray:
    """
    Avalia a árvore sintática produzindo uma linha de bits.

    Args:
        node: Nó da árvore.
        matrix: Matriz de incidência do corpus.

    Returns:
        Linha de bits empacotados com os documentos que satisfazem o nó.

    Raises:
        BooleanQueryError: Se o nó for de um tipo desconhecido.
    """
    if isinstance(node, TermNode):
        if not node.stems:
            # Termo que virou stopword: não restringe o conjunto.
            return matrix.full_row()
        result = matrix.term_row(node.stems[0])
        for stem in node.stems[1:]:
            result &= matrix.term_row(stem)
        return result

    if isinstance(node, NotNode):
        return matrix.complement(_evaluate(node.child, matrix))

    if isinstance(node, BinaryNode):
        left = _evaluate(node.left, matrix)
        right = _evaluate(node.right, matrix)
        if node.operator == "AND":
            return left & right
        if node.operator == "OR":
            return left | right
        raise BooleanQueryError(f"Operador desconhecido: {node.operator}")

    raise BooleanQueryError(f"Nó inesperado: {node!r}")


def _or_node(stems: Sequence[str]) -> Node:
    """
    Monta uma disjunção a partir de radicais já pré-processados.

    Construir a árvore diretamente evita reescrever os radicais como texto e
    submetê-los ao parser, que voltaria a aplicar o stemming sobre termos já
    reduzidos (``mouse`` -> ``mous`` -> algo diferente de ``mous``).

    Args:
        stems: Radicais de conteúdo da consulta.

    Returns:
        Árvore ``t1 OR t2 OR ... OR tn``.

    Raises:
        BooleanQueryError: Se ``stems`` estiver vazio.
    """
    if not stems:
        raise BooleanQueryError("Nenhum termo de conteúdo na consulta.")
    node: Node = TermNode(stems=(stems[0],))
    for stem in stems[1:]:
        node = BinaryNode("OR", node, TermNode(stems=(stem,)))
    return node


def _has_operators(query: str) -> bool:
    """
    Indica se a consulta usa sintaxe booleana explícita.

    Args:
        query: Consulta original.

    Returns:
        ``True`` se houver ``AND``, ``OR``, ``NOT`` ou parênteses.
    """
    tokens = _TOKEN_RE.findall(query)
    return any(token in utils.BOOLEAN_OPERATORS or token in "()" for token in tokens)


def _coordination_scores(
    doc_indices: np.ndarray,
    stems: Sequence[str],
    inverted_index: InvertedIndex,
) -> np.ndarray:
    """
    Conta quantos termos distintos da consulta cada documento contém.

    Args:
        doc_indices: Índices dos documentos que satisfizeram a expressão.
        stems: Radicais de conteúdo da consulta.
        inverted_index: Índice invertido do corpus.

    Returns:
        Vetor de contagens alinhado com ``doc_indices``.
    """
    scores = np.zeros(doc_indices.size, dtype=np.float64)
    if doc_indices.size == 0:
        return scores

    position = {doc: i for i, doc in enumerate(doc_indices.tolist())}
    for stem in set(stems):
        docs, _ = inverted_index.postings_arrays(stem)
        for doc in docs.tolist():
            index = position.get(doc)
            if index is not None:
                scores[index] += 1.0
    return scores


def boolean_search(
    query: str,
    inverted_index: InvertedIndex,
    matrix: IncidenceMatrix,
    top_k: int = config.TOP_K,
    default_operator: str = "OR",
) -> List[int]:
    """
    Executa uma consulta booleana sobre a matriz de incidência.

    Args:
        query: Consulta booleana ou texto livre. Sem operadores explícitos,
            os termos são combinados com ``default_operator``.
        inverted_index: Índice invertido pré-computado.
        matrix: Matriz de incidência binária.
        top_k: Número máximo de ``product_id`` retornados.
        default_operator: ``"OR"`` (padrão, favorece recall) ou ``"AND"``
            (favorece precisão) para consultas sem sintaxe explícita.

    Returns:
        Lista de ``product_id`` que satisfazem a expressão, ordenada por
        número de termos da consulta presentes no documento (*coordination
        level match*) e, no empate, por ``product_id`` crescente.

    Raises:
        BooleanQueryError: Se a consulta for malformada.

    Exemplo:
        >>> from src.indexer import build_inverted_index, build_boolean_index
        >>> corpus = [{"product_id": 1, "name": "Mouse",
        ...            "description": "mouse óptico", "category": "Informática"}]
        >>> bi = build_boolean_index(corpus)
        >>> boolean_search("mouse AND óptico", bi.inverted_index, bi.matrix)
        [1]
    """
    return [pid for pid, _ in boolean_search_scored(
        query, inverted_index, matrix, top_k, default_operator
    )]


def boolean_search_scored(
    query: str,
    inverted_index: InvertedIndex,
    matrix: IncidenceMatrix,
    top_k: int = config.TOP_K,
    default_operator: str = "OR",
) -> List[Tuple[int, float]]:
    """
    Variante de :func:`boolean_search` que expõe o score de coordenação.

    Args:
        query: Consulta booleana ou texto livre.
        inverted_index: Índice invertido pré-computado.
        matrix: Matriz de incidência binária.
        top_k: Número máximo de resultados.
        default_operator: Operador implícito para consultas sem sintaxe.

    Returns:
        Lista de ``(product_id, score)``, em que o score é o número de termos
        distintos da consulta presentes no documento.
    """
    stems = utils.preprocess_query(query)
    if not _has_operators(query) and default_operator.upper() == "OR":
        if not stems:
            return []
        node = _or_node(stems)
    else:
        node = parse_boolean_query(query)

    bits = _evaluate(node, matrix)
    doc_indices = matrix.to_doc_indices(bits)
    if doc_indices.size == 0:
        logger.debug("Consulta booleana %r não retornou documentos.", query)
        return []

    scores = _coordination_scores(doc_indices, stems, inverted_index)
    product_ids = inverted_index.product_ids

    order = sorted(
        range(doc_indices.size),
        key=lambda i: (-scores[i], product_ids[doc_indices[i]]),
    )[:top_k]

    return [
        (int(product_ids[doc_indices[i]]), float(scores[i])) for i in order
    ]


def boolean_match_count(
    query: str,
    inverted_index: InvertedIndex,
    matrix: IncidenceMatrix,
) -> int:
    """
    Conta quantos documentos satisfazem a consulta, sem montar o ranking.

    Útil para relatar o tamanho do conjunto-resposta, que no modelo booleano
    costuma ser grande demais (``OR``) ou vazio (``AND`` restritivo).

    Args:
        query: Consulta booleana.
        inverted_index: Índice invertido (não usado no cálculo, mantido por
            simetria de assinatura).
        matrix: Matriz de incidência binária.

    Returns:
        Número de documentos no conjunto-resposta.
    """
    if _has_operators(query):
        node = parse_boolean_query(query)
    else:
        stems = utils.preprocess_query(query)
        if not stems:
            return 0
        node = _or_node(stems)
    bits = _evaluate(node, matrix)
    return int(matrix.to_doc_indices(bits).size)
