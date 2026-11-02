#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FUCAPE BUSINESS SCHOOL — CIÊNCIA DE DADOS PARA NEGÓCIOS
Disciplina: Processamento de Linguagem Natural e Sistemas de Recuperação de Informação
Aula 06: Processamento de Consultas Lógicas, Posicionais e Rankeadas (Top-K & WAND)
Arquivo: solucao_laboratorio_aula06.py
================================================================================
Descrição:
Implementação completa e modular das 4 Micro-Etapas do Laboratório Prático:
  - Micro-Etapa 1: Índice Posicional, Interseção Booleana vs. Interseção Posicional
                   (Phrase Queries e Proximity Search).
  - Micro-Etapa 2: Indexação probabilística Okapi BM25 com pré-cálculo de Limites
                   Superiores (Upper Bounds - UB_t) e Seleção Top-K via Min-Heap.
  - Micro-Etapa 3: Motor de Poda Dinâmica Segura WAND (Weak AND - Broder et al., 2003).
  - Micro-Etapa 4: Benchmark de Escala em Corpus Sintético Zipfiano (2.000 docs)
                   para K in {3, 5, 10, 20} e geração de gráfico 300 DPI.
================================================================================
"""

import os
import re
import math
import time
import heapq
import bisect
import random
import unicodedata
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# Micro-Etapas 1, 2 e 3: Motor Posicional + BM25 Upper Bounds + Algoritmo WAND
# ==============================================================================
class PositionalAndWANDIndex:
    """
    Motor de Busca Híbrido integrando:
    1. Postagens Posicionais para Phrase Queries (k=1) e Proximity Search (k>1);
    2. Pré-cálculo de scores Okapi BM25 e Upper Bounds (UB_t) por termo;
    3. Avaliação Top-K Exaustiva (Baseline DAAT) vs. Poda Dinâmica Segura WAND (Weak AND).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        # termo -> [{'doc_id': int, 'tf': int, 'positions': List[int], 'score': float}]
        self.postings: Dict[str, List[Dict]] = {}
        # Vetor auxiliar de doc_ids por termo para busca binária rápida (Galloping/Skip)
        self.posting_doc_ids: Dict[str, List[int]] = {}
        self.upper_bounds: Dict[str, float] = {}
        self.idf_cache: Dict[str, float] = {}
        self.doc_lengths: Dict[int, int] = {}
        self.doc_names: Dict[int, str] = {}
        self.doc_texts: Dict[int, str] = {}
        self.N: int = 0
        self.avg_dl: float = 0.0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Normaliza acentuação (NFKD ASCII), converte para minúsculas e extrai tokens."""
        norm = (
            unicodedata.normalize("NFKD", text.lower())
            .encode("ASCII", "ignore")
            .decode("utf-8")
        )
        return re.findall(r"\b\w+\b", norm)

    def add_document(self, doc_id: int, doc_name: str, text: str) -> None:
        """Indexa um documento armazenando frequência (TF) e lista ordenada de offsets."""
        tokens = self.tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_names[doc_id] = doc_name
        self.doc_texts[doc_id] = text
        self.N += 1

        pos_map: Dict[str, List[int]] = {}
        for pos, term in enumerate(tokens):
            pos_map.setdefault(term, []).append(pos)

        for term, positions in pos_map.items():
            self.postings.setdefault(term, []).append(
                {
                    "doc_id": doc_id,
                    "tf": len(positions),
                    "positions": positions,
                    "score": 0.0,
                }
            )

    def _compute_bm25_term_score(self, term: str, doc_id: int, tf: int) -> float:
        """Calcula a contribuição exata Okapi BM25 de um termo t no documento doc_id."""
        idf = self.idf_cache[term]
        dl = self.doc_lengths[doc_id]
        tf_norm = (tf * (self.k1 + 1.0)) / (
            tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
        )
        return idf * tf_norm

    def finalize_and_compute_bounds(self) -> None:
        """
        Ordena listas de postagens por DocID e pré-calcula:
          - IDF e Score BM25(t, d) de cada postagem;
          - Limite Superior UB_t = max_{d in L_t} BM25(t, d) para cada termo t.
        """
        self.avg_dl = (
            sum(self.doc_lengths.values()) / self.N if self.N > 0 else 1.0
        )

        for term, plist in self.postings.items():
            plist.sort(key=lambda x: x["doc_id"])
            self.posting_doc_ids[term] = [entry["doc_id"] for entry in plist]
            df_t = len(plist)
            idf = math.log(1.0 + (self.N - df_t + 0.5) / (df_t + 0.5))
            self.idf_cache[term] = idf

            max_score = 0.0
            for entry in plist:
                score = self._compute_bm25_term_score(
                    term, entry["doc_id"], entry["tf"]
                )
                entry["score"] = score
                if score > max_score:
                    max_score = score

            self.upper_bounds[term] = max_score

    # --------------------------------------------------------------------------
    # Micro-Etapa 1: Interseção Booleana Simples vs. Interseção Posicional
    # --------------------------------------------------------------------------
    def boolean_intersect_and(self, term1: str, term2: str) -> List[int]:
        """
        Realiza a interseção booleana clássica (t1 AND t2) olhando apenas para DocID,
        ignorando a distância e a ordem das palavras no documento.
        """
        t1_norm = self.tokenize(term1)[0] if self.tokenize(term1) else term1
        t2_norm = self.tokenize(term2)[0] if self.tokenize(term2) else term2

        if t1_norm not in self.postings or t2_norm not in self.postings:
            return []

        p1 = self.postings[t1_norm]
        p2 = self.postings[t2_norm]
        i, j = 0, 0
        result: List[int] = []

        while i < len(p1) and j < len(p2):
            d1, d2 = p1[i]["doc_id"], p2[j]["doc_id"]
            if d1 == d2:
                result.append(d1)
                i += 1
                j += 1
            elif d1 < d2:
                i += 1
            else:
                j += 1
        return result

    def positional_intersect(
        self, term1: str, term2: str, k: int = 1, directional: bool = True
    ) -> List[Tuple[int, int, int]]:
        """
        Realiza a interseção posicional entre dois termos em tempo linear O(|L1| + |L2|).
        Retorna lista de tuplas: (doc_id, pos_term1, pos_term2).
          - Se directional=True e k=1: exige frase exata (pos_term2 - pos_term1 == 1).
          - Se directional=False e k>1: exige proximidade simétrica (|pos_term2 - pos_term1| <= k).
        """
        t1_norm = self.tokenize(term1)[0] if self.tokenize(term1) else term1
        t2_norm = self.tokenize(term2)[0] if self.tokenize(term2) else term2

        if t1_norm not in self.postings or t2_norm not in self.postings:
            return []

        p1 = self.postings[t1_norm]
        p2 = self.postings[t2_norm]
        i, j = 0, 0
        matches: List[Tuple[int, int, int]] = []

        while i < len(p1) and j < len(p2):
            d1, d2 = p1[i]["doc_id"], p2[j]["doc_id"]
            if d1 == d2:
                pos_list1 = p1[i]["positions"]
                pos_list2 = p2[j]["positions"]
                pi, pj = 0, 0

                while pi < len(pos_list1) and pj < len(pos_list2):
                    pos1, pos2 = pos_list1[pi], pos_list2[pj]
                    diff = pos2 - pos1
                    if directional:
                        if 1 <= diff <= k:
                            matches.append((d1, pos1, pos2))
                            pi += 1
                            pj += 1
                        elif pos1 < pos2:
                            pi += 1
                        else:
                            pj += 1
                    else:
                        if abs(diff) <= k:
                            matches.append((d1, pos1, pos2))
                            pi += 1
                            pj += 1
                        elif pos1 < pos2:
                            pi += 1
                        else:
                            pj += 1
                i += 1
                j += 1
            elif d1 < d2:
                i += 1
            else:
                j += 1

        return matches

    def phrase_query(self, phrase: str) -> List[Tuple[int, List[int]]]:
        """
        Resolve consultas por frase exata de múltiplos termos (ex: 'mercado de capitais')
        encadeando interseções posicionais onde pos(t_{i+1}) == pos(t_i) + 1.
        Retorna [(doc_id, [pos_t1, pos_t2, ...]), ...].
        """
        terms = self.tokenize(phrase)
        if not terms:
            return []
        for t in terms:
            if t not in self.postings:
                return []

        if len(terms) == 1:
            return [
                (entry["doc_id"], [p])
                for entry in self.postings[terms[0]]
                for p in entry["positions"]
            ]

        pair_matches = self.positional_intersect(
            terms[0], terms[1], k=1, directional=True
        )
        current_chains: List[Tuple[int, List[int]]] = [
            (doc_id, [p1, p2]) for doc_id, p1, p2 in pair_matches
        ]

        for next_term in terms[2:]:
            next_postings = {
                entry["doc_id"]: set(entry["positions"])
                for entry in self.postings[next_term]
            }
            updated_chains: List[Tuple[int, List[int]]] = []
            for doc_id, pos_chain in current_chains:
                expected_pos = pos_chain[-1] + 1
                if (
                    doc_id in next_postings
                    and expected_pos in next_postings[doc_id]
                ):
                    updated_chains.append((doc_id, pos_chain + [expected_pos]))
            current_chains = updated_chains

        return current_chains

    # --------------------------------------------------------------------------
    # Micro-Etapa 2: Avaliação Top-K Exaustiva DAAT com Min-Heap (Baseline)
    # --------------------------------------------------------------------------
    def search_exhaustive_topk(
        self, query: str, k: int = 3
    ) -> Tuple[List[Tuple[int, float]], int]:
        """
        Avalia exaustivamente em modo DAAT (Document-at-a-Time) TODOS os documentos
        candidatos que contêm pelo menos um termo da consulta, calculando o score BM25
        completo de cada candidato e mantendo os Top-K em um Min-Heap O(N log K).
        Retorna (ranking_top_k, total_documentos_avaliados).
        """
        q_terms = sorted(
            list({t for t in self.tokenize(query) if t in self.postings})
        )
        if not q_terms:
            return [], 0

        cursors: Dict[str, int] = {t: 0 for t in q_terms}
        min_heap: List[Tuple[float, int]] = []
        full_evaluations: int = 0

        while True:
            # Identifica o menor DocID atual entre todas as listas ativas (DAAT)
            min_doc = None
            for t in q_terms:
                idx = cursors[t]
                plist = self.postings[t]
                if idx < len(plist):
                    d = plist[idx]["doc_id"]
                    if min_doc is None or d < min_doc:
                        min_doc = d

            if min_doc is None:
                break

            # Calcula o score BM25 completo para o documento candidato min_doc
            full_evaluations += 1
            doc_score = 0.0
            for t in q_terms:
                idx = cursors[t]
                plist = self.postings[t]
                if idx < len(plist) and plist[idx]["doc_id"] == min_doc:
                    doc_score += self._compute_bm25_term_score(
                        t, min_doc, plist[idx]["tf"]
                    )
                    cursors[t] = idx + 1

            item = (round(doc_score, 10), -min_doc)
            if len(min_heap) < k:
                heapq.heappush(min_heap, item)
            elif item > min_heap[0]:
                heapq.heappushpop(min_heap, item)

        ranked = sorted(
            [(-neg_doc_id, round(score, 4)) for score, neg_doc_id in min_heap],
            key=lambda x: (-x[1], x[0]),
        )
        return ranked, full_evaluations

    # --------------------------------------------------------------------------
    # Micro-Etapa 3: Algoritmo WAND (Weak AND) para Poda Dinâmica Segura Top-K
    # --------------------------------------------------------------------------
    def search_wand_topk(
        self, query: str, k: int = 3
    ) -> Tuple[List[Tuple[int, float]], int, int]:
        """
        Executa a busca Top-K usando o algoritmo WAND (Weak AND - Broder et al., 2003).
        Garante resultado 100% idêntico ao algoritmo exaustivo (Rank-Safe Pruning),
        saltando via busca binária (Skip/Bisect) documentos cuja soma de Upper Bounds
        não supera o limiar dinâmico theta do Min-Heap.
        Retorna (ranking_top_k, full_evaluations, skipped_pivots).
        """
        q_terms = sorted(
            list({t for t in self.tokenize(query) if t in self.postings})
        )
        if not q_terms:
            return [], 0, 0

        cursors: Dict[str, int] = {t: 0 for t in q_terms}
        doc_id_lists: Dict[str, List[int]] = {
            t: self.posting_doc_ids[t] for t in q_terms
        }
        ub_map: Dict[str, float] = {t: self.upper_bounds[t] for t in q_terms}

        min_heap: List[Tuple[float, int]] = []
        threshold: float = 0.0
        full_evaluations: int = 0
        skipped_pivots: int = 0

        while True:
            # Passo 1: Filtrar termos ativos e ordená-los pelo DocID atual
            active_terms = [
                t for t in q_terms if cursors[t] < len(doc_id_lists[t])
            ]
            if not active_terms:
                break
            active_terms.sort(key=lambda t: (doc_id_lists[t][cursors[t]], t))

            # Passo 2: Encontrar o Termo Pivô onde a soma acumulada de UB_t supera o threshold
            cum_ub = 0.0
            pivot_idx: Optional[int] = None
            heap_full = len(min_heap) >= k
            for i, t in enumerate(active_terms):
                cum_ub += ub_map[t]
                if not heap_full or cum_ub > threshold + 1e-9:
                    pivot_idx = i
                    break

            # Se nem somando todos os Upper Bounds ativos superamos o threshold, encerramos!
            if pivot_idx is None:
                skipped_pivots += 1
                break

            pivot_term = active_terms[pivot_idx]
            pivot_doc = doc_id_lists[pivot_term][cursors[pivot_term]]
            first_doc = doc_id_lists[active_terms[0]][cursors[active_terms[0]]]

            # Passo 3: Decisão de Avaliação Completa ou Salto de Poda (Skip)
            if first_doc == pivot_doc:
                # Avalia integralmente o documento pivô (BM25 exato)
                full_evaluations += 1
                doc_score = 0.0
                for t in active_terms:
                    idx = cursors[t]
                    if doc_id_lists[t][idx] == pivot_doc:
                        tf = self.postings[t][idx]["tf"]
                        doc_score += self._compute_bm25_term_score(
                            t, pivot_doc, tf
                        )
                        cursors[t] = idx + 1

                item = (round(doc_score, 10), -pivot_doc)
                if len(min_heap) < k:
                    heapq.heappush(min_heap, item)
                    if len(min_heap) == k:
                        threshold = min_heap[0][0]
                elif item > min_heap[0]:
                    heapq.heappushpop(min_heap, item)
                    threshold = min_heap[0][0]
            else:
                # Poda Dinâmica: saltar cursores anteriores diretamente para >= pivot_doc via bisect!
                skipped_pivots += 1
                for i in range(pivot_idx):
                    t_skip = active_terms[i]
                    cursors[t_skip] = bisect.bisect_left(
                        doc_id_lists[t_skip], pivot_doc, lo=cursors[t_skip]
                    )

        ranked = sorted(
            [(-neg_doc_id, round(score, 4)) for score, neg_doc_id in min_heap],
            key=lambda x: (-x[1], x[0]),
        )
        return ranked, full_evaluations, skipped_pivots


# ==============================================================================
# Geração do Gráfico de Benchmark (300 DPI — Paleta Institucional FUCAPE)
# ==============================================================================
def generate_benchmark_chart(
    k_values: List[int],
    evals_exhaustive: List[int],
    evals_wand: List[int],
    pruning_rates: List[float],
    output_path: str,
) -> None:
    """
    Gera um gráfico de alta resolução (300 DPI) com dois painéis lado a lado:
      - Painel A: Documentos Avaliados (DAAT Exaustivo vs. WAND Top-K)
      - Painel B: Taxa de Poda Dinâmica (Pruning Rate %) para K in {3, 5, 10, 20}
    Paleta FUCAPE: Navy (#0E2841), Red (#C00000), Orange (#E97132).
    """
    FUCAPE_NAVY = "#0E2841"
    FUCAPE_RED = "#C00000"
    FUCAPE_ORANGE = "#E97132"
    BG_LIGHT = "#F8FAFC"

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.linewidth"] = 0.9

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)
    fig.patch.set_facecolor("white")

    x = np.arange(len(k_values))
    width = 0.36

    # --------------------------------------------------------------------------
    # Painel A: Documentos Avaliados (Exaustivo vs. WAND)
    # --------------------------------------------------------------------------
    ax1.set_facecolor(BG_LIGHT)
    bars_ex = ax1.bar(
        x - width / 2,
        evals_exhaustive,
        width,
        label="Avaliação Exaustiva DAAT (Baseline)",
        color=FUCAPE_NAVY,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    bars_wand = ax1.bar(
        x + width / 2,
        evals_wand,
        width,
        label="Poda Dinâmica WAND (Weak AND)",
        color=FUCAPE_ORANGE,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    ax1.set_title(
        "Painel A: Documentos Pontuados por Consulta (2.000 Docs)",
        fontsize=12,
        fontweight="bold",
        color=FUCAPE_NAVY,
        pad=14,
    )
    ax1.set_xlabel("Tamanho do Ranking Solicitado (Top-K)", fontsize=10.5, fontweight="bold", color=FUCAPE_NAVY)
    ax1.set_ylabel("Nº de Avaliações Completas de BM25", fontsize=10.5, fontweight="bold", color=FUCAPE_NAVY)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Top-{k}" for k in k_values], fontsize=10, fontweight="bold")
    ax1.set_ylim(0, max(evals_exhaustive) * 1.32)
    ax1.grid(axis="y", linestyle="--", alpha=0.5, color="#94A3B8", zorder=0)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9.5, loc="upper right")

    for bar in bars_ex:
        h = bar.get_height()
        ax1.annotate(
            f"{int(h)}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=FUCAPE_NAVY,
        )

    for bar in bars_wand:
        h = bar.get_height()
        ax1.annotate(
            f"{int(h)}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=FUCAPE_RED,
        )

    # --------------------------------------------------------------------------
    # Painel B: Taxa de Poda (% de Cálculos de Score Economizados)
    # --------------------------------------------------------------------------
    ax2.set_facecolor(BG_LIGHT)
    bars_prune = ax2.bar(
        x,
        pruning_rates,
        width=0.48,
        color=FUCAPE_RED,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label="Taxa de Poda WAND (% Economia de CPU)",
    )
    ax2.plot(
        x,
        pruning_rates,
        color=FUCAPE_NAVY,
        marker="o",
        markersize=7,
        linewidth=2.0,
        linestyle="-",
        zorder=4,
        label="Curva de Sensibilidade ao Limiar θ",
    )

    ax2.set_title(
        "Painel B: Taxa de Poda Dinâmica (% de Cálculos Evitados)",
        fontsize=12,
        fontweight="bold",
        color=FUCAPE_NAVY,
        pad=14,
    )
    ax2.set_xlabel("Tamanho do Ranking Solicitado (Top-K)", fontsize=10.5, fontweight="bold", color=FUCAPE_NAVY)
    ax2.set_ylabel("Taxa de Poda / Economia de CPU (%)", fontsize=10.5, fontweight="bold", color=FUCAPE_NAVY)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Top-{k}" for k in k_values], fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 112)
    ax2.grid(axis="y", linestyle="--", alpha=0.5, color="#94A3B8", zorder=0)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9.5, loc="lower left")

    for bar, rate in zip(bars_prune, pruning_rates):
        h = bar.get_height()
        ax2.annotate(
            f"{rate:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=FUCAPE_NAVY,
        )

    fig.suptitle(
        "FUCAPE Business School — Benchmark Algorítmico Aula 06: Poda Dinâmica WAND (Weak AND) vs. Busca Exaustiva",
        fontsize=13,
        fontweight="bold",
        color=FUCAPE_NAVY,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


# ==============================================================================
# Execução Principal: Micro-Etapas 1 a 4
# ==============================================================================
def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dir_corpora = os.path.join(base_dir, "corpora")

    engine = PositionalAndWANDIndex(k1=1.5, b=0.75)

    for i in range(1, 11):
        f_name = f"relatorio_{i:02d}.txt"
        path = os.path.join(dir_corpora, f_name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                engine.add_document(i, f_name, f.read().strip())

    engine.finalize_and_compute_bounds()

    print("=" * 92)
    print("🎓 FUCAPE BUSINESS SCHOOL — LABORATÓRIO AULA 06: CONSULTAS POSICIONAIS E WAND")
    print("=" * 92)
    print(f"📂 Corpus Corporativo Carregado: {engine.N} relatórios | Comprimento Médio (avg_dl): {engine.avg_dl:.1f} tokens")
    print(f"📖 Tamanho do Vocabulário Indexado: {len(engine.postings)} termos únicos")

    # --------------------------------------------------------------------------
    # MICRO-ETAPA 1: Interseção Booleana vs. Interseção Posicional
    # --------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("🔎 MICRO-ETAPA 1: INTERSEÇÃO BOOLEANA vs. INTERSEÇÃO POSICIONAL (PHRASE & PROXIMITY)")
    print("=" * 92)

    bool_div_prov = engine.boolean_intersect_and("dividendos", "proventos")
    prox_div_prov = engine.positional_intersect("dividendos", "proventos", k=5, directional=False)
    print("\n[1.1] Comparativo 'dividendos' E 'proventos' (Eliminando Falsos Positivos):")
    print(f"  • Interseção Booleana Simples (dividendos AND proventos) -> DocIDs: {bool_div_prov}")
    print(f"    (Nota: DocID 4 tem 'dividendos' na pos 10 e 'proventos' na pos 21 -> distância 11, frases distintas!)")
    print(f"  • Interseção Posicional ('dividendos ~5 proventos', k=5) -> Matches (DocID, pos1, pos2): {prox_div_prov}")

    bool_pag_jcp = engine.boolean_intersect_and("pagamento", "jcp")
    prox_pag_jcp = engine.positional_intersect("pagamento", "jcp", k=2, directional=True)
    print("\n[1.2] Comparativo 'pagamento' E 'jcp':")
    print(f"  • Interseção Booleana Simples (pagamento AND jcp)        -> DocIDs: {bool_pag_jcp}")
    print(f"  • Interseção Posicional ('pagamento [de] jcp', k=2, dir) -> Matches (DocID, pos1, pos2): {prox_pag_jcp}")

    print("\n[1.3] Consultas por Frase Exata (Phrase Queries):")
    f_lucro = engine.positional_intersect("lucro", "liquido", k=1, directional=True)
    f_receita = engine.positional_intersect("receita", "operacional", k=1, directional=True)
    f_mercado_2 = engine.positional_intersect("mercado", "capitais", k=2, directional=True)
    f_mercado_3 = engine.phrase_query("mercado de capitais")
    print(f"  • Frase Exata 'lucro liquido' (k=1, dir=True)            -> {f_lucro}")
    print(f"  • Frase Exata 'receita operacional' (k=1, dir=True)      -> {f_receita}")
    print(f"  • Frase Exata 'mercado de capitais' (k=2 / 3 termos)     -> {f_mercado_3} (pares: {f_mercado_2})")

    print("\n[1.4] Consultas por Proximidade (Proximity Search |pos1 - pos2| <= k):")
    p_risco = engine.positional_intersect("risco", "inadimplencia", k=5, directional=False)
    print(f"  • Proximidade 'risco ~5 inadimplencia' (k=5, dir=False)  -> {p_risco}")

    # --------------------------------------------------------------------------
    # MICRO-ETAPA 2: Upper Bounds (UB_t) do BM25
    # --------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("📊 MICRO-ETAPA 2: PRÉ-CÁLCULO DE LIMITES SUPERIORES (UPPER BOUNDS UB_t DO BM25)")
    print("=" * 92)
    sample_terms = ["lucro", "liquido", "dividendos", "proventos", "debentures", "inadimplencia", "ebitda"]
    print(f"{'Termo':<18} | {'DF_t':<6} | {'Upper Bound (UB_t)':<20}")
    print("-" * 50)
    for t in sample_terms:
        if t in engine.postings:
            print(f"{t:<18} | {len(engine.postings[t]):<6} | {engine.upper_bounds[t]:.4f}")

    # --------------------------------------------------------------------------
    # MICRO-ETAPA 3: Exatidão do WAND no Corpus Corporativo Financeiro
    # --------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("⚡ MICRO-ETAPA 3: EXATIDÃO E PODA DO ALGORITMO WAND NO CORPUS FINANCEIRO (10 DOCS)")
    print("=" * 92)
    query_fin = "lucro liquido dividendos proventos debentures"
    res_ex, evals_ex = engine.search_exhaustive_topk(query_fin, k=3)
    res_wand, evals_wand, skips_wand = engine.search_wand_topk(query_fin, k=3)
    pruning_fin = (1.0 - (evals_wand / evals_ex)) * 100.0 if evals_ex > 0 else 0.0

    print(f"Consulta Financeira: '{query_fin}'")
    print(f"  • Top-3 Exaustivo: {res_ex} | Docs Avaliados: {evals_ex}")
    print(f"  • Top-3 WAND:      {res_wand} | Docs Avaliados: {evals_wand} | Saltos de Pivô: {skips_wand}")
    print(f"  • Equivalência Matemática Exata (Rank-Safe)? {res_ex == res_wand} | Taxa de Poda: {pruning_fin:.2f}%")

    # --------------------------------------------------------------------------
    # MICRO-ETAPA 4: Benchmark em Escala (Corpus Sintético Zipfiano — 2.000 Docs)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("🚀 MICRO-ETAPA 4: BENCHMARK EM ESCALA (2.000 DOCS ZIPFIANOS — K in {3, 5, 10, 20})")
    print("=" * 92)

    scale_engine = PositionalAndWANDIndex(k1=1.5, b=0.75)
    random.seed(42)
    vocab_comum = ["mercado", "empresa", "relatorio", "financeiro", "trimestre", "dados"]
    vocab_medio = ["receita", "crescimento", "operacional", "credito", "carteira"]
    vocab_raro = ["cisplatina", "superavit", "debentures", "proventos", "inadimplencia"]

    for doc_id in range(1, 2001):
        words = random.choices(vocab_comum, k=25)
        if random.random() < 0.25:
            words += random.choices(vocab_medio, k=5)
        if random.random() < 0.04:
            words += random.choices(vocab_raro, k=2)
        scale_engine.add_document(doc_id, f"doc_{doc_id:04d}.txt", " ".join(words))

    scale_engine.finalize_and_compute_bounds()

    q_scale = "mercado financeiro crescimento debentures superavit"
    k_values = [3, 5, 10, 20]
    evals_ex_list: List[int] = []
    evals_wand_list: List[int] = []
    pruning_rates: List[float] = []

    print(f"Consulta de Benchmark: '{q_scale}'")
    print("-" * 92)
    print(
        f"{'Top-K':<7} | {'Docs Exaustivo':<14} | {'Docs WAND':<10} | "
        f"{'Saltos':<7} | {'Taxa Poda (%)':<13} | {'Tempo Ex (ms)':<13} | {'Tempo WAND (ms)':<15} | {'Rank-Safe?'}"
    )
    print("-" * 92)

    for k in k_values:
        t0 = time.perf_counter()
        for _ in range(50):
            r_ex, n_ex = scale_engine.search_exhaustive_topk(q_scale, k=k)
        t_ex_ms = ((time.perf_counter() - t0) / 50.0) * 1000.0

        t1 = time.perf_counter()
        for _ in range(50):
            r_wand, n_wand, n_skips = scale_engine.search_wand_topk(q_scale, k=k)
        t_wand_ms = ((time.perf_counter() - t1) / 50.0) * 1000.0

        pruning_pct = (1.0 - (n_wand / n_ex)) * 100.0 if n_ex > 0 else 0.0
        is_identical = r_ex == r_wand

        evals_ex_list.append(n_ex)
        evals_wand_list.append(n_wand)
        pruning_rates.append(pruning_pct)

        print(
            f"K = {k:<3} | {n_ex:<14} | {n_wand:<10} | "
            f"{n_skips:<7} | {pruning_pct:>11.2f}% | {t_ex_ms:>11.3f} ms | {t_wand_ms:>13.3f} ms | {str(is_identical)}"
        )

    print("-" * 92)

    chart_path = os.path.join(base_dir, "benchmark_wand_pruning.png")
    generate_benchmark_chart(
        k_values=k_values,
        evals_exhaustive=evals_ex_list,
        evals_wand=evals_wand_list,
        pruning_rates=pruning_rates,
        output_path=chart_path,
    )
    print(f"📈 Gráfico de alta resolução (300 DPI) salvo em: {chart_path}")
    print("=" * 92)


if __name__ == "__main__":
    main()
