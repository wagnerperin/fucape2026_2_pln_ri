# 🎓 Repositório Prático — Aula 06: Processamento de Consultas Posicionais, Top-K e Algoritmo WAND (*Weak AND*)

**Instituição:** FUCAPE Business School  
**Disciplina:** Processamento de Linguagem Natural e Sistemas de Recuperação de Informação  
**Professor:** Prof. Dr. Wagner Perin  
**Unidade:** Unidade 3 — Processamento e Resolução de Consultas  

---

## 🎯 Visão Geral do Laboratório

Este repositório contém a implementação completa, autossuficiente e validada do laboratório prático da **Aula 06**, abordando dois gargalos críticos de motores de busca corporativos em produção:

1. **Precisão em Consultas por Frase Exata (*Phrase Queries*) e Proximidade (*Proximity Search*):** Eliminação de falsos positivos gerados pela interseção booleana simples (`AND`) por meio de **Índices Invertidos Posicionais** ($\langle \text{DocID}, tf_{t,d}, [\text{pos}_1, \dots, \text{pos}_{tf}] \rangle$) e interseção linear com ponteiros aninhados.
2. **Escalabilidade e Latência de Cauda ($p99$) na Seleção Top-$K$:** Substituição da avaliação exaustiva de documentos candidatos pela **Poda Dinâmica Segura (*Rank-Safe Dynamic Pruning*)** com o algoritmo **WAND (*Weak AND* — Broder et al., 2003)** apoiado em limites superiores de score Okapi BM25 ($\text{UB}_t$) e filas de prioridade (*Min-Heap*).

---

## 📂 Estrutura de Arquivos do Repositório

```text
repositorio_aula06/
├── README.md                          # Documentação executiva e guia de execução
├── solucao_laboratorio_aula06.py      # Script Python modular das 4 Micro-Etapas + Gerador de Gráficos
├── laboratorio_aula_06_didatico.ipynb # Jupyter Notebook didático pré-executado (com outputs e gráficos)
├── benchmark_wand_pruning.png         # Gráfico de benchmark em alta resolução (300 DPI — Paleta FUCAPE)
└── corpora/                           # Acervo autossuficiente com 10 relatórios financeiros corporativos
    ├── relatorio_01.txt
    ├── relatorio_02.txt
    ├── relatorio_03.txt
    ├── relatorio_04.txt
    ├── relatorio_05.txt
    ├── relatorio_06.txt
    ├── relatorio_07.txt
    ├── relatorio_08.txt
    ├── relatorio_09.txt
    └── relatorio_10.txt
```

---

## 🔬 Síntese das 4 Micro-Etapas Implementadas

### 1. Micro-Etapa 1: Interseção Booleana vs. Interseção Posicional
* **Método `boolean_intersect_and(t1, t2)`:** Interseção clássica apenas por `DocID`.
* **Método `positional_intersect(t1, t2, k, directional)` e `phrase_query(phrase)`:**
  * **Frases Exatas ($k=1, \text{directional}=\text{True}$):** Localiza com precisão cirúrgica `"lucro liquido"` (`DocIDs 1, 2, 4`), `"receita operacional"` (`DocIDs 5, 6, 7`) e `"mercado de capitais"` (`DocIDs 9, 10`).
  * **Proximidade ($k=5, \text{directional}=\text{False}$):** Em `"dividendos ~5 proventos"`, retorna apenas `[(1, 17, 21)]`, eliminando o falso positivo `DocID 4` (onde os termos estão a $11$ tokens de distância em frases distintas). Em `"risco ~5 inadimplencia"`, localiza `[(8, 4, 9)]`.

### 2. Micro-Etapa 2: Pré-Cálculo de *Upper Bounds* ($\text{UB}_t$) e Seleção Top-$K$ via *Min-Heap*
* Pré-calcula e armazena no dicionário o limite superior $\text{UB}_t = \max_{d \in L_t} \text{Score}_{\text{BM25}}(t, d)$ para cada termo $t$.
* Mantém um *Min-Heap* (`heapq`) de capacidade fixa $K$, reduzindo a complexidade de seleção Top-$K$ de $O(N \log_2 N)$ para $O(N \log_2 K)$ e monitorando dinamicamente o limiar de corte $\theta$.

### 3. Micro-Etapa 3: Motor de Poda Dinâmica Segura WAND (*Weak AND*)
* Ordena cursores ativos por `DocID`, identifica o **Termo Pivô ($t_p$)** onde $\sum_{i=1}^{p} \text{UB}_{t_i} > \theta$ e salta via busca binária (`bisect_left`) todos os documentos anteriores ao pivô que não possuem massa de score suficiente para ingressar no Top-$K$.
* **Garantia *Rank-Safe*:** Produz $100\%$ de igualdade matemática nas SERPs em relação à busca exaustiva.

### 4. Micro-Etapa 4:Resultados do Benchmark em Escala (2.000 Documentos Zipfianos)

Consulta avaliada: `"mercado financeiro crescimento debentures superavit"`

| Top-$K$ | Docs Avaliados (Exaustivo) | Docs Avaliados (WAND) | Saltos de Pivô | Taxa de Poda (% Economia CPU) | Redução de Avaliações | Rank-Safe (SERP Idêntica)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 3$** | $2.000$ | **$40$** | $112$ | **$98,00\%$** | **$50,0\times$** | `True` |
| **$K = 5$** | $2.000$ | **$52$** | $104$ | **$97,40\%$** | **$38,5\times$** | `True` |
| **$K = 10$** | $2.000$ | **$122$** | $86$ | **$93,90\%$** | **$16,4\times$** | `True` |
| **$K = 20$** | $2.000$ | **$211$** | $111$ | **$89,45\%$** | **$9,5\times$** | `True` |

---

## 🚀 Como Executar

### 1. Executar o Script de Solução e Benchmark via Terminal
```bash
python3 solucao_laboratorio_aula06.py
```
O script executará os testes das 4 Micro-Etapas no terminal e regenerará automaticamente o gráfico `benchmark_wand_pruning.png` em 300 DPI.

### 2. Explorar o Jupyter Notebook Didático
Abra o arquivo `laboratorio_aula_06_didatico.ipynb` diretamente no VS Code, JupyterLab ou Google Colab. Todas as células já estão **pré-executadas** com tabelas HTML, métricas de telemetria e gráficos renderizados.
