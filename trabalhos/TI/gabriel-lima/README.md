# Benchmark de Modelos Clássicos de Recuperação da Informação

Comparação de **cinco algoritmos clássicos de RI** sobre um catálogo sintético
de e-commerce com 5.000 produtos, medindo **latência** (escalabilidade até
10.000 documentos) e **relevância** (P@5 e nDCG@5 contra julgamentos
conhecidos por construção).

| # | Modelo | Estrutura | Arquivo |
|---|--------|-----------|---------|
| A | Busca linear ingênua | nenhuma (baseline O(N)) | [`linear_search.py`](src/retrieval/linear_search.py) |
| B | Modelo booleano | matriz de incidência binária (bits empacotados) | [`boolean_search.py`](src/retrieval/boolean_search.py) |
| C | Espaço vetorial | TF-IDF + cosseno | [`vector_search.py`](src/retrieval/vector_search.py) |
| D | Okapi BM25 | índice invertido | [`bm25_search.py`](src/retrieval/bm25_search.py) |
| E | LSA | TF-IDF + TruncatedSVD (k=100) | [`lsa_search.py`](src/retrieval/lsa_search.py) |

**O relatório completo, com tabelas e discussão, está em
[`results/REPORT.md`](results/REPORT.md).**

---

## Setup

Requer **Python 3.10+** (testado em 3.14, Windows 11).

```bash
# 1. Ambiente virtual (opcional, recomendado)
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

# 2. Dependências
pip install -r requirements.txt

# 3. Recursos de português do NLTK (stopwords + stemmer RSLP)
python -c "import nltk; nltk.download('stopwords'); nltk.download('rslp')"
```

> O projeto funciona sem os recursos do NLTK: `src/utils.py` cai para uma
> lista de stopwords embutida e desliga o stemming, com perda de recall. Um
> aviso é registrado no log quando isso acontece.

## Execução

Os três comandos abaixo reproduzem todos os resultados do zero:

```bash
python src/corpus_generator.py          # 1. gera data/products.json + qrels
python benchmark/benchmark_scalability.py   # 2. latência e gráfico
python benchmark/benchmark_relevance.py     # 3. SERPs, P@5 / nDCG@5, tuning
```

Tempo total aproximado: **4 a 6 minutos** em uma máquina comum (a maior parte
indexando N = 10.000 cinco vezes).

Opções úteis:

```bash
python src/corpus_generator.py --size 1000            # corpus menor
python benchmark/benchmark_scalability.py --sizes 100 500 1000
python benchmark/benchmark_scalability.py --no-plot
python benchmark/benchmark_relevance.py --skip-tuning
```

### Demonstração interativa (live demo)

```bash
python demo.py
```

Abre um prompt onde cada consulta digitada é executada **nos cinco algoritmos
ao mesmo tempo**, exibindo lado a lado o TOP-5 de cada um com score, preço e a
latência real daquela busca. Quando a consulta é uma das cinco avaliadas no
estudo, os resultados saem marcados com ✓ / ~ / ✗ segundo os julgamentos de
`data/qrels.json` — fica visível na hora qual modelo acertou.

```bash
python demo.py "mouse óptico sem fio"   # consulta única, sem entrar no prompt
python demo.py --size 1000              # catálogo menor, sobe mais rápido
python demo.py --top-k 3                # profundidade do ranking
```

Comandos do prompt: `:ajuda`, `:exemplos` (as 5 consultas do estudo),
`:topk N`, `:produto N` (ficha completa), `:sair`.

A consulta aceita sintaxe booleana — `teclado AND (mecânico OR rgb) AND NOT
gamer`. Os operadores valem para o modelo booleano; os demais os ignoram, o
que é justamente o contraste que a Q4 expõe.

Os cinco modelos são aquecidos na inicialização: sem isso, a primeira busca
pagaria os custos de primeira chamada e exibiria uma latência várias vezes
maior que a real.

### Testes

```bash
python -m pytest tests/ -v
```

101 testes cobrem o pré-processamento, as estruturas de indexação, os cinco
algoritmos e as invariantes do gerador de corpus.

### Análise exploratória

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

Distribuições do catálogo, lei de Zipf, verificação da polissemia plantada e
curva de variância explicada do SVD.

---

## Estrutura

```
.
├── demo.py                         # demonstração interativa (live demo)
├── config.py                       # toda a configuração e hiperparâmetros
├── data/
│   ├── products.json               # corpus (12 MB, gerado; fora do git)
│   └── qrels.json                  # julgamentos de relevância graduados
├── src/
│   ├── corpus_generator.py         # gera o catálogo sintético + qrels
│   ├── indexer.py                  # índice invertido e matriz de incidência
│   ├── engine.py                   # fachada única sobre os cinco modelos
│   ├── utils.py                    # tokenização, stopwords, stemming RSLP
│   └── retrieval/                  # os cinco algoritmos
├── benchmark/
│   ├── benchmark_scalability.py    # latência x N, gráfico log-log
│   └── benchmark_relevance.py      # SERPs, P@5, nDCG@5, tuning do BM25
├── notebooks/exploratory_analysis.ipynb
├── tests/                          # pytest
└── results/                        # saídas (CSV, PNG, REPORT.md)
```

`src/engine.py` não constava do esqueleto original do enunciado: foi
acrescentado porque os dois benchmarks precisam montar exatamente os mesmos
índices, e concentrar isso em um lugar evita que cada script meça uma coisa
diferente.

## Saídas geradas

| Arquivo | Conteúdo |
|---------|----------|
| `results/benchmark_scalability.png` | gráfico log-log: latência e indexação x N |
| `results/benchmark_scalability.csv` | latência média (ms) por N e algoritmo |
| `results/benchmark_indexing.csv` | tempo de indexação (ms) por N e algoritmo |
| `results/serp_analysis.csv` | TOP-5 de cada consulta em cada algoritmo |
| `results/relevance_metrics.csv` | P@5, nDCG@5 e taxa de irrelevantes |
| `results/lexical_coverage.csv` | quanto de cada SERP é casamento literal |
| `results/bm25_tuning.csv` | varredura de `k1` x `b` |
| `results/REPORT.md` | **relatório final** |

---

## Decisões de projeto

**Um único pré-processamento para todos.** Índice invertido, TF-IDF, BM25 e
LSA compartilham `utils.preprocess` (caixa baixa → tokenização → stopwords →
stemming RSLP → remoção de acentos). Sem isso, diferenças de tokenização se
disfarçariam de diferenças de modelo.

**`sem` não é stopword.** A lista padrão do NLTK descartaria a palavra, e
`mouse sem fio` viraria `mouse fio` — indistinguível de um mouse com fio. O
conjunto de exceções está em `config.STOPWORDS_KEEP`.

**Relevância conhecida por construção.** O gerador sabe se um produto é um
mouse periférico, um rato de pelúcia ou um teclado musical, e grava esses
julgamentos graduados (0–3) em `data/qrels.json`. Isso permite medir nDCG@5
objetivamente, em vez de julgar as SERPs no olho.

**Corpus como superconjunto estável.** Cada produto é semeado pelo próprio
`product_id`, então `generate_corpus(10000)[:5000] == generate_corpus(5000)`.
Os pontos do benchmark são comparáveis entre si porque o corpus menor é um
prefixo exato do maior.

**Modelo booleano com ordenação.** O modelo clássico devolve um conjunto, não
um ranking. Para que a SERP fosse comparável à dos demais, o conjunto-resposta
é ordenado por *coordination level match* (quantos termos distintos da
consulta o documento contém), que é a extensão ranqueada tradicional do
modelo.

**Reprodutibilidade.** Semente única em `config.RANDOM_SEED` (42), aplicada ao
gerador de corpus e ao solver randomizado do `TruncatedSVD`.

## Limitações conhecidas

- As latências sub-milissegundo ficam na mesma ordem de grandeza do ruído do
  escalonador do SO. Reporta-se a mediana de 7 repetições por consulta, mas
  variações de ±30% entre execuções são esperadas nos modelos indexados.
- O corpus é sintético e montado a partir de moldes de frase. O vocabulário
  (cerca de mil radicais) é bem menor que o de um catálogo real, o que
  favorece os modelos de correspondência literal e limita o que o LSA
  consegue aprender de coocorrência.
- A avaliação usa cinco consultas. É suficiente para expor os contrastes
  qualitativos discutidos no relatório, não para uma conclusão estatística.
