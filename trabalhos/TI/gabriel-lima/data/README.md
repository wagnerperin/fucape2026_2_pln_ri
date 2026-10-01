# Dados

Esta pasta guarda o corpus do benchmark. Os dois arquivos abaixo **não são
versionados** (ver `.gitignore`): juntos passam de 12 MB e são inteiramente
reproduzíveis a partir da semente fixa `config.RANDOM_SEED = 42`.

| Arquivo | Tamanho | Conteúdo |
|---|---:|---|
| `products.json` | ~12,2 MB | 5.000 produtos de e-commerce gerados sinteticamente |
| `qrels.json` | ~36 KB | Julgamentos de relevância graduados (0–3) das 5 consultas de teste |

## Como gerar

```bash
python src/corpus_generator.py
```

Opções:

```bash
python src/corpus_generator.py --size 1000          # corpus menor
python src/corpus_generator.py --seed 7             # outra semente
python src/corpus_generator.py --output data/x.json # outro destino
```

A geração leva poucos segundos e é **determinística**: a mesma semente produz
exatamente o mesmo catálogo, byte a byte. Além disso, o corpus é um
*superconjunto estável* — cada produto é semeado pelo próprio `product_id`,
de modo que `generate_corpus(10000)[:5000] == generate_corpus(5000)`. É isso
que torna os pontos do benchmark de escalabilidade comparáveis entre si.

## Sobre o `qrels.json`

Como o corpus é gerado, o programa sabe por construção se um produto é um
mouse periférico, um rato de pelúcia ou um teclado musical. Esses julgamentos
são gravados automaticamente junto com o catálogo e permitem medir P@5 e
nDCG@5 de forma objetiva, sem anotação manual.

Formato: `{"Q1": {"<product_id>": <grau 0-3>, ...}, "Q2": {...}, ...}`
