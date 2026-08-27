# Trabalho Prático 1 (T1) - Benchmark de Modelos Clássicos de Recuperação da Informação

**Instituição:** FUCAPE Business School
**Graduação:** Ciência de Dados para Negócios
**Disciplina:** Processamento de Linguagem Natural e Sistemas de Recuperação da Informação
**Professor:** Prof. Dr. Wagner Perin
**Peso:** 30% da nota do 1º Bimestre (M1)
**Formato:** Grupos de 3 a 4 alunos

---

## 1. Escopo e Objetivo

Neste primeiro trabalho prático, vocês atuarão como Engenheiros de Dados e Sistemas de Busca. O objetivo é desenvolver, comparar e avaliar o desempenho de diferentes algoritmos clássicos de Recuperação da Informação (RI) em um **domínio corporativo real**.

Vocês têm **escopo aberto** para escolher o domínio de aplicação. Exemplos de áreas incluem, mas não se limitam a:
* **Direito:** Acórdãos, petições, jurisprudência.
* **E-commerce:** Descrições de produtos, catálogos.
* **Saúde:** Prontuários médicos, bulas, artigos científicos.
* **Finanças:** Relatórios da CVM, notícias de mercado.
* **TI/Operações:** Tickets de suporte, chamados técnicos, documentação interna.

A equipe deverá coletar ou selecionar um corpus contendo algo entre **1.000 e 10.000 documentos** representativos deste domínio.

## 2. Requisitos de Implementação (Algoritmos Obrigatórios)

O sistema deverá indexar o corpus e implementar os seguintes algoritmos de busca (retrieval):

1. **Busca Linear Ingênua (Baseline):** Implementação de busca sequencial ($O(N)$), varrendo strings em memória ou em disco sem uso de índice invertido. Servirá como *baseline* inferior de tempo.
2. **Modelo Booleano:** Implementação usando Matriz de Incidência Binária com suporte a operações *bitwise* (AND, OR, NOT).
3. **Modelo de Espaço Vetorial Clássico:** Implementação utilizando ponderação $TF\text{-}IDF$ e cálculo de similaridade através do Cosseno entre vetores.
4. **Modelo Probabilístico (Okapi BM25):** Implementação do BM25 original, exigindo parametrização dos hiperparâmetros de saturação de frequência ($k_1$) e normalização de comprimento do documento ($b$).
5. **Modelo Algébrico Alternativo (LSA):** Implementação de Análise Semântica Latente (Latent Semantic Analysis) usando redução de dimensionalidade (ex: TruncatedSVD) sobre a matriz termo-documento.

## 3. Requisitos de Engenharia de Software

O projeto deve ser tratado como um produto de software profissional e hospedado no GitHub. O repositório deverá apresentar:

* **Estrutura Profissional de Diretórios:** Padrão de mercado para projetos de Data Science.
  * `data/` (raw e processed - ignorados no `.gitignore` se muito grandes)
  * `src/` (código fonte modularizado)
  * `notebooks/` (exploração e prototipação)
  * `benchmark/` (scripts isolados para medição de desempenho)
  * `tests/` (testes unitários básicos)
* **Qualidade de Código:** Código modularizado, aderente à PEP8, uso de tipagem (Type Hints) e documentação de funções/classes via *docstrings*.
* **Controle de Versão (Git):** Histórico de commits semântico, demonstrando a evolução do projeto e a contribuição distribuída dos membros do grupo. Evitem "commits gigantes".
* **Documentação (README.md):** Um README executivo, contendo a descrição do projeto, instruções claras para instalação (ex: `requirements.txt` ou `Pipfile`), arquitetura da solução e como rodar o projeto e os benchmarks.

## 4. Análise e Benchmark

### 4.1 Análise Empírica Quantitativa (Escalabilidade)
A equipe deverá medir a latência (tempo de resposta em ms) dos 5 algoritmos ao crescer o tamanho do dataset.
* **Tamanhos do Corpus ($N$):** $100, 500, 1.000, 5.000, 10.000$ (ajuste o limite máximo de acordo com o tamanho final do seu corpus).
* **Métrica:** Tempo médio de resposta para um conjunto de queries padronizadas.
* **Entrega:** Gráficos de escalabilidade plotando Tamanho do Corpus vs. Tempo de Resposta.

### 4.2 Análise Qualitativa (Relevância Semântica)
O sistema deve ser avaliado em relação à qualidade da *Search Engine Results Page* (SERP) gerada.
* Elaborem de 3 a 5 **consultas desafiadoras** aplicáveis ao seu domínio (ex: envolvendo termos muito frequentes, termos extremamente raros, sinônimos, casos de polissemia).
* Comparem os Top-K documentos retornados por cada modelo (com foco entre Vetorial, BM25 e LSA) e discutam as diferenças de relevância e comportamento.

## 5. Culminância e Avaliação

A entrega final será composta por:
1. Link do Repositório GitHub (via AVA).
2. **Apresentação em Sala:** Pitch Executivo de 15 minutos + 5 minutos de Q&A.
3. **Live Demo:** Demonstração ao vivo do sistema (pode ser via terminal, script interativo ou interface simples como Streamlit) executando buscas reais.

### Rubrica de Avaliação (Total: 10,0 pontos)

| Pilar | Critério de Avaliação | Pontuação Máxima |
| :--- | :--- | :---: |
| **Pilar 1: Engenharia de Software, Complexidade e Repositório GitHub** | Estrutura de pastas, código limpo (PEP8, Type Hints, docstrings), qualidade dos commits no Git, completude dos 5 modelos obrigatórios e README com instruções de setup/reprodutibilidade. | 3,5 |
| **Pilar 2: Benchmark Quantitativo de Escala e Métricas de Performance** | Implementação e automação da medição de latência, baterias de testes escalando $N$, geração de gráficos comparativos claros ($O(N)$ vs Indexação) e análise de complexidade computacional observada. | 2,0 |
| **Pilar 3: Análise Qualitativa de Relevância e Semântica das SERPs** | Qualidade das queries elaboradas, profundidade da análise das diferenças entre os ranqueamentos (Vetorial vs BM25 vs LSA), sensibilidade a sinônimos/polissemia. | 2,0 |
| **Pilar 4: Apresentação Executiva, Live Demo e Defesa do ROI** | Postura profissional no pitch, clareza visual dos slides, demonstração funcional em tempo real sem falhas críticas, capacidade de responder perguntas e argumentação de viabilidade comercial/corporativa (ROI) do buscador construído. | 2,5 |
| **TOTAL** | | **10,0** |
