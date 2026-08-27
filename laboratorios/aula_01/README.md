# fucape2026_2_pln_ri
Processamento de Linguagem Natural

# Trabalho de Laboratório: Matriz de Incidência Binária e Buscas Booleanas

## 1. Objetivos Pedagógicos
Este laboratório prático tem como objetivos:
- Compreender na prática o modelo Booleano de Recuperação de Informação.
- Desenvolver habilidades de pré-processamento de texto (tokenização e normalização).
- Construir uma Matriz de Incidência Termo-Documento a partir de uma coleção de documentos.
- Implementar e executar consultas utilizando operações bitwise (AND, OR, NOT) sobre a matriz construída.

## 2. Descrição da Atividade
Nesta atividade, vocês irão processar os 10 relatórios disponíveis na pasta `repositorio_aula02/` (em formato `.txt`). O resultado final será a construção e persistência de uma Matriz de Incidência e a execução de consultas nela.

### Etapas da Atividade:

**Etapa 1: Leitura da Coleção**
- Implemente uma rotina para ler todos os arquivos `.txt` contidos no diretório `repositorio_aula02/`.
- Armazene o conteúdo e identifique cada documento unicamente (ex: Doc1, Doc2, etc.).

**Etapa 2: Tokenização, Normalização e Filtragem de Stop Words**
- Para cada documento, extraia os termos.
- Realize a normalização básica: conversão para letras minúsculas e remoção de pontuações.
- **Estratégia Obrigatória de Remoção de Stop Words:**
  - **Identificação Lexical:** Definir uma lista de stopwords em português contendo conectivos gramaticais (ex: `['de', 'e', 'sobre', 'por', 'com', 'para', 'em', 'da', 'do', 'dos', 'das', 'a', 'o']`).
  - **Identificação Estatística por Frequência ($DF_t$):** Alternativamente (ou complementarmente), identificar palavras que ocorrem em $DF_t \ge 80\%$ dos documentos ($DF_t \ge 8$) como stop words estruturais.
  - **Remoção:** Filtrar os tokens antes da montagem do vocabulário $V$, demonstrando a redução de dimensão da matriz e eliminação de ruídos semânticos.

**Etapa 3: Extração do Vocabulário (V)**
- Construa o conjunto de todos os termos únicos presentes em toda a coleção, que formará o vocabulário $V$.

**Etapa 4: Geração da Matriz de Incidência**
- Utilizando a biblioteca `pandas`, crie um DataFrame onde as linhas representam os termos do vocabulário $V$ e as colunas representam os documentos ($Doc_j$).
- Preencha a matriz de forma que a célula $(i, j)$ contenha $1$ se o termo $i$ ocorre no documento $j$, e $0$ caso contrário.

**Etapa 5: Exportação dos Dados**
- Salve o DataFrame gerado em um arquivo chamado `matriz_incidencia.csv`.

**Etapa 6: Desafios de Busca Booleana**
A partir da matriz gerada, implemente em Python (usando operações bitwise `&`, `|`, `~` em pandas) as seguintes consultas:
- **Desafio A:** (termo1 AND termo2) - Encontrar documentos que contenham duas palavras de sua escolha que existam no vocabulário.
- **Desafio B:** (termo3 OR termo4) - Encontrar documentos que contenham ao menos uma de duas outras palavras escolhidas.
- **Desafio C:** (termo5 AND NOT termo6) - Encontrar documentos que contenham um termo específico, mas excluam obrigatoriamente outro termo.

## 3. Entregas
Você deve entregar:
1. O script Python (ou Jupyter Notebook) contendo todo o pipeline (da leitura dos dados à resolução das consultas).
2. O arquivo `matriz_incidencia.csv` gerado pelo seu script.
3. Um breve documento/seção texto com as saídas (os documentos recuperados) para os 3 desafios de busca booleana executados.
