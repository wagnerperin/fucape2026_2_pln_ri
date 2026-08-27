import os
import re
import unicodedata
import pandas as pd

# Diretório contendo os relatórios .txt
dir_repositorio = os.path.dirname(os.path.abspath(__file__))

# 1. Leitura dos 10 relatórios jurídicos breves
documentos = {}
for i in range(1, 11):
    nome_arquivo = f"relatorio_{i:02d}.txt"
    caminho_completo = os.path.join(dir_repositorio, nome_arquivo)
    if os.path.exists(caminho_completo):
        with open(caminho_completo, "r", encoding="utf-8") as f:
            documentos[nome_arquivo] = f.read().strip()

# 2. Definição das Stop Words em Português (Estratégia Lexical + Frequência)
STOP_WORDS = set([
    'de', 'da', 'do', 'das', 'dos', 'em', 'para', 'por', 'com', 'e', 'a', 'o',
    'as', 'os', 'um', 'uma', 'uns', 'umas', 'sobre', 'ou', 'se'
])

# Função de Normalização de Acentos, Tokenização e Filtragem de Stop Words
def normalizar_texto(texto):
    # Remove acentos (ex: rescisão -> rescisao)
    nfkd_form = unicodedata.normalize('NFD', texto.lower())
    texto_sem_acento = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return texto_sem_acento

def tokenizar_e_filtrar(texto):
    texto_limpo = normalizar_texto(texto)
    tokens = re.findall(r"\b\w+\b", texto_limpo)
    # Remoção de Stop Words
    tokens_filtrados = [t for t in tokens if t not in STOP_WORDS and not t.isdigit()]
    return tokens_filtrados

# 3. Extração do Vocabulário Único V
todos_tokens_filtrados = [token for texto in documentos.values() for token in tokenizar_e_filtrar(texto)]
vocabulario = sorted(list(set(todos_tokens_filtrados)))

print(f"Tamanho do Vocabulário Filtrado e Normalizado |V|: {len(vocabulario)} palavras únicas.\n")

# 4. Construção da Matriz de Incidência Binária (0 ou 1)
matriz_incidencia = pd.DataFrame(0, index=vocabulario, columns=sorted(documentos.keys()))

for nome_arquivo, texto in documentos.items():
    tokens = tokenizar_e_filtrar(texto)
    for token in tokens:
        matriz_incidencia.loc[token, nome_arquivo] = 1

# Exportação da Matriz de Incidência para CSV
caminho_csv = os.path.join(dir_repositorio, "matriz_incidencia.csv")
matriz_incidencia.to_csv(caminho_csv, encoding="utf-8")
print(f"Matriz de Incidência exportada com sucesso para: {caminho_csv}\n")

# 5. Execução dos Desafios de Buscas Booleanas (Operações Bitwise)

# Desafio 1: 'contrato' AND 'rescisao'
v_contrato = matriz_incidencia.loc["contrato"] if "contrato" in matriz_incidencia.index else 0
v_rescisao = matriz_incidencia.loc["rescisao"] if "rescisao" in matriz_incidencia.index else 0
res_desafio1 = (v_contrato == 1) & (v_rescisao == 1)
print("=== DESAFIO 1: (contrato AND rescisao) ===")
print(res_desafio1[res_desafio1].index.tolist())

# Desafio 2: 'multa' AND NOT 'inadimplencia'
v_multa = matriz_incidencia.loc["multa"] if "multa" in matriz_incidencia.index else 0
v_inadimplencia = matriz_incidencia.loc["inadimplencia"] if "inadimplencia" in matriz_incidencia.index else 0
res_desafio2 = (v_multa == 1) & (v_inadimplencia == 0)
print("\n=== DESAFIO 2: (multa AND NOT inadimplencia) ===")
print(res_desafio2[res_desafio2].index.tolist())

# Desafio 3: 'sigilo' OR 'propriedade'
v_sigilo = matriz_incidencia.loc["sigilo"] if "sigilo" in matriz_incidencia.index else 0
v_propriedade = matriz_incidencia.loc["propriedade"] if "propriedade" in matriz_incidencia.index else 0
res_desafio3 = (v_sigilo == 1) | (v_propriedade == 1)
print("\n=== DESAFIO 3: (sigilo OR propriedade) ===")
print(res_desafio3[res_desafio3].index.tolist())
