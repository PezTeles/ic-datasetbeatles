import pandas as pd
import re

dataset = pd.read_csv('beatles_dataset.csv')
#print(dataset.head())

#trabalhar só com as colunas que interessam
colunas_desejadas = ['lyrics', 'year', 'album']
dataset_filtrado = dataset[colunas_desejadas].copy()

# Garante que 'year' seja numérico e substitui nulos por 0
dataset_filtrado['year'] = (
    pd.to_numeric(dataset_filtrado['year'], errors='coerce').fillna(0).astype(int)
)


def classificandoPeriodo(ano):
    if ano < 1965:
        return 'Fase inicial'
    elif ano < 1968:
        return 'Fase Psicodélica'
    else:
        return "Fase final"

dataset_filtrado['period'] = dataset_filtrado['year'].apply(classificandoPeriodo)
#print(dataset_filtrado.head())


#FUnção NLP para contar apenas as palavras
def contar_palavras(texto):
    if not isinstance(texto, str):
        return 0
    
    palavras = re.findall(r'\w+', texto)   
    return len(palavras)


#criar a coluna com a quantidade de palavras
dataset_filtrado['qtd_palavras'] = dataset_filtrado['lyrics'].apply(contar_palavras)

#agrupar as musicas pelos periodos e contar as palavras
media_por_periodo = dataset_filtrado.groupby('period')['qtd_palavras'].mean()

#print(media_por_periodo.round(2))

def contar_palavras_unicas(texto_periodo):
    #junta todas as letras de um periodo em um texto só
    #.dropna remove linhas vazias
    #''.join(), serve para juntar as strings e as aspas simples serve para adicionar um espaço entre as palavras
    texto_juntado = ' '.join(texto_periodo.dropna()).lower()

    #extrai as palavras do texto
    #re. está cahamndo a biblioteca re
    #findall busca no texto todas ocorrências do que foi pedido
    #r'\w+' define o que o texto está buscando, o r indica uma string bruta, tudo no texto é uma string
    #\w reconhece letras de a-z e numeros de 0-9, ignorando acentos e pontuações
    #o + juntas as palavras seguidas, ao invés de ser 'b'' 'e' 'a' 't' é beat
    todas_as_palavras = re.findall(r'\w+', texto_juntado)

    palavras_unicas = set(todas_as_palavras)

    return len(palavras_unicas)


palavras_unicas_por_periodo = dataset_filtrado.groupby('period')['lyrics'].apply(contar_palavras_unicas)

print(palavras_unicas_por_periodo)


