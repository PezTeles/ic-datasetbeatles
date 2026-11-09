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

print(media_por_periodo.round(2))
