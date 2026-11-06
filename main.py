
import pandas as pd
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

print(dataset_filtrado.head())