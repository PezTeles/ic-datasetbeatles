import pandas as pd
import re
from collections import Counter
import spacy
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

#carrega a bibloteca em inglês do spacy
nlp = spacy.load('en_core_web_sm')

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

    #cria uma variavel com as palavras usando o set que não armazena dados repetidos
    palavras_unicas = set(todas_as_palavras)

    return len(palavras_unicas)


palavras_unicas_por_periodo = dataset_filtrado.groupby('period')['lyrics'].apply(contar_palavras_unicas)

#print(palavras_unicas_por_periodo)

#LISTA set com as stopwords mais famosas da lingua inglesa
STOPWORDS = {'a',
    'about',
    'above',
    'after',
    'again',
    'against',
    'all',
    'am',
    'an',
    'and',
    'any',
    'are',
    'as',
    'at',
    'be',
    'because',
    'been',
    'before',
    'being',
    'below',
    'between',
    'both',
    'but',
    'by',
    'can',
    'did',
    'do',
    'does',
    'doing',
    'down',
    'during',
    'each',
    'few',
    'for',
    'from',
    'further',
    'had',
    'has',
    'have',
    'having',
    'he',
    'her',
    'here',
    'hers',
    'herself',
    'him',
    'himself',
    'his',
    'how',
    'i',
    'if',
    'in',
    'into',
    'is',
    'it',
    'its',
    'itself',
    'just',
    'me',
    'more',
    'most',
    'my',
    'myself',
    'no',
    'nor',
    'not',
    'now',
    'of',
    'off',
    'on',
    'once',
    'only',
    'or',
    'other',
    'our',
    'ours',
    'ourselves',
    'out',
    'over',
    'own',
    's',
    'same',
    'she',
    'should',
    'so',
    'some',
    'such',
    't',
    'than',
    'that',
    'the',
    'their',
    'theirs',
    'them',
    'themselves',
    'then',
    'there',
    'these',
    'they',
    'this',
    'those',
    'through',
    'to',
    'too',
    'under',
    'until',
    'up',
    'very',
    'was',
    'we',
    'were',
    'what',
    'when',
    'where',
    'which',
    'while',
    'who',
    'whom',
    'why',
    'will',
    'with',
    'you',
    'your',
    'yours',
    'yourself',
    'yourselves',
    'll',
    'm',}

def contar_palavras_unicas_sem_stopwords(texto_periodo):
    texto_juntado = ' '.join(texto_periodo.dropna()).lower()
    todas_as_palavras = re.findall(r'\w+', texto_juntado)

    #filtra as palavras sem stopwords
    palavras_filtradas = [
        palavra 
        for palavra in todas_as_palavras
        if palavra not in STOPWORDS
        ]

    palavras_unicas = set(palavras_filtradas)

    return len(palavras_unicas)


palavras_unicas_por_periodo_sem_STOPWORDS = dataset_filtrado.groupby('period')['lyrics'].apply(contar_palavras_unicas_sem_stopwords)
#print(palavras_unicas_por_periodo_sem_STOPWORDS)

#Função para buscar as top N palavras mais frequentes do período
def palavras_mais_frequentes(texto_periodo, top_n = 10):
    texto_juntado = ' '.join(texto_periodo.dropna()).lower()
    todas_as_palavras = re.findall(r'\w+', texto_juntado)
    palavras_filtradas = [
            palavra 
            for palavra in todas_as_palavras
            if palavra not in STOPWORDS
            ]
    #Counter conta a frequência de cada palavra na lista
    contagem = Counter(palavras_filtradas)

    #Retorna aspalavras mais comuns acompanhadas da contagem
    return contagem.most_common(top_n)

mais_frequentes_por_periodo = dataset_filtrado.groupby('period')['lyrics'].apply(palavras_mais_frequentes)
#print(mais_frequentes_por_periodo)

def analise_pos_tags(texto_periodo, top_n=10):
    texto_juntado = ' '.join(texto_periodo.dropna()).lower()
    doc = nlp(texto_juntado)

    verbos = []
    substantivos = []
    adjetivos = []

    for token in doc:
        # Ignora stopwords, pontuações, espaços e letras isoladas
        if (
            token.text in STOPWORDS
            or token.is_punct
            or token.is_space
            or len(token.text) <= 1
        ):
            continue

        if token.pos_ == 'VERB':
            verbos.append(token.text)
        elif token.pos_ == 'NOUN':
            substantivos.append(token.text)
        elif token.pos_ == 'ADJ':
            adjetivos.append(token.text)

    return {
        'Verbos': Counter(verbos).most_common(top_n),
        'Substantivos': Counter(substantivos).most_common(top_n),
        'Adjetivos': Counter(adjetivos).most_common(top_n),
    }


pos_por_periodo = dataset_filtrado.groupby('period')['lyrics'].apply(
    analise_pos_tags
)



# for (periodo, classe), lista in pos_por_periodo.items():
#     print(f'\n--- {periodo} | {classe} ---')
#     tabela = pd.DataFrame(lista, columns=['Palavra', 'Frequência'])
#     # index=False oculta os números 0, 1, 2... da esquerda
#     print(tabela.to_string(index=False))


def ngramas(texto_periodo, n=2, top_n=10):
    texto_juntado = ' '.join(texto_periodo.dropna()).lower()
    palavras = re.findall(r'\w+', texto_juntado)

    contagem_ngramas = defaultdict(int)

    for i in range(len(palavras) - n + 1):
        ngrama = ' '.join(palavras[i : i + n])

        contagem_ngramas[ngrama] += 1

    mais_comuns = sorted(
        contagem_ngramas.items(), key=lambda item: item[1], reverse=True
    )[:top_n]

    return mais_comuns



bigramas_por_periodo = dataset_filtrado.groupby('period')['lyrics'].apply(
    lambda texto: ngramas(texto, n=2, top_n=5)
)

# print('--- TOP 5 BIGRAMAS (2 PALAVRAS) POR PERÍODO ---')
# for periodo, lista in bigramas_por_periodo.items():
#     print(f'\n{periodo}:')
#     for ngrama, qtd in lista:
#         print(f'  "{ngrama}" -> {qtd} vezes')



# trigramas_por_periodo = dataset_filtrado.groupby('period')['lyrics'].apply(
#     lambda texto: ngramas(texto, n=3, top_n=5)
# )

# print('\n--- TOP 5 TRIGRAMAS (3 PALAVRAS) POR PERÍODO ---')
# for periodo, lista in trigramas_por_periodo.items():
#     print(f'\n{periodo}:')
#     for ngrama, qtd in lista:
#         print(f'  "{ngrama}" -> {qtd} vezes')    

def temas_lsa(serie_letras, n_temas=3, n_palavras=5):
    #remove linhas vazias, garantindo que cada musica seja um item na lista
    musicas = serie_letras.dropna().tolist()

    vetorizador = TfidfVectorizer(
    # 1. Pega apenas palavras com 3 ou mais letras (elimina 'll', 've', 'oh')
    token_pattern=r'(?u)\b[a-zA-Z]{3,}\b',
    # 2. Adiciona fragmentos de contração e interjeições às stopwords padrão
    stop_words=list(TfidfVectorizer(stop_words='english').get_stop_words())
    + ['don', 'ain', 'gon', 'wan', 'yeah', 'whoa', 'hoo', 'mon'],
    # 3. Ignora palavras que aparecem em mais de 50% das músicas (evita que 'love' e 'know' dominem todos os temas)
    max_df=0.5,
)
    matriz_tfidf = vetorizador.fit_transform(musicas)

    lsa = TruncatedSVD(n_components=n_temas, random_state=42)
    lsa.fit(matriz_tfidf)

    # Recupera o vocabulário mapeado pelo vetorizador
    termos = vetorizador.get_feature_names_out()

    temas_encontrados = []

    for idx, componente in enumerate(lsa.components_):
        # Ordena os índices das palavras do maior peso para o menor
        indices_top = componente.argsort()[: -n_palavras - 1 : -1]
        palavras_tema = [termos[i] for i in indices_top]
        temas_encontrados.append(f"Tema {idx + 1}: {', '.join(palavras_tema)}")

    return temas_encontrados


print('TEMAS RELEVANTES POR PERÍODO')
for periodo, grupo in dataset_filtrado.groupby('period'):
    print(f'\n--- {periodo.upper()} ---')
    temas = temas_lsa(grupo['lyrics'], n_temas=3, n_palavras=6)
    for t in temas:
        print(f'  {t}')