# Anotações: análise das letras dos Beatles com Python

> **Objetivo do código:** descobrir como o vocabulário e o estilo das letras dos Beatles mudaram ao longo da carreira, comparando três fases (inicial, psicodélica e final).
>
> **Como ler estas anotações:** cada bloco mostra um trecho do código e, logo abaixo, explica linha por linha o que acontece, o que cada palavra significa e por que ela está ali. Leia na ordem, pois os blocos se apoiam uns nos outros.

---

## 0. Vocabulário básico (leia primeiro)

| Termo | O que significa |
|---|---|
| **Variável** | Um "nome" que guarda um valor. Ex.: `idade = 20` guarda o número 20 no nome `idade`. |
| **Função** | Um bloco de código com nome que faz uma tarefa. Você escreve uma vez e usa várias vezes. |
| **Biblioteca** | Código pronto feito por outras pessoas, que você importa para usar. Evita reinventar a roda. |
| **String** | Um texto. Sempre entre aspas: `'amor'` ou `"amor"`. |
| **Lista** | Uma sequência ordenada de itens, entre colchetes: `[1, 2, 3]`. |
| **Set (conjunto)** | Uma coleção sem itens repetidos, entre chaves: `{1, 2, 3}`. |
| **Dicionário** | Pares "chave: valor", como `{'love': 120}`. |
| **Tupla** | Como uma lista, mas entre parênteses e que não muda: `('love', 120)`. |
| **DataFrame** | Uma tabela do pandas, com linhas e colunas (como uma planilha). |
| **Series** | Uma única coluna de um DataFrame. |
| **NLP** | *Natural Language Processing*, ou Processamento de Linguagem Natural: ensinar computadores a lidar com texto humano. |
| **Comentário** | Linha que começa com `#`. O Python ignora; serve só para explicar o código a humanos. |

---

## 1. Importando as bibliotecas

```python
import pandas as pd
import re
from collections import Counter
import spacy
from collections import defaultdict
```

**O que é `import`?** É a instrução que "traz" uma biblioteca para o seu programa. Sem ela, o Python não conhece as ferramentas.

**Linha por linha:**

- `import pandas as pd`
  - **pandas** é a principal biblioteca do Python para trabalhar com tabelas de dados.
  - `as pd` cria um **apelido**. Em vez de escrever `pandas.read_csv(...)` toda vez, escrevemos `pd.read_csv(...)`. É uma convenção usada no mundo todo.
- `import re`
  - **re** vem de *regular expressions* (expressões regulares). Serve para **procurar padrões em textos**, por exemplo "encontre todas as palavras".
  - Já vem instalada com o Python.
- `from collections import Counter`
  - `from X import Y` significa "do módulo X, traga apenas a ferramenta Y".
  - **Counter** é uma ferramenta que **conta quantas vezes cada item aparece** em uma lista.
- `import spacy`
  - **spaCy** é uma biblioteca de NLP. Ela entende a estrutura de uma frase: separa palavras e identifica se cada uma é verbo, substantivo, adjetivo etc.
- `from collections import defaultdict`
  - **defaultdict** é um dicionário "esperto": se você pedir uma chave que não existe, ele cria a chave com um valor inicial automático (veremos o uso na seção 12).
  - Observação: as duas linhas `from collections import ...` poderiam ser uma só: `from collections import Counter, defaultdict`.

---

## 2. Carregando o modelo de linguagem do spaCy

```python
#carrega a bibloteca em inglês do spacy
nlp = spacy.load('en_core_web_sm')
```

- `#carrega a bibloteca...` é um **comentário**. Não executa nada.
- `spacy.load('en_core_web_sm')` **carrega um modelo já treinado** para a língua inglesa.
  - `en` = English (inglês); `core_web` = treinado com textos gerais da internet; `sm` = *small* (pequeno e rápido).
  - Usamos inglês porque as letras dos Beatles estão em inglês.
- `nlp = ...` guarda o modelo na variável `nlp`. Daqui em diante, `nlp(texto)` analisa qualquer texto.

**Conceito de ciência de dados:** um *modelo* é um programa que aprendeu padrões a partir de muitos exemplos. Este aprendeu, lendo muito texto, a reconhecer classes gramaticais.

---

## 3. Lendo o arquivo de dados

```python
dataset = pd.read_csv('beatles_dataset.csv')
#print(dataset.head())
```

- `pd.read_csv('beatles_dataset.csv')`
  - **CSV** (*Comma-Separated Values*) é um arquivo de texto onde cada linha é uma linha da tabela e as colunas são separadas por vírgula.
  - `read_csv` lê o arquivo e devolve um **DataFrame**.
  - O arquivo precisa estar na **mesma pasta** do código, ou você deve informar o caminho completo.
- `dataset = ...` guarda a tabela na variável `dataset`.
- `#print(dataset.head())` está **comentada** (por causa do `#`), então não roda.
  - `print()` mostra algo na tela.
  - `.head()` mostra as **5 primeiras linhas** da tabela. É a primeira coisa que se faz para "espiar" os dados.
  - Dica: tire o `#` para ver a tabela.

---

## 4. Selecionando só as colunas necessárias

```python
#trabalhar só com as colunas que interessam
colunas_desejadas = ['lyrics', 'year', 'album']
dataset_filtrado = dataset[colunas_desejadas].copy()
```

- `colunas_desejadas = ['lyrics', 'year', 'album']`
  - Uma **lista** com os nomes das colunas que importam: letra (`lyrics`), ano (`year`) e álbum (`album`).
- `dataset[colunas_desejadas]`
  - Colchetes com uma lista de nomes **recortam** a tabela, mantendo só essas colunas.
- `.copy()`
  - Cria uma **cópia independente**. Sem ela, o pandas pode mostrar o aviso `SettingWithCopyWarning` quando você alterar a tabela nova, porque ela ainda estaria "ligada" à original.
- `dataset_filtrado` é a nova tabela, menor e mais fácil de trabalhar.

**Conceito:** selecionar apenas as variáveis (colunas) relevantes se chama **seleção de features**. Deixa o trabalho mais rápido e organizado.

---

## 5. Limpando a coluna do ano

```python
# Garante que 'year' seja numérico e substitui nulos por 0
dataset_filtrado['year'] = (
    pd.to_numeric(dataset_filtrado['year'], errors='coerce').fillna(0).astype(int)
)
```

Os parênteses externos servem só para quebrar a linha longa em várias, deixando-a legível. Dentro, há uma **cadeia de três operações**, executadas da esquerda para a direita:

1. `pd.to_numeric(dataset_filtrado['year'], errors='coerce')`
   - Tenta converter a coluna para **número**.
   - `errors='coerce'` significa: "se algum valor não puder virar número (ex.: texto `'desconhecido'`), não dê erro, transforme em `NaN`".
   - **NaN** (*Not a Number*) é o jeito do pandas representar **dado ausente** (célula vazia).
2. `.fillna(0)`
   - **Preenche** os `NaN` com o valor 0.
3. `.astype(int)`
   - Converte os valores para **número inteiro** (sem casa decimal). Era decimal (ex.: `1963.0`) porque a presença de `NaN` obriga o pandas a usar decimais.

O resultado é gravado de volta na própria coluna `year`.

> ⚠️ **Atenção, decisão com consequências:** trocar ausentes por 0 é uma escolha de limpeza. Veja o problema que isso causa na próxima seção.

**Conceito:** tratar **dados ausentes** (*missing data*) é uma das tarefas mais importantes da ciência de dados. As opções comuns são: remover as linhas, preencher com um valor (média, zero...) ou marcar como "desconhecido".

---

## 6. Criando a coluna de períodos (fases da banda)

```python
def classificandoPeriodo(ano):
    if ano < 1965:
        return 'Fase inicial'
    elif ano < 1968:
        return 'Fase Psicodélica'
    else:
        return "Fase final"

dataset_filtrado['period'] = dataset_filtrado['year'].apply(classificandoPeriodo)
#print(dataset_filtrado.head())
```

### A função

- `def classificandoPeriodo(ano):`
  - `def` = **define** uma função. `classificandoPeriodo` é o nome que escolhemos. `ano` é o **parâmetro**: o valor que a função recebe para trabalhar.
  - Os dois-pontos `:` abrem o bloco. Tudo que está **recuado (indentado)** pertence à função. Em Python, o recuo faz parte da sintaxe.
- `if ano < 1965:` → **Se** o ano for menor que 1965...
  - `return 'Fase inicial'`: `return` **devolve** um resultado e encerra a função.
- `elif ano < 1968:` → **Senão, se** for menor que 1968 (e, como falhou o primeiro teste, é ≥ 1965)...
  - `return 'Fase Psicodélica'`
- `else:` → **Senão** (qualquer outro caso, ou seja, ≥ 1968)...
  - `return "Fase final"`
  - Aspas simples `' '` e duplas `" "` funcionam igual para strings.

O Python testa as condições **de cima para baixo** e para na primeira verdadeira.

### Aplicando a função na coluna

- `dataset_filtrado['year'].apply(classificandoPeriodo)`
  - `.apply(função)` executa a função **em cada valor da coluna**, um por vez, e devolve uma nova coluna com os resultados.
  - Repare: passamos `classificandoPeriodo` **sem parênteses**. Estamos entregando a função em si, e não chamando ela.
- `dataset_filtrado['period'] = ...` cria uma **coluna nova** chamada `period`.

**Conceito:** criar uma variável nova a partir de outra é **feature engineering** (engenharia de atributos). Aqui transformamos um número contínuo (ano) em **categorias**, para poder comparar grupos.

> ⚠️ **Bug escondido:** músicas com ano ausente viraram `0` na seção 5. Como `0 < 1965` é verdadeiro, elas caem em "Fase inicial" por engano, distorcendo todos os resultados. Soluções: remover essas linhas com `dropna()`, ou criar uma categoria `'Desconhecido'`.

---

## 7. Contando palavras de uma música

```python
#FUnção NLP para contar apenas as palavras
def contar_palavras(texto):
    if not isinstance(texto, str):
        return 0
    
    palavras = re.findall(r'\w+', texto)   
    return len(palavras)
```

- `def contar_palavras(texto):` define a função, que recebe um `texto`.
- `if not isinstance(texto, str):`
  - `isinstance(texto, str)` pergunta: "`texto` é uma string?" (retorna `True` ou `False`).
  - `not` inverte: o `if` vale quando **não** é string.
  - Por que isso? Uma letra vazia vira `NaN`, que é do tipo número decimal (`float`), e não texto. Se tentássemos processá-la, o código daria erro. É uma **validação defensiva**.
- `return 0` → se não for texto, devolve 0 palavras.
- `palavras = re.findall(r'\w+', texto)`
  - `re.findall(padrão, texto)` **encontra todas as ocorrências** do padrão no texto e devolve uma lista.
  - O padrão é `r'\w+'`:
    - `r` antes das aspas = ***raw string***: o Python não interpreta a barra `\` de forma especial, passando-a intacta para a regex.
    - `\w` = qualquer **letra, número ou `_`**.
    - `+` = "**uma ou mais** vezes seguidas". Assim `\w+` captura palavras inteiras, e não letras soltas.
  - Exemplo: `"Love me do!"` → `['Love', 'me', 'do']`. Pontuação e espaços são ignorados.
- `return len(palavras)`
  - `len()` (de *length*, tamanho) conta quantos itens há na lista.

**Conceito:** quebrar um texto em pedaços (palavras) é a **tokenização**. Cada pedaço é um **token**.

---

## 8. Média de palavras por período (groupby)

```python
#criar a coluna com a quantidade de palavras
dataset_filtrado['qtd_palavras'] = dataset_filtrado['lyrics'].apply(contar_palavras)

#agrupar as musicas pelos periodos e contar as palavras
media_por_periodo = dataset_filtrado.groupby('period')['qtd_palavras'].mean()

#print(media_por_periodo.round(2))
```

- Primeira linha: aplica `contar_palavras` em **cada letra** e guarda o resultado na nova coluna `qtd_palavras`. Agora cada música tem seu número de palavras.
- `dataset_filtrado.groupby('period')`
  - **Agrupa** as linhas pelo valor da coluna `period`: um grupo para cada fase.
- `['qtd_palavras']` → escolhe a coluna que será analisada dentro de cada grupo.
- `.mean()` → calcula a **média** (soma ÷ quantidade) em cada grupo.
- `.round(2)` (na linha comentada) arredonda para 2 casas decimais.

**Conceito, o padrão *split-apply-combine*:**
1. **Split:** divide a tabela em grupos.
2. **Apply:** aplica uma função (aqui, a média) em cada grupo.
3. **Combine:** junta tudo em um resultado.

O resultado é uma **estatística descritiva**: resume muitos dados em poucos números. Aqui responde: "em média, quantas palavras tem uma música de cada fase?"

---

## 9. Contando palavras únicas (riqueza de vocabulário)

```python
def contar_palavras_unicas(texto_periodo):
    texto_juntado = ' '.join(texto_periodo.dropna()).lower()
    todas_as_palavras = re.findall(r'\w+', texto_juntado)
    palavras_unicas = set(todas_as_palavras)
    return len(palavras_unicas)

palavras_unicas_por_periodo = dataset_filtrado.groupby('period')['lyrics'].apply(contar_palavras_unicas)
#print(palavras_unicas_por_periodo)
```

Esta função recebe **todas as letras de um período** (uma Series) e conta quantas palavras diferentes existem.

- `texto_periodo.dropna()`
  - Remove os valores ausentes (`NaN`). Sem isso, a junção abaixo daria erro.
- `' '.join(...)`
  - O método `join` **junta** vários textos em um só. O texto antes do ponto (`' '`, um espaço) é o **separador** colocado entre eles.
  - Ex.: `' '.join(['a', 'b', 'c'])` → `'a b c'`.
- `.lower()`
  - Converte tudo para **minúsculas**. Assim "Love" e "love" contam como a mesma palavra. Isso é **normalização**.
- `re.findall(r'\w+', texto_juntado)` → extrai todas as palavras (tokenização), como na seção 7.
- `set(todas_as_palavras)`
  - Um **set não guarda repetidos**. Ao converter a lista em set, cada palavra aparece só uma vez.
- `len(palavras_unicas)` → quantidade de palavras diferentes.

A última parte aplica a função: `groupby('period')['lyrics'].apply(contar_palavras_unicas)` agrupa por fase e executa a função **sobre o grupo inteiro** (diferente da seção 8, em que o `apply` rodava linha a linha).

**Conceitos de linguística computacional:**
- **Token:** cada palavra do texto (com repetições).
- **Type:** cada palavra *diferente*.
- **TTR (*Type-Token Ratio*)** = types ÷ tokens. Mede riqueza de vocabulário.

> 💡 Comparar números **brutos** de palavras únicas entre grupos de tamanhos diferentes não é justo: uma fase com mais músicas terá mais palavras únicas só por ter mais texto. O ideal é calcular o TTR.

---

## 10. Stopwords

```python
STOPWORDS = {'a', 'about', 'above', 'after', ... 'll', 'm',}
```

(A lista completa está no código. Aqui explicamos o conceito.)

- **Stopwords** são palavras muito comuns que carregam pouco significado sozinhas: *the, and, of, to, I, you...*. Se não as removermos, as palavras mais frequentes serão sempre essas, e não revelam o tema das músicas.
- `STOPWORDS = { ... }` usa **chaves `{}` com itens soltos**, o que cria um **set**.
  - **Por que set e não lista?** Verificar se uma palavra está num set é **muito mais rápido**, porque ele usa uma estrutura chamada *tabela hash*. Com milhares de palavras, a diferença é grande.
- O nome em **MAIÚSCULAS** é uma convenção para **constantes**: valores que não devem mudar durante o programa.
- A vírgula depois do último item (`'m',`) é permitida no Python.
- As letras soltas `'s'`, `'t'`, `'m'`, `'ll'` estão na lista porque a regex `\w+` quebra contrações. `"don't"` vira `don` + `t`, `"I'm"` vira `I` + `m`. Esses resíduos são então descartados.

---

## 11. Palavras únicas sem stopwords

```python
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
```

É igual à função da seção 9, com um passo extra: **filtrar as stopwords** antes de contar. As duas primeiras linhas já foram explicadas.

### List comprehension (compreensão de lista)

```python
palavras_filtradas = [palavra for palavra in todas_as_palavras if palavra not in STOPWORDS]
```

Uma forma **compacta** de construir uma lista. Lê-se quase como uma frase em português:

> "Monte uma lista com cada `palavra`, percorrendo `todas_as_palavras`, **se** a palavra **não estiver** em `STOPWORDS`."

O mesmo código com `for` tradicional:

```python
palavras_filtradas = []
for palavra in todas_as_palavras:
    if palavra not in STOPWORDS:
        palavras_filtradas.append(palavra)
```

- `for palavra in lista:` → repete o bloco para **cada item** da lista, chamando o item atual de `palavra`.
- `not in` → verifica que o item **não** está na coleção.
- `.append(x)` → adiciona `x` ao final da lista.

O resto é igual: `set(...)` remove repetidos e `len(...)` conta.

**Para que serve comparar as duas contagens (com e sem stopwords)?** Para ver se o vocabulário "real", de palavras com significado, cresceu ou diminuiu entre as fases.

---

## 12. Palavras mais frequentes

```python
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
```

- `def palavras_mais_frequentes(texto_periodo, top_n = 10):`
  - `top_n = 10` é um **parâmetro com valor padrão**: se você não informar, vale 10. Você pode chamar com outro valor, como `top_n=5`.
- As linhas seguintes repetem o preparo de texto: juntar, minúsculas, tokenizar, remover stopwords.
- `Counter(palavras_filtradas)`
  - Recebe a lista e devolve algo parecido com um dicionário: `{palavra: quantidade de vezes}`.
- `contagem.most_common(top_n)`
  - Devolve as `top_n` palavras mais frequentes, como **lista de tuplas**: `[('love', 120), ('know', 95), ...]`.
- O `groupby(...).apply(...)` executa isso **para cada período**.

**Conceito:** isso é **análise de frequência**, base de técnicas como nuvem de palavras e TF-IDF.

---

## 13. Classes gramaticais com spaCy (POS tagging)

```python
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
```

**POS tagging** (*Part-of-Speech*) é descobrir a **classe gramatical** de cada palavra: verbo, substantivo, adjetivo, etc. Isso mostra, por exemplo, se uma fase fala mais de **ações** (verbos) ou de **coisas e sentimentos** (substantivos).

### Passo a passo

- `doc = nlp(texto_juntado)`
  - Passa o texto pelo modelo do spaCy. O resultado, `doc`, é um objeto que contém o texto já **analisado**, dividido em **tokens**, cada um com informações gramaticais.
- `verbos = []`, `substantivos = []`, `adjetivos = []`
  - Três **listas vazias** que serão preenchidas.
- `for token in doc:` → percorre cada token do texto analisado.

### O filtro com `continue`

```python
if (token.text in STOPWORDS or token.is_punct or token.is_space or len(token.text) <= 1):
    continue
```

- `or` significa "**ou**": basta uma condição ser verdadeira.
- `token.text in STOPWORDS` → a palavra é uma stopword?
- `token.is_punct` → é pontuação?
- `token.is_space` → é espaço em branco ou quebra de linha?
- `len(token.text) <= 1` → tem 1 caractere ou menos?
- `continue` → **pula** para o próximo token, ignorando o resto do bloco.

### Atributos de um token

| Atributo | O que é |
|---|---|
| `token.text` | O texto da palavra |
| `token.pos_` | A classe gramatical (`'VERB'`, `'NOUN'`, `'ADJ'`...) |
| `token.is_punct` | `True` se for pontuação |
| `token.is_space` | `True` se for espaço |
| `token.lemma_` | A forma base da palavra (*running* → *run*) |

### Classificando

- `if token.pos_ == 'VERB':` → atenção: **dois sinais de igual (`==`) comparam**; um só (`=`) **atribui**.
- Se for verbo, vai para `verbos`; se for substantivo (`NOUN`), para `substantivos`; se for adjetivo (`ADJ`), para `adjetivos`.

### O retorno

- A função devolve um **dicionário** com três chaves, e para cada uma, as `top_n` palavras mais comuns (usando `Counter` e `most_common`, como na seção 12). Guardar resultados múltiplos em um dicionário é um padrão muito comum.

> 💡 **Possíveis melhorias:**
> - Usar `token.lemma_` em vez de `token.text`, para que *love*, *loves* e *loved* sejam contadas juntas.
> - Não usar `.lower()` antes do spaCy: o modelo usa as maiúsculas como pista para acertar a classe gramatical.
> - O spaCy tem um limite de cerca de 1 milhão de caracteres por texto (`nlp.max_length`). Se o conjunto de um período passar disso, é preciso processar música por música ou aumentar o limite.
> - O código comentado que imprime os resultados (`for (periodo, classe), lista in pos_por_periodo.items():`) não funcionaria como está, porque `pos_por_periodo` é uma Series de **dicionários**, e não uma série com índice duplo `(periodo, classe)`. Seria preciso um laço que percorresse cada período e depois cada classe.

---

## 14. N-gramas (sequências de palavras)

```python
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
```

**N-grama** = uma sequência de **n palavras seguidas**.
- n = 2 → **bigrama** (2 palavras): `"love me"`
- n = 3 → **trigrama** (3 palavras): `"love me do"`

Eles capturam **expressões e contexto**, algo que contar palavras soltas não mostra.

### Linha por linha

- `def ngramas(texto_periodo, n=2, top_n=10):` → `n` é o tamanho da sequência (padrão 2) e `top_n` quantos resultados mostrar.
- As duas linhas seguintes juntam, normalizam e tokenizam. **Não há remoção de stopwords** aqui, de propósito: expressões como *"i love"* ou *"in the"* só fazem sentido com as palavras funcionais.
- `contagem_ngramas = defaultdict(int)`
  - Cria um dicionário em que, se a chave não existir, ela é criada com valor `0` (`int()` devolve 0). Isso evita verificar "essa chave já existe?" antes de somar.

### A janela deslizante

```python
for i in range(len(palavras) - n + 1):
    ngrama = ' '.join(palavras[i : i + n])
    contagem_ngramas[ngrama] += 1
```

- `range(x)` gera os números `0, 1, 2, ..., x-1`.
- `len(palavras) - n + 1` é o número de janelas possíveis. Garante que a janela **não ultrapasse o fim** da lista.
- `palavras[i : i + n]` é um **fatiamento (slice)**: pega os itens da posição `i` até `i + n` (o último **não** é incluído).
- `' '.join(...)` une as palavras da fatia com espaço, formando o n-grama.
- `contagem_ngramas[ngrama] += 1` → soma 1 à contagem desse n-grama. `+= 1` é o mesmo que `x = x + 1`.

**Exemplo** com `['love', 'me', 'do']` e n=2:

```
i = 0 → palavras[0:2] → "love me"
i = 1 → palavras[1:3] → "me do"
```

### Ordenando

```python
sorted(contagem_ngramas.items(), key=lambda item: item[1], reverse=True)[:top_n]
```

- `.items()` devolve os pares `(n-grama, quantidade)`.
- `sorted(...)` **ordena** a coleção.
- `key=lambda item: item[1]` diz **como** ordenar:
  - **`lambda`** cria uma função pequena e sem nome, escrita em uma linha.
  - `item[1]` é o **segundo elemento** do par (a posição começa em 0): a quantidade.
  - Ou seja: "ordene pela quantidade".
- `reverse=True` → do **maior para o menor**.
- `[:top_n]` → fatiamento que pega só os `top_n` primeiros.

> 💡 **Detalhe:** como todas as letras do período são unidas em um único texto, um n-grama pode **atravessar a fronteira entre duas músicas** (fim de uma + início da próxima). É um pequeno ruído estatístico.

---

## 15. Mostrando os bigramas na tela

```python
bigramas_por_periodo = dataset_filtrado.groupby('period')['lyrics'].apply(
    lambda texto: ngramas(texto, n=2, top_n=5)
)

print('--- TOP 5 BIGRAMAS (2 PALAVRAS) POR PERÍODO ---')
for periodo, lista in bigramas_por_periodo.items():
    print(f'\n{periodo}:')
    for ngrama, qtd in lista:
        print(f'  "{ngrama}" -> {qtd} vezes')
```

- `lambda texto: ngramas(texto, n=2, top_n=5)`
  - O `apply` só passa **um argumento** para a função. Como `ngramas` precisa de outros parâmetros (`n` e `top_n`), usamos uma `lambda` que recebe o `texto` e chama `ngramas` já com `n=2` e `top_n=5`.
- `print('--- TOP 5 ... ---')` → imprime um título.
- `for periodo, lista in bigramas_por_periodo.items():`
  - `.items()` em uma Series devolve pares `(índice, valor)`, ou seja, `(nome do período, lista de bigramas)`.
  - Escrever **duas variáveis** no `for` desmonta cada par automaticamente. Isso se chama **desempacotamento**.
- `print(f'\n{periodo}:')`
  - O `f` antes das aspas cria uma **f-string**: tudo entre `{ }` é substituído pelo valor da variável.
  - `\n` é uma **quebra de linha** (deixa uma linha em branco antes).
- `for ngrama, qtd in lista:` → percorre cada tupla `(n-grama, quantidade)` da lista, desempacotando nas variáveis `ngrama` e `qtd`.
- `print(f'  "{ngrama}" -> {qtd} vezes')` → mostra algo como `  "i love" -> 14 vezes`. Os dois espaços iniciais criam um recuo visual.

---

## 16. Mostrando os trigramas

```python
trigramas_por_periodo = dataset_filtrado.groupby('period')['lyrics'].apply(
    lambda texto: ngramas(texto, n=3, top_n=5)
)

print('\n--- TOP 5 TRIGRAMAS (3 PALAVRAS) POR PERÍODO ---')
for periodo, lista in trigramas_por_periodo.items():
    print(f'\n{periodo}:')
    for ngrama, qtd in lista:
        print(f'  "{ngrama}" -> {qtd} vezes')
```

É **exatamente o mesmo código** da seção 15, só mudando `n=2` para `n=3`. Isso mostra o benefício de ter escrito uma **função reutilizável**: a mesma função gera bigramas, trigramas ou qualquer tamanho, apenas trocando um parâmetro.

---

## 17. Mapa do programa inteiro

```
Ler CSV  →  Selecionar colunas  →  Limpar o ano  →  Criar 'period'
                                                         │
        ┌────────────────────────────────────────────────┤
        ▼                                                ▼
 Contar palavras por música                    Agrupar letras por período
 (média por fase)                                        │
                                  ┌──────────────┬───────┴───────┬──────────────┐
                                  ▼              ▼               ▼              ▼
                           Palavras únicas   Mais frequentes   Verbos/subst./   N-gramas
                           (com/sem stop)    (sem stopwords)   adjetivos (POS)  (2 e 3)
```

## 18. Resumo dos conceitos

| Conceito | Onde aparece no código |
|---|---|
| DataFrame / Series | `dataset`, `dataset_filtrado`, colunas |
| Dados ausentes (NaN) | `to_numeric`, `fillna`, `dropna` |
| Feature engineering | coluna `period` |
| Tokenização | `re.findall(r'\w+', ...)` |
| Normalização | `.lower()` |
| Stopwords | `STOPWORDS` |
| Frequência | `Counter`, `most_common` |
| Riqueza lexical | `set(...)` de palavras únicas |
| POS tagging | `nlp(...)`, `token.pos_` |
| N-gramas | função `ngramas` |
| Split-apply-combine | `groupby(...).apply(...)` |

## 19. Resumo dos comandos Python usados

| Comando | Para que serve |
|---|---|
| `def` | Cria uma função |
| `return` | Devolve o resultado de uma função |
| `if / elif / else` | Toma decisões |
| `for ... in ...` | Repete para cada item |
| `continue` | Pula para o próximo item do laço |
| `len()` | Conta itens |
| `set()` | Remove repetidos |
| `sorted()` | Ordena |
| `lambda` | Função curta sem nome |
| `print()` | Mostra na tela |
| `isinstance()` | Verifica o tipo de um valor |
| `' '.join()` | Une textos |
| `f'...{x}...'` | Insere variáveis em texto |
| `[a:b]` | Fatiamento (pega uma parte) |

