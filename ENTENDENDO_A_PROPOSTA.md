# Entendendo a proposta — do dado bruto até a decisão

Explicação para quem nunca viu este projeto. Sem pressupor conhecimento do
código, de previsão de séries temporais ou de aprendizado de máquina.

Todos os números deste texto são reais, extraídos de uma execução do sistema
sobre dados de venda de combustíveis (dataset `ANP_MONTHLY`).

---

## 1. O problema, com uma analogia

Imagine que você quer prever quanto de gasolina será vendido nos próximos 12
meses numa certa região. Você contrata **19 consultores diferentes**. Cada um
usa um método próprio — um olha tendência, outro olha sazonalidade, outro usa
aprendizado de máquina — e cada um te entrega uma previsão dos 12 meses.

Agora você tem 19 previsões diferentes e precisa entregar **uma só**.

Três saídas óbvias, e por que nenhuma resolve:

| ideia | problema |
|---|---|
| tirar a média das 19 | trata o pior consultor igual ao melhor |
| escolher só o melhor consultor | quem é "o melhor" muda de região para região |
| dar mais peso a quem acertou mais no passado | com pouquíssimo histórico, isso vira chute |

Este projeto resolve isso de um jeito específico: **decidir a combinação série
por série**, usando um agente de IA que escolhe a estratégia, mas com todas as
contas feitas por código determinístico.

Este documento explica a parte que acontece **antes** do agente entrar em cena —
que é onde mora a peça mais importante da proposta.

---

## 2. Os dados de entrada

### 2.1 O que é uma série

Uma **série temporal** é uma sequência de medições ao longo do tempo. No nosso
caso, cada série é a venda mensal de um produto numa região.

A série de índice 1 do nosso dataset tem **407 meses** de histórico:

```
primeiros meses: [9388.2, 4634.7, 2431.1, 3259.7, 3492.9, 3207.4, ...]
últimos meses:   [..., 15804.9, 16910.3, 15925.0, 14961.4, 15359.4, 15132.8]
```

### 2.2 O que é um dataset

Um **dataset** é um conjunto de séries do mesmo tipo. O `ANP_MONTHLY` tem
**182 séries** — 182 combinações diferentes de produto e região, cada uma com
seu próprio histórico.

Isso vai importar muito: a proposta depende de ter **várias séries**, não
apenas uma longa.

### 2.3 As previsões já existem

Ponto importante e contraintuitivo: **este sistema não faz previsão.** Os 19
modelos já rodaram antes, separadamente, e já deixaram suas previsões gravadas
em arquivos. O sistema recebe as previsões prontas e decide **como combiná-las**.

Os 19 modelos são:

| tipo | modelos |
|---|---|
| estatísticos clássicos | ARIMA, ETS, THETA |
| aprendizado de máquina | rf (floresta aleatória), catboost |
| ML + transformadas de sinal | CWT_rf, DWT_rf, FT_rf, CWT_catboost, DWT_catboost, FT_catboost |
| ML só sobre a transformada | ONLY_CWT_rf, ONLY_CWT_catboost, ONLY_DWT_rf, ONLY_DWT_catboost, ONLY_FT_rf, ONLY_FT_catboost |
| "ingênuos" (referência simples) | NaiveSeasonal, NaiveMovingAverage |

---

## 3. As janelas — a ideia central de toda a avaliação

### 3.1 O problema de avaliar previsão

Como saber se uma estratégia de combinação é boa? Você precisa comparar a
previsão com o que **de fato aconteceu**. Mas se você usar os mesmos dados para
decidir e para avaliar, está colando na prova.

A solução: dividir o histórico em pedaços com papéis diferentes.

### 3.2 Como fica na prática

Cada série tem várias "janelas" — pedaços de 12 meses que os modelos já
previram. O sistema usa **apenas as 4 últimas**:

```
validação 1    2020-12-31 a 2021-11-30   (12 meses)
               real       = [12505.7, 12721.1, 8702.6, 5941.3, ...]
               ARIMA prev = [12974.0, 12913.2, 6695.9, 7014.7, ...]

validação 2    2021-12-31 a 2022-11-30
               real       = [17195.5, 20469.1, 16387.2, 16755.4, ...]
               ARIMA prev = [15622.9, 15285.2, 13069.5, 10810.1, ...]

validação 3    2022-12-31 a 2023-11-30
               real       = [20031.2, 19713.7, 16773.0, 16425.7, ...]
               ARIMA prev = [17528.6, 18147.6, 16404.2, 15990.1, ...]

TESTE (cega)   2023-12-31 a 2024-11-30
               real       = [17321.5, 18859.1, 15978.7, 17282.1, ...]   ← ESCONDIDO
               ARIMA prev = [16772.5, 17804.0, 16026.9, 16172.5, ...]
```

**A regra de ouro:** as 3 janelas de validação são usadas para **decidir**. A
janela de teste é usada **uma única vez**, no fim, apenas para dar a nota final.
Os valores reais da janela de teste nem sequer são carregados na memória do
sistema durante a decisão — não é disciplina, é impossibilidade estrutural.

### 3.3 A restrição que molda tudo

Repare: são só **3 janelas** para decidir. Isso é muito pouco. Qualquer método
que tente "aprender" pesos olhando apenas essas 3 janelas vai acabar decorando
ruído em vez de aprender padrão.

É exatamente esse o problema que a peça central da proposta resolve.

---

## 4. Etapa 0 — a faxina: modelos desalinhados

Antes de qualquer coisa, uma verificação de sanidade.

Todos os 19 modelos deveriam concordar sobre **qual é a realidade** — eles
preveem coisas diferentes, mas o valor que de fato aconteceu é um só. Em alguns
datasets isso não acontece. Exemplo real, do dataset `ETTM1`:

```
ARIMA              início 2018-06-26 08:00, medições de 30 em 30 min
                   realidade = [2.88, -0.234, -4.856, -6.162, ...]

ETS                início 2018-06-26 08:00, de 30 em 30 min
                   realidade = [2.88, -0.234, -4.856, -6.162, ...]     ← concorda

ONLY_DWT_rf        início 2018-06-26 14:00, de 15 em 15 min
                   realidade = [-1.273, 0.0, -0.335, 0.335, ...]       ← DISCORDA
```

Os dois últimos foram gerados sobre **outro pedaço da série**, com outra
resolução. Combinar as previsões deles com as dos outros seria somar coisas de
períodos diferentes — e comparar o resultado com apenas um dos períodos.

O sistema detecta isso comparando os valores reais de cada modelo com um modelo
de referência, e **remove os divergentes** do conjunto. No `ETTM1`, 5 dos 19
modelos saem.

---

## 5. Etapa 1 — descrever cada série em 26 números

Agora começa a construção da peça central.

### 5.1 Por que resumir a série

Queremos ensinar um modelo a reconhecer que "séries com este formato costumam
ser bem previstas pelo THETA". Para isso, precisamos de uma forma compacta de
descrever o formato de uma série — não podemos jogar 407 números crus.

A solução é calcular **26 características** que resumem o comportamento da
série. Estes são os valores reais para a série 1:

**As 4 principais** (as mais fáceis de entender):

| característica | valor | o que significa |
|---|---|---|
| `trend_strength` | 0.9098 | força da tendência, de 0 a 1. Alto = a série sobe (ou desce) de forma consistente |
| `seasonal_strength` | 0.4120 | força da sazonalidade, de 0 a 1. Alto = repete um padrão todo ano |
| `spectral_entropy` | 0.3697 | quão "bagunçada" é a série, de 0 a 1. Baixo = comportamento previsível |
| `acf1` | 0.9444 | quanto o mês atual se parece com o anterior. Alto = mudanças suaves |

Traduzindo: *série com tendência muito forte, sazonalidade moderada, pouco
ruidosa, e que muda de forma suave.*

**As outras 22** vêm de um conjunto padronizado da literatura chamado
**catch22** — 22 medidas estatísticas escolhidas por pesquisadores para capturar
comportamentos variados de séries temporais. Elas têm nomes técnicos e não são
legíveis para humanos:

```
DN_HistogramMode_5                  = -0.5465
CO_f1ecac                           = 63.8676
SB_BinaryStats_mean_longstretch1    = 116.0000
PD_PeriodicityWang_th0_01           = 11.0000
... (mais 18)
```

Você não precisa entender cada uma. O ponto é: **cada série vira uma lista de
26 números.** É a "impressão digital" daquela série.

### 5.2 Detalhe importante

Essas 26 características descrevem **a série em si** — o comportamento das
vendas. Nenhuma delas fala sobre os modelos ou sobre os erros deles. Isso vai
ser relevante daqui a pouco.

---

## 6. Etapa 2 — medir quanto cada modelo errou

Agora a outra metade. Para cada série, medimos o erro de cada um dos 19 modelos
**nas 3 janelas de validação**.

A medida de erro usada chama-se **sMAPE**. Pense nela como "erro percentual":
0 é perfeito, e quanto maior, pior. Ela é usada em vez do erro absoluto porque
funciona igual em séries de escalas diferentes — uma série que vende 20 mil
litros e outra que vende 200 podem ser comparadas.

Erros reais dos 19 modelos na série 1, do melhor para o pior:

```
THETA                  0.1477   ← melhor nesta série
ETS                    0.1500
FT_rf                  0.1969
ONLY_DWT_rf            0.2227
ARIMA                  0.2308
rf                     0.2378
DWT_rf                 0.2379
ONLY_DWT_catboost      0.2470
FT_catboost            0.2507
ONLY_FT_rf             0.2595
ONLY_CWT_rf            0.2692
CWT_rf                 0.2693
ONLY_FT_catboost       0.2809
DWT_catboost           0.2836
ONLY_CWT_catboost      0.2905
CWT_catboost           0.3198
NaiveMovingAverage     0.3245
catboost               0.3632
NaiveSeasonal          0.4375   ← pior nesta série
```

Note que os 3 valores das 3 janelas viram **um número só** por modelo (é a média
sobre as janelas e sobre os 12 meses). Então cada série produz **19 números** de
erro.

---

## 7. Etapa 3 — montando a planilha de treino

Juntando as duas metades, cada série vira **uma linha** de uma planilha:

```
        ←── ENTRADA: 26 características ──→   ←── ALVO: 19 erros ──→
série   trend   seasonal  entropia   acf1  |  ARIMA    ETS   THETA     rf  NaiveSeas
──────────────────────────────────────────────────────────────────────────────────
  0    0.8481    0.0633    0.4906  0.9395  | 0.3298 0.4170  0.4525 0.3154   0.4922
  1    0.9098    0.4120    0.3697  0.9444  | 0.2308 0.1500  0.1477 0.2378   0.4375
  2    0.9790    0.4215    0.3154  0.9746  | 0.0636 0.0642  0.0642 0.0758   0.0864
  3    0.7913    0.2979    0.5165  0.8462  | 0.2473 0.2068  0.2266 0.2763   0.3098
```

(mostrando 4 das 26 características e 5 dos 19 erros, por espaço — a planilha
real do ANP tem **182 linhas × 26 colunas** de entrada e **182 × 19** de alvo)

### Dá para ler o padrão a olho nu

- **Série 2**: tendência altíssima (0.979), série muito comportada → *todos* os
  modelos erram pouco (~0.06). Série fácil.
- **Série 0**: sazonalidade quase zero (0.063) → `rf` (0.3154) e `ARIMA` (0.3298)
  vão melhor que `ETS` (0.4170) e `THETA` (0.4525).
- **Série 1**: sazonalidade moderada (0.412) → o oposto: `THETA` (0.1477) e
  `ETS` (0.1500) ganham de `rf` (0.2378).

É exatamente essa associação — **formato da série → quem erra menos** — que
queremos ensinar a uma máquina.

---

## 8. Etapa 4 — a regra do "deixa uma de fora"

Aqui entra uma sutileza que é fácil de errar e cara de descobrir depois.

### O problema

Se treinarmos um modelo com as 182 linhas e depois perguntarmos a ele sobre a
série 1, ele vai acertar — porque a série 1 estava no treino. Ele **decorou** a
resposta. Isso não prova nada sobre séries novas.

### A solução: leave-one-series-out

Treinamos **um modelo separado para cada série**, sempre removendo aquela série
das linhas de treino:

```
para responder sobre a série 0  →  treina com as linhas 1, 2, 3, ..., 181
para responder sobre a série 1  →  treina com as linhas 0, 2, 3, ..., 181
para responder sobre a série 2  →  treina com as linhas 0, 1, 3, ..., 181
...
```

São **182 modelos**, cada um treinado com 181 linhas. Cada um é cego para a
série que vai responder.

### Demonstração real

Rodando com 30 séries (para caber na tela), o resultado é um dicionário:

```
30 entradas, uma por série:
  modelo[0]  → treinado com 29 linhas (a série 0 ficou de fora)
  modelo[1]  → treinado com 29 linhas (a série 1 ficou de fora)
  modelo[2]  → treinado com 29 linhas (a série 2 ficou de fora)
```

E a prova de que são de fato **modelos diferentes** — alimentando os três com
exatamente a mesma entrada (as características da série 0):

```
modelo[0] → recomenda: ONLY_DWT_catboost, NaiveSeasonal, FT_catboost
modelo[1] → recomenda: catboost, FT_catboost, NaiveSeasonal
modelo[2] → recomenda: FT_catboost, NaiveSeasonal, catboost
```

Mesma pergunta, respostas diferentes — porque aprenderam com conjuntos
diferentes. E apenas o `modelo[0]` pode ser usado para a série 0; os outros dois
viram a série 0 no treino e estariam colando.

---

## 9. Etapa 5 — o que o modelo aprende, exatamente

O algoritmo usado chama-se **gradient boosting** (biblioteca XGBoost). Pense nele
como uma coleção de regras do tipo "se a tendência é maior que 0.9 e a
sazonalidade é menor que 0.2, então...", construídas automaticamente e
empilhadas até acertarem bem.

Mas há uma diferença crucial em relação ao uso comum desse algoritmo.

### O jeito óbvio (que NÃO é o usado)

O jeito natural seria: treinar 19 modelos separados, cada um prevendo o erro de
um modelo. "Quanto o ARIMA vai errar nesta série? 0.23." Depois, dar mais peso a
quem tem erro previsto menor.

Isso funciona, está implementado, mas **não é o padrão** — porque mede pior.

### O jeito usado

Um **único** modelo que, para cada série, produz 19 números de uma vez. Esses 19
números passam por uma função chamada **softmax**, que os transforma em
proporções que somam 100%:

```
19 números do modelo  →  [softmax]  →  19 pesos que somam 1
```

E o treino é feito de forma que **esses pesos, aplicados às 19 previsões,
produzam a menor combinação de erro possível**.

A diferença é sutil mas decisiva: o modelo não aprende "quanto cada consultor
erra isoladamente" — ele aprende "**qual mistura de consultores funciona
junto**". Ele pode descobrir que dois modelos medianos se complementam e, juntos,
batem o melhor modelo sozinho. Um modelo que prevê erros isolados nunca
enxergaria isso.

Essa forma de treinar vem do método **FFORMA** (Montero-Manso et al., 2020), uma
referência da área que este projeto usa como base de comparação.

---

## 10. Etapa 6 — usando o modelo treinado

Terminado o treino, o uso é direto. Para a série 1:

**Entrada** — as 26 características dela:

```
trend_strength     0.9098        acf1              0.9444
seasonal_strength  0.4120        spectral_entropy  0.3697
CO_f1ecac         63.8676        ... (mais 20)
```

**Saída** — depois de consultar o modelo e aplicar o softmax:

```
THETA          57.5%
ETS            32.6%
ONLY_DWT_rf     3.8%
ONLY_CWT_rf     3.4%
FT_catboost     2.8%
(os outros 14 modelos: praticamente 0%)
```

### Por que isso é impressionante

Volte à seção 6 e compare. Os dois modelos que de fato erram menos na série 1
são **THETA (0.1477) e ETS (0.1500)** — e o meta-modelo concentrou **90% do
peso** exatamente neles.

E ele fez isso **sem nunca ter visto os erros desta série**. Ele viu apenas o
formato dela (tendência forte, sazonalidade moderada, pouco ruído) e inferiu,
a partir das outras 181 séries, que séries com esse formato são bem previstas
pelo THETA e pelo ETS.

Esse é o valor da proposta: transferir conhecimento **entre séries**, contornando
a limitação de ter apenas 3 janelas por série.

---

## 11. Etapa 7 — o segundo pré-passo: o "prior do dataset"

Além do meta-modelo, roda uma segunda varredura pelo dataset inteiro, com um
objetivo diferente: descobrir **quais estratégias de combinação costumam
funcionar neste dataset**.

### Como funciona

Existem 10 estratégias "padrão" que o sistema sempre testa antes de qualquer
decisão (média simples, mediana, média dos 5 modelos mais estáveis, etc.).

**Passo 1** — testa as 10 em cada série e anota a nota de validação (menor =
melhor):

```
estratégia                            série0   série1   série2   série3   série4
média simples                         0.6909   0.6848   0.6545   0.6424   0.6606
mediana                               0.7064   0.7115   0.6971   0.6763   0.7185
dba                                   0.6884   0.7013   0.6226   0.6540   0.7568
média dos 5 mais estáveis             0.7739   0.6187   0.8215   0.6631   0.8263
média aparada dos 5 mais estáveis     0.7753   0.6131   0.8642   0.6755   0.7818
média dos 7 mais estáveis             0.7501   0.6667   0.8443   0.6741   0.8103
média aparada dos 7 mais estáveis     0.7413   0.6522   0.8638   0.6774   0.8307
média dos 9 mais estáveis             0.7356   0.6986   0.7621   0.6644   0.7639
média aparada dos 9 mais estáveis     0.7220   0.6929   0.7907   0.6665   0.7810
```

**Passo 2** — para cada série, calcula a média das **outras** colunas (a mesma
lógica de "deixa uma de fora"):

```
prior da série 0 = média das colunas série1, série2, série3, série4

estratégia                            série1   série2   série3   série4  →  prior
média simples                         0.6848   0.6545   0.6424   0.6606  →  0.6606
mediana                               0.7115   0.6971   0.6763   0.7185  →  0.7008
dba                                   0.7013   0.6226   0.6540   0.7568  →  0.6837
média dos 5 mais estáveis             0.6187   0.8215   0.6631   0.8263  →  0.7324
...
```

**Passo 3** — ordena, e isso vira uma "dica" entregue ao agente:

```
1. média simples                      0.6606   ← melhor em média neste dataset
2. dba                                0.6837
3. mediana                            0.7008
...
9. média aparada dos 7 mais estáveis  0.7560   ← pior
```

É por isso que o nome é "prior": é uma **crença prévia** sobre o que costuma
funcionar aqui, formada olhando as outras séries, antes de olhar esta.

---

## 12. Etapa 8 — o que finalmente chega ao agente

Só agora o agente de IA entra. E ele nunca começa do zero — quando lê o primeiro
prompt, já recebe tudo isto pronto:

| o que recebe | de onde veio |
|---|---|
| ficha da série (tendência, sazonalidade, etc.) | Etapa 1 |
| tabela de erro dos 19 modelos, ranking, redundância | calculado na hora |
| **dica do dataset** (quais estratégias funcionam aqui) | Etapa 7 |
| **10 estratégias já testadas e pontuadas** | testadas antes do agente abrir |
| entre elas, a do meta-modelo (`THETA 57.5%, ETS 32.6%...`) | Etapas 5 e 6 |

O agente então tem até 12 rodadas para tentar **melhorar** o que já está na
mesa: pode podar modelos redundantes, testar outra forma de ponderar, tentar uma
mediana em vez de média. Cada tentativa é testada nas 3 janelas de validação.

E há uma garantia estrutural: **o resultado final é sempre a melhor estratégia
de todo o histórico** — incluindo as 10 pré-testadas. Se nada do que o agente
propuser for melhor, o sistema aplica a melhor pré-testada. Ou seja: **o agente
só pode melhorar, nunca piorar**.

---

## 13. A linha do tempo completa

```
┌─ UMA VEZ POR DATASET (sem IA, só cálculo) ───────────────────────────┐
│                                                                       │
│  Etapa 0   remove modelos desalinhados                                │
│                                                                       │
│  Etapa 1-5 PRIMEIRA VARREDURA — o meta-modelo                         │
│            para cada uma das 182 séries:                              │
│              • extrai 26 características                              │
│              • mede o erro dos 19 modelos                             │
│            → monta planilha 182 × (26 + 19)                           │
│            → treina 182 modelos, cada um cego para sua própria série  │
│                                                                       │
│  Etapa 7   SEGUNDA VARREDURA — o prior do dataset                     │
│            testa as 10 estratégias padrão em cada série               │
│            → para cada série, a média das outras 181                  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─ PARA CADA SÉRIE (agora sim com IA) ─────────────────────────────────┐
│                                                                       │
│  • carrega os dados e a ficha da série                                │
│  • testa as 10 estratégias padrão (incluindo a do meta-modelo)        │
│  • entrega tudo ao agente + a dica do dataset                         │
│  • agente tenta melhorar, até 12 rodadas                              │
│  • aplica a MELHOR de todas na janela de teste (a que estava cega)    │
│  • calcula a nota final                                               │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 14. Resumo em cinco frases

1. Dezenove modelos já fizeram suas previsões; o sistema decide **como
   combiná-las**, série por série.
2. Cada série é resumida em **26 números** que descrevem seu formato, e mede-se
   quanto **cada um dos 19 modelos** errou nela.
3. Com isso monta-se uma planilha de 182 linhas e treina-se um modelo que aprende
   a associação **formato da série → melhor mistura de modelos** — sempre
   deixando de fora a série que vai responder.
4. Uma segunda varredura descobre quais estratégias funcionam bem **neste
   dataset** e vira uma dica.
5. O agente de IA recebe tudo isso pronto e tenta melhorar, com a garantia de que
   o resultado final nunca é pior do que o que já estava na mesa.

---

## 15. Por que isso é diferente do que já existe

O método de referência da área, o **FFORMA**, também treina um modelo usando
várias séries e características. A diferença da nossa proposta está no que vem
depois: em vez de aplicar os pesos direto, o resultado do meta-modelo é
apresentado a um **agente**, que pode aceitá-lo, refiná-lo (aplicando-o a um
subconjunto menor de modelos, por exemplo) ou descartá-lo em favor de outra
estratégia — e cada alternativa é testada nas janelas de validação antes de
qualquer decisão.

O outro método de referência, o **ADE**, resolve o problema de forma diferente:
em vez de aprender entre séries, ele aprende ao longo do tempo dentro de cada
série. Por isso funciona mesmo com uma série só — enquanto a nossa proposta e o
FFORMA precisam de um conjunto de séries para treinar (no nosso caso, no mínimo
20).
