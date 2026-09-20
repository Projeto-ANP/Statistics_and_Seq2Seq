# Entendendo as métricas — como o sistema mede, ranqueia e reporta

Terceiro documento da série (depois de `ENTENDENDO_A_PROPOSTA.md` e
`ENTENDENDO_O_AGENTE.md`). Este cobre a parte que costuma ser lida com mais
rigor por uma banca: **de onde vem cada número, o que ele mede, e para que
serve.**

Escrito assumindo que quem lê vai perguntar "isso é a métrica final, ou é uma
métrica interna?" em cada seção — porque essa é a confusão mais fácil de cometer
neste sistema, e a mais importante de evitar. Todos os números são reais,
extraídos e conferidos manualmente (a fórmula do score foi recalculada à mão e
bateu com o valor gravado, mostrado na seção 4).

---

## 1. O ponto que precisa ficar claro antes de tudo

Existem **dois sistemas de avaliação diferentes** rodando neste projeto, e eles
não podem ser confundidos:

| | **score interno** | **métricas reportadas** |
|---|---|---|
| para que serve | decidir qual estratégia é a melhor, durante a exploração | o número que vai para a tabela de resultados |
| calculado sobre | as 3 janelas de **validação** | a janela de **teste**, uma vez, no fim |
| fórmula | uma combinação normalizada de 4 métricas contra uma âncora | as 6 métricas cruas, sem normalização |
| onde aparece | dentro do histórico de tentativas, nunca é "o resultado" | as colunas `mape`, `smape`, `rmse`, `pocid`, `msmape`, `mae` do CSV |
| exemplo real (série 1, ANP) | `score = 0.4765` | `smape = 0.1566`, `rmse = 2870.80` |

Repare que são números em escalas completamente diferentes, sobre dados
diferentes. **O score de 0.4765 nunca aparece em nenhuma tabela de resultado do
artigo.** Ele existe só para o sistema comparar tentativas entre si durante a
decisão. Quem decide "este método teve sMAPE de 15,66% nesta série" é outro
cálculo, feito depois, sobre dados que a decisão nunca viu.

O resto deste documento explica os dois, na ordem em que acontecem.

---

## 2. As seis métricas de base

Toda avaliação de previsão neste projeto — seja para ranquear uma tentativa, seja
para o resultado final — usa combinações das mesmas seis métricas.

### MAPE — erro percentual absoluto médio

```
MAPE = média( |previsto - real| / |real| )
```

Erro como fração do valor real. Um MAPE de 0.20 significa "errei 20%, em média".
Problema conhecido: quando o valor real está perto de zero, o denominador
explode e o MAPE se torna instável — este projeto detecta esses casos e marca a
linha do CSV (`test_has_zero_actual`), em vez de deixar o número mentir
silenciosamente.

### SMAPE — MAPE simétrico

```
SMAPE = média( 2 × |previsto - real| / (|previsto| + |real|) )
```

Varia entre 0 e 2 (às vezes reportado como 0% a 200%). A diferença para o MAPE é
o denominador: em vez de usar só o valor real, usa a média entre previsto e
real, o que evita parte da instabilidade do MAPE. **É a métrica mais citada nas
comparações deste projeto** — praticamente todo número que aparece nos
documentos anteriores ("FFORMA 0.2166", "ADE 0.1178") é sMAPE, porque ela é
comparável entre séries de escalas muito diferentes (litros de combustível,
graus de temperatura, contagem de pessoas), e é o padrão de facto na literatura
de competições de previsão (M4, M5).

### RMSE — raiz do erro quadrático médio

```
RMSE = raiz( média( (previsto - real)² ) )
```

Está na mesma unidade da série original (litros, no caso do ANP). Ao quadrado
antes de tirar a média, então **erros grandes pesam desproporcionalmente mais**
que erros pequenos — uma previsão que erra por 10.000 litros custa 100× mais no
RMSE que uma que erra por 1.000, não 10× mais.

### MSMAPE — SMAPE modificado

```
MSMAPE = média( 2 × |previsto - real| / max(0.5 + ε, |previsto| + |real| + ε) )
```

Variante do SMAPE com um piso no denominador (`ε = 0.1`), para não explodir
quando previsto e real estão os dois perto de zero ao mesmo tempo.

### MAE — erro absoluto médio

```
MAE = média( |previsto - real| )
```

Como o RMSE, mas sem elevar ao quadrado — todo erro pesa proporcionalmente ao
seu tamanho, sem penalizar erros grandes de forma desproporcional.

### POCID — percentual de acerto de direção

```
POCID = 100 × (quantas vezes o sinal da variação prevista bateu com o sinal da
               variação real) / (número de passos - 1)
```

Diferente das outras cinco: não mede "quão perto" a previsão chegou do valor,
mede se ela **acertou a direção** (subiu quando devia subir, caiu quando devia
cair). Um modelo pode ter erro numérico pequeno e ainda errar a direção; POCID
captura isso, as outras cinco não.

---

## 3. O score interno — o que decide quem vence durante a exploração

### Por que não usar as métricas cruas direto

Se você somasse RMSE (na casa dos milhares) com SMAPE (entre 0 e 2) sem ajustar
nada, o RMSE dominaria completamente a soma — a diferença de escala é grande
demais. É preciso colocar tudo na mesma régua antes de combinar.

### A solução: razão contra uma âncora

A âncora é **a média simples de todos os 19 modelos**, calculada com o mesmo
protocolo de validação de qualquer outra estratégia. Cada métrica de uma
tentativa vira uma razão: "quantas vezes melhor (ou pior) que a média simples".

```python
score = peso_rmse  × (RMSE_da_tentativa  / RMSE_da_âncora)
      + peso_smape × (SMAPE_da_tentativa / SMAPE_da_âncora)
      + peso_mape  × (MAPE_da_tentativa  / MAPE_da_âncora)
      − peso_pocid × (POCID_da_tentativa / 100)
```

Repare no sinal: os três primeiros termos são **razões de erro** (menor é
melhor, então entram somando); o POCID é **taxa de acerto** (maior é melhor,
então entra subtraindo). Isso é o que faz o score inteiro seguir a convenção
"menor é melhor" de ponta a ponta.

### Os pesos — cinco combinações possíveis

| preset | RMSE | SMAPE | MAPE | POCID | quando usar |
|---|---|---|---|---|---|
| `balanced` (padrão) | 0.3 | 0.3 | 0.2 | 0.2 | uso geral |
| `rmse_focus` | 0.5 | 0.2 | 0.2 | 0.1 | penaliza mais erros grandes |
| `direction_focus` | 0.25 | 0.25 | 0.1 | 0.4 | prioriza acertar a direção |
| `robust_smape` | 0.2 | 0.5 | 0.1 | 0.2 | menos sensível a outliers |
| `scale_free_safe` | 0.7 | 0.1 | 0.0 | 0.2 | séries que cruzam zero (MAPE/SMAPE instáveis ali) |

Todas as execuções deste projeto usaram `balanced`.

### Conta feita à mão, com números reais

Série 1 do ANP_MONTHLY. A âncora (média simples, pool completo):

```
RMSE = 3477.3262    SMAPE = 0.2374    MAPE = 0.2354    POCID = 57.6
```

A tentativa `a5` (média aparada dos 5 modelos mais estáveis):

```
RMSE = 3169.1721    SMAPE = 0.2156    MAPE = 0.2148    POCID = 57.6
```

Calculando as razões:

```
razão RMSE  = 3169.1721 / 3477.3262 = 0.9114
razão SMAPE = 0.2156    / 0.2374    = 0.9079
razão MAPE  = 0.2148    / 0.2354    = 0.9126
POCID / 100 = 57.6 / 100            = 0.5758

score = 0.3×0.9114 + 0.3×0.9079 + 0.2×0.9126 − 0.2×0.5758
      = 0.27342 + 0.27237 + 0.18252 − 0.11516
      = 0.6131
```

**O valor gravado pelo sistema para essa tentativa é 0.6131.** Bate exatamente.

Interpretação prática: um score menor que 1.0 significa "melhor que a média
simples"; maior que 1.0 significa "pior que a média simples". A âncora sempre
tem razões = 1.0 em cada termo, então o score dela é uma constante fixa
determinada só pelo POCID (nesta série: `0.3+0.3+0.2 − 0.2×0.576 = 0.6848`).

---

## 4. Como as 3 janelas viram um número só

Ponto que já foi perguntado e vale registrar aqui de forma completa: **as 3
janelas de validação não são votadas — são concatenadas.**

```python
per_window = [métricas(janela 0), métricas(janela 1), métricas(janela 2)]

agg = métricas( todos_os_valores_reais_das_3_janelas_juntos,
                todas_as_previsões_das_3_janelas_juntas )
```

Para a série 1 (12 meses por janela), isso significa que as métricas agregadas
são calculadas sobre um vetor de **36 pontos** (3×12), como se fosse uma
sequência só. O `score` do passo anterior vem desse `agg` — um número por
tentativa, não três.

A única exceção é o POCID, que é calculado janela por janela (porque
"direção" só faz sentido dentro de uma sequência contínua) e depois tirada a
média das três.

Os três valores por janela **também** são guardados (`rmse_per_window` na
observação que o agente vê), mas servem para leitura qualitativa — "essa
estratégia é ótima na janela 3 e ruim nas outras duas" — não para decidir quem
ganha. Quem decide é sempre o `score` agregado.

---

## 5. Como as primeiras estratégias são escolhidas e pontuadas

Ponto de terminologia que vale fixar: **não são "os 10 melhores modelos"**. Os
19 modelos individuais não competem entre si diretamente aqui — o que é
escolhido e pontuado são **estratégias de combinação**, cada uma usando um
subconjunto e uma forma de misturar os modelos.

Antes do agente atuar, entre 9 e 10 estratégias são testadas:

| grupo | quantas | o quê |
|---|---|---|
| clássicas, pool completo | 3 | média, mediana, DBA sobre os 19 modelos |
| por estabilidade | 6 | média e média aparada sobre os k=5, k=7 e k=9 modelos mais consistentes entre as janelas |
| meta-modelo cross-series | 0 ou 1 | pesos aprendidos com as outras séries do dataset (só existe se houver séries suficientes — ver `ENTENDENDO_A_PROPOSTA.md`) |

**Cada uma dessas passa exatamente pelo mesmo cálculo de score da seção 3.** Não
há tratamento especial: a mesma função (`evaluate`) que pontua uma proposta do
agente pontua essas sementes. A única diferença registrada é um campo interno
(`origin`) que marca se veio do sistema (`baseline`) ou do agente (`agent`) —
usado só para análise depois, não para calcular o score de forma diferente.

**Como "k mais consistentes" é decidido:** não é o menor erro médio (isso seria
usar a mesma métrica de validação duas vezes — uma para escolher, outra para
pontuar, o que infla artificialmente o resultado). É a consistência do
**ranking** de cada modelo entre as 3 janelas: um modelo que é 2º, 3º e 4º nas
três janelas é mais "estável" que um que é 1º, 15º e 3º, mesmo que a média de
erro dos dois seja parecida.

---

## 6. Por que a ordem de cálculo respeita o protocolo anti-vazamento

Um score só significa algo se ele não colou na prova. Duas regras garantem
isso, e afetam diretamente os números da seção 3:

**Ajuste de pesos** segue o modo `expanding`: ao pontuar a janela 2, os pesos só
podem ter sido calculados usando as janelas 0 e 1 — nunca a própria janela 2.

**Escolha de quais modelos entram no pool** segue *leave-one-out*: ao pontuar a
janela 2, o pool "mais estável" é recalculado usando só as janelas 0 e 1 (não
todas as três, e não incluindo a 2).

Sem essas duas regras, o score de uma estratégia estaria artificialmente
otimista, porque ela teria "visto" parte do que está sendo usado para
avaliá-la.

---

## 7. Ranking — como o histórico decide quem está na frente

Toda tentativa pontuada entra numa lista. A lista é ordenada só pelo `score`,
do menor para o maior:

```python
ranked_attempts() = sorted(todas_as_tentativas, key=score)
best_attempt()    = ranked_attempts()[0]
```

Não existe empate técnico automático nem margem mínima para trocar de líder —
qualquer score menor, mesmo que por 0.0001, assume a liderança. (A pergunta "essa
diferença é real ou é ruído?" é respondida por outro mecanismo, explicado na
seção 8 — o ranking em si é cru.)

**Um detalhe que evita um viés:** se duas estratégias produzem exatamente a
mesma previsão numérica (por exemplo, pesos ajustados tão perto do uniforme que
o resultado é idêntico à média simples), o sistema detecta isso comparando os
resíduos das duas. Isso não muda o ranking, mas é usado para não comparar uma
estratégia com uma cópia disfarçada dela mesma na análise de confiança — ver a
seção seguinte.

---

## 8. A camada de confiança — o score sozinho não basta

Com apenas 3 janelas, a diferença entre a 1ª e a 2ª colocada pode ser sorte.
Este projeto não confia no ranking cru para afirmar "esta estratégia é
melhor" — ele testa estatisticamente.

### O que é comparado

A vencedora contra a **primeira concorrente genuinamente diferente** (pulando
qualquer uma que seja uma cópia numérica da vencedora, pelo motivo da seção 7).

### Dois testes, por motivos diferentes

**Diebold-Mariano** — compara os 36 resíduos (3 janelas × 12 passos) das duas
estratégias, com uma correção estatística para amostra pequena (Harvey,
Leybourne & Newbold, 1997). É o teste principal aqui porque 36 pontos, embora
poucos, ainda sustentam alguma coisa.

**Bootstrap pareado** — reamostra os 3 scores por janela repetidamente. Com
apenas 3 valores originais, esse teste é tratado como contexto, não como prova:
reamostrar 3 números produz muito poucas combinações possíveis, então o
resultado é grosseiro. Só quando há 5 ou mais janelas ele conta como voto
decisivo.

### O veredito

```
margem              diferença relativa de score entre 1º e 2º colocado
bootstrap_pvalue     resultado do bootstrap
dm_pvalue            resultado do Diebold-Mariano
verdict:
   "separated"           os testes considerados rejeitam a hipótese de empate
   "indistinguishable"   nenhum teste rejeita — a diferença é ruído
   "weak"                os testes discordam entre si
```

Medido nas execuções reais: entre 55% e 66% das séries (dependendo do dataset)
recebem `indistinguishable`. Isso não é falha do método — é honestidade sobre o
que 3 janelas conseguem provar. O texto do projeto trata isso como um resultado
em si, não como um problema a esconder.

### Por que isso substituiu a confiança que o próprio agente declarava

Pedia-se ao agente para declarar um número de confiança ao aceitar uma
estratégia. Medido: **94% das vezes ele respondia exatamente 0.9**, em todos os
datasets testados. Uma constante não sustenta nenhuma afirmação, então o
sistema passou a calcular a confiança de forma determinística — o valor que o
agente declara continua sendo gravado, mas não é mais o que qualquer análise
usa como evidência.

---

## 9. Fase final — a métrica que de fato é reportada

Depois que a estratégia vencedora é escolhida (usando só as 3 janelas de
validação, pelo processo acima), ela é aplicada **uma única vez** às previsões
da janela de teste — que até este ponto nunca foi vista por nenhum cálculo.

O cálculo aqui é **diferente** do score da seção 3, de propósito:

```python
def compute_metrics(previsão, valores_reais_do_teste):
    # usa a biblioteca "all_functions" do projeto, para bater byte a byte
    # com o formato histórico de resultado, e o MAPE do sklearn
    return {"mape", "pocid", "smape", "rmse", "msmape", "mae"}
```

Não há normalização contra âncora aqui, não há pesos, não há razão. São as seis
métricas cruas da seção 2, calculadas direto entre a previsão final e o valor
real do teste. **Essas seis colunas do CSV são o resultado do experimento** —
são elas que entram em qualquer tabela comparando este método com FFORMA, ADE,
média simples, etc.

### O contraste final, com a mesma série usada o documento inteiro

```
score de validação que decidiu a vencedora (seção 3, sobre 3 janelas):  0.4765
sMAPE no teste (o número reportado, sobre a janela nunca vista):        0.1566
RMSE no teste (idem):                                                2870.80
```

São três números diferentes, de três cálculos diferentes, sobre dados
diferentes. Nenhuma tabela do artigo deveria citar o "0.4765" — esse número
morre dentro do processo de decisão. O que vai para o artigo é o sMAPE e o RMSE
do teste.

---

## 10. Resumo — qual número está em qual coluna do CSV

| coluna do CSV | o que é | calculado sobre |
|---|---|---|
| `mape`, `pocid`, `smape`, `rmse`, `msmape`, `mae` | **o resultado reportado** | janela de teste, uma vez |
| `description` → validation → score | o score interno que decidiu o ranking | 3 janelas de validação |
| `description` → validation → per_window_rmse | RMSE de cada uma das 3 janelas, separado | 3 janelas de validação |
| `selection_margin`, `selection_dm_pvalue`, `selection_bootstrap_pvalue`, `selection_verdict` | a confiança estatística da escolha | resíduos e scores de validação |
| `accept_confidence` | o número que o próprio agente declarou (não usado como evidência) | — |
| `score_preset` | qual conjunto de pesos foi usado no score interno | — |

---

## 11. Cinco frases para fixar

1. Existem seis métricas cruas (MAPE, SMAPE, RMSE, POCID, MSMAPE, MAE); cinco
   medem "quão perto" e uma (POCID) mede "acertou a direção".
2. Durante a exploração, essas métricas viram **um score só** — uma soma
   ponderada de razões contra a média simples, calculada sobre as 3 janelas de
   validação concatenadas (não votadas).
3. O que é testado antes do agente agir não são "os 10 melhores modelos", são
   9-10 **estratégias de combinação** pré-definidas, pontuadas pelo mesmo cálculo
   que qualquer proposta do agente.
4. Antes de declarar uma vencedora, o sistema testa estatisticamente se a
   diferença para a segunda colocada é real ou ruído — em mais da metade das
   séries, é ruído, e o sistema diz isso.
5. O número que de fato vai para a tabela de resultados é calculado **depois**
   de tudo isso, sobre a janela de teste nunca vista, com uma fórmula diferente
   e sem relação numérica direta com o score que decidiu a escolha.
