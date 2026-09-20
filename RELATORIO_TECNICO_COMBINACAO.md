# Relatório técnico — repensando a combinação de previsões

Revisão densa da arquitetura atual à luz da literatura, com experimentos novos
rodados nos nossos próprios dados (NN5, 111 séries; ANP_MONTHLY, 182 séries).

Todo número marcado 📊 foi medido nesta investigação, não citado. Todo experimento
ajusta apenas nas 3 janelas de validação e toca a janela de teste uma única vez,
no fim, para pontuar.

Data: 2026-07-29.

---

## Sumário executivo

Cinco achados, em ordem de impacto sobre o que devemos fazer:

1. **Qual estratégia de combinação é a melhor NÃO transfere entre datasets.**
   Spearman entre o ranking de 16 estratégias no NN5 e no ANP: **+0.121 (p=0.66)**.
   Nossa escolha atual de sementes (`stable_k`) foi calibrada no NN5 — lá é 1º
   lugar, no ANP é **11º de 16**.

2. **Existe prêmio real na mesa: 8–11%.** O oráculo por série sobre as mesmas 16
   estratégias bate a melhor estratégia fixa em 10.9% (NN5) e 8.4% (ANP).

3. **Mas a melhor estratégia fixa é a vencedora de uma série específica em apenas
   ~3.5% das vezes.** O vencedor por série está espalhado por todas as 16
   estratégias, sem concentração. É por isso que o prêmio existe e é por isso que
   é difícil de capturar.

4. **Otimizar diretamente o erro combinado de validação é a PIOR estratégia de
   todas — nos dois datasets.** Seleção gulosa (adiciona o modelo que mais melhora
   a média combinada) fica em último lugar no NN5 e no ANP. É a demonstração mais
   limpa do sobreajuste de validação que já tínhamos diagnosticado.

5. **A restrição importa mais que o método.** Stacking ridge irrestrito é
   catastrófico (0.249 no ANP vs 0.221 da média simples); o *mesmo* stacking
   projetado no simplex (não-negativo, soma 1) fica competitivo (0.2183, empatado
   com a melhor estratégia encontrada). Não é "aprender pesos" que quebra — é
   aprender pesos sem restrição.

E um achado que muda onde devemos investir:

6. **A saturação de features do STL é específica do NN5, não um bug nosso.**
   No NN5, `seasonal_strength` tem desvio-padrão **0.0001** (constante); no ANP,
   **0.2262**. Toda a família de métodos baseada em features (diagnosticador,
   meta-modelo pooled) está rodando às cegas no NN5 e com sinal real no ANP.

---

## 1. Metodologia dos experimentos

Implementei 8 estratégias de seleção de pool, incluindo três da literatura que não
tínhamos, e testei cada uma com `mean` e `median` — 16 combinações, nos dois
datasets:

| estratégia | origem | como decide |
|---|---|---|
| `full` | baseline | todos os 19 modelos |
| `top5` | já tínhamos | 5 de menor erro |
| `stable5`, `stable7` | já tínhamos | menor `mean rank + std rank` |
| `islands` | **novo** — Kourentzes et al. 2019 | ordena por erro, corta na primeira "queda brusca" detectada por cerca de Tukey nos gaps |
| `greedy_combo` | **novo** | forward greedy: adiciona o modelo que mais reduz o erro da média combinada |
| `div_pen_k5/k7` | **novo** — literatura de acurácia×diversidade | greedy minimizando `erro_normalizado − λ·(1 − corr com os já escolhidos)` |

Mais um experimento separado de **stacking pooled entre séries** (§5).

---

## 2. Achado 1 — a estratégia vencedora não transfere 📊

```
Spearman(ranking no NN5, ranking no ANP), 16 estratégias:  +0.121   p=0.6564
```

| estratégia | pos. NN5 | pos. ANP | sMAPE NN5 | sMAPE ANP |
|---|---|---|---|---|
| `stable5_mean` | **1º** | **11º** | 0.114858 | 0.220891 |
| `stable7_mean` | 2º | 5º | 0.114860 | 0.219783 |
| `islands_median` | **13º** | **1º** | 0.120782 | 0.218109 |
| `top5_median` | 11º | 2º | 0.120287 | 0.218504 |
| `full_median` | 10º | 4º | 0.120134 | 0.219412 |
| `greedy_combo_mean` | 16º | 16º | 0.124157 | 0.225625 |

Duas inversões quase perfeitas: `stable5_mean` (1º→11º) e `islands_median`
(13º→1º).

**Implicação direta para o que já fizemos.** `seed_stable_pools` — que a
ARQUITETURA.md §2.4 registra como "o maior lever medido de todos", com ganho de
0.12036→0.11536 — foi calibrado inteiramente no NN5. No ANP, a mesma família de
sementes está na metade de baixo da tabela. **O ganho documentado é real, mas é
específico do NN5, e o documento não diz isso.** Precisa de ressalva.

Nota sobre a amplitude: no ANP a diferença entre a melhor e a pior das 16
estratégias é de só **3.4%** do valor da melhor (no NN5, 8.4%). Ou seja: no ANP,
escolher a estratégia "certa" importa menos do que no NN5 — o que por si só já é
um resultado sobre heterogeneidade entre datasets.

---

## 3. Achado 2 e 3 — o prêmio existe, mas está pulverizado 📊

```
NN5:  melhor estratégia FIXA 0.114858 | oráculo por série 0.102289 | lacuna 10.9%
ANP:  melhor estratégia FIXA 0.218109 | oráculo por série 0.199850 | lacuna  8.4%
```

Isso justifica a premissa do projeto: escolher por série **vale** 8–11%, se der
para escolher certo.

Mas:

```
com que frequência a melhor estratégia fixa é a vencedora daquela série?
  NN5: 3.6%      ANP: 3.3%
```

O vencedor por série se espalha por praticamente todas as 16 estratégias (no ANP:
`full_mean` 32 séries, `greedy_combo_mean` 28, `stable5_median` 18, `full_median`
13, `top5_median` 12...). **Não existe um "quase sempre certo" para convergir.**

Detalhe revelador: `greedy_combo` é a **pior em média** e ainda assim é a vencedora
em 28 séries do ANP. Ela não é ruim — ela é *volátil*. Isso é exatamente o perfil
de um estimador de alta variância: às vezes acerta na mosca, em média destrói.

E o teto absoluto, para dimensionar: o oráculo sobre **modelos individuais**
(escolher o melhor modelo único por série, sem combinar nada) dá 0.0933 no NN5 e
0.1736 no ANP — **melhor que qualquer combinação testada**. O limite superior do
problema está na *seleção*, não na *combinação*. Isso não é acionável diretamente
(é oráculo), mas define onde o potencial mora.

---

## 4. Achado 4 — otimizar validação diretamente é o pior caminho 📊

`greedy_combo` é a única estratégia que otimiza **exatamente o objetivo**:
adiciona modelos enquanto o erro da média combinada nas janelas de validação cair.
É a coisa mais "óbvia" de fazer. Resultado:

```
NN5: 16º de 16 (0.124157)     ANP: 16º de 16 (0.225625)
tamanho médio do pool escolhido: 2.62 (NN5), 2.16 (ANP)
```

Ela converge para 2–3 modelos porque com 24 pontos de validação sempre existe um
par que casa quase perfeitamente — e esse casamento não sobrevive à janela cega.

Isso fecha uma linha de raciocínio que atravessa todo o projeto: já vimos
`weights_ols` colapsar no vértice do simplex, já vimos o score de validação ficar
anticorrelacionado com o teste antes do `nested_selection`, e agora vemos a busca
gulosa direta ser a pior de todas. **Três mecanismos diferentes, mesma causa: 3
janelas não sustentam otimização livre.**

---

## 5. Achado 5 — a restrição vale mais que o método 📊

A literatura mais recente e de maior escala que encontrei ([Multi-layer Stack
Ensembles, 2025](https://arxiv.org/pdf/2511.15350), 50 datasets / 90K séries)
**contradiz frontalmente** o que assumimos: conclui que stacking aprendido bate
agregação simples de forma decisiva (Elo 1306 vs 1000), e que funciona mesmo com
K=2 folds.

Testei stacking pooled entre séries na nossa base — treinando o stacker nas
predições ponto a ponto de **todas as outras séries**, normalizadas por escala,
leave-one-series-out:

| | NN5 | ANP |
|---|---|---|
| `full_mean` (referência) | **0.119939** | 0.220649 |
| ridge irrestrito (α=0.1) | 0.126915 | **0.248973** ← catástrofe |
| ridge irrestrito (α=10) | 0.126340 | 0.249872 |
| **ridge projetado no simplex** | 0.123707 | **0.218266** ← melhor que a média |

Duas leituras:

**(a) A restrição é o que importa.** No ANP, o mesmo ridge vai de 0.249
(irrestrito) para 0.218 (não-negativo, soma 1) — uma melhora de 12%, só por
restringir. Irrestrito ele produz pesos grandes e de sinal trocado que não
generalizam; no simplex ele vira uma combinação de verdade.

**(b) O benchmark de 90K séries não transfere para 3 janelas.** No NN5 mesmo o
stacker restrito perde da média simples. A reconciliação provável: eles têm séries
longas com muitos folds; nós temos 24 pontos por série. Nosso regime é o caso de
fronteira onde o "puzzle da combinação" ainda vale — e o paper deles não o refuta,
ele mostra que ele desaparece com dados suficientes.

**Isso é material forte para o paper**: podemos citar o benchmark grande e mostrar
empiricamente onde fica a fronteira.

---

## 6. Achado 6 — onde a saturação de features realmente está 📊

Já tínhamos visto duas vezes que features baseadas em STL saturam. Agora medi nos
dois datasets:

| feature | desvio NN5 | desvio ANP | CV ANP |
|---|---|---|---|
| `trend_strength` | 0.023 | 0.151 | 0.18 |
| `seasonal_strength` | **0.0001** | **0.226** | **0.62** |
| `spectral_entropy` | 0.107 | 0.146 | 0.34 |
| `acf1` | 0.195 | 0.148 | 0.17 |

**Não é bug nosso — é característica do NN5.** Séries semanais de caixa eletrônico
têm tendência e sazonalidade quase perfeitas em todas as 111 séries. O ANP, que
mistura produtos e regiões, tem heterogeneidade real.

Consequência prática imediata, e é uma correção de rumo:

- O **meta-modelo pooled** (`weights_pooled_meta_model`) foi validado no NN5, onde
  2 das suas 4 features são constantes. Ele empatou com `softmax_neg_error` ali —
  mas **nunca foi testado no ANP**, onde tem sinal de verdade. Testar isso é a
  próxima ação de maior valor por esforço, e eu recomendaria isso *antes* do
  experimento do diagnosticador.
- O **experimento do diagnosticador** que preparamos está apontado para o NN5.
  Deveria rodar no **ANP**: um LLM lendo "trend=1.0, seasonal=1.0" em 111 séries
  idênticas não tem do que discordar. No ANP ele tem.

---

## 7. Revisão dos prompts

Reli `prompts.py` inteiro. Ele está bem construído — espaço de ação fechado,
handles em vez de números, histórico ranqueado (que é exatamente o formato que o
DCATS recomenda). Três observações concretas:

**7.1 O prompt não diz ao agente o que os dados desta conversa mostram.** A regra
atual sugere "se uma baseline semeada está liderando, tente o mesmo método num
pool melhor". Os dados dizem algo mais específico e mais útil: *pools pequenos
obtidos por critério de erro puro tendem a não generalizar*. Uma regra explícita
sobre isso é engenharia de prompt barata, que é justamente onde Ren & Wang (item 3
do `insights_trabalhos.md`) localizam o ganho em cenários zero-shot.

**7.2 O prompt não expõe o tamanho da amostra.** O agente vê `n_validation_windows: 3`
enterrado na ficha da série, mas nenhuma das regras conecta isso a "portanto pesos
ajustados são de alta variância". Ele descobre isso sozinho às vezes — vimos nas
justificativas ("only 3 windows: weight estimation is high variance") — mas por
acidente, não por instrução.

**7.3 A regra de justificativa está certa e deve ser mantida.** Exigir que o
`accept` explique em termos de características observáveis é exatamente o que
TSOrchestr e Nexus identificam como fonte de ganho, e é replicável sem fine-tuning.

---

## 8. O que a literatura oferece que ainda não testamos

| ideia | fonte | avaliação |
|---|---|---|
| **Forecast islands** (cerca de Tukey nos gaps de erro) | Kourentzes et al. 2019 | ✅ **testei nesta investigação** — 1º no ANP, 13º no NN5 |
| Acurácia × diversidade explícita | revisão de 50 anos; trimming 2022 | ✅ testei — sem ganho consistente (5º–14º) |
| Stacking multi-camada | benchmark 2025, 90K séries | ✅ testei versão de 1 camada pooled — ver §5 |
| Clustering de séries antes de escolher o pool | M4 (Montero-Manso) | ❌ não testado — plausível, e o ANP tem heterogeneidade para sustentar |
| Combinação por intervalos de predição | Treating & Pruning (2020) | ❌ não aplicável: nosso pool não fornece intervalos |
| Dirichlet / combinações infinitas | 2023 | ❌ exige muito mais dados de validação |

---

## 9. Recomendações, em ordem de valor por esforço

**(1) Corrigir a ressalva na ARQUITETURA.md sobre `seed_stable_pools`.** O ganho de
0.0050 está documentado como geral e é específico do NN5 (11º de 16 no ANP). É a
correção mais barata e a mais importante para a honestidade do paper.

**(2) Testar o meta-modelo pooled no ANP.** Já está implementado e testado; só
nunca rodou onde as features têm variância. Custo: uma rodada.

**(3) Redirecionar o experimento do diagnosticador para o ANP** pelo mesmo motivo.

**(4) Adicionar `select_islands` ao catálogo.** É a única estratégia nova da
literatura que ficou em 1º em algum dataset, é determinística e barata (ordena
erros, aplica cerca de Tukey). Dá ao agente uma opção que hoje ele não tem — e o
fato de ser 1ª no ANP e 13ª no NN5 é precisamente o tipo de escolha que um agente
por série poderia acertar e uma regra fixa não.

**(5) Adicionar a projeção no simplex ao stacking pooled** se formos investir
nessa linha: os dados mostram que a restrição vale 12% no ANP.

**(6) NÃO investir em:** busca gulosa direta, OLS irrestrito, ou qualquer método
que otimize livremente o erro de validação. Três experimentos independentes agora
mostram que isso é o caminho mais rápido para o sobreajuste no nosso regime de 3
janelas.

---

## 10. A questão de fundo, para o paper

A arquitetura assume que existe uma melhor estratégia por série e que a validação
consegue encontrá-la. Esta investigação sustenta a primeira metade e complica a
segunda:

- **existe** ganho por série (8–11% de oráculo),
- **mas** o vencedor está pulverizado (3.5% de acerto para a melhor regra fixa),
- **e** o sinal de validação de 3 janelas ordena estratégias mal (Spearman +0.55
  no NN5 com aninhamento, −0.29 no ANP),
- **e** qual estratégia é melhor nem sequer transfere entre datasets (+0.12).

A conclusão honesta não é "o agente falhou". É que **o problema de escolher a
combinação por série, com 3 janelas de validação, é próximo do limite do que é
estatisticamente decidível** — e isso é, em si, um resultado publicável, sustentado
por medições nossas em dois datasets independentes e coerente com o puzzle da
combinação de previsões. O valor da arquitetura de agente passa a ser tanto o
**instrumento de medição** (procedência auditável, protocolo anti-vazamento,
veredito de confiança calibrado) quanto o combinador em si.

---

## Referências

- [Forecast combinations: an over 50-year review](https://arxiv.org/pdf/2205.04216) — Wang, Hyndman et al.
- [Another look at forecast trimming for combinations: robustness, accuracy and diversity](https://arxiv.org/pdf/2208.00139)
- [Kourentzes — Another look at forecast selection and combination](https://kourentzes.com/forecasting/wp-content/uploads/2018/06/Kourentzes-2018-forecast-selection-combination.pdf) — forecast islands
- [Multi-layer Stack Ensembles for Time Series Forecasting](https://arxiv.org/pdf/2511.15350) — benchmark de 33 métodos, 90K séries
- [Optimizing accuracy and diversity: a multi-task approach to forecast combinations](https://arxiv.org/html/2310.20545v2)
- [Solving the Forecast Combination Puzzle](https://arxiv.org/pdf/2308.05263)
- [A study on Ensemble Learning for Time Series Forecasting and the need for Meta-Learning](https://arxiv.org/pdf/2104.11475) — FFORMA
- [M5 accuracy competition: Results, findings, and conclusions](https://www.sciencedirect.com/science/article/pii/S0169207021001874)
- [The M4 forecasting competition – A practitioner's view](https://www.sciencedirect.com/science/article/pii/S0169207019301189)
- [A Survey of Reasoning and Agentic Systems in Time Series with LLMs](https://arxiv.org/pdf/2509.11575)
- [Efficient Model Selection for Time Series Forecasting via LLMs](https://arxiv.org/html/2504.02119)
