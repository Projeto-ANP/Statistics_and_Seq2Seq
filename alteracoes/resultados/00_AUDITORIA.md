# Auditoria de `overleaf/resultados.tex`

Arquivo auditado: `overleaf/resultados.tex` (1421 linhas).
Referências principais: `alteracoes/resultados/*.md`, `outputs/resultados/` e os artefatos da execução `orchestrator_react_consist_seed11`.

## Resumo

- As 38 tabelas do capítulo conferem com os arquivos `.md` de origem.
- As médias, desvios-padrão, diferenças e contagens das seções por conjunto de dados, do ranking, da MCM de SMAPE, da contagem de vitórias e da ablação estão corretas.
- Há sete problemas críticos: figuras de POCID desatualizadas, texto da MCM agregada de POCID, uso de "estratégia final" no lugar de "líder", tabelas de contrafactual com a coluna CARTE inconsistente, rastreabilidade com 165 séries de uma execução antiga, ação de meta-modelo fora do catálogo e figuras de POCID sem referência no texto.
- Há problemas médios de custo, intervalos de confiança, nomenclatura da ablação, escopo do roadmap e uma nuance na ETTh2.
- A subseção `Uso do contexto na escolha das ações` (`18_agente_contexto.md`) não está no arquivo.

## Quadro por seção

| Seção | Status | Problema principal |
|---|---|---|
| Considerações iniciais | Ajustar | Roadmap fora de ordem e incompleto |
| Design experimental | OK | — |
| Escolha do modelo de linguagem | Ajustar | Gramática e esclarecimento do custo |
| Execuções e agregação | Ajustar | "dessa série" e custo |
| ANP | Substituir figura | MCM de POCID antiga |
| NN5 | Substituir figura | MCM de POCID antiga |
| M4 | Substituir figura | MCM de POCID antiga |
| ETTh1 | OK | Falta apenas referenciar a figura de POCID |
| ETTh2 | Substituir figura | MCM de POCID antiga e "treze modelos" |
| ETTm1 | Substituir figura | MCM de POCID antiga |
| ETTm2 | Substituir figura | MCM de POCID antiga |
| Síntese | Corrigir números | POCID agregado e contrafactual |
| Análise do agente | Corrigir | Líder × estratégia final, meta-modelo, rastreabilidade |
| Ablação | Ajustar | "Melhores 10" × "10 primeiras" |
| Considerações finais | Corrigir números | Contrafactual e meta-modelo |

## 1. Considerações iniciais

Linhas 3–11.

O texto está correto, mas o roadmap está fora de ordem e incompleto:

- a comparação entre modelos de linguagem é a Seção 3 e acontece antes dos resultados por conjunto, mas o roadmap a menciona no fim;
- o estudo de ablação e as considerações finais não são mencionados;
- o roadmap diz "Por fim, são apresentadas uma síntese ... e uma análise do comportamento do agente", mas o capítulo continua com a ablação e as considerações finais.

Sugestão:

> Inicialmente, são apresentados a configuração experimental, a escolha do modelo de linguagem e os critérios de avaliação. Em seguida, os resultados são discutidos em uma seção para cada conjunto de dados, contemplando o desempenho dos modelos individuais, das estratégias de combinação e a comparação geral entre os métodos. Por fim, são apresentadas uma síntese dos resultados entre os conjuntos, uma análise do comportamento do agente, um estudo de ablação e as considerações finais.

Também há um espaço duplo em "meta-aprendizagem  FFORMA" (linha 9).

## 2. Design experimental

Linhas 13–45.

Correto. A composição do pool (3 estatísticos + 2 regressores base + 12 variantes + 2 ingênuos = 19), a tabela de conjuntos (182 + 111 + 359 + 28 = 680), os horizontes e o protocolo de três janelas de validação e uma de teste estão certos.

## 3. Escolha do modelo de linguagem

Linhas 46–108.

As tabelas e as contas estão corretas. As médias dos sete conjuntos, os agregados e os desvios conferem. Os tempos médios por série, os multiplicadores (22,4× e 19,0×) e as horas em execução serial também conferem.

Problemas de texto:

- Linha 78: "Nota-se a partir dos resultados, que" → "Nota-se, a partir dos resultados, que" ou "Os resultados mostram que".
- Linha 80: "considera o agregados" → "considera o agregado".
- Linha 80: "o melhor SMAPE  (13,67\%)" tem espaço duplo.
- Linha 108: "Em tempo serial, os 680 séries custariam" → "Em uma execução serial, as 680 séries custariam".
- Linha 108: "os valores de SMAPE obtidos com Gemma4 e Qwen3.8 são inferiores a 1 ponto percentual" é ambíguo. O correto é "as diferenças de SMAPE em relação ao gpt-oss são inferiores a 1 ponto percentual".

## 4. Execuções e agregação dos resultados

Linhas 110–146.

A tabela `tab:execucoes-agregados` confere: médias 20,27 / 13,78 / 54,05 / 57,19 e desvios 0,66 / 0,12 / 0,68 / 0,27. Os mínimos e máximos citados também conferem.

Problemas:

- Linha 146: "usa a execução 1 dessa série" → "usa a execução 1 dessa lista" ou "desse conjunto de execuções". Não há uma série associada.
- O custo da execução representativa (17,63–22,95 s) é menor que o custo da execução de escolha do modelo (25,20 s). Ver a seção 13.

## 5. ANP

Linhas 147–234.

As tabelas e a aritmética estão corretas. O texto de SMAPE está correto. O texto de POCID está correto para a MCM nova: usa o ETS como melhor individual e a mediana como melhor estática; CARTE 63,28; ETS 64,69; p = 0,0025.

Problema: a figura `mcm_ANP_MONTHLY_pocid.png` em `overleaf/resultados_final/` é a versão antiga, que usa o Theta. A figura correta está em `outputs/resultados/mcm_carte_pocid_metric_png/mcm_ANP_MONTHLY_pocid.png`.

A figura de POCID também não é referenciada no texto. Acrescentar `(Figura~\ref{fig:mcm-anp-pocid})` no início do parágrafo de POCID.

## 6. NN5

Linhas 235–322.

As tabelas e o texto estão corretos. A MCM nova de POCID usa Random Forest (FT) 49,16, DBA 46,46, CARTE 46,04, ADE 45,05 e FFORMA 44,66; os p-valores citados conferem.

Problema: a figura `mcm_NN5_WEEKLY_DATASET_pocid.png` em `overleaf/resultados_final/` é a versão antiga, que usa CatBoost (Original + FT) e a média. A figura correta está em `outputs/resultados/mcm_carte_pocid_metric_png/`.

Acrescentar a referência `(Figura~\ref{fig:mcm-nn5-pocid})`.

## 7. M4

Linhas 324–413.

As tabelas e o texto estão corretos. A MCM nova de POCID usa CatBoost (Original + DWT) 59,84, FFORMA 59,45, mediana 58,45, CARTE 57,88 e ADE 57,29; os p-valores citados conferem.

Problema: a figura `mcm_M4_WEEKLY_DATASET_pocid.png` em `overleaf/resultados_final/` é a versão antiga, que usa Random Forest (CWT). A figura correta está em `outputs/resultados/mcm_carte_pocid_metric_png/`.

Acrescentar a referência `(Figura~\ref{fig:mcm-m4-pocid})`.

## 8. ETTh1

Linhas 414–503.

As tabelas, o texto e a MCM de POCID estão corretos. A figura de POCID usa Random Forest (CWT), que é o melhor individual por SMAPE e por POCID neste conjunto, então a figura antiga continua válida.

Falta apenas referenciar a figura: `(Figura~\ref{fig:mcm-etth1-pocid})`.

## 9. ETTh2

Linhas 504–590.

As tabelas e a MCM de POCID estão corretos: Random Forest (DWT), FFORMA, ADE e mediana empatam em 56,52; CARTE 54,22; p = 0,5520 para o Random Forest (DWT).

Problemas:

- A figura `mcm_ETTH2_pocid.png` em `overleaf/resultados_final/` é a versão antiga, que usa CatBoost (Original + CWT). A figura correta está em `outputs/resultados/mcm_carte_pocid_metric_png/`.
- Linha 541: "os treze modelos de aprendizagem de máquina, com ou sem transformação, ocupam as treze primeiras posições". O 13º é o Random Forest sobre a representação original; o CatBoost sobre a representação original é a exceção e fica em último (172,70%). A expressão "com ou sem transformação" é enganosa. Sugestão: "os treze primeiros são modelos de aprendizagem de máquina, sendo doze variantes transformadas e o Random Forest sobre a representação original; o CatBoost sobre a representação original fica em último."
- Acrescentar a referência `(Figura~\ref{fig:mcm-etth2-pocid})`.

## 10. ETTm1

Linhas 591–678.

As tabelas e o texto estão corretos. A MCM de POCID usa a mediana e o Random Forest (Original + DWT) empatados em 58,39; CARTE 54,47; p = 0,2969.

Problema: a figura `mcm_ETTM1_pocid.png` em `overleaf/resultados_final/` é a versão antiga, que usa Random Forest (DWT). A figura correta está em `outputs/resultados/mcm_carte_pocid_metric_png/`.

Acrescentar a referência `(Figura~\ref{fig:mcm-ettm1-pocid})`.

## 11. ETTm2

Linhas 679–766.

As tabelas e o texto estão corretos. A MCM de POCID usa o ARIMA 48,45, a média 43,48, o CARTE 40,37 e o ADE/FFORMA 31,06; os p-valores citados conferem.

Problema: a figura `mcm_ETTM2_pocid.png` em `overleaf/resultados_final/` é a versão antiga, que usa Random Forest (DWT) e a mediana. A figura correta está em `outputs/resultados/mcm_carte_pocid_metric_png/`.

Acrescentar a referência `(Figura~\ref{fig:mcm-ettm2-pocid})`.

## 12. Síntese entre conjuntos de dados

Linhas 767–989.

### 12.1 Impacto das transformações

A Tabela `tab:sintese-transformacoes` (62/70) está correta. A Tabela `tab:sintese-transformacoes-escolhas` e os percentuais 99,2%, 64,2% e 61,7% estão corretos.

As Tabelas `tab:sintese-transformacoes-remocao-smape` e `tab:sintese-transformacoes-remocao-pocid` têm um problema: a coluna "sem transformações" e os totais 22,42% e 53,06% conferem com o recálculo atual nas 670 séries, mas a coluna CARTE e os deltas não conferem. Os valores atuais do recálculo são:

| Conjunto | SMAPE CARTE | SMAPE sem transf. | Δ SMAPE | POCID CARTE | POCID sem transf. | Δ POCID |
|---|---:|---:|---:|---:|---:|---:|
| ANP | 22,24 | 22,99 | +0,75 | 63,28 | 65,24 | +1,96 |
| NN5 | 12,04 | 13,46 | +1,42 | 46,04 | 42,41 | −3,63 |
| M4 | 9,28 | 23,70 | +14,42 | 57,72 | 50,46 | −7,26 |
| ETTh1 | 14,97 | 35,04 | +20,07 | 55,65 | 50,08 | −5,57 |
| ETTh2 | 17,67 | 50,68 | +33,00 | 54,22 | 50,42 | −3,80 |
| ETTm1 | 41,03 | 48,12 | +7,09 | 54,47 | 46,86 | −7,61 |
| ETTm2 | 17,02 | 25,75 | +8,73 | 40,37 | 46,69 | +6,31 |
| Total 670 | 13,78 | 22,42 | +8,64 | 57,03 | 53,06 | −3,97 |

Consequências no texto:

- Linha 845: "sobe de 13,73% para 22,42%, um aumento de 8,69 pontos" → "sobe de 13,78% para 22,42%, um aumento de 8,64 pontos".
- Linha 845: os ganhos por conjunto passam a ser +33,00 (ETTh2), +20,07 (ETTh1) e +14,42 (M4).
- Linha 871: "57,01% contra 53,06%, uma diferença de 3,95 pontos" → "57,03% contra 53,06%, uma diferença de 3,97 pontos".
- Linha 871: a ANP passa a +1,96 e o ETTm2 a +6,31.
- Linhas 882 e 1418: repetir os valores corrigidos.

A conclusão qualitativa não muda: as transformações pioram o SMAPE nos sete conjuntos e só perdem no POCID na ANP e no ETTm2.

### 12.2 Ranking por conjunto de dados

A Tabela `tab:sintese-ranking` e o texto estão corretos. O rank médio 2,71 do CARTE, 3,00 do FFORMA, 3,43 da melhor estática e 3,86 do ADE confere. As posições por conjunto também.

### 12.3 MCM agregada

O texto de SMAPE está correto: CARTE 13,78, mediana 13,79, FFORMA 14,15, Random Forest (DWT) 14,28, ADE 14,70; os p-valores 0,2342, 0,2397, 0,2539 e 0,2249 conferem; 17 empates.

O texto de POCID está errado. A figura correta está em `outputs/resultados/mcm_carte_pocid_metric_png/mcm_agregado_pocid.png` e usa Random Forest (Original + FT), não Random Forest (DWT). Os valores são:

- FFORMA 58,21;
- Random Forest (Original + FT) 57,30;
- mediana 57,20;
- CARTE 57,18;
- ADE 56,37.

O CARTE é o quarto, não o terceiro. O FFORMA supera o CARTE (p = 0,0090), a mediana (p = 0,0392) e o ADE (p = 0,0001), mas não o Random Forest (Original + FT) (p = 0,0616). O CARTE não se separa da mediana (p = 0,6635), do Random Forest (Original + FT) (p = 0,7386) nem do ADE (p = 0,0777). A mediana não se separa do Random Forest (Original + FT) (p = 0,9661). A frase "A mediana supera o Random Forest (DWT) (p = 0,0369)" não existe na figura correta.

Parágrafo sugerido:

> Por POCID, o FFORMA obtém o maior valor médio (58,21\%), seguido do Random Forest (Original + FT) (57,30\%), da mediana (57,20\%), do CARTE (57,18\%) e do ADE (56,37\%). O FFORMA supera o CARTE ($p = 0{,}0090$), a mediana ($p = 0{,}0392$) e o ADE ($p = 0{,}0001$), com significância estatística; contra o Random Forest (Original + FT), a diferença não é significativa ($p = 0{,}0616$). O CARTE fica em quarto, praticamente empatado com a mediana ($p = 0{,}6635$), à frente do ADE ($p = 0{,}0777$) e sem diferença para o Random Forest (Original + FT) ($p = 0{,}7386$). A mediana não se separa do Random Forest (Original + FT) ($p = 0{,}9661$). As demais comparações não são significativas.

O parágrafo de fecho (linha 942) também precisa mudar: o CARTE fica em quarto no POCID, atrás do FFORMA, do Random Forest (Original + FT) e da mediana. O FFORMA lidera com significância sobre o CARTE, a mediana e o ADE, mas não sobre o Random Forest (Original + FT).

A frase de abertura (linha 922) também precisa distinguir os dois melhores individuais: Random Forest (DWT) no SMAPE e Random Forest (Original + FT) no POCID.

### 12.4 Contagem de vitórias

A Tabela `tab:sintese-vitorias` e o texto estão corretos. As 205 vitórias do Random Forest (DWT), 142 do CARTE, 127 do ADE, 114 do FFORMA e 92 da mediana somam 680. Os percentuais conferem.

### 12.5 Erro por passo do horizonte

Os valores de SMAPE e POCID do primeiro e do oitavo passo conferem com `outputs/resultados/sintese_horizonte/horizonte_resumo.csv`. A alternância de liderança (mediana nos passos 1, 2, 3, 5 e 6; CARTE nos passos 4, 7 e 8) também confere.

Problema: a linha 987 diz que as faixas de confiança de POCID "se sobrepõem entre os métodos, especialmente nos passos intermediários". No oitavo passo, a faixa do FFORMA (59,75–67,01) não se sobrepõe à do ADE (51,55–59,04). Sugestão: "As faixas se sobrepõem na maioria dos passos; no oitavo, a faixa do FFORMA não se sobrepõe à do ADE."

## 13. Análise do comportamento do agente

Linhas 990–1172.

### 13.1 Abertura

A execução representativa e os valores 20,33 / 13,87 / 53,99 / 57,22 estão corretos. O texto usa "usamos" e "calculamos" (linha 992), enquanto o restante do capítulo é impessoal. Padronizar para "utiliza-se" e "calcula-se".

### 13.2 Frequência de uso do catálogo

A Tabela `tab:agente-acoes` conta apenas chamadas bem-sucedidas, e os valores conferem com os traces: 12.675 avaliações, 550 seleções top-k, 480 pesos top-k, 109 melhor por passo, 99 pesos gulosos, 110 erro inverso, 46 seleção estável, 8 seleção gulosa, 8 tendência do erro, 1 meta-modelo. O total é 14.086 chamadas bem-sucedidas.

Problemas:

- Linha 1024: "a média é de 20,86 chamadas por série" está errado. 20,86 é a média de **turnos** (14.186/680). A média de chamadas bem-sucedidas é 20,71 (14.086/680). Sugestão: "A mediana é de 21 turnos por série, e a média é de 20,86 turnos por série."
- Linha 1024: "só então escolhe a melhor tentativa do histórico" → "só então define o líder do histórico". A escolha final pode trocar o líder na etapa sazonal.
- Linha 1015 e 1022: "Peso pelo meta-modelo" e "pelo meta-modelo (1)" não existem no catálogo de `07_agente_combinador.md`, que lista 10 ações e não inclui `weights_meta_model`. Ou a ação é acrescentada ao catálogo do Capítulo 4, ou a linha e a menção são removidas. O mesmo vale para as linhas 1292 e 1420, que dizem que o meta-modelo entre séries não é treinado nos ETT.
- A subseção `Uso do contexto na escolha das ações` (`18_agente_contexto.md`) não está no arquivo. Se ela deve entrar, o lugar é entre a frequência e o motivo de encerramento.

### 13.3 Motivo de encerramento

A Tabela `tab:agente-encerramento` está correta: 552 séries (81,18%) por orçamento de avaliação e 128 (18,82%) por orçamento de turnos. A mediana de 21 turnos e a média de 20,86 conferem.

Problema: linha 1046, "A escolha final recai sobre a melhor tentativa do histórico" → "O líder do histórico, que inclui todas as avaliações feitas, é o ponto de partida da decisão final."

### 13.4 Passo a passo de uma decisão

Os escores de validação, as iterações e as métricas de teste da série 87 conferem com o trace: a1 0,4131, a2 0,3105, a4 0,1294, a5 1,0140, a7 0,1268, a10 0,1254, a12 0,1126; teste final 0,52% de SMAPE e 75,00% de POCID.

Problemas:

- Linha 1049: "A escolha final é a de melhor pontuação entre as tentativas" é uma regra geral falsa no método atual. Na série 87, o líder foi mantido porque a condição sazonal estava inativa. Sugestão: "Nesta série, a condição sazonal está inativa, e a escolha final mantém o líder, que é a melhor pontuação entre as tentativas."
- Linha 1094: "escolhe a melhor tentativa do histórico quando o orçamento de avaliação se esgota" → "define o líder do histórico; nesta série, a condição sazonal não se aplica e o líder é mantido."
- A tabela é seletiva: ela mostra as iterações 1, 2, 4, 5, 7, 8, 13 e 16, mas omite 3, 6, 9–12 e 14–15. Acrescentar uma frase dizendo que a tabela apresenta os passos mais informativos, não a trajetória completa. A iteração 12 (a9, 0,1267) chegou a assumir o líder e não aparece.
- O teste da primeira avaliação (7,97% de SMAPE e 33,33% de POCID) não está nos artefatos atuais; só o teste final está em `seed_preset_tests`. Confirme a origem desse valor antes de manter a tabela.

### 13.5 Interpretabilidade das previsões

Os dois casos conferem com `outputs/resultados/interpretabilidade_carte/`:

- M4 série 206: CARTE 10,92%, ADE 13,90%, FFORMA 14,15%, mediana 15,39%, DBA 21,87%.
- ANP série 13: CARTE 20,03%, ETS 16,31%, demais combinações 21,17–22,27%, POCID 100,00%.

As descrições das trajetórias e os valores de teste também conferem. Não há erro.

### 13.6 Distribuição da melhoria relativa

Os números 509 / 171 / 0, o ganho máximo de 0,232 na série 9 do M4 e a mediana de 0,011 estão corretos, mas descrevem o **líder**, não a estratégia final. O cálculo compara o líder de validação com a melhor das primeiras dez tentativas (a1–a10), com tolerância de 1e-6. Na execução representativa, 48 séries têm troca sazonal e a estratégia final fica com pontuação de validação pior que a do líder.

Correção: trocar "estratégia final" por "líder" em todo o parágrafo e acrescentar uma frase dizendo que a análise usa o líder e que a escolha final pode diferir dele quando a condição sazonal é ativada.

### 13.7 Custo computacional

A Tabela `tab:agente-custo` confere: 17,63–22,95 s por série, desvio-padrão 1,89–5,97 s, média ponderada 21,24 s.

Problema: a Seção 3 diz que o gpt-oss leva 25,20 s por série e usa esse valor para as 4,8 horas e para os multiplicadores de 22,4× e 19,0×. A tabela do agente usa a execução representativa, com 17,63–22,95 s. A diferença não é explicada. Acrescentar uma frase dizendo que a Seção 3 usa a execução de escolha do modelo e esta subseção usa a execução representativa das dez. Se a conclusão de custo usar 21,24 s, os multiplicadores mudam para cerca de 26,6× e 22,6×; mantenha a comparação de 22,4× e 19,0× apenas dentro da execução de escolha do modelo.

### 13.8 Rastreabilidade da decisão do agente

Os números 165 séries, Jaccard 0,85 / 0,93 / 0,94 / 0,84 e 160/165 são de uma execução antiga. Na execução representativa atual, 632 das 680 justificativas contêm "an agent proposal". A subseção precisa ser recalculada para o conjunto atual ou atualizada com os novos valores. A interpretação qualitativa (o histórico é mais informativo que a frase final) pode ser mantida, mas os números não.

## 14. Estudo de ablação

Linhas 1173–1414.

### 14.1 Buscas sem LLM

As tabelas `tab:ablacao-buscas-smape`, `tab:ablacao-buscas-pocid`, `tab:ablacao-buscas-vantagem` e `tab:ablacao-buscas-vitorias` conferem com `outputs/resultados/ablacao_search_carte_resumo/`. As diferenças 0,81 / 0,66 / 2,06, os p-valores de Holm (0,0003 para a gulosa), as contagens C/E/P e as vitórias 150/119/196/158/57 estão corretos. O texto de POCID e a leitura da política escolhida também estão corretos.

Observação: "a direção é consistente" (linha 1284) é forte, porque há duas exceções de POCID na média dos sete. Talvez "a direção é predominantemente consistente".

### 14.2 CARTE-oráculo

As tabelas e toda a aritmética estão corretas: restrito 18,64 / 11,61; 10 primeiras 18,87 / 11,74; validação 21,91 / 14,11; oráculo 17,39 / 10,55; individual 17,55 / 10,04. As diferenças 1,63 / 2,17, 1,40 / 2,04, 2,88 / 3,23 e 2,72 / 3,74 conferem. As contagens O/E/C também.

Problemas:

- "Melhores 10" é uma expressão enganosa. O oráculo usa a melhor entre as **10 primeiras tentativas** (a1–a10), não as 10 melhores tentativas. A tabela já usa "10 primeiras"; o texto deveria usar "melhor das 10 primeiras" em todas as ocorrências (linhas 1290, 1300, 1320, 1322, 1400 e legendas).
- Linha 1292: "nos ETT, fica em 216 ou menos porque o meta-modelo entre séries não é treinado com sete séries" é a mesma inconsistência do meta-modelo. Se o CARTE não tem meta-modelo entre séries, explique o que $U$ contém ou remova a menção.
- O texto "Todos, exceto o de validação, usam a janela de teste" está correto.

### 14.3 Síntese da ablação

Os valores e as comparações conferem. A conclusão de que o gargalo está na seleção, não na geração, é sustentada pelos números. A expressão "melhores 10" deve ser trocada por "melhor das 10 primeiras".

## 15. Considerações finais

Linhas 1415–1422.

O texto está bem estruturado e as limitações são pertinentes. Problemas:

- Linha 1418: os valores 13,73 / 22,42 e 57,01 / 53,06 precisam ser atualizados para 13,78 / 22,42 e 57,18 / 53,06, com deltas +8,64 e −4,12. Ver a seção 12.1.
- Linha 1420: "O meta-modelo entre séries não é usado nesses conjuntos porque o número de séries não atinge o mínimo do protocolo" contradiz a descrição do método, que diz que não há meta-modelo entre séries. Alinhar com o Capítulo 4.
- As demais afirmações conferem: melhor rank médio, menor SMAPE agregado, top-2 em quatro conjuntos, ANP como pior caso, ETTm1 com maior variabilidade, quatro conjuntos em que o melhor individual supera o CARTE.

## 16. Figuras

### 16.1 Figuras de POCID a substituir

As imagens em `overleaf/resultados_final/` estão desatualizadas para POCID. As corretas estão em `outputs/resultados/mcm_carte_pocid_metric_png/`:

- `mcm_ANP_MONTHLY_pocid.png`
- `mcm_NN5_WEEKLY_DATASET_pocid.png`
- `mcm_M4_WEEKLY_DATASET_pocid.png`
- `mcm_ETTH2_pocid.png`
- `mcm_ETTM1_pocid.png`
- `mcm_ETTM2_pocid.png`
- `mcm_agregado_pocid.png`

`mcm_ETTH1_pocid.png` está correta.

### 16.2 Referências de figura

As sete figuras de POCID por conjunto não são referenciadas no texto. Acrescentar `(Figura~\ref{fig:mcm-...-pocid})` no parágrafo de POCID de cada seção. A figura agregada de POCID já é referenciada na abertura da subseção, mas o parágrafo de POCID não a cita; não é obrigatório, mas ajuda.

### 16.3 Caminho das figuras

O `.tex` aponta para `images/resultados_final/...`; no repositório local, as figuras estão em `overleaf/resultados_final/...`. Confirme no Overleaf se o prefixo `images/` existe. Se não, ajuste o caminho ou o `\graphicspath`.

## 17. Problemas menores de texto

- Linha 9: "meta-aprendizagem  FFORMA" (espaço duplo).
- Linha 78: "Nota-se a partir dos resultados, que" (vírgula).
- Linha 80: "o agregados" e "SMAPE  (13,67\%)" (espaço duplo).
- Linha 108: "os 680 séries" e "Em tempo serial".
- Linha 146: "dessa série".
- Linha 541: "treze modelos ... com ou sem transformação" (nuance da ETTh2).
- Linha 987: faixas de confiança de POCID no oitavo passo.
- Linha 992: "usamos" e "calculamos".
- Linha 1024: "20,86 chamadas".
- Linha 1046: "A escolha final recai sobre a melhor tentativa".
- Linha 1049: "A escolha final é a de melhor pontuação".
- Linha 1094: "escolhe a melhor tentativa do histórico".
- Linha 1168: "165 séries" e Jaccard antigos.
- Linha 1292: meta-modelo entre séries.
- Linha 1420: meta-modelo entre séries.
- Linhas 1320, 1322, 1400: "melhores 10".

## 18. O que está correto e não precisa mudar

- Design experimental e tabela de conjuntos.
- Tabela e texto de escolha do modelo de linguagem (desempenho e custo), ressalvadas as correções de texto acima.
- Tabela de execuções e agregados.
- Todas as tabelas de modelos individuais e de estratégias de combinação dos sete conjuntos.
- Texto de SMAPE das MCMs por conjunto.
- Texto de POCID das MCMs por conjunto, ressalvadas as figuras antigas.
- Ranking por conjunto e rank médio.
- MCM agregada de SMAPE.
- Contagem de vitórias.
- Erro por passo do horizonte, ressalvada a sobreposição das faixas de POCID no oitavo passo.
- Casos de interpretabilidade (M4 206 e ANP 13).
- Tabelas e aritmética da ablação com buscas sem LLM.
- Tabelas e aritmética do CARTE-oráculo, ressalvada a nomenclatura "melhores 10".

## 19. Prioridade de correção

1. Substituir as sete figuras de POCID.
2. Corrigir o texto da MCM agregada de POCID (CARTE em quarto, Random Forest (Original + FT)).
3. Trocar "estratégia final" por "líder" na Subseção 13.6 e ajustar as frases sobre a decisão final nas Subseções 13.2, 13.3 e 13.4.
4. Corrigir as Tabelas de contrafactual e os valores 13,73/57,01 para 13,78/57,18.
5. Recalcular a rastreabilidade para a execução atual.
6. Alinhar o meta-modelo entre o Capítulo 4 e o Capítulo 5.
7. Acrescentar as referências das figuras de POCID.
8. Corrigir o roadmap, o custo e os problemas menores de texto.

## Segunda passada — revisão completa após as correções

Depois de reler o arquivo inteiro, foram encontrados e corrigidos mais cinco pontos:

- ETTm1: o texto dizia que o pool completo não foi usado em nenhuma execução. Na verdade, ele foi usado em cerca de 3,4 das 7 séries por execução; o tamanho do pool variou entre 3 e 14 modelos, com mediana 8,0.
- ETTm2: o texto dizia que o pool completo não foi usado em nenhuma execução. Na verdade, ele foi usado em cerca de 1,4 das 7 séries por execução.
- A frase sobre a janela de teste dizia que ela "não é observada em nenhuma etapa de decisão". O correto é que os valores reais não são observados; as previsões de teste podem ser usadas como evidência de forma. A frase foi ajustada.
- A justificativa da escolha do gpt-oss dizia que as diferenças de SMAPE eram inferiores a 1 ponto percentual sem qualificar. Isso vale na média dos sete conjuntos, não em todos os conjuntos. A frase foi qualificada.
- A tabela de vitórias atribui os quatro empates entre CARTE e mediana à mediana; isso agora está explícito na legenda e no texto.

Também foram ajustados:

- "tetos de seleção" → "os oráculos";
- "CARTE-oráculo amplo" → "CARTE-oráculo";
- "removi/recalculei" → "foram removidas/recalculou-se";
- "Nos seis conjuntos que não o ETTm1" → "Nos seis conjuntos diferentes do ETTm1";
- "a direção é consistente" → "a direção é predominantemente consistente";
- a frase ambígua do NN5 sobre superar o melhor individual em SMAPE.

Permanecem fora do `resultados.tex`:

- o catálogo de ações do Capítulo 4 não lista `weights_meta_model`, embora a tabela de frequência registre uma chamada bem-sucedida a essa ação;
- a subseção `Uso do contexto na escolha das ações` (`18_agente_contexto.md`) não está integrada.
