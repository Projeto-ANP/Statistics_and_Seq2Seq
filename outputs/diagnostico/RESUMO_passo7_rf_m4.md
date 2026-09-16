# Resumo -- divergência do Random Forest no M4: mesmo mecanismo do CatBoost, ou outro?

Fatos apurados em `outputs/diagnostico/passo7_rf_m4_casos.json` e `outputs/diagnostico/passo7_rf_m4_classificacao.csv` (12 casos: os `(regressor, dataset_index)` da família RF marcados divergentes em `passo5_contexto_divergencia.csv` para `M4_WEEKLY_DATASET`), cruzados com `all_functions.py` e `run_tsf_regressors.py`. Sem recomendação de tratamento.

## 1. Quantos são "retroalimentação instável" e quantos "denominador pequeno"?

**0 de 12 são "denominador pequeno" — a hipótese do MAPE está descartada.** `test_pequeno_no_ponto` é `False` nos 12 casos: no ponto do horizonte com maior APE de cada série, o valor real (`test`) nunca é pequeno (varia de 240 a 9945, sempre em escala normal para a série). A previsão nesse mesmo ponto nunca é "comparável às demais" — é sempre ela mesma o valor extremo (`pred_comparavel_as_demais = False` em 11/12; o único `True`, `ONLY_DWT_rf` idx 9, ainda assim tem `test_pequeno_no_ponto = False`, então não conta como "denominador pequeno" pela definição do Passo 2).

Pela classificação literal do prompt (limiar de `log10(max|pred|/max|test|) > 10`, "dezenas de ordens de grandeza"): **7/12 "retroalimentação instável"**, **5/12 "ambíguo"**. Mas os 5 "ambíguos" não são ambíguos quanto à família de causa — só têm razão pred/test menor (10²·⁰⁷ a 10⁶·⁸, contra 10¹⁰·⁹ a 10²⁶·² dos 7 "instáveis"). Em nenhum dos 12 casos a previsão fica em escala comparável ao `test`; a diferença entre os dois grupos é de grau, não de natureza.

Os 12 casos se dividem em dois grupos por `dataset_index`, que não se sobrepõem:

- **`dataset_index=208`** afeta as 7 variantes RF (`rf`, `CWT_rf`, `DWT_rf`, `FT_rf`, `ONLY_CWT_rf`, `ONLY_DWT_rf`, `ONLY_FT_rf`), razão de 10¹⁰·⁹ a 10²⁶·² — mecanismo identificado abaixo.
- **`dataset_index` 9, 75, 217** afetam exclusivamente `DWT_rf`/`ONLY_DWT_rf` (nunca `rf`, `CWT_rf` ou `FT_rf` puros) — razão bem menor, 10²·⁰⁷ a 10⁶·⁸. Não investiguei o mecanismo específico desses 3 casos (ver seção "Em aberto").

## 2. A previsão do RF respeita o intervalo teórico `[min(y_train), max(y_train)]`?

**Sim, para o caso em que o teste é válido — mas o intervalo teórico não é o que parece.** O RF não é treinado para prever o valor bruto da série: `rolling_window` (`all_functions.py:190-209`) normaliza cada linha de treino por `(alvo - média_da_janela) / desvio_padrão_da_janela`, usando a média/desvio **daquela janela específica**, não da série inteira. `reverse_regressors` (`all_functions.py:1060-1066`) desfaz isso na predição final com um único `mean`/`std` fixo (da última janela de `y_train`), diferente do usado em cada linha de treino. Comparar a predição final direto contra `[min(y_train), max(y_train)]` bruto testa o limite errado.

Reconstruí a predição bruta do modelo (revertendo essa desnormalização final) e a distribuição real dos alvos normalizados vistos no treino (reaplicando a normalização de `rolling_window` a `y_train`), só para `regressor="rf"` puro (o único caso onde confirmei em código que o treino passa por `rolling_window`; as 6 variantes com wavelet/Fourier treinam por um caminho diferente, ver abaixo). Resultado para `rf`, `dataset_index=208`:

- Alvos normalizados de treino: variam de **-9,16×10¹⁴ a +1,75×10¹⁵** (921 janelas de treino; 35 delas com alvo normalizado acima de 10⁶).
- As 13 predições brutas reconstruídas do horizonte caem **todas dentro** desse intervalo (`predictions_dentro_do_limite_bagging = True`, 0 pontos fora).

**O RF respeita seu próprio limite teórico.** O limite em si é que está corrompido: a causa raiz é `rolling_window` (`all_functions.py:198-201`) checar `if std > 0` (positividade estrita) em vez de uma tolerância como a usada em `znorm` (`all_functions.py:55`, `abs(std) < NORM_EPS`). Quando uma janela de treino é uma sequência de valores repetidos (ex.: `[2492.05]×13`), `np.std()` não devolve exatamente `0.0` por ruído de ponto flutuante (`4.5×10⁻¹³`, não zero) — o código não cai no caso de proteção, e `(alvo - média) / 4.5×10⁻¹³` produz um "alvo normalizado" da ordem de 10¹⁵ a partir de um valor real perfeitamente normal (confirmado diretamente: janela `[2492.05]×13`, alvo real `3288.18`, alvo normalizado `1,75×10¹⁵`). Esse valor entra no conjunto de treino do RF como um alvo legítimo (envenenado). A série `M4_WEEKLY_DATASET` idx 208 tem 35 janelas assim entre suas 921 janelas de treino — provavelmente por ter trechos longos de valores idênticos/arredondados, uma característica dos dados semanais do M4 que não vi nos outros datasets do estudo.

Para as 6 variantes com wavelet/Fourier, **não testei o limite** (`limite_bagging_testavel = False`, coluna vazia no CSV): elas treinam via `run_tsf_image_series` → `rolling_window_image` (`run_tsf_regressors.py:411`), que normaliza o alvo com `znorm_by` (`all_functions.py:85-91`) — essa função **tem** a guarda de tolerância (`abs(std) < NORM_EPS` → devolve `0.0`), diferente de `rolling_window`. Isso sugere que essas 6 variantes não sofrem exatamente o mesmo bug de ponto flutuante, mas eu não reconstruí a distribuição de treino correta para confirmar (exigiria reimplementar a transformada wavelet/Fourier exata usada, fora do escopo deste passo).

## 3. Se respeita o intervalo, isso confirma outra causa (não uma falha do próprio RF)?

**Sim, para `dataset_index=208`, confirmado.** O Random Forest em si funciona exatamente como a teoria do bagging prevê — a agregação das árvores não extrapola além do que foi visto no treino. A causa não é "RF falha em generalizar" nem "denominador pequeno do MAPE" (descartado no item 1): é um bug de tolerância numérica em `rolling_window` que insere alvos de treino astronomicamente grandes sempre que uma janela de `horizon` observações consecutivas é (quase) constante. O CatBoost, alimentado pelos mesmos alvos de treino corrompidos, herda o mesmo problema na raiz — mas, por fazer *boosting* (soma, não média, de correções), tem margem para amplificar ainda mais via a retroalimentação recursiva já documentada; o RF fica matematicamente travado no pior alvo corrompido que existir no seu próprio conjunto de treino, o que também explica por que a escala de erro do RF (`10¹⁰` a `10²⁶`) fica consistentemente abaixo da do CatBoost no mesmo dataset (`10¹¹⁸` a `10¹³³`, ver diagnóstico anterior) mesmo nos casos mais graves.

## Em aberto

- Mecanismo de `DWT_rf`/`ONLY_DWT_rf` em `dataset_index` 9, 75 e 217 (razão pred/test de 10² a 10⁶⁻⁷, bem menor que idx 208, e que não afeta `rf`/`CWT_rf`/`FT_rf` na mesma série) -- não investigado aqui.
- Confirmação do limite de bagging para as 6 variantes com wavelet/Fourier em `dataset_index=208`, que teriam que ser testadas contra a distribuição de alvos normalizados por `rolling_window_image`/`znorm_by`, não por `rolling_window`.

Arquivos: `outputs/diagnostico/passo7_rf_m4_casos.json`, `outputs/diagnostico/passo7_rf_m4_classificacao.csv`.
