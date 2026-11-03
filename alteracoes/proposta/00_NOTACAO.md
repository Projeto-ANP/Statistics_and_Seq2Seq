# Notação canônica do documento

Esta é a convenção que os arquivos de `alteracoes/` seguem e que deve ser usada
em `proposta.tex`, `resultados.tex` e nos demais capítulos.

## Símbolos do método

| símbolo | significado | observação |
|---|---|---|
| $\mathcal{D}$ | conjunto de históricos de séries | não usar `D` para série individual |
| $\mathcal{S}_i$ | histórico da série $i$ | **S de série**; caligráfico para não colidir com $S_t$ (sazonalidade) |
| $x_{i,t}$ | observação $t$ da série $i$ | valor escalar |
| $n_i$ | número de observações de $\mathcal{S}_i$ | substitui o antigo $T_i$ de comprimento |
| $N$ | número de séries | índice $i$ |
| $M$ | número de preditores | índice $j$ |
| $\mathcal{M}$ | conjunto de preditores | $\{M_j\}_{j=1}^{M}$ |
| $M_j$ | preditor $j$ | nome do modelo |
| $h$ | horizonte | passos $k=1,\dots,h$ |
| $w$ | janela de validação | $w\in\{1,2,3\}$ |
| $\mathbf{y}_i^{(w)}$ | valores reais da janela de validação | alvo de validação |
| $\hat{\mathbf{y}}_{i,j}^{(w)}$ | previsão do preditor $j$ na janela | previsão de validação |
| $\mathbf{y}_i^{(\mathrm{teste})}$ | valores reais de teste | não acessado na decisão |
| $\hat{\mathbf{y}}_{i,j}^{(\mathrm{teste})}$ | previsão de teste do preditor $j$ | visível como evidência de forma |
| $\hat{\mathbf{y}}_i^{(\mathrm{comb})}$ | previsão combinada final | aplicação da estratégia |
| $f_i$ | função de combinação da série $i$ | definida pela estratégia escolhida |
| $s$ | estratégia candidata | combinação de pool + pesos + agregação |
| $s_0$ | média simples de $\mathcal{M}$ | referência de pontuação |
| $\mathcal{M}_{i,s}^{(w)}$ | subconjunto de preditores de $s$ na janela $w$ | pode variar por janela no backtest |
| $\mathbf{p}_{i,s}^{(w)}$ | pesos de $s$ na janela $w$ | pode variar por janela no backtest |
| $\mathrm{score}_i(s)$ | pontuação de validação de $s$ | menor é melhor |
| $\mathbf{z}_i$ | perfil da série $i$ | features de forma/estatística |
| $\mathcal{E}_i$ | avaliação do pool na série $i$ | erros, ranking, correlação |
| $\mathcal{C}_i$ | diagnóstico interpretativo | regime, previsibilidade, indicação |
| $\mathcal{F}_i$ | representação da forma das previsões | validação + previsões de teste, sem o real |
| $\mathcal{H}_i$ | histórico de tentativas | pares $(s,\mathrm{score}_i(s))$ |
| $\ell_i$ | líder de validação | menor score |
| $\mathcal{K}_i$ | shortlist por família | candidatos da condição sazonal |
| $s_i^{\star}$ | candidato da shortlist com menor distância sazonal | comparado ao líder |
| $c_i$ | estratégia final escolhida | pode ser o líder ou um candidato sazonal |
| $\mathcal{A}$ | conjunto de ações | preparação + avaliação |
| $d_i(s)$ | distância ao sazonal ingênuo | usada na condição sazonal |
| $\mathbf{q}_i$ | previsão sazonal ingênua | referência da condição sazonal |
| $\sigma_i$ | escala recente da série | normaliza a distância sazonal |
| $\alpha_w$ | peso fixo da janela $w$ na pontuação | prioriza a janela mais recente |

## Regras de uso

- No corpo do texto, usar nomes conceituais em português para ações e métodos. Identificadores de código aparecem apenas nas tabelas de correspondência, em `\texttt{}`.
- Rótulos literais de blocos do prompt aparecem apenas na Tabela `tab:blocos_contexto`, em versalete; no texto, usar a designação em português.
- No capítulo do método, **não usar $X_i$**; usar $\mathcal{S}_i$.
- **Não usar $T_i$** para comprimento; usar $n_i$. O símbolo $T_t$ fica reservado
  para a tendência no background.
- **Não usar $L_i$** para líder; usar $\ell_i$. O símbolo $L_t$ fica reservado
  para o componente linear em `revisao.tex`.
- **Não usar $S_i$** para shortlist; usar $\mathcal{K}_i$.
- **Não usar $S_t$** para série no método: $S_t$ é o componente sazonal em
  `background.tex`. A série no método é $\mathcal{S}_i$ (caligráfico).
- $Y_t$, $T_t$, $S_t$ e $R_t$ permanecem a notação padrão de decomposição no
  background; são escopos diferentes de $\mathcal{S}_i$ e $\mathbf{z}_i$.
- Se for exigido um único “S” itálico para série em todo o documento, será
  necessário renomear o componente sazonal do background de $S_t$ para $Z_t$.
  Recomendação: manter $\mathcal{S}_i$ no método e $S_t$ para sazonalidade.
- $M_j$ e $\mathcal{M}$ são os mesmos em todos os capítulos; não usar $P$ para
  preditor nem $\mathcal{P}$ para pool.
- Previsões usam sempre $\hat{\mathbf{y}}$; valores reais usam $\mathbf{y}$.
  Evitar $\hat{y}$ sem vetor quando a grandeza for o vetor completo.

## Auditoria atual dos `.tex`

- `background.tex`: usa $Y_t$ (série observada), $T_t$ (tendência), $S_t$
  (sazonalidade) e $R_t$ (ruído). Não há conflito com o método, mas $S_t$ **não**
  é a série.
- `revisao.tex`: usa $Y_t$ (série), $L_t$ (componente linear), $R_t$ (resíduo) e
  $A_l$ (modelo). O método usa $\ell_i$ para líder e $\mathcal{A}$ para ações;
  não reutilizar $L_t$ nem $A_l$ no método.
- `resultados.tex`: usa $\mathcal{M}$ para o pool; manter. Ainda usa `CREST`;
  a renomeação global para `CARTE` está no índice.
- `proposta.tex`: as seções antigas ainda usam $X_i$, $T_i$ e $S_i$ com outros
  sentidos; elas serão substituídas pelos arquivos 01–09 e deverão seguir esta
  tabela.

## Comandos de verificação

Antes de fechar um capítulo, verificar no `.tex`:

```bash
rg -n 'X_i|\bT_i\b|\bL_i\b|S_i' overleaf/proposta.tex
rg -n 'CREST|crest' overleaf/*.tex
```

O primeiro comando deve retornar apenas ocorrências em seções ainda não
reescritas; o segundo deve retornar zero após a renomeação global.
