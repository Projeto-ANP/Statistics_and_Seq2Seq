# Alterações da proposta — CARTE

Este diretório reúne as substituições da proposta em formato de seções LaTeX prontas
para integração em `overleaf/proposta.tex`. O arquivo `.tex` ainda **não** foi
alterado.

## Regras editoriais

- O método passa a se chamar **CARTE**:
  `\textit{Context-Aware Reasoning for Time-series Ensembles}`.
- A notação canônica está em [`00_NOTACAO.md`](00_NOTACAO.md): série
  `\mathcal{S}_i`, perfil `\mathbf{z}_i`, avaliação do pool `\mathcal{E}_i`,
  diagnóstico `\mathcal{C}_i`, shortlist `\mathcal{K}_i`, líder `\ell_i`.
- A descrição deve refletir o método efetivamente executado, não flags, ablações,
  ativadores ou parâmetros de experimento.
- Todos os elementos de configuração usados na execução são tratados como
  **decisões fixas de projeto**. Não mencionar nomes de flags, valores de seed,
  temperatura, número de iterações, limites de turnos ou orçamentos numéricos.
- Remover da arquitetura:
  - meta-modelo entre séries;
  - prior de estratégias do conjunto de dados;
  - estratégias fixas semeadas;
  - aceite explícito, estagnação e paciência;
  - catálogo com contagem fixa de ações;
  - qualquer noção de “semente” ou “ativador”.
- Métricas: o **RMSE** é usado na construção do contexto e na preparação de pools/pesos; o **SMAPE** é a métrica que ordena as tentativas; o **POCID** é reportado fora do ranking. MAPE, MAE e MSMAPE não entram no texto.
- Manter:
  - três janelas de validação e uma janela de teste;
  - previsões de teste visíveis apenas como forma, sem os valores reais de teste;
  - perfil da série, diagnóstico do pool e diagnóstico interpretativo como contexto;
  - ações determinísticas de preparação e avaliação;
  - top-k por SMAPE e pesos top-k por horizonte como uma das ações possíveis;
  - histórico de tentativas, líder e shortlist por família;
  - condição sazonal determinística após o ciclo;
  - aplicação da estratégia escolhida às previsões de teste e cálculo das métricas.

## Mapa de alterações

| Seção em `proposta.tex` | Linhas atuais (aprox.) | Arquivo de alteração | Status |
|---|---:|---|---|
| `Considerações iniciais` | 3–8 | [`01_consideracoes_iniciais.md`](01_consideracoes_iniciais.md) | pronto para integrar |
| `Formulação do problema` | 20–102 | [`02_formulacao_problema.md`](02_formulacao_problema.md) | pronto para integrar |
| `Visão geral do CARTE` | 104–125 | [`03_visao_geral_carte.md`](03_visao_geral_carte.md) | pronto para integrar |
| `Pseudocódigo` | 127–289 | [`04_pseudocodigo_carte.md`](04_pseudocodigo_carte.md) | pronto para integrar |
| Notação de todo o documento | todos os `.tex` | [`00_NOTACAO.md`](00_NOTACAO.md) | referência ativa |
| `Preparação em nível de conjunto de dados` | 290–365 | [`05_mecanismo_pontuacao.md`](05_mecanismo_pontuacao.md) | pronto para integrar; meta-modelo e prior removidos |
| `Caracterização e diagnóstico por série` | 367–408 | [`06_contexto_por_serie.md`](06_contexto_por_serie.md) | pronto para integrar; inclui a forma das previsões |
| `Agente combinador` | 409–480 | [`07_agente_combinador.md`](07_agente_combinador.md) | pronto para integrar; sem sementes, aceite ou estagnação |
| `Aplicação da estratégia e rastreabilidade` | 481–497 | [`08_aplicacao_rastreabilidade.md`](08_aplicacao_rastreabilidade.md) | pronto para integrar; decisão final + aplicação + auditoria |
| `Considerações finais` | 498–506 | [`09_consideracoes_finais.md`](09_consideracoes_finais.md) | pronto para integrar; síntese alinhada ao CARTE |

## Renomeação global

Enquanto as seções seguintes não forem reescritas, aplicar:

- `CREST` → `CARTE` em todo o capítulo (incluindo comentários e legenda);
- `\label{sec:crest}` → `\label{sec:carte}`;
- `\ref{sec:crest}` → `\ref{sec:carte}`;
- `\label{alg:crest}` → `\label{alg:carte}`;
- `\ref{alg:crest}` → `\ref{alg:carte}`;
- `\label{fig:crest_arquitetura}` → `\label{fig:carte_arquitetura}`;
- `\ref{fig:crest_arquitetura}` → `\ref{fig:carte_arquitetura}`;
- caminho da figura: `images/proposta/carte_arquitetura.pdf` (exportar o novo
  diagrama do `.excalidraw`; se for usado PNG, trocar a extensão).

Com a integração dos arquivos 01–09, o nome `CREST` não deve permanecer no
capítulo; a substituição ainda precisa ser aplicada ao arquivo `.tex`.

## Dependências

- A introdução, a formulação, a visão geral e o pseudocódigo já foram reescritos
  nos arquivos 01–04; a numeração dos arquivos segue a ordem do capítulo.
- As seções de pontuação, contexto por série e agente combinador já foram
  reescritas nos arquivos 05–07. Elas removem meta-modelo, prior, sementes,
  aceite, estagnação e o catálogo com contagem fixa.
- A decisão final, a aplicação e a auditoria foram reescritas no arquivo 08. A
  seção `Aplicação` antiga, que descrevia o argmin simples, foi substituída por
  líder + shortlist por família + condição sazonal.
- A Tabela `tab:blocos_contexto`, no arquivo 07, é a referência única para os
  nomes dos blocos do prompt; 06 e 08 usam os mesmos rótulos em versalete.
- A Tabela `tab:acoes_carte`, também no arquivo 07, lista as ações efetivas do
  CARTE sem usar uma contagem fixa de entradas, com a correspondência entre o
  nome conceitual e o identificador de código. A Tabela `tab:metodos_pesos`
  faz o mesmo para os métodos de pesos.
- A seção `Considerações finais` foi reescrita no arquivo 09 e agora sintetiza a
  arquitetura descrita nos arquivos 03–08. Todos os trechos do capítulo possuem
  arquivo de alteração; resta integrar os arquivos no `.tex` e atualizar
  `resultados.tex`.
- A Equação `eq:score` já foi substituída em `05_mecanismo_pontuacao.md` pela
  razão de SMAPE agregada com pesos fixos; o POCID não entra no ranking.
- `resultados.tex` ainda referencia `sec:preparacao`, que deixou de existir. Ao
  integrar o arquivo 05, essa referência precisa ser atualizada para
  `sec:pontuacao` ou o trecho correspondente precisa ser reescrito.
