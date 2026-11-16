# Análise do comportamento do agente - Uso do contexto

Atualiza a seção `Análise do comportamento do agente` com a subseção `Uso do contexto na escolha das ações`, que deve entrar depois da frequência de uso do catálogo e antes do motivo de encerramento.

```latex
\subsection{Uso do contexto na escolha das ações} \label{subsec:agente-contexto}

O agente recebe, a cada turno, o perfil da série, o resumo do pool, o histórico de tentativas, o diagnóstico determinístico, a forma das previsões e o resumo do conjunto de dados. O diagnóstico é uma leitura determinística do perfil e do resumo do pool. A forma das previsões traz a última janela de validação, as previsões cegas dos modelos e a concordância entre elas, sem o teste real. A série bruta e o teste real não entram no prompt. O perfil traz tendência, sazonalidade, estacionariedade e outliers. O resumo do pool traz erro, estabilidade do ranking e correlação dos erros. O resumo do conjunto de dados traz o que funcionou nas outras séries do mesmo conjunto, com base apenas na validação.

O prompt pede que o agente nomeie uma característica da série em cada pensamento. A menção isolada não prova uso, porque o agente pode apenas cumprir a instrução. A evidência mais forte é a correspondência entre o que ele cita e o valor real do perfil e do resumo do pool. Nos 14.186 pensamentos da execução representativa, a menção à sazonalidade ocorre em 54,9\% dos pensamentos nas séries do quartil superior de força sazonal e em 1,7\% nas do quartil inferior. A menção ao ranking instável ocorre em 75,1\% das séries com veredito instável e em 0,1\% das moderadas. Na série 339 do M4, o pensamento diz \textit{a série tem ranking instável (tau -0,279)}. O tau é o coeficiente de Kendall definido na Seção~\ref{sec:diagnostico} do Capítulo~\ref{chp:proposta}, e valores baixos indicam rankings que mudam entre janelas. Na série 312 do M4, o pensamento diz \textit{a série tem tendência 0,0 e tau 0,072}. A menção à tendência é mais fraca, 42,2\% contra 28,0\%, porque quase todas as séries têm tendência alta. A correlação também é mais fraca, 17,5\% contra 9,1\%.

A Tabela~\ref{tab:agente-contexto} mostra o sinal diretamente associado a cada ação, não todos os termos citados.

\begin{table}[H]
\centering
\small
\caption{Sinal diretamente associado a cada ação e fração de pensamentos que o citam, na execução representativa.}
\label{tab:agente-contexto}
\begin{tabular}{@{}llr@{}}
\toprule
\textbf{Ação} & \textbf{Sinal citado} & \textbf{Pensamentos que citam} \\
\midrule
Seleção pelos $k$ melhores & ranking instável & 51,6\% \\
Peso pela média dos melhores por passo & horizonte & 100,0\% \\
Peso pelo erro inverso & erro & 75,0\% \\
Peso pela seleção gulosa & correlação de erros & 43,4\% \\
\bottomrule
\end{tabular}
\end{table}

O horizonte aparece em 100,0\% dos pensamentos antes das ações de horizonte e em 9,8\% antes do erro inverso. O erro aparece em 75,0\% antes do erro inverso e em 10,3\% antes das ações de horizonte. A associação também aparece no uso das ações. Nos testes qui-quadrado, nas séries de maior sazonalidade, o peso pela média dos melhores por passo aparece em 58,8\% delas, contra 44,1\% nas de menor ($p = 0{,}009$). O peso pelo melhor preditor por passo aparece em 20,0\% contra 11,2\% ($p = 0{,}036$). A seleção pelos mais estáveis aparece em 7,7\% das séries com ranking instável e 3,5\% das moderadas ($p = 0{,}039$). A seleção pelos $k$ melhores não varia com a sazonalidade, 68,2\% nos dois grupos. O peso pelo erro inverso e o peso pela seleção gulosa têm associações mais fracas com tendência e correlação. O perfil desloca a probabilidade de algumas ações, não determina a escolha.

Na série 295 do M4, com sazonalidade 1,0 e tau 0,251, o pensamento registra \textit{a série tem sazonalidade forte e ranking instável} e o agente chama \texttt{weights\_topk\_per\_horizon}. A ação cria o vetor de pesos w4, usado em uma avaliação seguinte. O caso mostra a cadeia perfil, pensamento e ação, sem uma regra fixa.

Os traces mostram uso, não necessidade causal. A evidência vem da correspondência com o valor real, não da simples menção. Os blocos de forma das previsões, resumo do conjunto de dados e histórico são pouco verbalizados. A evidência forte desta seção é para o perfil da série e o resumo do pool. Essas associações são descritivas e os testes são exploratórios. Uma ablação de contexto, removendo ou permutando cada bloco, ainda precisa ser feita.
```
