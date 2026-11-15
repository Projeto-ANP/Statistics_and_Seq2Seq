# Análise do comportamento do agente - Interpretabilidade das previsões

Atualiza a subseção `Interpretabilidade das previsões` de `overleaf/resultados.tex` (linhas 941–963) com a execução representativa. Os dois casos continuam sendo a série 206 do M4 e a série 13 da ANP, mas agora usam a estratégia final dessa execução. As figuras `caso_sucesso.png` e `caso_fracasso.png` foram regeradas.

```latex
\subsection{Interpretabilidade das previsões} \label{subsec:interpretabilidade-previsoes}

Para ilustrar como a estratégia final se traduz em previsões, esta subseção apresenta duas séries em extremos opostos de desempenho, uma do M4 semanal (série 206) e uma da ANP (série 13), ambas com estratégia final proposta pelo agente e combinando múltiplos preditores, não um único modelo isolado.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{images/resultados_final/caso_sucesso.png}
    \caption{Previsões individuais, valor real, combinação final do CARTE e demais métodos de combinação na série 206 do M4, um caso de sucesso. O painel inferior amplia a faixa das estratégias de combinação.}
    \label{fig:caso-sucesso}
\end{figure}

Na série 206 do M4 (Figura~\ref{fig:caso-sucesso}), o CARTE obtém SMAPE de 10,92\%, combinando sete modelos, entre Random Forest e CatBoost sobre as transformações wavelet contínua, discreta e de Fourier, além do Random Forest na representação original por média ponderada, com pesos médios entre 0,12 e 0,17 e sem concentração em um único modelo. As sete previsões individuais acompanham o nível geral da série, mas divergem nos picos e vales ao longo do horizonte. A combinação final segue a alta de forma mais fiel que as demais estratégias e é a que mais se aproxima do valor real entre as combinações, embora fique abaixo do pico isolado do sexto passo, quando o valor observado salta para 20.567 e nenhuma combinação acompanha o movimento. As demais estratégias ficam atrás. Contra 10,92\% do CARTE, o ADE obtém 13,90\%, o FFORMA 14,15\%, a mediana 15,39\% e o DBA 21,87\%. A justificativa registrada é \textit{"Selected 'weighted on pool4 weighted by w5' (an agent proposal) as the best of 11 backtested attempts, score 0.771."}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{images/resultados_final/caso_fracasso.png}
    \caption{Previsões individuais, valor real, combinação final do CARTE e demais métodos de combinação na série 13 da ANP, um caso de fracasso. O painel inferior amplia a faixa das estratégias de combinação.}
    \label{fig:caso-fracasso}
\end{figure}

Na série 13 da ANP (Figura~\ref{fig:caso-fracasso}), o CARTE obtém SMAPE de 20,03\%, combinando sete modelos (ARIMA, ETS, Theta, Random Forest na representação original, CatBoost e Random Forest sobre a transformação de Fourier e o Naive Moving Average) por média simples. A figura mostra por que a combinação não teve sucesso. O valor real recua de aproximadamente 1.540 para pouco mais de 900 nos primeiros passos, recupera-se em parte e volta a subir até 1.884 no fim do horizonte, enquanto a média dos sete modelos fica confinada entre 1.279 e 1.432. A média dilui os poucos preditores que acompanham a mudança de nível e produz uma previsão suavizada. O melhor modelo individual isolado, o ETS, teria obtido 16,31\% de SMAPE. As demais combinações ficam entre 21,17\% e 22,27\% de SMAPE. O CARTE é a melhor combinação nesta série, mas ainda assim não acompanha a magnitude. O POCID do CARTE é 100,00\%, o que indica que a direção é correta em todos os passos. O erro está na magnitude, comprimida pela média. As demais combinações acertam a direção em 54,55\% a 72,73\% dos passos. A justificativa registrada é \textit{"Selected 'mean on pool4' (an agent proposal) as the best of 12 backtested attempts, score 0.9119."}
```
