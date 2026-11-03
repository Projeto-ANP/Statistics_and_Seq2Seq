# Pseudocódigo do CARTE

Substitui a subseção `Pseudocódigo` de `overleaf/proposta.tex`
(linhas 127–289), incluindo os Algoritmos 1–3.

```latex
\subsection{Pseudocódigo}

A seguir, o Algoritmo~\ref{alg:carte} apresenta a visão geral do método proposto. O mecanismo de pontuação é detalhado no Algoritmo~\ref{alg:pontuar}, e o diagnóstico interpretativo, no Algoritmo~\ref{alg:diagnostico}. A notação empregada é a da formulação da Seção~\ref{sec:formulacao}, complementada por: $\mathbf{z}_i$, o perfil da série $\mathcal{S}_i$; $\mathcal{E}_i$, a avaliação do pool de preditores nas janelas de validação; $\mathcal{C}_i$, o diagnóstico interpretativo; $\mathcal{F}_i$, a representação da forma das previsões, que reúne o último trecho real de validação e as previsões dos modelos para a janela de teste, sem os valores reais de teste; $\mathcal{H}_i$, o histórico de tentativas da série $i$, formado por pares $(s, \text{score}_i(s))$; $\mathcal{M}_{i,s}^{(w)}$ e $\mathbf{p}_{i,s}^{(w)}$, o subconjunto de preditores e os pesos usados por $s$ na janela $w$; $\ell_i$, o líder de validação; $\mathcal{K}_i$, a shortlist de famílias de estratégias; $d_i(s)$, a distância entre a previsão de teste de $s$ e a referência sazonal ingênua; $s_i^{\star}$, o candidato da shortlist com menor distância sazonal; $c_i$, a estratégia final escolhida; $\text{margem}$, a margem fixa para troca do líder; $\mathcal{A}$, o conjunto de ações disponíveis; e $\hat{\mathbf{Y}}_i^{(\mathrm{teste})}$, a coleção das previsões de teste dos $M$ preditores. Em cada iteração, o agente responde exclusivamente com um \textit{Thought}, uma \textit{Action} e uma \textit{Action Input}. O ciclo usa um orçamento fixo de avaliações e de turnos e termina quando esse orçamento se esgota.

\begin{algorithm}[H]
\footnotesize
\caption{CARTE: visão geral do método proposto}
\label{alg:carte}
\Entrada{$\mathcal{D}=\{\mathcal{S}_i\}_{i=1}^{N}$; $\mathcal{M}=\{M_j\}_{j=1}^{M}$;
         horizonte $h$; janelas de validação
         $\{(\mathbf{y}_i^{(w)}, \hat{\mathbf{y}}_{i,j}^{(w)})\}_{i,j,w}$ com
         $w \in \{1,2,3\}$; previsões de teste
         $\hat{\mathbf{Y}}_i^{(\mathrm{teste})}$; ações $\mathcal{A}$;
         orçamento fixo}
\Saida{$\{\hat{\mathbf{y}}_i^{(\mathrm{comb})}\}_{i=1}^{N}$ e métricas de
       desempenho}

\For{$i \gets 1$ \KwTo $N$}{
  $\mathbf{z}_i \gets \Act{Perfil}(\mathcal{S}_i)$
  $\mathcal{E}_i \gets \Act{AvaliarConjunto}(i)$
  $\mathcal{C}_i \gets \Act{Diagnosticar}(\mathbf{z}_i, \mathcal{E}_i)$
  $\mathcal{F}_i \gets \Act{FormaPrevisao}(i)$
  $\mathcal{H}_i \gets \emptyset$
  $\ell_i \gets \text{nulo}$
  \While{\text{orçamento de turnos e avaliações não esgotado}}{
    $u \gets \Act{Contexto}(\mathbf{z}_i, \mathcal{E}_i, \mathcal{C}_i,
                              \mathcal{F}_i, \mathcal{H}_i, \ell_i)$
    $r \gets \Act{Agente}(u)$
    $(\text{ação}, \text{argumentos}) \gets \Act{Validar}(r)$
    \eIf{formato inválido, ação fora do catálogo \textbf{ou} argumentos inválidos}{
      devolver erro estruturado ao agente; consumir um turno
    }{
      \eIf{$\text{ação}$ é de preparação}{
        $o \gets \Act{Executar}(\text{ação}, \text{argumentos})$
        registrar handle de pool ou de pesos; consumir um turno; não pontuar
      }{
        $s \gets \Act{ConstruirEstrategia}(\text{ação}, \text{argumentos})$
        $sc \gets \Act{Pontuar}(i, s)$
        \eIf{$s$ já foi avaliada em $\mathcal{H}_i$}{
          recuperar a tentativa existente
        }{
          $\mathcal{H}_i \gets \mathcal{H}_i \cup \{(s, sc)\}$
          \If{$\ell_i = \text{nulo}$ \textbf{ou} $sc < \text{score}_i(\ell_i)$}{
            $\ell_i \gets s$
          }
          consumir uma avaliação
        }
        consumir um turno
      }
    }
  }
  $\mathcal{K}_i \gets \Act{ShortlistFamilias}(\mathcal{H}_i, \mathcal{F}_i)$
  \eIf{$\mathcal{K}_i = \emptyset$}{
    $c_i \gets \ell_i$
  }{
    \If{$\ell_i \notin \mathcal{K}_i$}{
      $\ell_i \gets$ primeiro candidato plausível de $\mathcal{K}_i$
    }
    $c_i \gets \ell_i$
    \If{$\Act{CondicaoSazonal}(\mathbf{z}_i)$ \textbf{e}
        $|\mathcal{K}_i| \geq 2$}{
      $s_i^{\star} \gets \arg\min_{s \in \mathcal{K}_i} d_i(s)$
      \If{$d_i(\ell_i) - d_i(s_i^{\star}) > \text{margem}$}{
        $c_i \gets s_i^{\star}$
      }
    }
  }
  $f_i \gets$ regra de combinação de $c_i$
  $\hat{\mathbf{y}}_i^{(\mathrm{comb})} \gets
      f_i(\hat{\mathbf{Y}}_i^{(\mathrm{teste})})$
  $\text{SMAPE}_i, \text{POCID}_i \gets
      \text{métricas}(\hat{\mathbf{y}}_i^{(\mathrm{comb})},
                       \mathbf{y}_i^{(\mathrm{teste})})$
}
\KwRet{$\{\hat{\mathbf{y}}_i^{(\mathrm{comb})}\}_{i=1}^{N}$; métricas; registros}
\end{algorithm}

A shortlist $\mathcal{K}_i$ contém a melhor tentativa de cada família de estratégia presente em $\mathcal{H}_i$ e é construída aplicando cada tentativa às previsões de teste dos preditores, sem acessar o valor real de teste. Estratégias ponderadas são separadas pelo método de pesos, de modo que a família top-k por horizonte seja preservada como uma família própria. Se a shortlist tiver menos de dois candidatos, o líder é mantido e o caso é registrado como fallback. A condição sazonal só é considerada quando a série apresenta sazonalidade estimável; ela compara as previsões dos candidatos da shortlist com uma previsão sazonal ingênua e troca o líder apenas quando a melhora na distância sazonal supera uma margem fixa. A condição não acessa os valores reais da janela de teste. O pseudocódigo pressupõe ao menos uma tentativa avaliada; caso o orçamento se esgote sem nenhuma avaliação, a série é registrada como falha e não produz previsão combinada.

\begin{algorithm}[H]
\footnotesize
\caption{Mecanismo de pontuação de estratégias}
\label{alg:pontuar}
\Entrada{Série $i$; estratégia $s$; janelas de validação}
\Saida{$\text{score}_i(s)$}

$s_0 \gets$ média simples sobre $\mathcal{M}$
\For{$w \gets 1$ \KwTo $3$}{
  $\mathcal{M}_{i,s}^{(w)} \gets$ pool fixo de $s$
  $\mathbf{p}_{i,s}^{(w)} \gets$ pesos fixos de $s$ (se houver)
  \If{$s$ seleciona o subconjunto a partir dos dados}{
    $\mathcal{M}_{i,s}^{(w)} \gets$ seleção sobre as janelas
        $\{1, 2, 3\} \setminus \{w\}$
  }
  \If{$s$ ajusta pesos a partir dos dados}{
    \eIf{$w = 1$}{
      $\mathbf{p}_{i,s}^{(w)} \gets$ pesos uniformes
    }{
      $\mathbf{p}_{i,s}^{(w)} \gets$ ajuste sobre as janelas $w' < w$
    }
  }
  $\hat{\mathbf{y}}_{i,s}^{(w)} \gets
      \Act{Combinar}(s, \mathcal{M}_{i,s}^{(w)}, \mathbf{p}_{i,s}^{(w)})$
  $\text{SMAPE}_{i,s}^{(w)} \gets
      \text{SMAPE}(\hat{\mathbf{y}}_{i,s}^{(w)}, \mathbf{y}_i^{(w)})$
  $\text{SMAPE}_{i,s_0}^{(w)} \gets
      \text{SMAPE}(\hat{\mathbf{y}}_{i,s_0}^{(w)}, \mathbf{y}_i^{(w)})$
  $\text{score}_{i,s}^{(w)} \gets
      \text{SMAPE}_{i,s}^{(w)} / \text{SMAPE}_{i,s_0}^{(w)}$
}

$\text{POCID}_i(s) \gets \frac{1}{3}\sum_{w=1}^{3}
    \text{POCID}_i^{(w)}(s)$
$\text{score}_i(s) \gets \sum_{w=1}^{3}
    \alpha_w \text{score}_{i,s}^{(w)}$
\KwRet{$\text{score}_i(s)$}
\end{algorithm}

Os pesos $\alpha_w$ são fixos e priorizam a janela mais recente. O POCID é reportado separadamente e não participa da pontuação usada para ordenar as tentativas. O mecanismo preserva as salvaguardas contra vazamento: a seleção do pool usa as demais janelas, e o ajuste de pesos usa apenas as janelas anteriores à janela pontuada.

\begin{algorithm}[H]
\caption{Diagnóstico interpretativo}
\footnotesize
\label{alg:diagnostico}
\Entrada{Perfil $\mathbf{z}_i$; avaliação do conjunto de preditores $\mathcal{E}_i$}
\Saida{Diagnóstico $\mathcal{C}_i$}

\eIf{força de tendência alta \textbf{e} força sazonal alta}{
  $\text{regime} \gets \text{misto}$
}{
  \eIf{força sazonal alta}{
    $\text{regime} \gets \text{dominado por sazonalidade}$
  }{
    \eIf{força de tendência alta}{
      $\text{regime} \gets \text{dominado por tendência}$
    }{
      $\text{regime} \gets \text{ruidoso}$
    }
  }
}
\eIf{entropia espectral alta}{
  $\text{previsibilidade} \gets \text{baixa}$
}{
  \eIf{entropia espectral baixa}{
    $\text{previsibilidade} \gets \text{alta}$
  }{
    $\text{previsibilidade} \gets \text{média}$
  }
}
\eIf{$\tau$ baixo}{
  $\text{indicação} \gets \text{combinação robusta}$
}{
  \eIf{dispersão de erro alta \textbf{e} $\tau$ razoavelmente estável}{
    $\text{indicação} \gets \text{combinação seletiva}$
  }{
    \eIf{dispersão de erro alta}{
      $\text{indicação} \gets \text{combinação ponderada}$
    }{
      $\text{indicação} \gets \text{conjunto completo}$
    }
  }
}
$\text{riscos} \gets \emptyset$
\If{$\tau$ baixo}{$\text{riscos} \gets \text{riscos} \cup
    \{\text{instabilidade do ranking}\}$}
\If{há grupo de erro correlacionado}{$\text{riscos} \gets \text{riscos} \cup
    \{\text{redundância}\}$}
\If{previsibilidade baixa}{$\text{riscos} \gets \text{riscos} \cup
    \{\text{sobreajuste}\}$}
$\text{riscos} \gets \text{riscos} \cup \{\text{apenas três janelas}\}$
$\text{narrativa} \gets$ síntese textual das categorias
$\mathcal{C}_i \gets (\text{regime}, \text{previsibilidade},
    \text{indicação}, \text{riscos}, \text{narrativa})$
\KwRet{$\mathcal{C}_i$}
\end{algorithm}
```

## Observações de integração

- O algoritmo não contém as variáveis antigas de orçamento `K`, paciência `P`, limite `L`, `S_fix`, `G_{-i}` nem `prior`.
- O histórico começa vazio; não há estratégias iniciais avaliadas.
- As ações de preparação criam handles de pool e de pesos, mas não geram
  pontuação; apenas a avaliação de uma estratégia entra no histórico.
- A pontuação agora é a razão de SMAPE contra a média do pool, agregada pelas
  três janelas com pesos fixos; a subseção `Mecanismo de pontuação` e a Equação
  `eq:score` foram alinhadas no arquivo 05.
- A seção `Preparação em nível de conjunto de dados` foi substituída pelo
  mecanismo de pontuação no arquivo 05; o perfil e o diagnóstico migraram para o
  contexto por série no arquivo 06.
- A condição sazonal não usa os valores reais de teste: ela usa o perfil da
  série, a shortlist de tentativas e as previsões de teste dos candidatos.
- O Algoritmo 3 é determinístico; não há leitura opcional por modelo de
  linguagem.
