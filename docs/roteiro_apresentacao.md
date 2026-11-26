# Roteiro da apresentação (10 minutos)

Abrir `reports/primeiro_degrau.html`, ou o notebook no JupyterLab com o código recolhido. Os números abaixo são os da execução de 26/11/2026. Se os dados forem atualizados, vale conferir no notebook.

| # | O que mostrar | Tempo |
|---|---|---|
| 1 | Seção 0: resposta curta + tabela de vereditos | 1 min 30 |
| 2 | Gráfico 0 (horizonte METR), rápido | 1 min |
| 3 | Gráfico 3 (Stack Overflow vs. tendência anterior) | 1 min 30 |
| 4 | Ato 2: só a linha do veredito e a tabela | 1 min |
| 5 | Gráfico 5 (fatia empregada ≤2 vs. ≥10 anos) | 1 min 30 |
| 6 | Gráfico 7 (conta do gestor) | 1 min 30 |
| 7 | Seção 7 (recomendações e KPIs) | 1 min 30 |
| | Folga para perguntas | ~30 s |

## 1. Seção 0: resumo executivo (1 min 30)
- A pergunta: a IA tirou o primeiro degrau da carreira, ou é só o ciclo de juros?
- A resposta curta vem antes dos gráficos: o lugar onde o iniciante aprendia esvaziou e a fatia empregada de quem tem até 2 anos caiu muito mais que a do sênior.
- Apontar a tabela de vereditos. As regras foram escritas **antes** de rodar os modelos, e por isso dois testes aparecem como "inconclusivo" e "consistente com ciclo". A gente manteve assim.

## 2. Gráfico 0: a IA ficou mais autônoma (1 min)
- Em 3 anos, de tarefas de 5 minutos para perto de 16 horas, que é o limite do que o METR mede.
- As três ondas (chat, assistente, agente) aparecem como linhas verticais; os outros gráficos usam os mesmos marcos.
- Não ficar aqui: é contexto.

## 3. Gráfico 3: Stack Overflow vs. tendência anterior (1 min 30)
- O SARIMAX foi treinado até out/2022 e projeta a tendência que já existia. O real ficou 64% abaixo dela em média nos 48 meses seguintes.
- O modelo erra menos que um ingênuo sazonal no passado, então a projeção é confiável.
- Mesmo assim o veredito é **inconclusivo**. No primeiro ano a queda (−24%) tem o tamanho do placebo da COVID (−30%), e a regra que fixamos antes olha 12 meses.

## 4. Ato 2: vagas (1 min)
- Ler só a linha do veredito: **consistente com ciclo**.
- Com juros e a inércia da série no modelo, o efeito pós-ChatGPT fica em −3,3% (p = 0,67).
- Este é o ponto contra a nossa tese, e é bom dizer isso em voz alta: nas vagas, os dados não separam IA do fim do boom de 2021–22.

## 5. Gráfico 5: o primeiro degrau (1 min 30)
- É o resultado principal. De 2025 para 2026, a fatia empregada de quem tem até 2 anos caiu entre 9 e 20 pontos, conforme o método. A do sênior caiu ~2.
- Usamos três métodos porque quem respondeu à pergunta de experiência mudou entre os anos.
- Bate com estudos dos EUA (Stanford/ADP e Harvard). É associação, não causa.

## 6. Gráfico 7: a conta do gestor (1 min 30)
- É simulação, não observação de empresas.
- Se o júnior rende metade de um sênior, basta a IA render +5,8% para o sênior sozinho sair mais barato. Se rende 70%, precisa de +19,9%.
- H2 é parcial porque tudo depende dessa premissa sobre o júnior.

## 7. Seção 7: o que fazer (1 min 30)
- Estudante: a fatia empregada caiu e a razão salarial subiu para 2,3×, então vale buscar entrada antes de se formar e acumular experiência que dê para provar. Usar agentes é requisito, não proteção.
- Gestor: cortar o júnior pode sair caro depois, porque tende a secar a fonte de sêniores. A sugestão é manter um pipeline júnior menor e intencional.
- Fechar com os KPIs e com a próxima rodada: refazer tudo com o survey de 2027.

## Perguntas prováveis do professor

**Por que o Ato 1 é inconclusivo, se o gráfico mostra uma queda enorme?**
Porque a regra foi definida antes: o efeito do primeiro ano precisa ser maior que o dobro de qualquer placebo. O placebo de jun/2020, quando a COVID se desfez, deu −30%, e o primeiro ano real deu −24%. Nos 48 meses nenhum placebo chega perto, mas mudar a regra depois de ver o resultado seria exatamente o que quisemos evitar.

**Por que o Ato 2 mudou para AR(1)?**
A primeira regressão (erros HAC) dava −28% com p < 0,001. Na revisão, os resíduos se mostraram muito autocorrelacionados (Durbin-Watson 0,20), e com cerca de 80 meses o HAC não corrige isso. O modelo com erros AR(1) leva essa inércia em conta, e aí o efeito some (−3,3%, p = 0,67). O modelo antigo está no Apêndice, marcado como substituído.

**O survey não tem viés de seleção?**
Tem. Quem responde é quem visita o Stack Overflow, e em 2026 há mais estudantes entre os iniciantes. Por isso usamos três métodos, entre eles a reponderação pela chance de responder e o recorte só com quem tem 2 anos, e tiramos 2023–24 da comparação de emprego. A queda aparece nos três e também sem os estudantes.

**Isso prova que a IA causou a queda?**
Não. Tudo é "consistente com" ou "associado a". Nenhum dos nossos dados tem grupo de controle aleatório; o mais perto disso são os estudos citados, e os próprios autores chamam os resultados de descritivos.

**Por que não usar dados do Brasil?**
A série do Indeed Hiring Lab que usamos é dos EUA e não tem o Brasil, e montar o CAGED por ocupação e idade não cabia no prazo de um dia. O survey tem respondentes brasileiros, mas não testamos se o recorte de até 2 anos com salário teria amostra suficiente. Fica para a próxima iteração.
