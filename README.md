# Análise de Grafo — Wikipedia 2006

Script Python (`wiki_graph_analysis.py`) para analisar o grafo de links
internos da Wikipedia em inglês (2006), a partir de um CSV no formato:

```
page_id_from, page_title_from, page_id_to, page_title_to
```

## Instalação

```bash
pip install networkx pandas matplotlib numpy scipy
```//
(`scipy` acelera bastante o cálculo de eigenvector centrality / PageRank
via `nx.eigenvector_centrality_numpy`, e é opcional.)

## Uso básico

```bash
python wiki_graph_analysis.py caminho/para/wikipedia_2006.csv
```

Isso roda o pipeline completo e imprime um relatório no console com todas
as métricas pedidas.

## Opções importantes

| Flag | Descrição |
|---|---|
| `--nrows N` | Lê apenas as primeiras N linhas do CSV. Ótimo para testar rápido antes de rodar no arquivo completo (que pode ter milhões de linhas). |
| `--exact` | Força o cálculo **exato** de métricas caras (average path length, diâmetro, betweenness centrality). Pode levar horas/dias em grafos com centenas de milhares de nós — use com cautela. |
| `--sample-nodes N` | Tamanho da amostra usada nas aproximações (padrão: 500). Quanto maior, mais preciso e mais lento. |
| `--top-k N` | Quantos nós mostrar em cada ranking de centralidade (padrão: 10). |
| `--plot` | Gera `degree_distribution.png` com o gráfico log-log da distribuição de graus. |
| `--plot-out caminho.png` | Define o caminho de saída do gráfico. |

## Por que existe modo aproximado?

O grafo de links da Wikipedia 2006 tende a ter **centenas de milhares de
nós e milhões de arestas**. Algumas métricas clássicas são computacionalmente
caras nesse tamanho:

- **Betweenness centrality**: complexidade O(n·m) no algoritmo exato de
  Brandes — inviável para grafos grandes. O script usa a versão com
  amostragem (`k=sample_nodes`) do próprio NetworkX, que estima a métrica
  a partir de um subconjunto de nós-fonte.
- **Average path length** e **diâmetro**: exigem BFS a partir de (idealmente)
  todos os nós. O script faz BFS apenas a partir de uma amostra de nós e
  estima a média (average path length) ou um limite inferior (diâmetro).
- **Clustering coefficient**: por padrão também calculado sobre amostra
  quando o grafo é grande.

Use `--exact` apenas em grafos pequenos/médios (até dezenas de milhares de
nós, dependendo do hardware) ou tenha paciência — pode ser bem lento.

## Recomendação de fluxo de trabalho

1. Primeiro rode com `--nrows 5000` (ou similar) para validar que o CSV
   está sendo lido corretamente e ver um relatório rápido.
2. Depois rode no arquivo completo, sem `--exact`, para obter as métricas
   aproximadas em tempo razoável.
3. Se quiser refinar alguma métrica específica (ex: diâmetro exato do maior
   componente), pode chamar as funções individualmente em um script/
   notebook próprio, importando `wiki_graph_analysis.py` como módulo — todas
   as funções (`densidade`, `pagerank`, `betweenness_centrality` etc.) podem
   ser usadas separadamente:

```python
from wiki_graph_analysis import carregar_dados, construir_grafo, pagerank

df = carregar_dados("wikipedia_2006.csv")
G = construir_grafo(df)
pr = pagerank(G)
```

## Sobre a direção do grafo

O grafo é **direcionado** (um link de A para B não implica B para A), o que
é o correto para links da Wikipedia. Por isso:

- **PageRank** é calculado diretamente sobre o grafo direcionado (é a
  aplicação original/clássica do algoritmo).
- **Componentes conexos** são reportados tanto na versão fraca (ignorando
  direção) quanto forte (respeitando direção).
- **Clustering coefficient**, **average path length** e **diâmetro** usam
  a versão não-direcionada do grafo, seguindo a convenção clássica dessas
  métricas.
- **Closeness centrality** é calculada sobre o grafo revertido (`G.reverse()`),
  medindo o quão "central" um nó é do ponto de vista de quem chega até ele
  — interpretação usual em redes de citação/links.
