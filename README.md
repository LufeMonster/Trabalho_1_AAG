# Graph Analysis — Wikipedia Link Graph

Python script (`wiki_graph_analysis.py`) for analyzing the internal link
graph of English Wikipedia, from a CSV file in the format:

```
page_id_from, page_title_from, page_id_to, page_title_to
```

The script works with any dump in this format — the examples below use an
English Wikipedia 2003 link-graph dataset, but any similarly-shaped
edge list will do.

## Installation

```bash
pip install networkx pandas matplotlib numpy scipy
```

(`scipy` significantly speeds up eigenvector centrality / PageRank via
`nx.eigenvector_centrality_numpy`, and is optional.)

## Basic usage

```bash
python wiki_graph_analysis.py path/to/wikipedia_2003.csv
```

This loads the graph once and opens an interactive menu where you can run
individual metrics, or all of them at once, without re-reading the CSV.

For a one-shot, non-interactive run (useful for scripts/automation):

```bash
python wiki_graph_analysis.py path/to/wikipedia_2003.csv --all
```

## Important options

| Flag | Description |
|---|---|
| `--nrows N` | Reads only the first N rows of the CSV. Great for quick testing before running on the full file (which can have millions of rows). |
| `--exact` | Forces **exact** computation of the expensive metrics (clustering coefficient, average path length, diameter, betweenness centrality). Can take a very long time on graphs with hundreds of thousands of nodes — use with caution. |
| `--sample-nodes N` | Sample size used for the approximations (default: 500). The larger it is, the more accurate and the slower. |
| `--top-k N` | How many nodes to show in each centrality ranking (default: 10). |
| `--plot` | Generates `degree_distribution.png` with a log-log plot of the degree distribution. |
| `--plot-out path.png` | Sets the output path for the plot. |
| `--sep` | Forces the field separator manually (e.g. `'\t'` for TAB). Auto-detected by default. |
| `--all` | Runs every metric once and exits, skipping the interactive menu. |

## Why is there an approximate mode?

The Wikipedia link graph tends to have **hundreds of thousands of nodes
and millions of edges**. Some classic metrics are computationally
expensive at that scale:

- **Betweenness centrality**: O(n·m) complexity in Brandes' exact
  algorithm — infeasible for large graphs. The script uses NetworkX's own
  sampling variant (`k=sample_nodes`), which estimates the metric from a
  subset of source nodes.
- **Average path length**: an exact computation requires a BFS from every
  node in the largest component. The script instead runs BFS from a
  random sample of nodes and averages the resulting distances.
- **Diameter**: an exact computation needs the eccentricity of every node.
  The script estimates a lower bound by taking the largest eccentricity
  found from a random sample of nodes — cheap to compute and always a
  safe (conservative) estimate.
- **Clustering coefficient**: by default also estimated over a sample of
  nodes when `--exact` is not passed.

By default (no `--exact`), all four of these metrics use the sampling
approximation described above with `--sample-nodes` (default 500) source
nodes. Pass `--exact` to compute them precisely instead — recommended
only for small/medium graphs (up to a few tens of thousands of nodes,
depending on your hardware), or if you're prepared to wait.

All other metrics (node/edge counts, average degree, degree distribution,
density, connected components, degree centrality, closeness centrality,
eigenvector centrality, PageRank) are always computed exactly — they are
cheap enough even on large graphs.

## Recommended workflow

1. First run with `--nrows 5000` (or similar) to confirm the CSV is being
   read correctly and to see a quick report.
2. Then run on the full file without `--exact`, to get the approximate
   metrics in a reasonable amount of time.
3. If you want to refine a specific metric (e.g. the exact diameter of
   the largest component), you can call the individual functions from
   your own script/notebook by importing `wiki_graph_analysis.py` as a
   module — every function (`get_density`, `compute_pagerank`,
   `get_diameter`, etc.) can be used on its own:

```python
from graph import Graph
from wiki_graph_analysis import get_density, get_diameter
import centrality_metrics

g = Graph()
df = g.load_data("wikipedia_2003.csv")
G = g.build_graph(df)

density = get_density(G)
pr = centrality_metrics.compute_pagerank(G)
```

## About the direction of the graph

The graph is **directed** (a link from A to B does not imply B to A),
which is the correct model for Wikipedia links. Because of that:

- **PageRank** is computed directly on the directed graph (this is the
  algorithm's original/classic application).
- **Connected components** are reported both as weakly connected
  (ignoring direction) and strongly connected (respecting direction).
- **Clustering coefficient**, **average path length** and **diameter**
  use the undirected version of the graph, following the classic
  convention for these metrics.
- **Closeness centrality** is computed on the reversed graph
  (`G.reverse()`), measuring how "central" a node is from the point of
  view of the nodes that link to it — the usual interpretation in
  citation/link networks.

## Project files

| File | Purpose |
|---|---|
| `wiki_graph_analysis.py` | CLI entry point, orchestration, interactive menu, and the basic structural metrics (degree, density, clustering, path length, diameter, components). |
| `centrality_metrics.py` | Centrality metrics: degree, closeness, betweenness, eigenvector, and PageRank. |
| `graph.py` | CSV/TSV loading (with separator auto-detection) and graph construction. |
| `utilities.py` | Small shared helpers: timestamped logging and a `@timeit` decorator used to time every step. |
