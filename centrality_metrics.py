import networkx as nx
import igraph as ig
from utilities import Utilities

@Utilities.timeit
def compute_degree_centrality(G):
    """Degree centrality (in, out e total) para cada nó."""
    return {
        "in": nx.in_degree_centrality(G),
        "out": nx.out_degree_centrality(G),
        "total": {
            n: (G.in_degree(n) + G.out_degree(n)) / (G.number_of_nodes() - 1)
            for n in G.nodes()
        } if G.number_of_nodes() > 1 else {n: 0 for n in G.nodes()},
    }

@Utilities.timeit
def compute_closeness_centrality(G):
    """
    Closeness centrality. Usa a implementação do NetworkX, que já lida
    corretamente com grafos desconexos (normaliza pelo tamanho do
    componente alcançável). Para grafos direcionados, closeness é
    calculada com base no caminho de entrada (in-distances) por padrão
    aqui usamos G.reverse() para medir "o quão perto os outros nós
    conseguem chegar até este nó" (interpretação mais comum em redes de
    citação/links).
    """
    return nx.closeness_centrality(G.reverse())

@Utilities.timeit
def compute_betweenness_centrality(G, amostra_k=None, seed=42):
    """
    Betweenness centrality.

    Custo exato: O(n*m) — proibitivo para grafos grandes. Se `amostra_k`
    for informado, usa o parâmetro `k` do NetworkX para aproximar via
    amostragem de nós-fonte (algoritmo de Brandes com amostragem).
    """
    if amostra_k:
        return nx.betweenness_centrality(G, k=amostra_k, seed=seed, normalized=True)
    return nx.betweenness_centrality(G, normalized=True)

@Utilities.timeit
def ig_compute_betweenness_centrality(g, cutoff=None):
    n = g.vcount()
    raw = g.betweenness(directed=True, cutoff=cutoff)

    # igraph returns unnormalized betweenness; normalize the same way
    # networkx does with normalized=True (divide by the number of ordered
    # pairs of other nodes)
    norm = (n - 1) * (n - 2) if n > 2 else 1
    return {i: v / norm for i, v in enumerate(raw)}

@Utilities.timeit
def compute_eigenvector_centrality(G, max_iter=1000, tol=1e-06):
    """
    Eigenvector centrality. Pode não convergir em alguns grafos
    direcionados com estrutura patológica (ex: muitos nós sem
    in-edges); nesses casos, caímos de volta para a versão via numpy
    (mais robusta) e, em último caso, retornamos None com aviso.
    """
    try:
        return nx.eigenvector_centrality(G, max_iter=max_iter, tol=tol)
    except nx.PowerIterationFailedConvergence:
        Utilities.log("  Aviso: eigenvector_centrality (power iteration) não convergiu; "
            "tentando eigenvector_centrality_numpy...")
        try:
            return nx.eigenvector_centrality_numpy(G)
        except Exception as e:
            Utilities.log(f"  Aviso: eigenvector_centrality_numpy também falhou ({e}). "
                "Retornando None.")
            return None

@Utilities.timeit
def compute_pagerank(G, alpha=0.85):
    """
    PageRank — métrica natural e especialmente adequada para grafos de
    links da Wikipedia, já que foi originalmente desenhada para esse tipo
    de rede (links de páginas web).
    """
    return nx.pagerank(G, alpha=alpha)