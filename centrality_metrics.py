import networkx as nx
from utilities import Utilities


@Utilities.timeit
def compute_degree_centrality(G):
    """Degree centrality (in, out and total) for every node."""
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
    Closeness centrality. Uses NetworkX's implementation, which already
    handles disconnected graphs correctly (normalizing by the size of the
    reachable component). For directed graphs, closeness is computed based
    on in-distances by default; here we use G.reverse() to measure "how
    easily other nodes can reach this node" (the most common
    interpretation for citation/link networks).
    """
    return nx.closeness_centrality(G.reverse())

@Utilities.timeit
def compute_betweenness_centrality(G, sample_k=None, seed=42):
    """
    Betweenness centrality.

    Exact cost: O(n*m) — prohibitive for large graphs. If `sample_k` is
    given, uses NetworkX's `k` parameter to approximate via source-node
    sampling (Brandes' algorithm with sampling).
    """
    if sample_k:
        # NetworkX raises if k exceeds the number of nodes (e.g. a small
        # test graph with the default sample size) — clamp defensively.
        sample_k = min(sample_k, G.number_of_nodes())
        return nx.betweenness_centrality(G, k=sample_k, seed=seed, normalized=True)
    return nx.betweenness_centrality(G, normalized=True)

@Utilities.timeit
def compute_eigenvector_centrality(G, max_iter=1000, tol=1e-06):
    """
    Eigenvector centrality. May fail to converge on some directed graphs
    with pathological structure (e.g. many nodes with no in-edges); in
    that case we fall back to the numpy-based version (more robust) and,
    as a last resort, return None with a warning.
    """
    try:
        return nx.eigenvector_centrality(G, max_iter=max_iter, tol=tol)
    except nx.PowerIterationFailedConvergence:
        Utilities.log("  Warning: eigenvector_centrality (power iteration) did not "
            "converge; trying eigenvector_centrality_numpy...")
        try:
            return nx.eigenvector_centrality_numpy(G)
        except Exception as e:
            Utilities.log(f"  Warning: eigenvector_centrality_numpy also failed ({e}). "
                "Returning None.")
            return None

@Utilities.timeit
def compute_pagerank(G, alpha=0.85):
    """
    PageRank — a natural metric that fits Wikipedia link graphs
    particularly well, since it was originally designed for that kind of
    network (web page links).
    """
    return nx.pagerank(G, alpha=alpha)
