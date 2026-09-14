"""
Wikipedia Link Graph Analysis
=============================

Reads a CSV file with columns:
    page_id_from, page_title_from, page_id_to, page_title_to

and computes a standard set of network/graph analysis metrics:

Basic structure:
    - number of nodes and edges
    - average degree
    - degree distribution
    - density
    - clustering coefficient
    - average path length
    - diameter
    - connected components

Centrality (most important nodes):
    - degree centrality
    - closeness centrality
    - betweenness centrality
    - eigenvector centrality
    - PageRank

The Wikipedia link graph is DIRECTED (a link from A to B does not imply
the reverse). The script builds the directed version (for PageRank,
strongly/weakly connected components, etc.) and, when needed, also works
with the undirected version (traditionally used for clustering
coefficient, "classic" diameter, etc.).

Since real graphs like this can have hundreds of thousands or millions of
nodes and edges, some metrics (average path length, diameter, betweenness)
have prohibitive complexity for an exact computation. Because of that, the
script:
    - computes these metrics EXACTLY when the graph is small enough;
    - automatically falls back to SAMPLING (approximation) when the graph
      is large, and always makes this explicit in the output.

Usage:
    python wiki_graph_analysis.py path/to/file.csv
    python wiki_graph_analysis.py path/to/file.csv --sample-nodes 12000
    python wiki_graph_analysis.py path/to/file.csv --exact
    python wiki_graph_analysis.py path/to/file.csv --top-k 15 --plot
"""

import argparse
import random
import sys
from collections import Counter

import networkx as nx

import centrality_metrics
from graph import Graph
from utilities import Utilities

graph: Graph = Graph()


# --------------------------------------------------------------------------
# 1. Data loading and graph construction -> see Graph.load_data / build_graph
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 2. Basic structural metrics
# --------------------------------------------------------------------------

@Utilities.timeit
def count_nodes_and_edges(G):
    """Returns (number of nodes, number of edges)."""
    return G.number_of_nodes(), G.number_of_edges()

@Utilities.timeit
def get_average_degree(G):
    """
    Average degree of the graph.
    For directed graphs, returns the average in-degree and average
    out-degree (which are always equal in total value, since every edge
    contributes +1 to one out-degree and +1 to one in-degree), plus the
    average total degree (in + out) per node.
    """
    n = G.number_of_nodes()
    if n == 0:
        return {"in_average": 0, "out_average": 0, "total_average": 0}

    in_sum = sum(d for _, d in G.in_degree())
    out_sum = sum(d for _, d in G.out_degree())

    return {
        "in_average": in_sum / n,
        "out_average": out_sum / n,
        "total_average": (in_sum + out_sum) / n,
    }


@Utilities.timeit
def get_degree_distribution(G, kind="total"):
    """
    Returns the degree distribution as a Counter {degree: number_of_nodes}.

    kind: 'in', 'out' or 'total' (in+out), applicable to directed graphs.
    """
    if kind == "in":
        degrees = [d for _, d in G.in_degree()]
    elif kind == "out":
        degrees = [d for _, d in G.out_degree()]
    else:
        degrees = [G.in_degree(n) + G.out_degree(n) for n in G.nodes()]

    return Counter(degrees)


@Utilities.timeit
def get_density(G):
    """Graph density (ratio of existing edges to the maximum possible)."""
    return nx.density(G)


@Utilities.timeit
def get_clustering_coefficient(G, exact=True, sample_nodes=12000, seed=42, undirected=None):
    """
    Average clustering coefficient (average local transitivity).
    NetworkX computes clustering on undirected graphs (or handles directed
    graphs with its own definition); here we convert to undirected, which
    is the classic (Watts-Strogatz) approach.

    If `exact` is False, the average is estimated over a random sample of
    `sample_nodes` nodes instead of the whole graph — much faster on large
    graphs, at the cost of being an approximation.

    `undirected` can be passed in to reuse an already-computed
    `G.to_undirected()` instead of recomputing it here.
    """
    Gu = undirected if undirected is not None else G.to_undirected()

    if exact:
        average = nx.average_clustering(Gu)
        return {"average": average, "exact": True}

    n = Gu.number_of_nodes()
    k = min(sample_nodes, n)
    rng = random.Random(seed)
    sample = rng.sample(list(Gu.nodes()), k) if k > 0 else []
    approximate_average = nx.average_clustering(Gu, nodes=sample) if sample else 0.0
    return {"approximate_average": approximate_average, "sampled_nodes": k, "exact": False}


@Utilities.timeit
def get_connected_components(G):
    """
    For a directed graph, computes:
      - the number and size of WEAKLY connected components
      - the number and size of STRONGLY connected components
    """
    weak = sorted(
        (len(c) for c in nx.weakly_connected_components(G)), reverse=True
    )
    strong = sorted(
        (len(c) for c in nx.strongly_connected_components(G)), reverse=True
    )

    return {
        "weak_component_count": len(weak),
        "largest_weak_component": weak[0] if weak else 0,
        "weak_top5_sizes": weak[:5],
        "strong_component_count": len(strong),
        "largest_strong_component": strong[0] if strong else 0,
        "strong_top5_sizes": strong[:5],
    }


def _largest_component_as_subgraph(G):
    """
    Returns the subgraph corresponding to the largest connected component.
    Accepts either a directed graph (uses weakly connected components) or
    an already-undirected graph (uses "regular" connected components).
    """
    if G.is_directed():
        largest = max(nx.weakly_connected_components(G), key=len)
    else:
        largest = max(nx.connected_components(G), key=len)
    return G.subgraph(largest).copy()


@Utilities.timeit
def get_average_path_length(G, exact=True, sample_nodes=12000, seed=42,
                             undirected=None, component=None):
    """
    Average shortest path length.

    NetworkX's exact calculation requires the graph to be (strongly, if
    directed) connected. Since real graphs are rarely fully connected, we
    apply the computation to the largest connected component.

    If `exact` is False, the average is estimated by running a
    single-source BFS from a random sample of `sample_nodes` nodes in the
    largest component and averaging the resulting distances — this avoids
    the O(n^2) cost of an all-pairs computation on large graphs.

    `undirected` / `component` can be passed in to reuse graphs already
    computed by the caller (see `generate_complete_log`).
    """
    Gu = undirected if undirected is not None else G.to_undirected()
    comp = component if component is not None else _largest_component_as_subgraph(Gu)
    n = comp.number_of_nodes()

    if exact:
        average = nx.average_shortest_path_length(comp)
        return {"average": average, "largest_component_nodes": n, "exact": True}

    k = min(sample_nodes, n)
    rng = random.Random(seed)
    sample = rng.sample(list(comp.nodes()), k) if k > 0 else []

    total_distance = 0
    total_pairs = 0
    for source in sample:
        lengths = nx.single_source_shortest_path_length(comp, source)
        total_distance += sum(d for target, d in lengths.items() if target != source)
        total_pairs += len(lengths) - 1

    approximate_average = total_distance / total_pairs if total_pairs else 0.0
    return {
        "approximate_average": approximate_average,
        "sampled_nodes": k,
        "largest_component_nodes": n,
        "exact": False,
    }


@Utilities.timeit
def get_diameter(G, exact=True, sample_nodes=12000, seed=42, undirected=None, component=None):
    """
    Graph diameter (largest shortest-path distance between two nodes),
    computed on the largest connected component (undirected version).

    If `exact` is False, estimates a LOWER BOUND for the diameter: it runs
    a BFS-based eccentricity computation from a random sample of
    `sample_nodes` nodes and returns the largest eccentricity found. This
    is much cheaper than the exact algorithm (which needs eccentricities
    from every node) and, being a lower bound, is a safe, conservative
    estimate for large graphs.
    """
    Gu = undirected if undirected is not None else G.to_undirected()
    comp = component if component is not None else _largest_component_as_subgraph(Gu)
    n = comp.number_of_nodes()

    if exact:
        d = nx.diameter(comp)
        return {"diameter": d, "largest_component_nodes": n, "exact": True}

    k = min(sample_nodes, n)
    rng = random.Random(seed)
    sample = rng.sample(list(comp.nodes()), k) if k > 0 else []

    lower_bound = 0
    for source in sample:
        lengths = nx.single_source_shortest_path_length(comp, source)
        if lengths:
            lower_bound = max(lower_bound, max(lengths.values()))

    return {
        "approximate_diameter_lower_bound": lower_bound,
        "sampled_nodes": k,
        "largest_component_nodes": n,
        "exact": False,
    }

# --------------------------------------------------------------------------
# 3. Reports / result presentation
# --------------------------------------------------------------------------

def top_k(values_dict, G, k=10):
    """Returns the k nodes with the highest value in `values_dict`, with labels (titles)."""
    ranked = sorted(values_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(graph.label(G, nid), nid, value) for nid, value in ranked]


def print_top_k(metric_name, top_list, k=10):
    print(f"\nTop {min(k, len(top_list))} nodes by {metric_name}:")
    for i, (title, nid, value) in enumerate(top_list, start=1):
        print(f"  {i:2d}. {title!r} (id={nid})  ->  {value:.6f}")


def plot_degree_distribution(degree_dist, title="Degree distribution", output_path=None):
    """Generates a log-log plot of the degree distribution (typical of scale-free networks)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degrees = sorted(degree_dist.keys())
    counts = [degree_dist[d] for d in degrees]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(degrees, counts, s=12, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree (k)")
    ax.set_ylabel("Number of nodes with degree k")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        Utilities.log(f"Plot saved to: {output_path}")
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. Per-metric print functions (used both by the full report and by the
#    interactive menu, to avoid code duplication)
# --------------------------------------------------------------------------

def print_graph_basic_data(G):
    num_nodes, num_edges = count_nodes_and_edges(G)
    print(f"Number of nodes: {num_nodes:,}")
    print(f"Number of edges: {num_edges:,}")
    return num_nodes, num_edges


def print_average_degree(G):
    avg_degree = get_average_degree(G)
    print(f"Average degree (in):    {avg_degree['in_average']:.4f}")
    print(f"Average degree (out):   {avg_degree['out_average']:.4f}")
    print(f"Average degree (total): {avg_degree['total_average']:.4f}")
    return avg_degree


def print_degree_distribution(G, plot=False, plot_output_path="degree_distribution.png"):
    dist = get_degree_distribution(G, kind="total")
    print(f"Degree distribution: {len(dist)} distinct degree values "
          f"(maximum degree observed: {max(dist)})")
    print("  (degree : number of nodes with that degree) — first 10 values:")
    for d in sorted(dist.keys())[:10]:
        print(f"    {d:>4} : {dist[d]:,}")
    if plot:
        plot_degree_distribution(
            dist, title="Wikipedia Link Graph - Degree Distribution",
            output_path=plot_output_path,
        )
    return dist


def print_density(G):
    density = get_density(G)
    print(f"Density: {density:.8f}")
    return density


def print_clustering_coefficient(G, exact=True, sample_nodes=12000, undirected=None):
    clustering = get_clustering_coefficient(G, exact=exact, sample_nodes=sample_nodes, undirected=undirected)
    if clustering.get("exact"):
        print(f"Clustering coefficient (average, exact): {clustering['average']:.6f}")
    else:
        print(f"Clustering coefficient (average, approximate, "
              f"n={clustering['sampled_nodes']}): {clustering['approximate_average']:.6f}")
    return clustering


def print_components(G):
    comp = get_connected_components(G)
    print(f"Weakly connected components: {comp['weak_component_count']} "
          f"(largest: {comp['largest_weak_component']:,} nodes)")
    print(f"  Top 5 sizes: {comp['weak_top5_sizes']}")
    print(f"Strongly connected components: {comp['strong_component_count']} "
          f"(largest: {comp['largest_strong_component']:,} nodes)")
    print(f"  Top 5 sizes: {comp['strong_top5_sizes']}")
    return comp


def print_average_path_length(G, exact=True, sample_nodes=12000, undirected=None, component=None):
    apl = get_average_path_length(G, exact=exact, sample_nodes=sample_nodes,
                                   undirected=undirected, component=component)
    if apl.get("exact"):
        print(f"Average path length (exact, largest component, "
              f"n={apl['largest_component_nodes']:,}): {apl['average']:.4f}")
    else:
        print(f"Average path length (approximate, sample={apl['sampled_nodes']}, "
              f"largest component n={apl['largest_component_nodes']:,}): "
              f"{apl['approximate_average']:.4f}")
    return apl


def print_diameter(G, exact=True, sample_nodes=12000, undirected=None, component=None):
    diam = get_diameter(G, exact=exact, sample_nodes=sample_nodes,
                         undirected=undirected, component=component)
    if diam.get("exact"):
        print(f"Diameter (exact, largest component): {diam['diameter']}")
    else:
        print(f"Diameter (approximate lower bound, sample="
              f"{diam['sampled_nodes']}): {diam['approximate_diameter_lower_bound']}")
    return diam


def print_degree_centrality(G, top_k_n):
    dc = centrality_metrics.compute_degree_centrality(G)
    print_top_k("degree centrality (total)", top_k(dc["total"], G, top_k_n), top_k_n)
    return dc


def print_closeness_centrality(G, top_k_n):
    Utilities.log("Computing closeness centrality (can take a while on large graphs)...")
    clo = centrality_metrics.compute_closeness_centrality(G)
    print_top_k("closeness centrality", top_k(clo, G, top_k_n), top_k_n)
    return clo


def print_betweenness_centrality(G, exact, sample_nodes, top_k_n):
    sample_size = None if exact else min(sample_nodes, G.number_of_nodes())
    Utilities.log("Computing betweenness centrality (can take a long time)...")
    bet = centrality_metrics.compute_betweenness_centrality(G, sample_k=sample_size)
    print_top_k(
        f"betweenness centrality {'(approximate, k=' + str(sample_size) + ')' if sample_size else '(exact)'}",
        top_k(bet, G, top_k_n), top_k_n,
    )
    return bet


def print_eigenvector_centrality(G, top_k_n):
    Utilities.log("Computing eigenvector centrality...")
    eig = centrality_metrics.compute_eigenvector_centrality(G)
    if eig is not None:
        print_top_k("eigenvector centrality", top_k(eig, G, top_k_n), top_k_n)
    else:
        print("\nEigenvector centrality: could not be computed (did not converge).")
    return eig


def print_pagerank(G, top_k_n):
    Utilities.log("Computing PageRank...")
    pr = centrality_metrics.compute_pagerank(G)
    print_top_k("PageRank", top_k(pr, G, top_k_n), top_k_n)
    return pr


# --------------------------------------------------------------------------
# 5. Orchestration — full report (all metrics at once)
# --------------------------------------------------------------------------

def generate_complete_log(G, exact=False, sample_nodes=12000, top_k_n=10,
                           plot=False, plot_output_path="degree_distribution.png"):
    """
    Computes and prints ALL metrics, in the requested order, for an
    already-loaded graph G. Returns a dictionary with all the results.
    """
    print("\n" + "=" * 70)
    print("BASIC STRUCTURAL METRICS")
    print("=" * 70)

    num_nodes, num_edges = print_graph_basic_data(G)
    avg_degree = print_average_degree(G)
    dist = print_degree_distribution(G, plot=plot, plot_output_path=plot_output_path)
    density = print_density(G)

    # The undirected graph and its largest connected component are reused
    # by clustering coefficient, average path length and diameter instead
    # of being recomputed for each metric — on a large graph, converting
    # to undirected and finding weakly connected components are themselves
    # expensive steps, so computing them once here avoids doing that work
    # three times over.
    undirected = G.to_undirected()
    largest_component = _largest_component_as_subgraph(undirected)

    clustering = print_clustering_coefficient(G, exact=exact, sample_nodes=sample_nodes, undirected=undirected)
    comp = print_components(G)
    apl = print_average_path_length(G, exact=exact, sample_nodes=sample_nodes,
                                     undirected=undirected, component=largest_component)
    diam = print_diameter(G, exact=exact, sample_nodes=sample_nodes,
                           undirected=undirected, component=largest_component)

    print("\n" + "=" * 70)
    print("CENTRALITY — MOST IMPORTANT NODES")
    print("=" * 70)

    dc = print_degree_centrality(G, top_k_n)
    clo = print_closeness_centrality(G, top_k_n)
    bet = print_betweenness_centrality(G, exact, sample_nodes, top_k_n)
    eig = print_eigenvector_centrality(G, top_k_n)
    pr = print_pagerank(G, top_k_n)

    return {
        "graph": G,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "average_degree": avg_degree,
        "degree_distribution": dist,
        "density": density,
        "clustering": clustering,
        "components": comp,
        "average_path_length": apl,
        "diameter": diam,
        "degree_centrality": dc,
        "closeness_centrality": clo,
        "betweenness_centrality": bet,
        "eigenvector_centrality": eig,
        "pagerank": pr,
    }


def analyze_graph(csv_path, nrows=None, exact=False, sample_nodes=12000,
                   top_k_n=10, plot=False, plot_output_path="degree_distribution.png",
                   sep=None):
    """
    Reads the CSV, builds the graph and runs the full analysis pipeline
    (all metrics at once), printing a report to the console.

    Kept for non-interactive / programmatic use (e.g. calling from another
    script or notebook, or via the --all flag in CLI mode).
    """
    df = graph.load_data(csv_path, nrows=nrows, sep=sep)
    G = graph.build_graph(df)
    return generate_complete_log(
        G, exact=exact, sample_nodes=sample_nodes, top_k_n=top_k_n,
        plot=plot, plot_output_path=plot_output_path,
    )


# --------------------------------------------------------------------------
# 6. Interactive menu
# --------------------------------------------------------------------------

MENU_OPTIONS = [
    ("1", "Number of nodes and edges"),
    ("2", "Average degree"),
    ("3", "Degree distribution"),
    ("4", "Density"),
    ("5", "Clustering coefficient"),
    ("6", "Average path length"),
    ("7", "Diameter"),
    ("8", "Connected components"),
    ("9", "Degree centrality"),
    ("10", "Closeness centrality"),
    ("11", "Betweenness centrality"),
    ("12", "Eigenvector centrality"),
    ("13", "PageRank"),
    ("14", "Run ALL metrics (full report)"),
    ("15", "Change settings (exact/approximate mode, sample size, top-k, plot)"),
    ("0", "Exit"),
]


def show_menu(G, config):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    mode = "EXACT" if config["exact"] else f"approximate (sample={config['sample_nodes']})"

    print("\n" + "=" * 70)
    print(" MAIN MENU — Wikipedia Graph Analysis")
    print("=" * 70)
    print(f" Graph loaded: {num_nodes:,} nodes, {num_edges:,} edges")
    print(f" Current settings: mode={mode} | top-k={config['top_k']} | "
          f"plot={'on' if config['plot'] else 'off'}")
    print("-" * 70)
    print(" --- Basic structural metrics ---")
    for key, name in MENU_OPTIONS[0:8]:
        print(f"  {key:>2}. {name}")
    print(" --- Centrality (most important nodes) ---")
    for key, name in MENU_OPTIONS[8:13]:
        print(f"  {key:>2}. {name}")
    print(" --- Other options ---")
    for key, name in MENU_OPTIONS[13:]:
        print(f"  {key:>2}. {name}")
    print("=" * 70)


def menu_settings(config):
    """Submenu for changing computation settings at runtime."""
    while True:
        print("\n" + "-" * 70)
        print(" SETTINGS")
        print("-" * 70)
        print(f"  1. Computation mode (current: "
              f"{'exact' if config['exact'] else 'approximate'})")
        print(f"  2. Sample size for approximations (current: {config['sample_nodes']})")
        print(f"  3. Top-k nodes shown in rankings (current: {config['top_k']})")
        print(f"  4. Generate degree-distribution plot (current: "
              f"{'yes' if config['plot'] else 'no'})")
        print(f"  0. Back to main menu")
        print("-" * 70)
        choice = input("Choose an option: ").strip()

        if choice == "1":
            response = input("Use EXACT mode for expensive metrics? "
                          "(can be very slow on large graphs) [y/N]: ").strip().lower()
            config["exact"] = response == "y"
        elif choice == "2":
            try:
                config["sample_nodes"] = int(input("New sample size: ").strip())
            except ValueError:
                print("Invalid value, keeping the previous one.")
        elif choice == "3":
            try:
                config["top_k"] = int(input("New top-k value: ").strip())
            except ValueError:
                print("Invalid value, keeping the previous one.")
        elif choice == "4":
            response = input("Generate the degree-distribution plot whenever that "
                          "metric is computed? [y/N]: ").strip().lower()
            config["plot"] = response == "y"
        elif choice == "0":
            return
        else:
            print("Invalid option.")


def execute_option(choice, G, config):
    """Runs the metric that corresponds to the chosen menu option."""
    exact = config["exact"]
    sample_nodes = config["sample_nodes"]
    top_k_n = config["top_k"]

    print()  # blank line before the result
    if choice == "1":
        print_graph_basic_data(G)
    elif choice == "2":
        print_average_degree(G)
    elif choice == "3":
        print_degree_distribution(G, plot=config["plot"], plot_output_path=config["plot_out"])
    elif choice == "4":
        print_density(G)
    elif choice == "5":
        print_clustering_coefficient(G, exact=exact, sample_nodes=sample_nodes)
    elif choice == "6":
        print_average_path_length(G, exact=exact, sample_nodes=sample_nodes)
    elif choice == "7":
        print_diameter(G, exact=exact, sample_nodes=sample_nodes)
    elif choice == "8":
        print_components(G)
    elif choice == "9":
        print_degree_centrality(G, top_k_n)
    elif choice == "10":
        print_closeness_centrality(G, top_k_n)
    elif choice == "11":
        print_betweenness_centrality(G, exact, sample_nodes, top_k_n)
    elif choice == "12":
        print_eigenvector_centrality(G, top_k_n)
    elif choice == "13":
        print_pagerank(G, top_k_n)
    elif choice == "14":
        generate_complete_log(
            G, exact=exact, sample_nodes=sample_nodes, top_k_n=top_k_n,
            plot=config["plot"], plot_output_path=config["plot_out"],
        )
    else:
        print("Invalid option. Please try again.")


def execute_interactive_menu(G, config):
    """
    Main menu loop: shows the options, runs the chosen metric and shows the
    menu again, until the user chooses to exit (option 0).
    """
    while True:
        show_menu(G, config)
        choice = input("Choose an option: ").strip()

        if choice == "0":
            print("Exiting. See you next time!")
            break
        elif choice == "15":
            menu_settings(config)
        elif choice in dict(MENU_OPTIONS):
            execute_option(choice, G, config)
            input("\nPress Enter to return to the main menu...")
        else:
            print("Invalid option. Please try again.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Wikipedia link graph analysis from a CSV file."
    )
    parser.add_argument("csv", help="Path to the edge CSV file.")
    parser.add_argument("--nrows", type=int, default=None,
                         help="Limit the number of CSV rows read (for quick tests).")
    parser.add_argument("--exact", dest="exact", action="store_true",
                         help="Force exact computation of expensive metrics (clustering "
                              "coefficient, average path length, diameter, betweenness "
                              "centrality). Can be very slow on large graphs. In menu "
                              "mode, this can also be changed later via 'Settings'.")
    parser.add_argument("--sample-nodes", type=int, default=12000,
                         help="Sample size used for approximate metrics (default: 12000).")
    parser.add_argument("--top-k", type=int, default=10,
                         help="How many nodes to show in each centrality ranking (default: 10).")
    parser.add_argument("--plot", action="store_true",
                         help="Generate the degree-distribution plot (degree_distribution.png).")
    parser.add_argument("--plot-out", default="degree_distribution.png",
                         help="Output path for the degree-distribution plot.")
    parser.add_argument("--sep", default=None,
                         help="Force the field separator manually (e.g. '\\t' for TAB). "
                              "Auto-detected by default.")
    parser.add_argument("--all", dest="run_all", action="store_true",
                         help="Run ALL metrics at once, without the interactive menu "
                              "(non-interactive behaviour, useful for scripts/automation).")

    args = parser.parse_args()

    # allows passing --sep '\t' literally on the command line
    sep = args.sep.encode().decode("unicode_escape") if args.sep else None

    if args.run_all:
        # non-interactive mode: runs everything at once and exits
        analyze_graph(
            csv_path=args.csv,
            nrows=args.nrows,
            exact=args.exact,
            sample_nodes=args.sample_nodes,
            top_k_n=args.top_k,
            plot=args.plot,
            plot_output_path=args.plot_out,
            sep=sep,
        )
        return

    # interactive mode (default): loads the graph once and opens the menu,
    # allowing individual metrics to be chosen without re-reading the CSV
    df = graph.load_data(args.csv, nrows=args.nrows, sep=sep)
    G = graph.build_graph(df)

    config = {
        "exact": args.exact,
        "sample_nodes": args.sample_nodes,
        "top_k": args.top_k,
        "plot": args.plot,
        "plot_out": args.plot_out,
    }

    execute_interactive_menu(G, config)


if __name__ == "__main__":
    sys.exit(main())
