import networkx as nx
import pandas as pd
from utilities import Utilities


class Graph:
    def __init__(self):
        pass

    def _detect_separator(self, csv_path):
        """
        Tries to auto-detect the field separator of the file.

        Wikipedia link-graph datasets (e.g. the "wikilinkgraphs" dataset by
        Consonni et al.) often ship with a .csv extension but are actually
        TAB-separated, not comma-separated. Reading a TAB-separated file as
        if it were comma-separated makes pandas see "1 column" for most
        rows and breaks (ParserError) as soon as a page title contains a
        comma.

        Strategy: look at the first non-empty line and count occurrences of
        each candidate separator; pick whichever appears most often.
        """
        candidates = ["\t", ",", ";", "|"]

        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()

        counts = {sep: first_line.count(sep) for sep in candidates}
        best_separator = max(counts, key=counts.get)

        if counts[best_separator] == 0:
            # no candidate separator found; fall back to comma as a last resort
            return ","

        return best_separator

    @Utilities.timeit
    def load_data(self, csv_path, nrows=None, sep=None):
        """
        Reads the Wikipedia edge CSV/TSV file.

        Expected columns: page_id_from, page_title_from, page_id_to, page_title_to
        (common column-name variants are accepted, see `rename_map` below).

        The `sep` parameter allows the separator to be forced manually
        (e.g. '\\t'). If `sep=None` (default), the separator is
        auto-detected — important because several datasets of this kind
        (e.g. "wikilinkgraphs") use TAB as the separator despite the .csv
        extension.
        """
        if sep is None:
            sep = self._detect_separator(csv_path)
            sep_label = {"\t": "TAB", ",": "comma", ";": "semicolon", "|": "pipe"}.get(sep, repr(sep))
            Utilities.log(f"Auto-detected separator: {sep_label}")

        Utilities.log(f"Reading CSV file: {csv_path}")
        try:
            df = pd.read_csv(csv_path, nrows=nrows, sep=sep, engine="c")
        except pd.errors.ParserError:
            # fallback: more tolerant parser (slower, but more robust to
            # malformed rows / inconsistent separators)
            Utilities.log("  Warning: fast parser failed; retrying with "
                "engine='python' (slower, but more tolerant)...")
            df = pd.read_csv(csv_path, nrows=nrows, sep=sep, engine="python",
                            on_bad_lines="warn")

        # normalize column names (tolerates case and minor naming variations)
        rename_map = {}
        cols_lower = {c.lower().strip(): c for c in df.columns}

        aliases = {
            "page_id_from": ["page_id_from", "id_from", "source_id", "from_id"],
            "page_title_from": ["page_title_from", "title_from", "source", "from_title"],
            "page_id_to": ["page_id_to", "id_to", "target_id", "to_id"],
            "page_title_to": ["page_title_to", "title_to", "target", "to_title"],
        }

        for pattern, options in aliases.items():
            for opt in options:
                if opt in cols_lower:
                    rename_map[cols_lower[opt]] = pattern
                    break

        df = df.rename(columns=rename_map)

        expected_columns = ["page_id_from", "page_title_from", "page_id_to", "page_title_to"]
        missing = [c for c in expected_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns in CSV: {missing}. "
                f"Columns found: {list(df.columns)}"
            )

        df = df.dropna(subset=["page_id_from", "page_id_to"])
        Utilities.log(f"  {len(df):,} rows (raw edges) loaded.")
        return df

    @Utilities.timeit
    def build_graph(self, df, use_titles_as_label=True):
        """
        Builds a directed (DiGraph) NetworkX graph from the edge DataFrame.

        Uses page_id as the node identifier (more reliable than the title,
        which can have duplicates/ambiguities) and stores the title as the
        node's 'title' attribute, when available.

        Note: nodes are added via `add_nodes_from` with a generator of
        (id, attrs) pairs instead of iterating the DataFrame row by row
        with `iterrows()`. For graphs with hundreds of thousands of nodes
        this is dramatically faster, since `iterrows()` re-boxes every row
        into a pandas Series (expensive) and is one of the most common
        hidden bottlenecks when building graphs from large CSVs.
        """
        G = nx.DiGraph()

        # collect (id, title) pairs from both endpoints and de-duplicate by id
        nodes_from = df[["page_id_from", "page_title_from"]].rename(
            columns={"page_id_from": "id", "page_title_from": "title"}
        )
        nodes_to = df[["page_id_to", "page_title_to"]].rename(
            columns={"page_id_to": "id", "page_title_to": "title"}
        )
        nodes = pd.concat([nodes_from, nodes_to]).drop_duplicates(subset="id")

        if use_titles_as_label:
            node_records = zip(nodes["id"], nodes["title"])
        else:
            node_records = zip(nodes["id"], nodes["id"])

        G.add_nodes_from((node_id, {"title": title}) for node_id, title in node_records)

        # add edges (duplicate self-loops are automatically deduped by the graph)
        edges = list(zip(df["page_id_from"], df["page_id_to"]))
        G.add_edges_from(edges)

        Utilities.log(f"  Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges.")
        return G

    def label(self, G, node_id):
        """Returns a readable label (title) for a node id, if available."""
        return G.nodes[node_id].get("title", str(node_id))
