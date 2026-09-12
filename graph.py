import networkx as nx
import igraph as ig
import pandas as pd
from utilities import Utilities

class Graph:
    def __init__(self):
        pass
    
    def _detectar_separador(self, caminho_csv):
        """
        Tenta detectar automaticamente o separador de campos do arquivo.

        Datasets de grafo de links da Wikipedia (ex: o dataset "wikilinkgraphs"
        de Consonni et al.) costumam vir com extensão .csv mas separados por
        TAB, não por vírgula. Ler um arquivo TAB-separado como se fosse
        vírgula-separado faz o pandas enxergar "1 coluna" na maior parte das
        linhas e quebrar (ParserError) assim que algum título de página contiver
        uma vírgula.

        Estratégia: olhamos a primeira linha não vazia e contamos ocorrências
        de cada separador candidato; escolhemos o que aparece mais vezes.
        """
        candidatos = ["\t", ",", ";", "|"]

        with open(caminho_csv, "r", encoding="utf-8", errors="replace") as f:
            primeira_linha = f.readline()

        contagens = {sep: primeira_linha.count(sep) for sep in candidatos}
        melhor_sep = max(contagens, key=contagens.get)

        if contagens[melhor_sep] == 0:
            # nenhum separador candidato encontrado; assume vírgula como último recurso
            return ","

        return melhor_sep


    @Utilities.timeit
    def carregar_dados(self, caminho_csv, nrows=None, sep=None):
        """
        Lê o CSV/TSV de arestas da Wikipedia.

        Espera colunas: page_id_from, page_title_from, page_id_to, page_title_to
        (aceita variações comuns de nome de coluna, ver `rename_map` abaixo).

        O parâmetro `sep` permite forçar o separador manualmente (ex: '\\t').
        Se `sep=None` (padrão), o separador é detectado automaticamente —
        importante porque vários datasets desse tipo (ex: o dataset
        "wikilinkgraphs") usam TAB como separador apesar da extensão .csv.
        """
        if sep is None:
            sep = self._detectar_separador(caminho_csv)
            rotulo_sep = {"\t": "TAB", ",": "vírgula", ";": "ponto-e-vírgula", "|": "pipe"}.get(sep, repr(sep))
            Utilities.log(f"Separador detectado automaticamente: {rotulo_sep}")

        Utilities.log(f"Lendo arquivo CSV: {caminho_csv}")
        try:
            df = pd.read_csv(caminho_csv, nrows=nrows, sep=sep, engine="c")
        except pd.errors.ParserError:
            # fallback: parser mais tolerante (mais lento, porém mais robusto
            # a linhas malformadas / separadores inconsistentes)
            Utilities.log("  Aviso: falha ao ler com o parser rápido; tentando novamente "
                "com engine='python' (mais lento, porém mais tolerante)...")
            df = pd.read_csv(caminho_csv, nrows=nrows, sep=sep, engine="python",
                            on_bad_lines="warn")

        # normaliza nomes de coluna (tolera maiúsculas/minúsculas e pequenas variações)
        rename_map = {}
        cols_lower = {c.lower().strip(): c for c in df.columns}

        aliases = {
            "page_id_from": ["page_id_from", "id_from", "source_id", "from_id"],
            "page_title_from": ["page_title_from", "title_from", "source", "from_title"],
            "page_id_to": ["page_id_to", "id_to", "target_id", "to_id"],
            "page_title_to": ["page_title_to", "title_to", "target", "to_title"],
        }

        for padrao, opcoes in aliases.items():
            for op in opcoes:
                if op in cols_lower:
                    rename_map[cols_lower[op]] = padrao
                    break

        df = df.rename(columns=rename_map)

        colunas_esperadas = ["page_id_from", "page_title_from", "page_id_to", "page_title_to"]
        faltando = [c for c in colunas_esperadas if c not in df.columns]
        if faltando:
            raise ValueError(
                f"Colunas ausentes no CSV: {faltando}. "
                f"Colunas encontradas: {list(df.columns)}"
            )

        df = df.dropna(subset=["page_id_from", "page_id_to"])
        Utilities.log(f"  {len(df):,} linhas (arestas brutas) carregadas.")
        return df


    @Utilities.timeit
    def construir_grafo(self, df, usar_titulos_como_rotulo=True):
        """
        Constrói um grafo direcionado (DiGraph) do NetworkX a partir do
        DataFrame de arestas.

        Usa page_id como identificador do nó (mais confiável que o título,
        que pode ter duplicatas/ambiguidades) e guarda o título como atributo
        'title' do nó, se disponível.
        """
        nodes_from = df[["page_id_from", "page_title_from"]].rename(
            columns={"page_id_from": "id", "page_title_from": "title"}
        )
        nodes_to = df[["page_id_to", "page_title_to"]].rename(
            columns={"page_id_to": "id", "page_title_to": "title"}
        )
        nodes = pd.concat([nodes_from, nodes_to]).drop_duplicates(subset="id").reset_index(drop=True)

        id_to_index = {node_id: idx for idx, node_id in enumerate(nodes["id"])}

        ig_g = ig.Graph(directed=True)
        ig_g.add_vertices(len(nodes))
        ig_g.vs["page_id"] = nodes["id"].tolist()
        ig_g.vs["title"] = (nodes["title"].tolist() if usar_titulos_como_rotulo
                          else nodes["id"].tolist())

        edges = [
            (id_to_index[src], id_to_index[dst])
            for src, dst in zip(df["page_id_from"], df["page_id_to"])
        ]
        ig_g.add_edges(edges)

        Utilities.log(f"  Graph built: {ig_g.vcount():,} nodes, {ig_g.ecount():,} edges.")
        return ig_g

        """
        G = nx.DiGraph()
        # adiciona nós com atributo de título
        nos_from = df[["page_id_from", "page_title_from"]].rename(
            columns={"page_id_from": "id", "page_title_from": "title"}
        )
        nos_to = df[["page_id_to", "page_title_to"]].rename(
            columns={"page_id_to": "id", "page_title_to": "title"}
        )
        nos = pd.concat([nos_from, nos_to]).drop_duplicates(subset="id")

        for _, row in nos.iterrows():
            G.add_node(row["id"], title=row["title"] if usar_titulos_como_rotulo else row["id"])

        # adiciona arestas (remove self-loops duplicados automaticamente via set)
        arestas = list(zip(df["page_id_from"], df["page_id_to"]))
        G.add_edges_from(arestas)

        Utilities.log(f"  Grafo construído: {G.number_of_nodes():,} nós, {G.number_of_edges():,} arestas.")
        return G
        """


    def rotulo(self, G, node_id):
        """Retorna um rótulo legível (título) para um id de nó, se existir."""
        return G.nodes[node_id].get("title", str(node_id))

    def label(self, g, vertex_index):
        """Returns a human-readable label (title) for a vertex index, if available."""
        title = g.vs[vertex_index]["title"]
        return title if title else str(g.vs[vertex_index]["page_id"])