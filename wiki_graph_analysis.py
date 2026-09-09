"""
Análise de grafo de links da Wikipedia (2006)
================================================

Lê um arquivo CSV com colunas:
    page_id_from, page_title_from, page_id_to, page_title_to

e calcula um conjunto de métricas clássicas de análise de redes/grafos:

Estrutura básica:
    - número de nós e arestas
    - grau médio
    - distribuição de graus
    - densidade
    - clustering coefficient (coeficiente de agrupamento)
    - average path length (comprimento médio do caminho)
    - diâmetro
    - componentes conexos

Centralidade (nós mais importantes):
    - degree centrality
    - closeness centrality
    - betweenness centrality
    - eigenvector centrality
    - PageRank

O grafo de links da Wikipedia é DIRECIONADO (um link de A para B não implica
o inverso). O script constrói tanto a versão direcionada (para PageRank,
componentes fortemente/fracamente conexos, etc.) quanto, quando necessário,
trabalha com a versão não-direcionada (usada tradicionalmente para
clustering coefficient, diâmetro "clássico" etc.).

Como esse tipo de grafo real pode ter centenas de milhares/milhões de nós e
arestas, algumas métricas (average path length, diâmetro, betweenness)
possuem complexidade proibitiva para cálculo exato. Por isso o script:
    - calcula essas métricas de forma EXATA quando o grafo é pequeno o
      suficiente;
    - usa AMOSTRAGEM (aproximação) automaticamente quando o grafo é grande,
      deixando isso claro no output.

Uso:
    python wiki_graph_analysis.py caminho/para/arquivo.csv
    python wiki_graph_analysis.py caminho/para/arquivo.csv --sample-nodes 500
    python wiki_graph_analysis.py caminho/para/arquivo.csv --exact
    python wiki_graph_analysis.py caminho/para/arquivo.csv --top-k 15 --plot
"""

import argparse
import sys
from collections import Counter

import networkx as nx

import centrality_metrics
from graph import Graph
from utilities import Utilities

graph: Graph = Graph()


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 1. Leitura dos dados e construção do grafo
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 2. Métricas estruturais básicas
# --------------------------------------------------------------------------

@Utilities.timeit
def count_nodes_and_edges(G):
    """Retorna (número de nós, número de arestas)."""
    return G.number_of_nodes(), G.number_of_edges()

@Utilities.timeit
def get_average_degree(G):
    """
    Grau médio do grafo.
    Para grafos direcionados, retorna in-degree médio e out-degree médio
    (que são sempre iguais em valor total, pois cada aresta contribui +1
    para um out-degree e +1 para um in-degree), além do grau total médio
    (in + out) por nó.
    """
    n = G.number_of_nodes()
    if n == 0:
        return {"in_medio": 0, "out_medio": 0, "total_medio": 0}

    soma_in = sum(d for _, d in G.in_degree())
    soma_out = sum(d for _, d in G.out_degree())

    return {
        "in_medio": soma_in / n,
        "out_medio": soma_out / n,
        "total_medio": (soma_in + soma_out) / n,
    }


@Utilities.timeit
def get_degree_distribution(G, tipo="total"):
    """
    Retorna a distribuição de graus como um Counter {grau: quantidade_de_nós}.

    tipo: 'in', 'out' ou 'total' (in+out), aplicável a grafos direcionados.
    """
    if tipo == "in":
        graus = [d for _, d in G.in_degree()]
    elif tipo == "out":
        graus = [d for _, d in G.out_degree()]
    else:
        graus = [G.in_degree(n) + G.out_degree(n) for n in G.nodes()]

    return Counter(graus)


@Utilities.timeit
def get_density(G):
    """Densidade do grafo (proporção de arestas existentes sobre o máximo possível)."""
    return nx.density(G)


@Utilities.timeit
def get_clustering_coefficient(G):
    """
    Coeficiente de clustering médio (transitividade local média).
    NetworkX calcula clustering em grafos não-direcionados (ou trata o
    direcionado com sua própria definição); aqui convertemos para
    não-direcionado, que é a abordagem clássica (Watts-Strogatz).

    Se `amostra` for informado (int), calcula em uma amostra de nós para
    grafos muito grandes (mais rápido, mas aproximado).
    """
    Gu = G.to_undirected()

    media = nx.average_clustering(Gu)
    return {"media": media, "exato": True}


@Utilities.timeit
def get_connected_components(G):
    """
    Para grafo direcionado, calcula:
      - número e tamanho dos componentes FRACAMENTE conexos (weakly connected)
      - número e tamanho dos componentes FORTEMENTE conexos (strongly connected)
    """
    fracos = sorted(
        (len(c) for c in nx.weakly_connected_components(G)), reverse=True
    )
    fortes = sorted(
        (len(c) for c in nx.strongly_connected_components(G)), reverse=True
    )

    return {
        "n_componentes_fracos": len(fracos),
        "maior_componente_fraco": fracos[0] if fracos else 0,
        "tamanhos_fracos_top5": fracos[:5],
        "n_componentes_fortes": len(fortes),
        "maior_componente_forte": fortes[0] if fortes else 0,
        "tamanhos_fortes_top5": fortes[:5],
    }


def _biggest_component_as_subgraph(G):
    """
    Retorna o subgrafo correspondente ao maior componente conexo.
    Aceita tanto grafo direcionado (usa componentes fracamente conexos)
    quanto já não-direcionado (usa componentes conexos "normais").
    """
    if G.is_directed():
        maior = max(nx.weakly_connected_components(G), key=len)
    else:
        maior = max(nx.connected_components(G), key=len)
    return G.subgraph(maior).copy()


@Utilities.timeit
def get_average_path_length(G):
    """
    Comprimento médio do caminho mais curto.

    Exige que o grafo seja (fortemente, se direcionado) conexo para o
    cálculo exato do NetworkX. Como grafos reais raramente são totalmente
    conexos, aplicamos o cálculo sobre o maior componente conexo.
    """
    Gu = G.to_undirected()
    componente = _biggest_component_as_subgraph(Gu)

    media = nx.average_shortest_path_length(componente)
    return {
        "media": media,
        "n_nos_maior_componente": componente.number_of_nodes(),
        "exato": True,
    }


@Utilities.timeit
def get_diameter(G):
    """
    Diâmetro do grafo (maior distância mínima entre dois nós), calculado
    sobre o maior componente conexo (versão não-direcionada).
    """
    Gu = G.to_undirected()
    componente = _biggest_component_as_subgraph(Gu)
    n = componente.number_of_nodes()

    d = nx.diameter(componente)
    return {"diametro": d, "n_nos_maior_componente": n, "exato": True}

# --------------------------------------------------------------------------
# 4. Relatórios / apresentação dos resultados
# --------------------------------------------------------------------------

def top_k(dicionario, G, k=10):
    """Retorna os k nós com maior valor em `dicionario`, já com rótulo (título)."""
    ordenado = sorted(dicionario.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(graph.rotulo(G, nid), nid, valor) for nid, valor in ordenado]


def imprimir_top_k(nome_metrica, lista_top, k=10):
    print(f"\nTop {min(k, len(lista_top))} nós por {nome_metrica}:")
    for i, (titulo, nid, valor) in enumerate(lista_top, start=1):
        print(f"  {i:2d}. {titulo!r} (id={nid})  ->  {valor:.6f}")


def plotar_distribuicao_de_graus(dist_graus, titulo="Distribuição de graus", caminho_saida=None):
    """Gera um gráfico log-log da distribuição de graus (típico de redes livres de escala)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graus = sorted(dist_graus.keys())
    quantidades = [dist_graus[g] for g in graus]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(graus, quantidades, s=12, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Grau (k)")
    ax.set_ylabel("Número de nós com grau k")
    ax.set_title(titulo)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout()

    if caminho_saida:
        fig.savefig(caminho_saida, dpi=150)
        Utilities.log(f"Gráfico salvo em: {caminho_saida}")
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. Funções de impressão por métrica (usadas tanto pelo relatório completo
#    quanto pelo menu interativo, para evitar duplicação de código)
# --------------------------------------------------------------------------

def print_graph_basic_data(G):
    n_nos, n_arestas = count_nodes_and_edges(G)
    print(f"Número de nós:     {n_nos:,}")
    print(f"Número de arestas: {n_arestas:,}")
    return n_nos, n_arestas


def print_average_degree(G):
    gm = get_average_degree(G)
    print(f"Grau médio (in):    {gm['in_medio']:.4f}")
    print(f"Grau médio (out):   {gm['out_medio']:.4f}")
    print(f"Grau médio (total): {gm['total_medio']:.4f}")
    return gm


def print_degree_distribution(G, plot=False, caminho_grafico="degree_distribution.png"):
    dist = get_degree_distribution(G, tipo="total")
    print(f"Distribuição de graus: {len(dist)} valores distintos de grau "
          f"(grau máximo observado: {max(dist)})")
    print("  (grau : nº de nós com esse grau) — 10 primeiros valores:")
    for g in sorted(dist.keys())[:10]:
        print(f"    {g:>4} : {dist[g]:,}")
    if plot:
        plotar_distribuicao_de_graus(
            dist, titulo="Distribuição de graus - Wikipedia 2006",
            caminho_saida=caminho_grafico,
        )
    return dist


def print_density(G):
    dens = get_density(G)
    print(f"Densidade: {dens:.8f}")
    return dens


def print_clustering_coefficient(G):
    cc = get_clustering_coefficient(G)
    if cc.get("exato"):
        print(f"Clustering coefficient (médio, exato): {cc['media']:.6f}")
    else:
        print(f"Clustering coefficient (médio, aproximado, "
              f"n={cc['n_amostrados']}): {cc['media_aproximada']:.6f}")
    return cc


def print_components(G):
    comp = get_connected_components(G)
    print(f"Componentes fracamente conexos: {comp['n_componentes_fracos']} "
          f"(maior: {comp['maior_componente_fraco']:,} nós)")
    print(f"  Top 5 tamanhos: {comp['tamanhos_fracos_top5']}")
    print(f"Componentes fortemente conexos: {comp['n_componentes_fortes']} "
          f"(maior: {comp['maior_componente_forte']:,} nós)")
    print(f"  Top 5 tamanhos: {comp['tamanhos_fortes_top5']}")
    return comp


def print_average_path_length(G):
    apl = get_average_path_length(G)
    if apl.get("exato"):
        print(f"Average path length (exato, maior componente, "
              f"n={apl['n_nos_maior_componente']:,}): {apl['media']:.4f}")
    else:
        print(f"Average path length (aproximado, amostra={apl['n_amostrados']}, "
              f"maior componente n={apl['n_nos_maior_componente']:,}): "
              f"{apl['media_aproximada']:.4f}")
    return apl


def print_diameter(G):
    diam = get_diameter(G)
    if diam.get("exato"):
        print(f"Diâmetro (exato, maior componente): {diam['diametro']}")
    else:
        print(f"Diâmetro (limite inferior aproximado, amostra="
              f"{diam['n_amostrados']}): {diam['diametro_aproximado_limite_inferior']}")
    return diam


def print_degree_centrality(G, top_k_n):
    dc = centrality_metrics.compute_degree_centrality(G)
    imprimir_top_k("degree centrality (total)", top_k(dc["total"], G, top_k_n), top_k_n)
    return dc


def print_closeness_centrality(G, top_k_n):
    Utilities.log("Calculando closeness centrality (pode demorar em grafos grandes)...")
    clo = centrality_metrics.compute_closeness_centrality(G)
    imprimir_top_k("closeness centrality", top_k(clo, G, top_k_n), top_k_n)
    return clo


def print_betweenness_centrality(G, exato, sample_nodes, top_k_n):
    amostra_bet = None if exato else sample_nodes
    Utilities.log("Calculando betweenness centrality (pode demorar bastante)...")
    bet = centrality_metrics.compute_betweenness_centrality(G, amostra_k=amostra_bet)
    imprimir_top_k(
        f"betweenness centrality {'(aproximado, k=' + str(amostra_bet) + ')' if amostra_bet else '(exato)'}",
        top_k(bet, G, top_k_n), top_k_n,
    )
    return bet


def imprimir_eigenvector_centrality(G, top_k_n):
    Utilities.log("Calculando eigenvector centrality...")
    eig = centrality_metrics.compute_eigenvector_centrality(G)
    if eig is not None:
        imprimir_top_k("eigenvector centrality", top_k(eig, G, top_k_n), top_k_n)
    else:
        print("\nEigenvector centrality: não foi possível calcular (não convergiu).")
    return eig


def print_pagerank(G, top_k_n):
    Utilities.log("Calculando PageRank...")
    pr = centrality_metrics.compute_pagerank(G)
    imprimir_top_k("PageRank", top_k(pr, G, top_k_n), top_k_n)
    return pr


# --------------------------------------------------------------------------
# 6. Orquestração — relatório completo (todas as métricas de uma vez)
# --------------------------------------------------------------------------

def generate_complete_log(G, exato=False, sample_nodes=500, top_k_n=10,
                              plot=False, caminho_grafico="degree_distribution.png"):
    """
    Calcula e imprime TODAS as métricas, na ordem solicitada, para um grafo
    G já carregado. Retorna um dicionário com todos os resultados.
    """
    print("\n" + "=" * 70)
    print("MÉTRICAS ESTRUTURAIS BÁSICAS")
    print("=" * 70)

    n_nos, n_arestas = print_graph_basic_data(G)
    gm = print_average_degree(G)
    dist = print_degree_distribution(G, plot=plot, caminho_grafico=caminho_grafico)
    dens = print_density(G)
    cc = print_clustering_coefficient(G)
    comp = print_components(G)
    apl = print_average_path_length(G)
    diam = print_diameter(G)

    print("\n" + "=" * 70)
    print("CENTRALIDADE — NÓS MAIS IMPORTANTES")
    print("=" * 70)

    dc = print_degree_centrality(G, top_k_n)
    clo = print_closeness_centrality(G, top_k_n)
    bet = print_betweenness_centrality(G, exato, sample_nodes, top_k_n)
    eig = imprimir_eigenvector_centrality(G, top_k_n)
    pr = print_pagerank(G, top_k_n)

    return {
        "grafo": G,
        "n_nos": n_nos,
        "n_arestas": n_arestas,
        "grau_medio": gm,
        "distribuicao_graus": dist,
        "densidade": dens,
        "clustering": cc,
        "componentes": comp,
        "average_path_length": apl,
        "diametro": diam,
        "degree_centrality": dc,
        "closeness_centrality": clo,
        "betweenness_centrality": bet,
        "eigenvector_centrality": eig,
        "pagerank": pr,
    }


def analyze_graph(caminho_csv, nrows=None, exato=False, sample_nodes=500,
                    top_k_n=10, plot=False, caminho_grafico="degree_distribution.png",
                    sep=None):
    """
    Lê o CSV, constrói o grafo e executa o pipeline completo de análise
    (todas as métricas de uma vez), imprimindo um relatório no console.

    Mantido para uso não-interativo / programático (ex: chamar a partir de
    outro script ou notebook, ou via a flag --all no modo CLI).
    """
    df = graph.carregar_dados(caminho_csv, nrows=nrows, sep=sep)
    G = graph.construir_grafo(df)
    return generate_complete_log(
        G, exato=exato, sample_nodes=sample_nodes, top_k_n=top_k_n,
        plot=plot, caminho_grafico=caminho_grafico,
    )


# --------------------------------------------------------------------------
# 7. Menu interativo
# --------------------------------------------------------------------------

MENU_OPTIONS = [
    ("1", "Número de nós e arestas"),
    ("2", "Grau médio"),
    ("3", "Distribuição de graus"),
    ("4", "Densidade"),
    ("5", "Clustering coefficient"),
    ("6", "Average path length"),
    ("7", "Diâmetro"),
    ("8", "Componentes conexos"),
    ("9", "Degree centrality"),
    ("10", "Closeness centrality"),
    ("11", "Betweenness centrality"),
    ("12", "Eigenvector centrality"),
    ("13", "PageRank"),
    ("14", "Executar TODAS as métricas (relatório completo)"),
    ("15", "Alterar configurações (modo exato/aproximado, amostra, top-k, gráfico)"),
    ("0", "Sair"),
]


def show_menu(G, config):
    n_nos = G.number_of_nodes()
    n_arestas = G.number_of_edges()
    modo = "EXATO" if config["exato"] else f"aproximado (amostra={config['sample_nodes']})"

    print("\n" + "=" * 70)
    print(" MENU PRINCIPAL — Análise de Grafo da Wikipedia")
    print("=" * 70)
    print(f" Grafo carregado: {n_nos:,} nós, {n_arestas:,} arestas")
    print(f" Configurações atuais: modo={modo} | top-k={config['top_k']} | "
          f"gráfico={'ligado' if config['plot'] else 'desligado'}")
    print("-" * 70)
    print(" --- Métricas estruturais básicas ---")
    for chave, nome in MENU_OPTIONS[0:8]:
        print(f"  {chave:>2}. {nome}")
    print(" --- Centralidade (nós mais importantes) ---")
    for chave, nome in MENU_OPTIONS[8:13]:
        print(f"  {chave:>2}. {nome}")
    print(" --- Outras opções ---")
    for chave, nome in MENU_OPTIONS[13:]:
        print(f"  {chave:>2}. {nome}")
    print("=" * 70)


def menu_settings(config):
    """Submenu para alterar as configurações de cálculo em tempo de execução."""
    while True:
        print("\n" + "-" * 70)
        print(" CONFIGURAÇÕES")
        print("-" * 70)
        print(f"  1. Modo de cálculo (atual: "
              f"{'exato' if config['exato'] else 'aproximado'})")
        print(f"  2. Tamanho da amostra para aproximações (atual: {config['sample_nodes']})")
        print(f"  3. Top-k de nós exibidos nos rankings (atual: {config['top_k']})")
        print(f"  4. Gerar gráfico da distribuição de graus (atual: "
              f"{'sim' if config['plot'] else 'não'})")
        print(f"  0. Voltar ao menu principal")
        print("-" * 70)
        escolha = input("Escolha uma opção: ").strip()

        if escolha == "1":
            resp = input("Usar modo EXATO para métricas caras? "
                          "(pode ser muito lento em grafos grandes) [s/N]: ").strip().lower()
            config["exato"] = resp == "s"
        elif escolha == "2":
            try:
                config["sample_nodes"] = int(input("Novo tamanho de amostra: ").strip())
            except ValueError:
                print("Valor inválido, mantendo o anterior.")
        elif escolha == "3":
            try:
                config["top_k"] = int(input("Novo valor de top-k: ").strip())
            except ValueError:
                print("Valor inválido, mantendo o anterior.")
        elif escolha == "4":
            resp = input("Gerar gráfico da distribuição de graus quando essa "
                          "métrica for calculada? [s/N]: ").strip().lower()
            config["plot"] = resp == "s"
        elif escolha == "0":
            return
        else:
            print("Opção inválida.")


def execute_option(escolha, G, config):
    """Executa a métrica correspondente à opção escolhida no menu."""
    exato = config["exato"]
    sample_nodes = config["sample_nodes"]
    top_k_n = config["top_k"]

    print()  # linha em branco antes do resultado
    if escolha == "1":
        print_graph_basic_data(G)
    elif escolha == "2":
        print_average_degree(G)
    elif escolha == "3":
        print_degree_distribution(G, plot=config["plot"], caminho_grafico=config["plot_out"])
    elif escolha == "4":
        print_density(G)
    elif escolha == "5":
        print_clustering_coefficient(G)
    elif escolha == "6":
        print_average_path_length(G)
    elif escolha == "7":
        print_diameter(G)
    elif escolha == "8":
        print_components(G)
    elif escolha == "9":
        print_degree_centrality(G, top_k_n)
    elif escolha == "10":
        print_closeness_centrality(G, top_k_n)
    elif escolha == "11":
        print_betweenness_centrality(G, exato, sample_nodes, top_k_n)
    elif escolha == "12":
        imprimir_eigenvector_centrality(G, top_k_n)
    elif escolha == "13":
        print_pagerank(G, top_k_n)
    elif escolha == "14":
        generate_complete_log(
            G, exato=exato, sample_nodes=sample_nodes, top_k_n=top_k_n,
            plot=config["plot"], caminho_grafico=config["plot_out"],
        )
    else:
        print("Opção inválida. Tente novamente.")


def execute_interactive_menu(G, config):
    """
    Loop principal do menu: exibe as opções, executa a métrica escolhida e
    volta a mostrar o menu, até o usuário optar por sair (opção 0).
    """
    while True:
        show_menu(G, config)
        escolha = input("Escolha uma opção: ").strip()

        if escolha == "0":
            print("Encerrando. Até mais!")
            break
        elif escolha == "15":
            menu_settings(config)
        elif escolha in dict(MENU_OPTIONS):
            execute_option(escolha, G, config)
            input("\nPressione Enter para voltar ao menu principal...")
        else:
            print("Opção inválida. Tente novamente.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Análise de grafo de links da Wikipedia (2006) a partir de um CSV."
    )
    parser.add_argument("csv", help="Caminho para o arquivo CSV de arestas.")
    parser.add_argument("--nrows", type=int, default=None,
                         help="Limitar número de linhas lidas do CSV (para testes rápidos).")
    parser.add_argument("--exact", dest="exato", action="store_true",
                         help="Forçar cálculo exato das métricas caras (pode ser muito lento). "
                              "No modo menu, pode ser alterado depois na opção 'Configurações'.")
    parser.add_argument("--sample-nodes", type=int, default=500,
                         help="Tamanho da amostra para métricas aproximadas (default: 500).")
    parser.add_argument("--top-k", type=int, default=10,
                         help="Quantos nós mostrar em cada ranking de centralidade (default: 10).")
    parser.add_argument("--plot", action="store_true",
                         help="Gerar gráfico da distribuição de graus (degree_distribution.png).")
    parser.add_argument("--plot-out", default="degree_distribution.png",
                         help="Caminho de saída do gráfico de distribuição de graus.")
    parser.add_argument("--sep", default=None,
                         help="Forçar separador de campos manualmente (ex: '\\t' para TAB). "
                              "Por padrão, é detectado automaticamente.")
    parser.add_argument("--all", dest="modo_all", action="store_true",
                         help="Executar TODAS as métricas de uma vez, sem menu interativo "
                              "(comportamento não-interativo, útil para scripts/automação).")

    args = parser.parse_args()

    # permite passar --sep '\t' de forma literal na linha de comando
    sep = args.sep.encode().decode("unicode_escape") if args.sep else None

    if args.modo_all:
        # modo não-interativo: roda tudo de uma vez e encerra
        analyze_graph(
            caminho_csv=args.csv,
            nrows=args.nrows,
            exato=args.exato,
            sample_nodes=args.sample_nodes,
            top_k_n=args.top_k,
            plot=args.plot,
            caminho_grafico=args.plot_out,
            sep=sep,
        )
        return

    # modo interativo (padrão): carrega o grafo uma vez e abre o menu,
    # permitindo escolher métricas individualmente sem recarregar o CSV
    df = graph.carregar_dados(args.csv, nrows=args.nrows, sep=sep)
    G = graph.construir_grafo(df)

    config = {
        "exato": args.exato,
        "sample_nodes": args.sample_nodes,
        "top_k": args.top_k,
        "plot": args.plot,
        "plot_out": args.plot_out,
    }

    execute_interactive_menu(G, config)


if __name__ == "__main__":
    sys.exit(main())
