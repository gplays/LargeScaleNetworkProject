from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from graph import load_or_create_graph, add_vertices_attributes


def main(**kwargs):
    graph, parsed_data = load_or_create_graph(**kwargs)
    withIgraph = kwargs.get("withIgraph", True)
    n = parsed_data["n_nodes"]
    papers = parsed_data["papers"]

    compute_tfidf(graph, papers, n, withIgraph)
    add_standard_properties(g, papers, n, withIgraph)
    add_standard_metrics(g, withIgraph)


def compute_tfidf(g, papers, n, withIgraph):
    for origin in ["abstract", "title"]:
        # compute TFIDF vector of each paper
        corpus = ['' for _ in range(n)]
        for k, v in papers:
            corpus[k] = v[origin]
        vectorizer = TfidfVectorizer(stop_words="english")
        # each row is a node in the order of node_info
        features_TFIDF = vectorizer.fit_transform(corpus)
        add_vertices_attributes(g, "tfidf_" + origin, features_TFIDF,
                                withIgraph,
                                value_type="vector<int>")


def add_standard_properties(g, papers, n, withIgraph):
    title = ["" for _ in range(n)]
    venue = ["" for _ in range(n)]
    year = [0 for _ in range(n)]
    authors = [[] for _ in range(n)]
    for k, v in papers:
        title[k] = v["title"]
        venue[k] = v["venue"]
        year[k] = v["year"]
        authors[k] = v["authors"]

    g.add_vertices_attributes(g, title, title, withIgraph, "string")
    g.add_vertices_attributes(g, venue, venue, withIgraph, "string")
    g.add_vertices_attributes(g, year, year, withIgraph, "int")
    g.add_vertices_attributes(g, authors, authors, withIgraph, "vector<string>")


def add_standard_metrics(g, withIgraph):
    ## TFIDF cosine similarity, in other words similarity between texts based
    #  on word occurences
    add_metric(g, "tfidf_abstract", "tfidf_abstract_cosine",
               cosine_similarity, withIgraph)
    add_metric(g, "tfidf_title", "tfidf_title_cosine", cosine_similarity,
               withIgraph)

    ## Temporal difference between publication dates
    diff = lambda x, y: int(x) - int(y)
    add_metric(g, "year", "temp_diff", diff, withIgraph)
    ## number of authors in common
    common_item = lambda x, y: len(set(x).intersection(set(y)))
    add_metric(g, "authors", "comm_auth", common_item, withIgraph)


def add_metric(g, v_attr, m_attr, func, withIgraph):
    tuples = [edge.tuple for edge in g.es]
    metric = [func(g.vs[v_attr][t[0]], g.vs[v_attr][t[1]])
              for t in tuples]
    add_edges_attributes(g, m_attr, metric, withIgraph, "double")


if __name__ == "__main__":

    main()
