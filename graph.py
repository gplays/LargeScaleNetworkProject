from os import path

import igraph as ig

# default is igraph, should not output error if not loaded
# This was done because it's non trivial to install graph-tool
try:
    import graph_tool.all as gt
except ModuleNotFoundError:
    gt = None
from preprocessing import DEFAULT_DATASET, get_data_path, preprocess

IGRAPH = True
VALUE_TYPES = {"title": "string",
               "authors": "string",
               "venue": "string",
               "year": "int",
               "abstract": "string", }


def load_or_create_graph(try_load=True, write=True, withIgraph=IGRAPH,
                         dataset=DEFAULT_DATASET, version=1, data_path=None):
    """

    :param version: version of the graph
    :type version: int
    :param data_path: path for data storage if not given initialized as ./.data
    :type data_path: str
    :param write: Set to True to write the preprocessed data to files
    :type write: bool
    :param try_load: Set to True to check for existing preprocessed files
    :type try_load: bool
    :param dataset: Set to v10 to preprocess DBLP-citation network V10
              Set to v8 to preprocess ACM-citation network V8
    :type dataset: str
    :param withIgraph: set to True to use igraph library else graph-tool is
    used
    :type withIgraph: bool
    :return: A graph
    :rtype: Graph
    """
    data_path, out_dir = get_data_path(dataset, version, data_path)

    graph = None
    if try_load:
        graph = maybe_load_graph(out_dir, withIgraph)

    if graph is None:
        parsed_data = preprocess(data_path=data_path, dataset=dataset,
                                 version=version)
        graph = create_graph(parsed_data, withIgraph, out_dir, dump_graph=write)

    return graph


def maybe_load_graph(data_path, withIgraph):
    """
    Try to load graph, if file doesn't exist return None
    :param data_path: path for data storage
    :type data_path: str
    :param withIgraph: set to True to use igraph library else graph-tool is
    used
    :type withIgraph: bool
    :return: graph object
    :rtype:
    """
    try:
        if withIgraph:
            g = ig.Graph.Read_Picklez(path.join(data_path, "graphi"))
        else:
            g = gt.load_graph("my_graph.xml.gz")
    except FileNotFoundError:
        g = None
    return g


def create_graph(parsed_data, withIgraph, data_path, dump_graph=False):
    """
    Create a graph from basic data with the right API

    :param parsed_data:
    :type parsed_data: dict
    :param data_path: path for data storage
    :type data_path: str
    :param withIgraph: set to True to use igraph library else graph-tool is
    used
    :type withIgraph: bool
    :param dump_graph: Set to True to save model after creation
    :type dump_graph: bool
    :return: graph object
    :rtype:
    """
    if withIgraph:
        g = ig.Graph(directed=True)
        g.add_vertices([i for i in range(len(parsed_data["papers"]))])
        g.add_edges(parsed_data["references_flat"])
        if dump_graph:
            dump(g, path.join(data_path, "graphi"), withIgraph)

    else:
        g = gt.Graph()
        g.add_vertex(n=parsed_data[len(parsed_data["papers"])])
        g.add_edge_list(parsed_data["references_flat"])
        if dump_graph:
            dump(g, path.join(data_path, "grapht"), withIgraph)
    return g


def dump(g, data_path, withIgraph):
    """
    Save graph model with right API

    :param g: The graph object
    :type g:
    :param data_path: path for data storage
    :type data_path: str
    :param withIgraph: set to True to use igraph library else graph-tool is
    used
    :type withIgraph: bool
    """
    if withIgraph:
        g.write_picklez(data_path)
    else:
        g.save(data_path)


def add_vertices_attributes(g, attr, vals, withIgraph=True,
                            value_type=None):
    """
    Add attributes directly to the vertices of the graph using their own API
    Afterward values of attributes can be read/write through .v attribute of
    the graph indifferently
    :param g: The graph object
    :type g:
    :param attr: name of the attribute
    :type attr: str
    :param vals: Values for the attributes
    :type vals: list
    :param withIgraph: set to True to use igraph library else graph-tool is
    used
    :type withIgraph: bool
    :param value_type: for graph tool need to give indication on data type
    for more efficient handling (C backend)
    :type value_type: str
    :return:
    :rtype:
    """

    if withIgraph:
        g.vs[attr] = vals
        # magic alias
        g.v = g.vs
    else:
        if value_type is None:
            value_type = VALUE_TYPES.get(attr, None)
        assert value_type is not None, "with graph tool you need to provide \
                                        the value_type"
        g.vp[attr] = gt.new_vp(value_type, vals=vals)
        # magic alias
        g.v = g.vp
