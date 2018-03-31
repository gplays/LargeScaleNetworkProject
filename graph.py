

import igraph as ig

# Do not need to use graph_tool
try:
    import graph_tool.all as gt
except ModuleNotFoundError:
    gt = None


def create_graph(parsed_data, withIgraph=True,dump_graph=False,data_path=''):
    if withIgraph:
        g = ig.Graph(directed=True)
        g.add_vertices([i for i in range(len(parsed_data["papers"]))])
        g.add_edges(parsed_data["references_flat"])
    else:
        g = gt.Graph()
        g.add_vertex(n=parsed_data[len(parsed_data["papers"])])
        g.add_edge_list(parsed_data["references_flat"])
    if dumpGraph:
        graph.write_picklez(data_path)
    return g, withIgraph


def add_vertices_attributes(g, attr, vals, withIgraph=True,
                            value_type=None):
    if withIgraph:
        g.vs[attr] = vals
    else:
        assert value_type is not None, "with graph tool you need to provide \
                                        the value_type"
        g.vp[attr] = gt.new_vp(value_type, vals=None)