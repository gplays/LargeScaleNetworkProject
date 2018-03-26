import csv
import json
import os
from os import path

import graph_tool.all as gt
import igraph

VALUE_TYPES = {"title": "string",
               "authors": "string",
               "venue": "string",
               "year": "int",
               "abstract": "string", }


def preprocess(dumpGraph=True, dumpRawData=False, v10=True, v8=True,
               dumpLight=False):
    """
    Preprocessing function.
    Warning! can be RAM intensive if experiencing serious slow down consider
    asking for preprocessed files from a comrade

    :param dumpGraph: Set to True to Dump the iGraph version
    :type dumpGraph: bool
    :param dumpRawData: Set to True to dump roughly preprocessed files
    :type dumpRawData: bool
    :param v10: Set to true to preprocess DBLP-citation network V10
    :type v10: bool
    :param v8: Set to true to preprocess ACM-citation network V8
    :type v8: bool
    :param dumpLight: Set to True to dump a subgraph for testing purposes
    Currently unsupported
    :type dumpLight: bool
    :return:
    :rtype:
    """

    # Automatically get the path of the file
    my_path = path.dirname(path.realpath(__file__))
    # Assume data is in an already existing data directory at the same level as
    # this file
    data_path = path.join(my_path, "data")

    if v10 and v8:
        preprocess(dumpGraph, dumpRawData, False, True, dumpLight)

    if v10:
        parsed_data = read_v10(data_path)
    elif v8:
        parsed_data = read_v8(data_path)

    if dumpRawData:
        write_raw(data_path, parsed_data)
    graph, withIgraph = create_graph(parsed_data)
    if dumpGraph:
        graph.write_pickle()



def create_graph(parsed_data, withIgraph=True):
    if withIgraph:
        g = igraph.Graph(directed=True)
        g.add_vertices([i for i in range(parsed_data["papers"])])
        g.add_edges(parsed_data["references_flat"])
    else:
        g = gt.Graph()
        g.add_vertex(n=parsed_data[len(parsed_data["papers"])])
        g.add_edge_list(parsed_data["references_flat"])

    return g, withIgraph


def add_vertices_attributes(g, attr, vals, withIgraph=True,
                            value_type=None):
    if withIgraph:
        g.vs[attr] = vals
    else:
        assert value_type is not None, "with graph tool you need to provide \
                                        the value_type"
        g.vp[attr] = gt.new_vp(value_type, vals=None)


def read_v10(data_path):
    papers = []
    first_authors = {}
    collaboration_authors = {}
    references_flat = []
    id2idx = {}
    idx = 0
    for root, dirs, files in os.walk(path.join(data_path, "dblp-ref")):
        for file in files:

            with open(path.join(data_path, "dblp.v10", file))as dblp:
                for line in dblp:
                    paper = json.loads(line)
                    id2idx[paper["id"]] = idx
                    papers[idx] = {"title": paper["title"],
                                   "authors": paper["authors"],
                                   "venue": paper["venue"],
                                   "year": paper["year"],
                                   "abstract": paper["abstract"],
                                   }

                    safe_append(first_authors, paper["authors"][0], idx)

                    for author in paper["authors"]:
                        safe_append(collaboration_authors, author, idx)

                    for reference in paper["references"]:
                        references_flat.append([[paper["id"]], reference])

                idx += 1
    references_flat = [[id2idx[e[0]], id2idx[e[1]]] for e in references_flat]
    parsed_data = {"papers": papers,
                   "first_authors": first_authors,
                   "collaboration_authors": collaboration_authors,
                   "references_flat": references_flat}
    return parsed_data


def write_raw(data_path, parsed_data):
    with open(path.join(data_path, "papers", "w")) as f:
        json.dump(parsed_data["papers"], f)
    with open(path.join(data_path, "papers_id", "w")) as f:
        for key in parsed_data["papers"].keys():
            f.write(key)
    with open(path.join(data_path, "first_authors", "w")) as f:
        json.dump(parsed_data["first_authors"], f)
    with open(path.join(data_path, "collaboration_authors", "w")) as f:
        json.dump(parsed_data["collaboration_authors"], f)
    with open(path.join(data_path, "references_flat", "w")) as f:
        writer = csv.writer(f)
        for ref in parsed_data["references_flat"]:
            writer.writerow(ref)


def safe_append(dict, id, elem):
    try:
        dict[id].append(elem)
    except KeyError:
        dict[id] = [elem]
