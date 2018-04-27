import csv
import json
import os
from os import path

DEFAULT_DATASET = "daten"


def preprocess(try_load=True, write=True, dataset=DEFAULT_DATASET, version=1,
               data_path=None):
    """
    Preprocessing function.
    Warning! can be RAM intensive if experiencing serious slow down consider
    asking for preprocessed files from a comrade

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
    :return: preprocessed network
    :rtype: dict
    """

    data_path, out_dir = get_data_path(dataset, version, data_path)

    parsed_data = None
    if try_load:
        parsed_data = maybe_load_raw(data_path, dataset)

    if parsed_data is None:

        if dataset == "v10":
            parsed_data = read_v10(data_path)
        elif dataset == "v8":
            parsed_data = read_v8(data_path)
        elif dataset == "daten":
            parsed_data = read_daten(data_path)
        else:
            raise ValueError("v10 or v8 value must be set to True, otherwise "
                             "there is no target data to preprocess")

        if write:
            write_raw(data_path, parsed_data, dataset)

    return parsed_data


def read_v10(data_path):
    """
    Preprocess data following the format of dblp v10 from aminer website:
    a dir dblp-ref containing json files
    return dict containing:
        papers
        first_authors
        collaboration_authors
        references_flat
        n_nodes
    :param data_path:
    :type data_path:
    :return:
    :rtype:
    """
    papers = {}
    first_authors = {}
    collaboration_authors = {}
    references_flat = []
    ledger = Ledger()
    dblp_path = path.join(data_path, "dblp-ref")
    for root, dirs, files in os.walk(dblp_path):
        for file in files:

            with open(path.join(dblp_path, file))as dblp:
                for line in dblp:
                    paper = json.loads(line)
                    idx = ledger.id2idx(paper["id"])
                    papers[idx]({"title": paper.get("title", ''),
                                 "authors": paper.get("authors", []),
                                 "venue": paper.get("venue", ''),
                                 "year": paper.get("year", 0),
                                 "abstract": paper.get("abstract", ''),
                                 })

                    safe_append(first_authors, paper["authors"][0], idx)

                    for author in paper.get("authors", []):
                        safe_append(collaboration_authors, author, idx)

                    for reference in paper.get("references", []):
                        references_flat.append((paper["id"], reference))

    references_flat = [(ledger.id2idx(x), ledger.id2idx(y))
                       for x, y in references_flat]
    parsed_data = {"papers": papers,
                   "first_authors": first_authors,
                   "collaboration_authors": collaboration_authors,
                   "references_flat": references_flat,
                   "n_nodes": ledger.index}
    return parsed_data


def read_daten(data_path):
    """
       Preprocess data following the format of dblp v10 from aminer website:
       a dir dblp-ref containing json files
       return dict containing:
           papers
           first_authors
           collaboration_authors
           references_flat
           n_nodes
       :param data_path:
       :type data_path:
       :return:
       :rtype:
       """
    papers = {}
    first_authors = {}
    collaboration_authors = {}
    ledger = Ledger()
    with open(path.join(data_path, "astro-ALP-2003-2010.csv")) as f:
        reader = csv.reader(f)
        first_line = next(reader)
        id_idx = first_line.index("UT")
        title_idx = first_line.index("TI")
        authors_idx = first_line.index("AU")
        venue_idx = first_line.index("SO")
        year_idx = first_line.index("PY")
        abstract_idx = first_line.index("AB")
        for line in reader:
            paper_id = line[id_idx]
            authors = line[authors_idx].split(";")
            authors = [a.replace("\"", "").strip() for a in authors]
            idx = ledger.id2idx(paper_id)
            papers[idx] = {"title": line[title_idx],
                           "authors": authors,
                           "venue": line[venue_idx],
                           "year": line[year_idx],
                           "abstract": line[abstract_idx],
                           }

            safe_append(first_authors, authors[0], idx)

            for author in authors:
                safe_append(collaboration_authors, author, idx)

    with open(path.join(data_path, "direct_citations.txt")) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        references_flat = [(ledger.id2idx(x), ledger.id2idx(y))
                           for (x, y) in reader]
    parsed_data = {"papers": papers,
                   "first_authors": first_authors,
                   "collaboration_authors": collaboration_authors,
                   "references_flat": references_flat,
                   "n_nodes": ledger.index}
    return parsed_data


def read_v8(data_path):
    """

    Preprocess data following the format of acm v8 from aminer website:
    a single text file citation-acm-v8.txt
    with specific code at he beginning of each line
    return dict containing:
        papers
        first_authors
        collaboration_authors
        references_flat
        n_nodes
    :param data_path:
    :type data_path:
    :return:
    :rtype:
    """
    papers = {}
    first_authors = {}
    collaboration_authors = {}
    references_flat = []
    ledger = Ledger()
    idx = 0
    with open(path.join(data_path, "citation-acm-v8.txt")) as acm:
        # TODO the default value for the year is 0, might neeed further
        # preprocessing for missing values
        paper_id, title, authors, venue, year, abstract = -1, '', [], '', 0, ''
        for line in acm:
            if "#" == line[0]:
                # Need to cut last character which is a breakline
                # Need to cut first 'code' characters
                if "*" == line[1]:
                    title = line[2:-1]
                if "@" == line[1]:
                    authors = line[2:-1].split(",")
                if "t" == line[1]:
                    year = int(line[2:-1])
                if "c" == line[1]:
                    venue = line[2:-1]
                if "i" == line[1]:
                    paper_id = line[6:-1]
                    idx = ledger.id2idx(paper_id)
                if "%" == line[1]:
                    references_flat.append((paper_id, line[2:-1]))
                if "!" == line[1]:
                    abstract = line[2:-1]
            else:
                papers[idx] = {"title": title,
                               "authors": authors,
                               "venue": venue,
                               "year": year,
                               "abstract": abstract,
                               }
                try:
                    safe_append(first_authors, authors[0], idx)
                except IndexError:
                    pass
                for author in authors:
                    safe_append(collaboration_authors, author, idx)

                paper_id, title, authors, venue, year, abstract = (-1, '', [],
                                                                   '', 0, '')

    references_flat = [(ledger.id2idx(x), ledger.id2idx(y))
                       for x, y in references_flat]
    parsed_data = {"papers": papers,
                   "first_authors": first_authors,
                   "collaboration_authors": collaboration_authors,
                   "references_flat": references_flat,
                   "n_nodes": ledger.index}
    return parsed_data


def write_raw(data_path, parsed_data, dataset):
    """
    Write the content of the parsed_data dict to a file
    :param dataset: dataset being preprocessed
    :type dataset: str
    :param data_path: path for data storage
    :type data_path: str
    :param parsed_data:
    :type parsed_data:
    """
    data_path = path.join(data_path, str(dataset))
    with open(path.join(data_path, "papers"), "w") as f:
        json.dump(parsed_data["papers"], f)
    with open(path.join(data_path, "n_nodes"), "w") as f:
        f.write(str(parsed_data["n_nodes"]))
    with open(path.join(data_path, "first_authors"), "w") as f:
        json.dump(parsed_data["first_authors"], f)
    with open(path.join(data_path, "collaboration_authors"), "w") as f:
        json.dump(parsed_data["collaboration_authors"], f)
    with open(path.join(data_path, "references_flat"), "w") as f:
        writer = csv.writer(f)
        for ref in parsed_data["references_flat"]:
            writer.writerow(ref)


def maybe_load_raw(data_path, dataset):
    """
    Load the content of the files.
    If they exist if not don't raise error but return parsed_data=None
    :param data_path: path for data storage
    :type data_path: str
    :param dataset: dataset being preprocessed
    :type dataset: str
    :return: parsed data
    :rtype: dict
    """
    data_path = path.join(data_path, str(dataset))
    parsed_data = {}
    try:
        with open(path.join(data_path, "n_nodes"), "r") as f:
            parsed_data["n_nodes"] = int(f.read())
        with open(path.join(data_path, "references_flat"), "r") as f:
            reader = csv.reader(f)
            parsed_data["references_flat"] = [ref for ref in reader]
        with open(path.join(data_path, "papers"), "r") as f:
            parsed_data["papers"] = json.load(f)
        with open(path.join(data_path, "first_authors"), "r") as f:
            parsed_data["first_authors"] = json.load(f)
        with open(path.join(data_path, "collaboration_authors"), "r") as f:
            parsed_data["collaboration_authors"] = json.load(f)
    except FileNotFoundError:
        parsed_data = None
        print("load at {} failed will reprocess file".format(data_path))
    return parsed_data


###################################
## Utility functions and classes ##

class Ledger(object):
    """
    Ledger object to keep track of papers id
    Main functon used is id2idx
    Might be one day extended to allow node selection, filtering and reindexing
    """

    def __init__(self):
        self.index = 0
        self.ids = []
        self.id2index = {}

    def id2idx(self, my_id):
        """
        Return the index attributed to this id and if it is the first time
        this id is processed, will attribute it an id

        :param my_id: paper id could be a string or a number
        :type my_id: str
        :return: index of the paper
        :rtype: int
        """
        try:
            idx = self.id2index[my_id]
        except KeyError:
            self.id2index[my_id] = self.index
            self.ids.append(my_id)
            idx = self.index
            self.index += 1
        return idx

    def idx2id(self, idx):
        return self.ids[idx]


def get_data_path(dataset, version, root=None):
    """

    :param dataset: dataset name
    :type dataset: str
    :param version: project version
    :type version: int
    :param root: If given, is used as data_path
    :type root: str
    :return: path to dataset and path to output_dirs
    :rtype: (str,str)
    """
    if root is None:
        # Automatically get the path of the file
        # Assume data is in an already existing data directory at the same
        # level as this file
        my_path = path.dirname(path.realpath(__file__))
        data_path = path.join(my_path, ".data")
    else:
        data_path = root
    out_dir = path.join(data_path, dataset + "." + str(version))
    # Creating the dir corresponding to the datatset (raw data) and for the
    # version (graph)
    for my_path in [out_dir, path.join(data_path, dataset)]:
        try:
            os.mkdir(my_path)
        except FileExistsError:
            pass

    return data_path, out_dir


def safe_append(my_dict, my_id, elem):
    """
    Simple safe appending function for adding an element to a list in a dict
    when the list might not have been initialised
    """
    try:
        my_dict[my_id].append(elem)
    except KeyError:
        my_dict[my_id] = [elem]


if __name__ == "__main__":
    preprocess()
