"""Module to parse biomedical knowledge bases into several types of 
objects (dictionaries and Networkx graph)."""
import csv
import networkx as nx
import obonet


class KnowledgeBase:
    """Represents a knowledge base that is loaded from a given local file."""

    def __init__(
        self,
        kb=None,
        input_format=None,
        terms_filename=None,
        edges_filename=None,
        kb_filename=None,
    ):

        self.kb = kb
        self.data_dir = f"data/kbs/{kb}/"
        self.terms_filename = terms_filename
        self.edges_filename = edges_filename
        self.kb_filename = kb_filename
        self.input_format = input_format  # obo, tsv, csv, txt

        self.root_dict = {
            "go_bp": ("GO_0008150", "biological_process"),
            "go_cc": ("GO_0005575", "cellular_component"),
            "chebi": ("CHEBI_00", "root"),
            "hp": ("HP_0000001", "All"),
            "medic": ("MESH_C", "Diseases"),
            "ctd_anatomy": ("MESH_A", "Anatomy"),
            "ctd_chemicals": ("MESH_D", "Chemicals"),
            "ctd_gene": ("", ""),
            "ncbi_taxon": ("", ""),
            "do": "DOID_4",
        }

        self.name_2_id = None
        self.id_2_info = None
        self.synonym_2_id = None
        self.id_2_synonym = None
        # self.graph = None
        self.id_2_info = None
        self.alt_id_2_id = None
        self.edges = None

        # ---------------------------------------------------------------------
        #                 Load the info about the given KB
        # ---------------------------------------------------------------------

        if self.input_format == "obo":
            self.load_obo()

        elif self.input_format == "tsv" and kb != "ncbi_taxon":
            self.load_tsv()

        elif self.kb == "ncbi_taxon":
            self.load_ncbi_taxon()

    def load_obo(self):
        """Load KBs from local .obo files (ChEBI, HP, MEDIC, GO) into
        structured dicts containing the mappings name_2_id, id_2_name,
        id_2_info, synonym_2_id, and the list of
        edges between concepts. For 'chebi', only the concepts in the subset
        3_STAR are included, which correpond to manually validated entries.

        :param kb: target ontology to load, has value 'medic', 'chebi',
            'go_bp' or 'hp'
        :type kb: str
        """

        filepath = self.data_dir
        filepaths = {
            "medic": "CTD_diseases",
            "chebi": "chebi",
            "go_bp": "go-basic",
            "go_cc": "go-basic",
            "do": "doid",
            "hp": "hp",
            "cellosaurus": "cellosaurus",
            "cl": "cl-basic",
            "uberon": "uberon-basic",
        }

        if self.kb in filepaths.keys():
            filepath += f"{filepaths[self.kb]}.obo"

        else:
            filepath += +self.kb_filename

        name_2_id = {}
        id_2_name = {}
        id_2_info = {}
        synonym_2_id = {}
        alt_id_2_id = {}

        graph = obonet.read_obo(filepath)
        edges = []

        for node in graph.nodes(data=True):
            add_node = False

            if "name" in node[1].keys():
                node_id, node_name = node[0], node[1]["name"]

                if self.kb == "go_bp":
                    # For go_bp, ensure that only Biological Process
                    # concepts are considered

                    if node[1]["namespace"] == "biological_process":
                        name_2_id[node_name] = node_id
                        id_2_name[node_id] = node_name
                        add_node = True

                elif self.kb == "go_cc":

                    if node[1]["namespace"] == "cellular_component":
                        name_2_id[node_name] = node_id
                        id_2_name[node_id] = node_name
                        add_node = True

                elif self.kb == "medic":
                    name_2_id[node_name] = node_id.replace("MESH:", "")
                    id_2_name[node_id] = node_name
                    add_node = True

                else:
                    name_2_id[node_name] = node_id
                    id_2_name[node_id] = node_name
                    add_node = True

                if "alt_id" in node[1].keys():

                    for alt_id in node[1]["alt_id"]:
                        alt_id_2_id[alt_id.replace("MESH:", "")] = node_id

                if "is_obsolete" in node[1].keys() and node[1]["is_obsolete"] == True:
                    add_node = False
                    del name_2_id[node_name]
                    del id_2_name[node_id]

                # Check parents for this node
                if "is_a" in node[1].keys() and add_node:
                    # The root node of the ontology does not
                    # have is_a relationships

                    for parent in node[1]["is_a"]:
                        # To build the edges list, consider all
                        # concepts with at least one ancestor
                        edges.append((parent.replace("MESH:", ""), node_id))

                if self.kb == "cellosaurus":

                    if "relationship" in node[1].keys() and add_node:
                        relations = node[1]["relationship"]

                        for relation in relations:

                            if relation[:13] == "derived_from ":
                                parent = relation.split("derived_from")[1][1:]
                                edges.append((parent, node_id))

                if "synonym" in node[1].keys() and add_node:
                    # Check for synonyms for node (if they exist)

                    for synonym in node[1]["synonym"]:
                        synonym_name = synonym.split('"')[1]
                        synonym_2_id[synonym_name] = node_id

        if self.kb in self.root_dict.keys():
            root_concept_name = self.root_dict[self.kb][1]
            root_id = str()

            if root_concept_name not in name_2_id.keys():
                root_id = self.root_dict[self.kb][0]
                name_2_id[root_concept_name] = root_id
                id_2_name[root_id] = root_concept_name

        # ----------------------------------------------------------------------
        # Add misssing edges between the ontology root and
        # sub-ontology root concepts
        if self.kb == "chebi":
            chemical_entity = "CHEBI_24431"
            edges.append((chemical_entity, root_id))
            role = "CHEBI_50906"
            edges.append((role, root_id))
            subatomic_particle = "CHEBI_36342"
            edges.append((subatomic_particle, root_id))
            application = "CHEBI_33232"
            edges.append((application, root_id))

        kb_graph = nx.DiGraph([edge for edge in edges])

        # Build id_2_info (KB-ID: (outdegree, indegree, num_descendants))
        for node in kb_graph.nodes:
            num_descendants = len(nx.descendants(kb_graph, node))

            id_2_info[node] = (
                kb_graph.out_degree(node),
                kb_graph.in_degree(node),
                num_descendants,
            )

        node_2_node = {}

        for edge in edges:

            if edge[0] in node_2_node:
                node_2_node[edge[0]].append(edge[1])

            else:
                node_2_node[edge[0]] = [edge[1]]

            if edge[1] in node_2_node:
                node_2_node[edge[1]].append(edge[0])

            else:
                node_2_node[edge[1]] = [edge[0]]

        self.name_2_id = name_2_id
        self.id_2_name = id_2_name
        self.id_2_info = id_2_info
        self.synonym_2_id = synonym_2_id
        self.alt_id_2_id = alt_id_2_id
        self.node_2_node = node_2_node
        self.edges = edges

    def load_tsv(self):
        """Load KBs from local .tsv files (CTD-Chemicals, CTD-Anatomy)
           into structured dicts containing the mappings name_2_id,
           id_2_info, synonym_2_id, and the list of edges
           between concepts.

        :param kb: target ontology to load, has value 'ctd_chem' or 'ctd_anat'
        :type kb: str
        """

        kb_dict = {
            "ctd_chemicals": "CTD_chemicals",
            "ctd_anatomy": "CTD_anatomy",
            "ctd_gene": "CTD_genes",
            "medic": "CTD_diseases",
        }
        filepath = f"{self.data_dir}{kb_dict[self.kb]}.tsv"

        name_2_id = {}
        id_2_name = {}
        id_2_info = {}
        synonym_2_id = {}
        id_2_synonym = {}
        child2parent = {}
        edges = []

        with open(filepath) as kb_file:
            reader = csv.reader(kb_file, delimiter="\t")
            row_count = int()

            for row in reader:
                row_count += 1

                if row_count >= 30:
                    node_name = row[0]
                    node_id = row[1].replace("MESH:", "")

                    node_parents = row[4].split("|")
                    synonyms = row[7].split("|")
                    name_2_id[node_name] = node_id
                    id_2_name[node_id] = node_name

                    id_2_synonym[node_id] = []

                    for synonym in synonyms:
                        synonym_2_id[synonym] = node_id
                        id_2_synonym[node_id].append(synonym)

                    for parent in node_parents:
                        # To build the edges list, consider
                        # all concepts with at least one ancestor
                        edges.append((parent.replace("MESH:", ""), node_id))

        root_concept_name = self.root_dict[self.kb][1]
        root_concept_id = self.root_dict[self.kb][0]
        name_2_id[root_concept_name] = root_concept_id
        id_2_name[root_concept_id] = root_concept_name

        kb_graph = nx.DiGraph([edge for edge in edges])

        # Build id_2_info (KB-ID: (outdegree, indegree, num_descendants))
        for node in kb_graph.nodes:
            num_descendants = len(nx.descendants(kb_graph, node))

            id_2_info[node] = (
                kb_graph.out_degree(node),
                kb_graph.in_degree(node),
                num_descendants,
            )

        node_2_node = {}

        for edge in edges:

            if edge[0] in node_2_node:
                node_2_node[edge[0]].append(edge[1])

            else:
                node_2_node[edge[0]] = [edge[1]]

            if edge[1] in node_2_node:
                node_2_node[edge[1]].append(edge[0])

            else:
                node_2_node[edge[1]] = [edge[0]]

        self.name_2_id = name_2_id
        self.id_2_name = id_2_name
        self.id_2_info = id_2_info
        self.synonym_2_id = synonym_2_id
        self.id_2_synonym = id_2_synonym
        self.edges = edges
        self.node_2_node = node_2_node

    def load_ncbi_taxon(self):
        """Load KBs from local .csv files (NCBITaxon) into structured dicts
            containing the mappings name_to_id, id_to_info, synonym_to_id.

        :param kb: target ontology to load, has value 'ncbi_taxon'
        :type kb: str
        """

        filepath = data_dir

        if self.kb == "ncbi_taxon":
            filepath += "NCBITAXON.csv"

        name_to_id = {}
        id_to_name = {}
        id_to_info = {}
        synonym_to_id = {}
        child_to_parent = {}
        edges = []

        with open(filepath) as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            row_count = int()

            for row in reader:
                row_count += 1

                if row_count > 1 and "NCBITAXON/" in row[0]:
                    rank_node = row[9]

                    if rank_node == "species":
                        node_name = row[1]
                        node_id = "NCBITaxon_" + row[0].split("NCBITAXON/")[1]
                        synonyms = row[2].split("|")
                        name_to_id[node_name] = node_id
                        id_to_name[node_id] = node_name

                        if row[7] != "":
                            parent_id = "NCBITaxon_" + row[7].split("NCBITAXON/")[1]
                            relationship = (node_id, parent_id)
                            edges.append(relationship)
                            child_to_parent[node_id] = parent_id

                        for synonym in synonyms:
                            synonym_to_id[synonym] = node_id

        # Create a MultiDiGraph object with only "is-a" relations
        # this will allow the further calculation of shorthest path lenght
        kb_graph = nx.DiGraph([edge for edge in edges])

        # Build id_to_info (KB-ID: (outdegree, indegree, num_descendants))
        for node in kb_graph.nodes:
            num_descendants = len(nx.descendants(kb_graph, node))

            id_to_info[node] = (
                kb_graph.out_degree(node),
                kb_graph.in_degree(node),
                num_descendants,
            )

        node_to_node = {}

        for edge in edges:

            if edge[0] in node_to_node:
                node_to_node[edge[0]].append(edge[1])

            else:
                node_to_node[edge[0]] = [edge[1]]

            if edge[1] in node_to_node:
                node_to_node[edge[1]].append(edge[0])

            else:
                node_to_node[edge[1]] = [edge[0]]

        self.name_to_id = name_to_id
        self.id_to_name = id_to_name
        self.id_to_info = id_to_info
        self.synonym_to_id = synonym_to_id
        self.graph = kb_graph
        self.node_to_node = node_to_node
