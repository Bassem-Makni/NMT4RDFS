import codecs

import numpy as np
from rdflib import Graph
from tqdm import tqdm

from utils import ResourceDictionary


def encode_graphs_list(graph_words_encoder, files_df, rdf_format):
    input_graphs_encodings = []
    inference_graphs_encodings = []
    resources_dictionaries = []
    for row in tqdm(files_df.itertuples(), total=len(files_df)):
        input_graph_encoding, resources_dictionary = encode_input_graph(graph_words_encoder, row, rdf_format)
        inference_graph_encoding = encode_inference_graph(graph_words_encoder, row, resources_dictionary, rdf_format)
        input_graphs_encodings.append(input_graph_encoding)
        inference_graphs_encodings.append(inference_graph_encoding)
        resources_dictionaries.append(resources_dictionary)
    files_df['resources_dictionary'] = resources_dictionaries
    files_df['input_graph_encoding'] = input_graphs_encodings
    files_df['inference_graph_encoding'] = inference_graphs_encodings
    return files_df


def encode_input_graph(graph_words_encoder, file_names, rdf_format):
    input_graph = Graph()
    input_file = file_names.input_graph_file
    input_graph.parse(data=codecs.open(input_file, encoding="UTF-8").read(), format=rdf_format)
    input_graph_encoding, resources_dictionary = graph_words_encoder.encode_graph(input_graph)
    return input_graph_encoding, resources_dictionary


def encode_inference_graph(graph_words_encoder, file_names, resources_dictionary, rdf_format):
    inference_graph = Graph()
    inference_file = file_names.inference_file
    inference_graph.parse(data=codecs.open(inference_file, encoding="UTF-8").read(), format=rdf_format)
    inference_graph_encoding, _ = graph_words_encoder.encode_graph(inference_graph, resources_dictionary,
                                                                   inference=True)
    return inference_graph_encoding


def create_graph_words(graph_encodings, active_properties_size):
    graph_words_array = np.zeros((len(graph_encodings), active_properties_size), dtype=np.int16)
    graph_words_catalogue = ResourceDictionary()
    for graph_index, graph_encoding in enumerate(graph_encodings):
        for k in sorted(graph_encoding):
            p = graph_encoding[k]
            sorted_p = sorted(p, key=lambda element: (element[0], element[1]))
            tpl = tuple(sorted_p)
            graph_words_catalogue.add(tpl)
            graph_words_array[graph_index, k - 1] = graph_words_catalogue[tpl]
    return graph_words_array, graph_words_catalogue
