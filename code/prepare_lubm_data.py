import configparser
import json
import logging
import os
import re
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

from graph_words_encoder import GraphWordsEncoder
from graph_words_utils import encode_graphs_list, create_graph_words
from hope_embedding_utils import embed_layer_hope, reconstruct_layer
from prepare_data_utils import get_properties_dict, get_classes_dict, get_properties_groups
from utils import serialize, deserialize

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

config = configparser.ConfigParser()
with open('../config.json', 'r') as f:
    config = json.load(f)

LUBM_ONTOLOGY_ENDPOINT = config['LUBM']['SPARQL_PARAMS']['ONTOLOGY_ENDPOINT']
LUBM_ONTOLOGY_REPOSITORY = config['LUBM']['SPARQL_PARAMS']['ONTOLOGY_REPOSITORY']
LUBM_ONTOLOGY_ENDPOINT_TYPE = config['LUBM']['SPARQL_PARAMS']['ONTOLOGY_ENDPOINT_TYPE']
PROPERTIES_QUERY = config['LUBM']['SPARQL_PARAMS']['PROPERTIES_QUERY']
QUERY_LIMIT = config['LUBM']['SPARQL_PARAMS']['QUERY_LIMIT']
PROPERTIES_DICT_FILE = config['LUBM']['ENCODING_PARAMS']['PROPERTIES_DICT_FILE']
CLASSES_QUERY = config['LUBM']['SPARQL_PARAMS']['CLASSES_QUERY']
LUBM_ENCODING_DIRECTORY = config['LUBM']['ENCODING_PARAMS']['ENCODING_DIRECTORY']
CLASSES_DICT_FILE = config['LUBM']['ENCODING_PARAMS']['CLASSES_DICT_FILE']
PROPERTIES_GROUPS_FILE = config['LUBM']['ENCODING_PARAMS']['PROPERTIES_GROUPS_FILE']
GRAPH_WORDS_ENCODER = config['LUBM']['ENCODING_PARAMS']['GRAPH_WORDS_ENCODER']
SUB_PROPERTIES_QUERY = config['LUBM']['SPARQL_PARAMS']['SUB_PROPERTIES_QUERY']
INPUT_GRAPHS_FOLDER = config['LUBM']['DATA']['INPUT_GRAPHS_FOLDER']
INFERENCE_GRAPHS_FOLDER = config['LUBM']['DATA']['INFERENCE_GRAPHS_FOLDER']
DATASET_ENCODING_FILE = config['LUBM']['ENCODING_PARAMS']['DATASET_ENCODING_FILE']
INPUT_GRAPH_WORDS_CATALOGUE_FILE = config['LUBM']['ENCODING_PARAMS']['INPUT_GRAPH_WORDS_CATALOGUE_FILE']
INFERENCE_GRAPH_WORDS_CATALOGUE_FILE = config['LUBM']['ENCODING_PARAMS']['INFERENCE_GRAPH_WORDS_CATALOGUE_FILE']
EMBEDDING_MATRIX_CATALOGUE_FILE = config['LUBM']['ENCODING_PARAMS']['EMBEDDING_MATRIX_CATALOGUE_FILE']
MAXIMUM_MATRIX_SIZE = config['LUBM']['ENCODING_PARAMS']['MAXIMUM_MATRIX_SIZE']
HOPE_EMBEDDING_SIZE = config['LUBM']['ENCODING_PARAMS']['HOPE_EMBEDDING_SIZE']


def get_lubm_properties():
    logging.info("Creating ResourceDictionary for LUBM properties")
    return get_properties_dict(LUBM_ENCODING_DIRECTORY + PROPERTIES_DICT_FILE, PROPERTIES_QUERY,
                               LUBM_ONTOLOGY_REPOSITORY, LUBM_ONTOLOGY_ENDPOINT,
                               LUBM_ONTOLOGY_ENDPOINT_TYPE, QUERY_LIMIT)


def get_lubm_classes():
    logging.info("Creating ResourceDictionary for LUBM classes")
    return get_classes_dict(LUBM_ENCODING_DIRECTORY + CLASSES_DICT_FILE, CLASSES_QUERY, LUBM_ONTOLOGY_REPOSITORY,
                            LUBM_ONTOLOGY_ENDPOINT,
                            LUBM_ONTOLOGY_ENDPOINT_TYPE, QUERY_LIMIT)


def get_lubm_properties_groups(lubm_properties_dict):
    logging.info("Creating ResourceDictionary for LUBM properties groups")
    return get_properties_groups(LUBM_ENCODING_DIRECTORY + PROPERTIES_GROUPS_FILE, SUB_PROPERTIES_QUERY,
                                 LUBM_ONTOLOGY_REPOSITORY,
                                 LUBM_ONTOLOGY_ENDPOINT,
                                 LUBM_ONTOLOGY_ENDPOINT_TYPE, lubm_properties_dict, QUERY_LIMIT)


def get_lubm_graph_type(s):
    s = s.split('/')[-1]
    s = s.split('_')[-1]
    m = re.search("\d", s)
    return s[:m.start()]


def get_lubm_files_df():
    logging.info("Creating dataframe for LUBM1 input/inference pairs")
    rdf_files = []
    for input_graph_path in tqdm(sorted(glob(INPUT_GRAPHS_FOLDER + "*"))):
        input_graph_file = os.path.basename(input_graph_path)
        inference_path = INFERENCE_GRAPHS_FOLDER + input_graph_file
        graph_type = get_lubm_graph_type(input_graph_path)
        rdf_pair = {"input_graph_file": input_graph_path, "inference_file": inference_path, "graph_type": graph_type}
        rdf_files.append(rdf_pair)
    files_df = pd.DataFrame.from_dict(rdf_files)
    return files_df


def encode_lubm_dataset():
    try:
        dataset_encoding = pd.read_pickle(LUBM_ENCODING_DIRECTORY + DATASET_ENCODING_FILE)
    except:
        dataset_encoding = []
    lubm_graph_words_encoder = deserialize(LUBM_ENCODING_DIRECTORY + GRAPH_WORDS_ENCODER)
    if len(dataset_encoding) and lubm_graph_words_encoder:
        logging.info("Loading pre-created LUBM1 dataset")
        return dataset_encoding, lubm_graph_words_encoder

    lubm_classes = get_lubm_classes()
    lubm_properties = get_lubm_properties()
    lubm_properties_groups = get_lubm_properties_groups(lubm_properties)

    lubm_graph_words_encoder = GraphWordsEncoder(lubm_properties, lubm_properties_groups, lubm_classes)

    lubm_files_df = get_lubm_files_df()
    logging.info("Encoding LUBM1 dataset")
    lubm_files_df = encode_graphs_list(lubm_graph_words_encoder, lubm_files_df, "nt")
    lubm_files_df.to_pickle(LUBM_ENCODING_DIRECTORY + DATASET_ENCODING_FILE)
    serialize(lubm_graph_words_encoder, LUBM_ENCODING_DIRECTORY + GRAPH_WORDS_ENCODER)
    return lubm_files_df, lubm_graph_words_encoder


def create_lubm_graph_words(lubm_dataset, lubm_graph_words_encoder):
    logging.info("Creating graph words for LUBM1")
    # lubm_dataset, lubm_graph_words_encoder = encode_lubm_dataset()
    input_graph_words, input_graph_words_catalogue = create_graph_words(lubm_dataset['input_graph_encoding'],
                                                                        len(lubm_graph_words_encoder.active_properties))
    inference_graph_words, inference_graph_words_catalogue = create_graph_words(
        lubm_dataset['inference_graph_encoding'], len(lubm_graph_words_encoder.active_properties))
    lubm_dataset['input_graph_words'] = input_graph_words.tolist()
    lubm_dataset['inference_graph_words'] = inference_graph_words.tolist()
    serialize(input_graph_words_catalogue, LUBM_ENCODING_DIRECTORY + INPUT_GRAPH_WORDS_CATALOGUE_FILE)
    serialize(inference_graph_words_catalogue, LUBM_ENCODING_DIRECTORY + INFERENCE_GRAPH_WORDS_CATALOGUE_FILE)
    lubm_dataset.to_pickle(LUBM_ENCODING_DIRECTORY + DATASET_ENCODING_FILE)
    return lubm_dataset, input_graph_words_catalogue, inference_graph_words_catalogue


def create_matrix_embedding_catalogue(input_graph_words_catalogue, lubm_graph_words_encoder):
    # input_graph_words_catalogue = deserialize(LUBM_ENCODING_DIRECTORY + INPUT_GRAPH_WORDS_CATALOGUE_FILE)
    # lubm_graph_words_encoder = deserialize(LUBM_ENCODING_DIRECTORY + GRAPH_WORDS_ENCODER)
    matrix_embedding_catalogue = deserialize(LUBM_ENCODING_DIRECTORY + EMBEDDING_MATRIX_CATALOGUE_FILE)
    if len(matrix_embedding_catalogue):
        logging.info("Loading embeddings for the graph words")

        return matrix_embedding_catalogue

    logging.info("Creating embeddings for the graph words")

    matrix_embedding_catalogue = np.zeros(
        (len(input_graph_words_catalogue) + 1, MAXIMUM_MATRIX_SIZE * HOPE_EMBEDDING_SIZE))

    for graph_word in tqdm(input_graph_words_catalogue):
        layer_adjacency_matrix = lubm_graph_words_encoder.layer_to_np(graph_word,
                                                                      len(lubm_graph_words_encoder.active_classes),
                                                                      MAXIMUM_MATRIX_SIZE)
        x1, x2 = embed_layer_hope(layer_adjacency_matrix, int(HOPE_EMBEDDING_SIZE / 2))
        reconstructed_layer = reconstruct_layer(x1 * 100, x2 * 100, 0.1)
        matrix_embedding_catalogue[input_graph_words_catalogue[graph_word]] = np.concatenate(
            (x1.flatten() * 100, x2.flatten() * 100), axis=0)
        if not np.allclose(layer_adjacency_matrix, reconstructed_layer):
            print("Reconstruction layer failed, increase hope embedding size. Current embedding size = %",
                  HOPE_EMBEDDING_SIZE)

    serialize(matrix_embedding_catalogue, LUBM_ENCODING_DIRECTORY + EMBEDDING_MATRIX_CATALOGUE_FILE)


if __name__ == "__main__":
    lubm_dataset, lubm_graph_words_encoder = encode_lubm_dataset()
    lubm_dataset, input_graph_words_catalogue, inference_graph_words_catalogue = create_lubm_graph_words(lubm_dataset,
                                                                                                         lubm_graph_words_encoder)

    create_matrix_embedding_catalogue(input_graph_words_catalogue, lubm_graph_words_encoder)