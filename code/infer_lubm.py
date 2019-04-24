import configparser
import json
import logging
from datetime import datetime

import numpy as np
from keras.models import model_from_json
from utils import deserialize
import glob
import os
import codecs
from rdflib import Graph
from graphs_words_decoder import GraphWordsDecoder

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_graph")
parser.add_argument("--format", nargs='?', default="n3")
parser.add_argument("--model_path", nargs='?')

args = parser.parse_args()

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

config = configparser.ConfigParser()
with open('../config.json', 'r') as f:
    config = json.load(f)

LUBM_ENCODING_DIRECTORY = config['LUBM']['ENCODING_PARAMS']['ENCODING_DIRECTORY']
GRAPH_WORDS_ENCODER = config['LUBM']['ENCODING_PARAMS']['GRAPH_WORDS_ENCODER']
DATASET_ENCODING_FILE = config['LUBM']['ENCODING_PARAMS']['DATASET_ENCODING_FILE']
MODEL_FOLDER = config['LUBM']['TRAINING']['MODEL_FOLDER']
INPUT_GRAPH_WORDS_CATALOGUE_FILE = config['LUBM']['ENCODING_PARAMS']['INPUT_GRAPH_WORDS_CATALOGUE_FILE']
INFERENCE_GRAPH_WORDS_CATALOGUE_FILE = config['LUBM']['ENCODING_PARAMS']['INFERENCE_GRAPH_WORDS_CATALOGUE_FILE']
EMBEDDING_MATRIX_CATALOGUE_FILE = config['LUBM']['ENCODING_PARAMS']['EMBEDDING_MATRIX_CATALOGUE_FILE']
MODEL_ARCHITECTURE = config['LUBM']["TRAINING"]["MODEL_ARCHITECTURE"]


def get_most_recent_model_path(models_paths):
    list_of_models = glob.glob(models_paths+"/*")  # * means all if need specific format then *.csv
    latest_file = max(list_of_models, key=os.path.getctime)
    return latest_file


def encode_graph(lubm_graph_words_encoder, input_graph_file, rdf_format):
    input_graph = Graph()
    input_graph.parse(data=codecs.open(input_graph_file, encoding="UTF-8").read(), format=rdf_format)
    logging.info("Input graph: \n%s\n\n%s\n%s", '-'*200, input_graph.serialize(format="nt").decode(), '-'*200)
    input_graph_encoding, resources_dictionary = lubm_graph_words_encoder.encode_graph(input_graph)
    return input_graph_encoding, resources_dictionary


def create_graph_words(graph_encoding, graph_words_catalogue, active_properties_size):
    graph_words_array = np.zeros((active_properties_size), dtype=np.int16)

    for k in sorted(graph_encoding):
        p = graph_encoding[k]
        sorted_p = sorted(p, key=lambda element: (element[0], element[1]))
        tpl = tuple(sorted_p)
        if tpl not in graph_words_catalogue:
            logging.error("Unseen graph word. Needs to compute the embedding of this graph word. Will be added in "
                          "future versions of the code.")
            exit()
        graph_words_array[k - 1] = graph_words_catalogue[tpl]

    return graph_words_array


def load_model_architecture(model_file):
    inference_model = model_from_json(open(model_file).read())
    return inference_model


def main():
    input_graph_file = args.input_graph
    model_path = args.model_path
    rdf_format = args.format
    if not input_graph_file:
        logging.error("Please specify an input graph path")
        exit()
        
    if not model_path:
        model_path = get_most_recent_model_path(MODEL_FOLDER)

    logging.info("Input graph file: %s", input_graph_file)


    logging.info("Loading LUBM input graph words vocabulary")
    input_graph_words_catalogue = deserialize(LUBM_ENCODING_DIRECTORY + INPUT_GRAPH_WORDS_CATALOGUE_FILE)

    logging.info("Loading LUBM inference graph words vocabulary")
    inference_graph_words_catalogue = deserialize(LUBM_ENCODING_DIRECTORY + INFERENCE_GRAPH_WORDS_CATALOGUE_FILE)

    logging.info("Loading LUBM graph words encoder")
    lubm_graph_words_encoder = deserialize(LUBM_ENCODING_DIRECTORY + GRAPH_WORDS_ENCODER)

    logging.info("Creating LUBM graph words decoder")
    lubm_graph_words_decoder = GraphWordsDecoder(lubm_graph_words_encoder.active_properties,
                                                 lubm_graph_words_encoder.properties_groups,
                                                 lubm_graph_words_encoder.active_classes,
                                                 inference_graph_words_catalogue)

    logging.info("Loading embeddings of LUBM graph words vocabulary")
    embedding_matrix =  deserialize(LUBM_ENCODING_DIRECTORY + EMBEDDING_MATRIX_CATALOGUE_FILE)

    logging.info("Loading inference neural network architecture from: %s", model_path+"/"+MODEL_ARCHITECTURE)
    inference_model = load_model_architecture(model_path+"/"+MODEL_ARCHITECTURE)

    logging.info("Loaded inference nn architecture: ")
    inference_model.summary()

    logging.info("Loading trained nn weights from: %s", model_path+"/best_model")
    inference_model.load_weights(model_path+"/best_model")




    input_graph_encoding, resources_dictionary = encode_graph(lubm_graph_words_encoder, input_graph_file, rdf_format)
    logging.info("The input graph is encoded as: %s", input_graph_encoding)
    input_graph_words = create_graph_words(input_graph_encoding, input_graph_words_catalogue, len(lubm_graph_words_encoder.active_properties))
    logging.info("The input graph is encoded as a sequence of graph words: %s", input_graph_words)
    input_embedded = embedding_matrix[input_graph_words]
    input_embedded = np.expand_dims(input_embedded, axis=0)
    logging.info("The shape of the embedded graph words sequence is: %s", input_embedded.shape)

    logging.info("Using the trained nn to predict inference graph words sequence")
    predicted_inference_gw = inference_model.predict(input_embedded)
    predicted_inference_gw = predicted_inference_gw.argmax(axis=-1)
    logging.info("Predicted inference graph words sequence: %s", predicted_inference_gw)
    logging.info("Decoding the inference sequence of graph words")
    predicted_inference_graph_encoding = lubm_graph_words_decoder.graph_words_to_graph_encoding(predicted_inference_gw[0])
    predicted_inference_graph = lubm_graph_words_decoder.decode_graph(predicted_inference_graph_encoding, resources_dictionary)
    logging.info("Predicted inference graph: \n%s\n\n%s\n%s", '-'*200, predicted_inference_graph.serialize(format="nt").decode(), '-'*200)


if __name__ == "__main__":
    main()


